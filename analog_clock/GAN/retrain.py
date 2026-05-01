#!/usr/bin/env python3
"""
GAN retraining script with anti-vanishing-gradient training.

End-to-end workflow:
  0. Pre-flight validation (GPU, disk, memory, model sanity)
  1. Generate synthetic dataset (with "in the wild" realism effects)
  2. Train GAN with stabilised training (spectral-norm D, R1 gradient penalty,
     label smoothing, linear LR decay, D trained every step)
  3. Save weights to canonical path
  4. Run evaluation (val-set L1/PSNR + sample grid)

Supports both pipelines:
  --model sketch     ->  Sketch-cGAN (Pix2Pix, 6-channel input)
  --model inpainting ->  Inpainting GAN (4-channel input)

Logging:
  - Console output (INFO level)
  - File log at <data-dir>/retrain_<model>_<timestamp>.log (DEBUG level)

Example:
  python -m analog_clock.GAN.retrain --model sketch --epochs 100 --samples 20000
  python -m analog_clock.GAN.retrain --model inpainting --epochs 100 --samples 20000
  python -m analog_clock.GAN.retrain --model sketch --device cuda --r1-every 16
"""

from __future__ import annotations

import argparse
import datetime
import logging
import math
import os
import platform
import shutil
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import DataLoader, Dataset

# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from analog_clock.GAN.inpainting.generator_model import (
    InpaintDiscriminator,
    InpaintGenerator,
)
from analog_clock.GAN.sketch.generator_model import GeneratorUNet, SketchDiscriminator
from analog_clock.shared.config import InpaintConfig, SketchConfig

logger = logging.getLogger(__name__)

# Default: apply R1 penalty every 16 steps (StyleGAN2 "lazy regularization").
# torch.autograd.grad with create_graph=True is ~4x slower on MPS than CPU,
# so running it every step dominates training time on Apple Silicon.
R1_EVERY_N_STEPS: int = 16


# ============================================================================
# Logging setup
# ============================================================================


def _setup_logging(log_dir: Path, model_name: str) -> Path:
    """Configure dual console + file logging.

    Returns the path to the log file.
    """
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"retrain_{model_name}_{timestamp}.log"

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)

    # Clear any existing handlers (e.g. from basicConfig)
    root_logger.handlers.clear()

    # Console handler (INFO)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_fmt = logging.Formatter(
        "%(asctime)s %(levelname)-7s %(message)s", datefmt="%H:%M:%S"
    )
    console_handler.setFormatter(console_fmt)
    root_logger.addHandler(console_handler)

    # File handler (DEBUG — captures everything)
    file_handler = logging.FileHandler(str(log_path), mode="w", encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_fmt = logging.Formatter(
        "%(asctime)s %(levelname)-7s [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler.setFormatter(file_fmt)
    root_logger.addHandler(file_handler)

    logger.info("Log file: %s", log_path)
    return log_path


# ============================================================================
# Pre-flight validation
# ============================================================================


def _preflight_check(
    device: torch.device,
    model_name: str,
    batch_size: int,
    data_dir: str,
) -> None:
    """Validate system readiness before dataset generation or training.

    Checks: Python version, PyTorch build, GPU/CUDA availability and health,
    disk space, RAM, model instantiation, and a forward-pass sanity test.
    Logs everything at DEBUG level (file) and WARN/ERROR at console level.
    """
    logger.info("=" * 60)
    logger.info("PRE-FLIGHT VALIDATION")
    logger.info("=" * 60)
    errors: list[str] = []
    warnings: list[str] = []

    # ── 1. System info ────────────────────────────────────────────────────
    logger.info("OS:              %s %s", platform.system(), platform.release())
    logger.info("Python:          %s", sys.version.split()[0])
    logger.info("PyTorch:         %s", torch.__version__)
    logger.info("Torchvision:     %s", _safe_import_version("torchvision"))
    logger.info("NumPy:           %s", np.__version__)
    logger.info("PIL/Pillow:      %s", _safe_import_version("PIL"))

    try:
        import psutil
        mem = psutil.virtual_memory()
        logger.info("RAM:             %.1f GB total, %.1f GB available (%.0f%% used)",
                     mem.total / 1e9, mem.available / 1e9, mem.percent)
        if mem.available < 4e9:
            warnings.append(f"Low RAM: only {mem.available / 1e9:.1f} GB available")
    except ImportError:
        logger.debug("psutil not installed — skipping RAM check")

    # ── 2. Disk space ────────────────────────────────────────────────────
    data_path = Path(data_dir).resolve()
    data_path.mkdir(parents=True, exist_ok=True)
    disk = shutil.disk_usage(str(data_path))
    free_gb = disk.free / 1e9
    logger.info("Disk free:       %.1f GB (at %s)", free_gb, data_path)
    if free_gb < 10:
        warnings.append(f"Low disk space: {free_gb:.1f} GB free (recommend 10+ GB)")
    if free_gb < 2:
        errors.append(f"Critically low disk: {free_gb:.1f} GB free")

    # ── 3. GPU / CUDA validation ──────────────────────────────────────────
    logger.info("Target device:   %s", device)
    logger.info("CUDA available:  %s", torch.cuda.is_available())
    logger.info("CUDA version:    %s", getattr(torch.version, "cuda", "N/A"))
    logger.info("cuDNN version:   %s", torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else "N/A")
    logger.info("cuDNN enabled:   %s", torch.backends.cudnn.enabled if torch.backends.cudnn.is_available() else "N/A")
    logger.info("MPS available:   %s", torch.backends.mps.is_available())

    if device.type == "cuda":
        if not torch.cuda.is_available():
            errors.append(
                "Device set to CUDA but torch.cuda.is_available() is False. "
                "Install CUDA PyTorch: pip install torch torchvision "
                "--index-url https://download.pytorch.org/whl/cu124"
            )
        else:
            gpu_count = torch.cuda.device_count()
            logger.info("GPU count:       %d", gpu_count)
            for i in range(gpu_count):
                props = torch.cuda.get_device_properties(i)
                total_mem = props.total_mem / 1e9
                logger.info(
                    "  GPU %d: %s  |  %.1f GB  |  CC %d.%d  |  %d SMs",
                    i, props.name, total_mem,
                    props.major, props.minor, props.multi_processor_count,
                )
                if total_mem < 4:
                    warnings.append(f"GPU {i} has only {total_mem:.1f} GB VRAM — may OOM with batch_size={batch_size}")

            # Quick CUDA health check: alloc + matmul
            try:
                logger.debug("Running CUDA health check (alloc + matmul)...")
                a = torch.randn(256, 256, device="cuda")
                b = torch.randn(256, 256, device="cuda")
                c = a @ b
                torch.cuda.synchronize()
                del a, b, c
                torch.cuda.empty_cache()
                logger.info("CUDA health:     OK (alloc + matmul passed)")
            except RuntimeError as e:
                errors.append(f"CUDA health check failed: {e}")

            # cuDNN benchmark (faster convolutions for fixed input sizes)
            if torch.backends.cudnn.is_available():
                torch.backends.cudnn.benchmark = True
                logger.info("cuDNN benchmark: enabled (fixed 256x256 inputs)")

    elif device.type == "mps":
        if not torch.backends.mps.is_available():
            errors.append("Device set to MPS but torch.backends.mps.is_available() is False")
        else:
            try:
                t = torch.randn(64, 64, device="mps")
                _ = t @ t
                torch.mps.synchronize()
                logger.info("MPS health:      OK")
            except RuntimeError as e:
                errors.append(f"MPS health check failed: {e}")

    elif device.type == "cpu":
        warnings.append(
            "Training on CPU — this will be very slow. "
            "Consider using --device cuda or --device mps"
        )

    # ── 4. Model sanity check ────────────────────────────────────────────
    logger.info("Model:           %s", model_name)
    logger.info("Batch size:      %d", batch_size)
    try:
        logger.debug("Instantiating models on %s for sanity check...", device)
        if model_name == "sketch":
            g = GeneratorUNet().to(device)
            d = SketchDiscriminator().to(device)
            dummy_in = torch.randn(1, 6, 256, 256, device=device)
            with torch.no_grad():
                out = g(dummy_in)
            assert out.shape == (1, 3, 256, 256), f"G output shape mismatch: {out.shape}"
            g_params = sum(p.numel() for p in g.parameters())
            d_params = sum(p.numel() for p in d.parameters())
        else:
            g = InpaintGenerator().to(device)
            d = InpaintDiscriminator().to(device)
            dummy_in = torch.randn(1, 4, 256, 256, device=device)
            with torch.no_grad():
                out = g(dummy_in)
            assert out.shape == (1, 3, 256, 256), f"G output shape mismatch: {out.shape}"
            g_params = sum(p.numel() for p in g.parameters())
            d_params = sum(p.numel() for p in d.parameters())

        logger.info("G params:        %s", f"{g_params:,}")
        logger.info("D params:        %s", f"{d_params:,}")
        logger.info("Model sanity:    OK (forward pass verified)")

        # Estimate VRAM usage
        param_bytes = (g_params + d_params) * 4  # float32
        # Rough estimate: activations ~3x params for batch_size=1, scales linearly
        est_vram_gb = (param_bytes * 3 * batch_size) / 1e9
        logger.info("Est. VRAM usage: ~%.1f GB (batch_size=%d)", est_vram_gb, batch_size)

        if device.type == "cuda":
            gpu_mem = torch.cuda.get_device_properties(0).total_mem / 1e9
            if est_vram_gb > gpu_mem * 0.9:
                warnings.append(
                    f"Estimated VRAM ({est_vram_gb:.1f} GB) may exceed GPU memory ({gpu_mem:.1f} GB). "
                    f"Consider reducing --batch-size"
                )

        del g, d, dummy_in, out
        if device.type == "cuda":
            torch.cuda.empty_cache()
    except Exception as e:
        errors.append(f"Model sanity check failed: {e}")

    # ── 5. TensorBoard check ────────────────────────────────────────────
    try:
        from torch.utils.tensorboard import SummaryWriter
        logger.info("TensorBoard:     available")
    except ImportError:
        logger.info("TensorBoard:     not installed (pip install tensorboard)")

    # ── Report ────────────────────────────────────────────────────────────
    logger.info("-" * 60)
    if warnings:
        for w in warnings:
            logger.warning("WARN: %s", w)
    if errors:
        for e in errors:
            logger.error("ERROR: %s", e)
        logger.error("Pre-flight validation FAILED — aborting.")
        sys.exit(1)
    else:
        logger.info("Pre-flight validation PASSED")
    logger.info("=" * 60)


def _safe_import_version(module_name: str) -> str:
    """Get version string for a module, returning 'N/A' on failure."""
    try:
        import importlib
        mod = importlib.import_module(module_name)
        return getattr(mod, "__version__", getattr(mod, "VERSION", "installed"))
    except ImportError:
        return "not installed"


# ============================================================================
# Device and worker helpers
# ============================================================================


def _select_device(override: str | None = None) -> torch.device:
    """Pick the best available device, with optional manual override."""
    if override:
        dev = torch.device(override)
        logger.info("Device (manual override): %s", dev)
        return dev

    if torch.cuda.is_available():
        dev = torch.device("cuda")
    elif torch.backends.mps.is_available():
        dev = torch.device("mps")
    else:
        dev = torch.device("cpu")
    logger.info("Device (auto-detected): %s", dev)
    return dev


def _optimal_num_workers(device: torch.device) -> int:
    """Choose DataLoader num_workers based on device/OS.

    - CUDA: use half the CPU cores (capped at 8) for efficient GPU feeding
    - MPS: use 2 workers — more can cause fork-safety issues with MPS tensors
    - CPU: 0 (main-process loading avoids IPC overhead when CPU is the bottleneck)
    """
    if device.type == "cuda":
        return min(os.cpu_count() or 4, 8) // 2
    if device.type == "mps":
        return 2
    return 0


# ============================================================================
# Dataset classes
# ============================================================================


class SketchDataset(Dataset):
    """(source, sketch, target) triplets for sketch-cGAN."""

    def __init__(self, root_dir: str, mode: str = "train", transform=None) -> None:
        self.transform = transform
        self.source_dir = os.path.join(root_dir, mode, "source")
        self.sketch_dir = os.path.join(root_dir, mode, "sketch")
        self.target_dir = os.path.join(root_dir, mode, "target")
        self.files = sorted(
            f for f in os.listdir(self.source_dir) if f.endswith((".png", ".jpg"))
        )

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int):
        fn = self.files[idx]
        src = Image.open(os.path.join(self.source_dir, fn)).convert("RGB")
        skc = Image.open(os.path.join(self.sketch_dir, fn)).convert("RGB")
        tgt = Image.open(os.path.join(self.target_dir, fn)).convert("RGB")
        if self.transform:
            src, skc, tgt = self.transform(src), self.transform(skc), self.transform(tgt)
        return src, skc, tgt


class InpaintDataset(Dataset):
    """(source, mask, target) triplets for inpainting GAN."""

    def __init__(self, root_dir: str, mode: str = "train", transform=None) -> None:
        self.transform = transform
        self.source_dir = os.path.join(root_dir, mode, "source")
        self.mask_dir = os.path.join(root_dir, mode, "mask")
        self.target_dir = os.path.join(root_dir, mode, "target")
        self.files = sorted(
            f for f in os.listdir(self.source_dir) if f.endswith((".png", ".jpg"))
        )

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int):
        fn = self.files[idx]
        src = Image.open(os.path.join(self.source_dir, fn)).convert("RGB")
        mask = Image.open(os.path.join(self.mask_dir, fn)).convert("L")
        tgt = Image.open(os.path.join(self.target_dir, fn)).convert("RGB")
        if self.transform:
            src, tgt = self.transform(src), self.transform(tgt)
            mask = transforms.ToTensor()(mask)
        return src, mask, tgt


# ============================================================================
# Training utilities
# ============================================================================


def _weights_init_normal(m: nn.Module) -> None:
    classname = m.__class__.__name__
    if "Conv" in classname and hasattr(m, "weight"):
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif "BatchNorm" in classname and hasattr(m, "weight") and m.weight is not None:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0.0)


def _r1_gradient_penalty(
    discriminator: nn.Module,
    real_pred_inputs: list[torch.Tensor],
    real_output: torch.Tensor,
    gamma: float = 10.0,
) -> torch.Tensor:
    """
    R1 gradient penalty (Mescheder et al. 2018).

    Penalises the gradient magnitude of D on *real* samples, stabilising
    training and preventing vanishing gradients.
    """
    grad_outputs = torch.ones_like(real_output, requires_grad=False)
    gradients = torch.autograd.grad(
        outputs=real_output,
        inputs=real_pred_inputs,
        grad_outputs=grad_outputs,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )
    grad_norm_sq = sum(g.pow(2).sum() for g in gradients)
    return gamma * 0.5 * grad_norm_sq / real_output.shape[0]


def _linear_lr_lambda(epoch: int, total_epochs: int, decay_start_frac: float = 0.5) -> float:
    """Linear LR decay starting at decay_start_frac of total training."""
    decay_start = int(total_epochs * decay_start_frac)
    if epoch < decay_start:
        return 1.0
    return max(0.0, 1.0 - (epoch - decay_start) / (total_epochs - decay_start + 1))


def _psnr(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Peak Signal-to-Noise Ratio (in [-1,1] space, converted to [0,1])."""
    pred01 = (pred * 0.5 + 0.5).clamp(0, 1)
    tgt01 = (target * 0.5 + 0.5).clamp(0, 1)
    mse = torch.mean((pred01 - tgt01) ** 2).item()
    if mse < 1e-10:
        return 50.0
    return 10 * math.log10(1.0 / mse)


def _save_sample_grid(
    images: list[torch.Tensor],
    path: str,
    nrow: int = 4,
) -> None:
    """Save a row of images as a single PNG for visual inspection."""
    try:
        import torchvision.utils as vutils
        grid = vutils.make_grid(torch.cat(images, dim=0), nrow=nrow, normalize=True, value_range=(-1, 1))
        img = grid.permute(1, 2, 0).cpu().numpy()
        img = (img * 255).clip(0, 255).astype(np.uint8)
        Image.fromarray(img).save(path)
    except Exception as e:
        logger.warning("Could not save sample grid: %s", e)


def _log_gpu_memory(prefix: str = "") -> None:
    """Log current GPU memory usage (CUDA only)."""
    if not torch.cuda.is_available():
        return
    allocated = torch.cuda.memory_allocated() / 1e9
    reserved = torch.cuda.memory_reserved() / 1e9
    max_allocated = torch.cuda.max_memory_allocated() / 1e9
    logger.debug(
        "%sGPU mem: %.2f GB allocated, %.2f GB reserved, %.2f GB peak",
        f"[{prefix}] " if prefix else "", allocated, reserved, max_allocated,
    )


def _get_tb_writer(log_dir: Path, model_name: str):
    """Try to create a TensorBoard SummaryWriter, return None on failure."""
    try:
        from torch.utils.tensorboard import SummaryWriter
        tb_dir = log_dir / f"tensorboard_{model_name}"
        tb_dir.mkdir(parents=True, exist_ok=True)
        writer = SummaryWriter(log_dir=str(tb_dir))
        logger.info("TensorBoard logs: %s", tb_dir)
        logger.info("  View with: tensorboard --logdir %s", tb_dir)
        return writer
    except ImportError:
        logger.info("TensorBoard not available — skipping (pip install tensorboard)")
        return None


# ============================================================================
# Data generation
# ============================================================================


def _generate_sketch_data(n_samples: int, data_dir: str) -> None:
    from analog_clock.GAN.sketch.dataset_generator.dataset_generator import ClockDatasetGenerator

    cfg = replace(SketchConfig(), N_SAMPLES=n_samples, OUTPUT_DIR=data_dir,
                  CGAN_DIR=f"{data_dir}/cgan", YOLO_DIR=f"{data_dir}/yolo")
    gen = ClockDatasetGenerator(config=cfg)
    gen.generate_dataset()


def _generate_inpaint_data(n_samples: int, data_dir: str) -> None:
    from analog_clock.GAN.inpainting.dataset_generator.dataset_generator import ClockDatasetGenerator

    cfg = replace(InpaintConfig(), N_SAMPLES=n_samples, OUTPUT_DIR=data_dir,
                  INPAINT_DIR=f"{data_dir}/inpainting", YOLO_DIR=f"{data_dir}/yolo_seg")
    gen = ClockDatasetGenerator(config=cfg)
    gen.generate_dataset()


# ============================================================================
# Training — Sketch cGAN
# ============================================================================


def train_sketch_cgan(
    data_dir: str,
    epochs: int,
    batch_size: int,
    lr: float,
    l1_lambda: float,
    r1_gamma: float,
    device: torch.device,
    resume_path: str | None = None,
    r1_every: int = R1_EVERY_N_STEPS,
    tb_writer=None,
) -> tuple[nn.Module, dict]:
    """
    Train sketch-cGAN with stabilised GAN training.

    Returns (generator, metrics_dict).
    """
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3),
    ])

    nw = _optimal_num_workers(device)
    train_ds = SketchDataset(data_dir, "train", transform)
    val_ds = SketchDataset(data_dir, "val", transform)
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=nw, pin_memory=(device.type == "cuda"), persistent_workers=(nw > 0),
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=nw, pin_memory=(device.type == "cuda"), persistent_workers=(nw > 0),
    )

    generator = GeneratorUNet().to(device)
    discriminator = SketchDiscriminator().to(device)

    if resume_path and Path(resume_path).exists():
        logger.info("Resuming G from %s", resume_path)
        generator.load_state_dict(torch.load(resume_path, map_location=device, weights_only=True))
    else:
        generator.apply(_weights_init_normal)
    # D uses spectral norm — skip normal init (it conflicts)

    criterion_gan = nn.MSELoss()
    criterion_l1 = nn.L1Loss()

    opt_g = optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
    opt_d = optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))

    sched_g = optim.lr_scheduler.LambdaLR(opt_g, lambda ep: _linear_lr_lambda(ep, epochs))
    sched_d = optim.lr_scheduler.LambdaLR(opt_d, lambda ep: _linear_lr_lambda(ep, epochs))

    logger.info(
        "Training sketch-cGAN: %d epochs, %d train / %d val samples, "
        "batch=%d, lr=%.5f, workers=%d, R1 every %d steps",
        epochs, len(train_ds), len(val_ds), batch_size, lr, nw, r1_every,
    )
    _log_gpu_memory("pre-train")

    # Label smoothing: real=0.9
    REAL_LABEL = 0.9

    history: dict[str, list] = {"g_loss": [], "d_loss": [], "val_l1": [], "val_psnr": []}
    out_dir = Path(data_dir).parent
    global_step = 0
    train_start = time.time()

    for epoch in range(epochs):
        generator.train()
        discriminator.train()
        t0 = time.time()
        epoch_g, epoch_d = 0.0, 0.0
        n_batches = 0

        for src, skc, tgt in train_loader:
            src, skc, tgt = src.to(device), skc.to(device), tgt.to(device)

            # --- Train Generator ---
            opt_g.zero_grad()
            gen_input = torch.cat((src, skc), 1)
            fake_tgt = generator(gen_input)
            pred_fake = discriminator(src, skc, fake_tgt)
            valid = torch.full_like(pred_fake, REAL_LABEL)
            loss_gan = criterion_gan(pred_fake, valid)
            loss_pixel = criterion_l1(fake_tgt, tgt)
            loss_g = loss_gan + l1_lambda * loss_pixel
            loss_g.backward()
            opt_g.step()

            # --- Train Discriminator (every step) ---
            opt_d.zero_grad()
            apply_r1 = (global_step % r1_every == 0)
            # Real — only need grads when computing R1
            if apply_r1:
                tgt.requires_grad_(True)
            pred_real = discriminator(src, skc, tgt)
            loss_real = criterion_gan(pred_real, torch.full_like(pred_real, REAL_LABEL))
            # R1 penalty (lazy: every r1_every steps, scaled up to compensate)
            if apply_r1:
                r1_pen = _r1_gradient_penalty(discriminator, [tgt], pred_real, gamma=r1_gamma) * r1_every
            else:
                r1_pen = torch.tensor(0.0, device=device)
            # Fake
            pred_fake_d = discriminator(src, skc, fake_tgt.detach())
            loss_fake = criterion_gan(pred_fake_d, torch.zeros_like(pred_fake_d))
            loss_d = 0.5 * (loss_real + loss_fake) + r1_pen
            loss_d.backward()
            opt_d.step()
            if apply_r1:
                tgt.requires_grad_(False)

            epoch_g += loss_g.item()
            epoch_d += loss_d.item()
            n_batches += 1
            global_step += 1

            # Per-step debug logging
            if global_step % 100 == 0:
                logger.debug(
                    "  step %d — G: %.4f  D: %.4f  R1: %s",
                    global_step, loss_g.item(), loss_d.item(),
                    f"{r1_pen.item():.4f}" if apply_r1 else "skip",
                )
                _log_gpu_memory(f"step-{global_step}")

        sched_g.step()
        sched_d.step()

        avg_g = epoch_g / max(n_batches, 1)
        avg_d = epoch_d / max(n_batches, 1)
        history["g_loss"].append(avg_g)
        history["d_loss"].append(avg_d)

        # Validation
        val_l1, val_psnr, val_n = 0.0, 0.0, 0
        generator.eval()
        with torch.no_grad():
            for src, skc, tgt in val_loader:
                src, skc, tgt = src.to(device), skc.to(device), tgt.to(device)
                fake = generator(torch.cat((src, skc), 1))
                val_l1 += criterion_l1(fake, tgt).item() * src.size(0)
                val_psnr += _psnr(fake, tgt) * src.size(0)
                val_n += src.size(0)
        val_l1 /= max(val_n, 1)
        val_psnr /= max(val_n, 1)
        history["val_l1"].append(val_l1)
        history["val_psnr"].append(val_psnr)

        elapsed = time.time() - t0
        eta_s = elapsed * (epochs - epoch - 1)
        logger.info(
            "Epoch %3d/%d — G: %.4f  D: %.4f  val_L1: %.4f  val_PSNR: %.1f dB  "
            "lr: %.6f  (%.0fs, ETA %dh%02dm)",
            epoch + 1, epochs, avg_g, avg_d, val_l1, val_psnr,
            sched_g.get_last_lr()[0], elapsed,
            int(eta_s // 3600), int(eta_s % 3600 // 60),
        )

        # TensorBoard logging
        if tb_writer is not None:
            tb_writer.add_scalar("train/G_loss", avg_g, epoch + 1)
            tb_writer.add_scalar("train/D_loss", avg_d, epoch + 1)
            tb_writer.add_scalar("val/L1", val_l1, epoch + 1)
            tb_writer.add_scalar("val/PSNR", val_psnr, epoch + 1)
            tb_writer.add_scalar("train/lr", sched_g.get_last_lr()[0], epoch + 1)
            if device.type == "cuda":
                tb_writer.add_scalar(
                    "system/gpu_mem_GB",
                    torch.cuda.max_memory_allocated() / 1e9,
                    epoch + 1,
                )

        if (epoch + 1) % 10 == 0:
            ckpt = out_dir / f"generator_{epoch + 1}.pth"
            torch.save(generator.state_dict(), str(ckpt))
            logger.info("  Checkpoint saved: %s", ckpt)
            # Save sample grid
            generator.eval()
            with torch.no_grad():
                sample_batch = next(iter(val_loader))
                s, sk, t = [x[:4].to(device) for x in sample_batch]
                fake_s = generator(torch.cat((s, sk), 1))
                grid_path = str(out_dir / f"samples_epoch_{epoch+1}.png")
                _save_sample_grid([s, sk, fake_s, t], grid_path)
                # Also log to TensorBoard
                if tb_writer is not None:
                    import torchvision.utils as vutils
                    grid = vutils.make_grid(
                        torch.cat([s, sk, fake_s, t], dim=0),
                        nrow=4, normalize=True, value_range=(-1, 1),
                    )
                    tb_writer.add_image("samples/sketch", grid, epoch + 1)

    total_time = time.time() - train_start
    logger.info(
        "Training complete: %d epochs in %dh %dm %ds",
        epochs, int(total_time // 3600), int(total_time % 3600 // 60), int(total_time % 60),
    )
    _log_gpu_memory("post-train")
    return generator, history


# ============================================================================
# Training — Inpainting GAN
# ============================================================================


def train_inpainting_gan(
    data_dir: str,
    epochs: int,
    batch_size: int,
    lr: float,
    l1_lambda: float,
    mask_weight: float,
    r1_gamma: float,
    device: torch.device,
    resume_path: str | None = None,
    r1_every: int = R1_EVERY_N_STEPS,
    tb_writer=None,
) -> tuple[nn.Module, dict]:
    """
    Train inpainting GAN with stabilised training.

    Returns (generator, metrics_dict).
    """
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3),
    ])

    nw = _optimal_num_workers(device)
    train_ds = InpaintDataset(data_dir, "train", transform)
    val_ds = InpaintDataset(data_dir, "val", transform)
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=nw, pin_memory=(device.type == "cuda"), persistent_workers=(nw > 0),
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=nw, pin_memory=(device.type == "cuda"), persistent_workers=(nw > 0),
    )

    generator = InpaintGenerator().to(device)
    discriminator = InpaintDiscriminator().to(device)

    if resume_path and Path(resume_path).exists():
        logger.info("Resuming G from %s", resume_path)
        generator.load_state_dict(torch.load(resume_path, map_location=device, weights_only=True))
    else:
        generator.apply(_weights_init_normal)

    criterion_gan = nn.MSELoss()
    criterion_pixel = nn.L1Loss(reduction="none")

    opt_g = optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
    opt_d = optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))

    sched_g = optim.lr_scheduler.LambdaLR(opt_g, lambda ep: _linear_lr_lambda(ep, epochs))
    sched_d = optim.lr_scheduler.LambdaLR(opt_d, lambda ep: _linear_lr_lambda(ep, epochs))

    logger.info(
        "Training inpainting GAN: %d epochs, %d train / %d val samples, "
        "batch=%d, lr=%.5f, workers=%d, R1 every %d steps",
        epochs, len(train_ds), len(val_ds), batch_size, lr, nw, r1_every,
    )
    _log_gpu_memory("pre-train")

    REAL_LABEL = 0.9
    history: dict[str, list] = {"g_loss": [], "d_loss": [], "val_l1": [], "val_psnr": []}
    out_dir = Path(data_dir).parent
    global_step = 0
    train_start = time.time()

    for epoch in range(epochs):
        generator.train()
        discriminator.train()
        t0 = time.time()
        epoch_g, epoch_d = 0.0, 0.0
        n_batches = 0

        for src, mask, tgt in train_loader:
            src, mask, tgt = src.to(device), mask.to(device), tgt.to(device)
            gen_input = torch.cat((src, mask), 1)

            # --- Generator ---
            opt_g.zero_grad()
            fake_clean = generator(gen_input)
            pred_fake = discriminator(gen_input, fake_clean)
            valid = torch.full_like(pred_fake, REAL_LABEL)
            loss_gan = criterion_gan(pred_fake, valid)

            pixel_diff = torch.abs(fake_clean - tgt)
            weighted = pixel_diff * (mask * mask_weight + (1 - mask))
            loss_pixel = weighted.mean()

            loss_g = loss_gan + l1_lambda * loss_pixel
            loss_g.backward()
            opt_g.step()

            # --- Discriminator (every step) ---
            opt_d.zero_grad()
            apply_r1 = (global_step % r1_every == 0)
            if apply_r1:
                tgt.requires_grad_(True)
            pred_real = discriminator(gen_input.detach(), tgt)
            loss_real = criterion_gan(pred_real, torch.full_like(pred_real, REAL_LABEL))
            # R1 penalty (lazy: every r1_every steps, scaled up to compensate)
            if apply_r1:
                r1_pen = _r1_gradient_penalty(discriminator, [tgt], pred_real, gamma=r1_gamma) * r1_every
            else:
                r1_pen = torch.tensor(0.0, device=device)
            pred_fake_d = discriminator(gen_input.detach(), fake_clean.detach())
            loss_fake = criterion_gan(pred_fake_d, torch.zeros_like(pred_fake_d))
            loss_d = 0.5 * (loss_real + loss_fake) + r1_pen
            loss_d.backward()
            opt_d.step()
            if apply_r1:
                tgt.requires_grad_(False)

            epoch_g += loss_g.item()
            epoch_d += loss_d.item()
            n_batches += 1
            global_step += 1

            if global_step % 100 == 0:
                logger.debug(
                    "  step %d — G: %.4f  D: %.4f  R1: %s",
                    global_step, loss_g.item(), loss_d.item(),
                    f"{r1_pen.item():.4f}" if apply_r1 else "skip",
                )
                _log_gpu_memory(f"step-{global_step}")

        sched_g.step()
        sched_d.step()

        avg_g = epoch_g / max(n_batches, 1)
        avg_d = epoch_d / max(n_batches, 1)
        history["g_loss"].append(avg_g)
        history["d_loss"].append(avg_d)

        # Validation
        val_l1, val_psnr, val_n = 0.0, 0.0, 0
        generator.eval()
        with torch.no_grad():
            for src, mask, tgt in val_loader:
                src, mask, tgt = src.to(device), mask.to(device), tgt.to(device)
                fake = generator(torch.cat((src, mask), 1))
                val_l1 += nn.L1Loss()(fake, tgt).item() * src.size(0)
                val_psnr += _psnr(fake, tgt) * src.size(0)
                val_n += src.size(0)
        val_l1 /= max(val_n, 1)
        val_psnr /= max(val_n, 1)
        history["val_l1"].append(val_l1)
        history["val_psnr"].append(val_psnr)

        elapsed = time.time() - t0
        eta_s = elapsed * (epochs - epoch - 1)
        logger.info(
            "Epoch %3d/%d — G: %.4f  D: %.4f  val_L1: %.4f  val_PSNR: %.1f dB  "
            "lr: %.6f  (%.0fs, ETA %dh%02dm)",
            epoch + 1, epochs, avg_g, avg_d, val_l1, val_psnr,
            sched_g.get_last_lr()[0], elapsed,
            int(eta_s // 3600), int(eta_s % 3600 // 60),
        )

        # TensorBoard logging
        if tb_writer is not None:
            tb_writer.add_scalar("train/G_loss", avg_g, epoch + 1)
            tb_writer.add_scalar("train/D_loss", avg_d, epoch + 1)
            tb_writer.add_scalar("val/L1", val_l1, epoch + 1)
            tb_writer.add_scalar("val/PSNR", val_psnr, epoch + 1)
            tb_writer.add_scalar("train/lr", sched_g.get_last_lr()[0], epoch + 1)
            if device.type == "cuda":
                tb_writer.add_scalar(
                    "system/gpu_mem_GB",
                    torch.cuda.max_memory_allocated() / 1e9,
                    epoch + 1,
                )

        if (epoch + 1) % 10 == 0:
            ckpt = out_dir / f"inpaint_gen_{epoch + 1}.pth"
            torch.save(generator.state_dict(), str(ckpt))
            logger.info("  Checkpoint saved: %s", ckpt)
            generator.eval()
            with torch.no_grad():
                sample_batch = next(iter(val_loader))
                s, m, t = [x[:4].to(device) for x in sample_batch]
                fake_s = generator(torch.cat((s, m), 1))
                grid_path = str(out_dir / f"inpaint_samples_{epoch+1}.png")
                _save_sample_grid([s, fake_s, t], grid_path)
                if tb_writer is not None:
                    import torchvision.utils as vutils
                    grid = vutils.make_grid(
                        torch.cat([s, fake_s, t], dim=0),
                        nrow=4, normalize=True, value_range=(-1, 1),
                    )
                    tb_writer.add_image("samples/inpainting", grid, epoch + 1)

    total_time = time.time() - train_start
    logger.info(
        "Training complete: %d epochs in %dh %dm %ds",
        epochs, int(total_time // 3600), int(total_time % 3600 // 60), int(total_time % 60),
    )
    _log_gpu_memory("post-train")
    return generator, history


# ============================================================================
# Evaluation
# ============================================================================


def evaluate_and_report(
    model_name: str,
    history: dict,
    output_dir: Path,
) -> None:
    """Print final metrics and save training curves."""
    logger.info("=" * 60)
    logger.info("EVALUATION REPORT: %s", model_name)
    logger.info("=" * 60)

    if history["val_l1"]:
        best_l1_epoch = int(np.argmin(history["val_l1"])) + 1
        best_psnr_epoch = int(np.argmax(history["val_psnr"])) + 1
        logger.info("Best val L1:   %.4f  (epoch %d)", min(history["val_l1"]), best_l1_epoch)
        logger.info("Best val PSNR: %.1f dB  (epoch %d)", max(history["val_psnr"]), best_psnr_epoch)
        logger.info("Final val L1:  %.4f", history["val_l1"][-1])
        logger.info("Final val PSNR: %.1f dB", history["val_psnr"][-1])

    # Save loss curves as CSV for easy plotting
    csv_path = output_dir / f"{model_name}_training_log.csv"
    with open(csv_path, "w") as f:
        f.write("epoch,g_loss,d_loss,val_l1,val_psnr\n")
        for i in range(len(history["g_loss"])):
            f.write(f"{i+1},{history['g_loss'][i]:.6f},{history['d_loss'][i]:.6f},"
                    f"{history['val_l1'][i]:.6f},{history['val_psnr'][i]:.2f}\n")
    logger.info("Training log saved to %s", csv_path)
    logger.info("Sample grids saved to %s/samples_*.png", output_dir)
    logger.info("=" * 60)


# ============================================================================
# CLI
# ============================================================================


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Retrain GAN models with stabilised training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", choices=["sketch", "inpainting"], required=True)
    parser.add_argument("--samples", type=int, default=20000, help="Synthetic training samples")
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=0.0002)
    parser.add_argument("--l1-lambda", type=float, default=None,
                        help="L1 weight (default: 1000 sketch, 500 inpaint)")
    parser.add_argument("--mask-weight", type=float, default=50.0,
                        help="Mask region penalty (inpainting only)")
    parser.add_argument("--r1-gamma", type=float, default=10.0,
                        help="R1 gradient penalty weight")
    parser.add_argument("--data-dir", type=str, default="./dataset")
    parser.add_argument("--resume", type=str, default=None, help="Checkpoint to resume from")
    parser.add_argument("--skip-datagen", action="store_true")
    parser.add_argument("--skip-preflight", action="store_true",
                        help="Skip pre-flight validation checks")
    parser.add_argument("--device", type=str, default=None,
                        choices=["cpu", "cuda", "mps"],
                        help="Override device (default: auto-detect)")
    parser.add_argument("--r1-every", type=int, default=R1_EVERY_N_STEPS,
                        help="Apply R1 penalty every N steps (lazy regularization)")
    args = parser.parse_args()

    # Set up dual logging (console + file)
    out_dir = Path(args.data_dir)
    log_path = _setup_logging(out_dir, args.model)

    # Log all CLI arguments
    logger.info("CLI arguments: %s", vars(args))
    logger.info("Project root: %s", _PROJECT_ROOT)

    device = _select_device(args.device)

    # Pre-flight validation
    if not args.skip_preflight:
        _preflight_check(device, args.model, args.batch_size, args.data_dir)
    else:
        logger.info("Skipping pre-flight validation (--skip-preflight)")

    gan_dir = Path(__file__).resolve().parent

    # TensorBoard writer
    tb_writer = _get_tb_writer(out_dir, args.model)

    if args.model == "sketch":
        l1_lambda = args.l1_lambda if args.l1_lambda is not None else 1000.0
        cgan_dir = os.path.join(args.data_dir, "cgan")

        if not args.skip_datagen:
            logger.info("Generating %d sketch-cGAN samples...", args.samples)
            t0 = time.time()
            _generate_sketch_data(args.samples, args.data_dir)
            logger.info("Dataset generation took %.0fs", time.time() - t0)

        generator, history = train_sketch_cgan(
            data_dir=cgan_dir, epochs=args.epochs, batch_size=args.batch_size,
            lr=args.lr, l1_lambda=l1_lambda, r1_gamma=args.r1_gamma,
            device=device, resume_path=args.resume, r1_every=args.r1_every,
            tb_writer=tb_writer,
        )

        final = gan_dir / "sketch" / f"generator_{args.epochs}.pth"
        torch.save(generator.state_dict(), str(final))
        logger.info("Final weights: %s", final)
        evaluate_and_report("sketch_cgan", history, out_dir)

    elif args.model == "inpainting":
        l1_lambda = args.l1_lambda if args.l1_lambda is not None else 500.0
        inpaint_dir = os.path.join(args.data_dir, "inpainting")

        if not args.skip_datagen:
            logger.info("Generating %d inpainting samples...", args.samples)
            t0 = time.time()
            _generate_inpaint_data(args.samples, args.data_dir)
            logger.info("Dataset generation took %.0fs", time.time() - t0)

        generator, history = train_inpainting_gan(
            data_dir=inpaint_dir, epochs=args.epochs, batch_size=args.batch_size,
            lr=args.lr, l1_lambda=l1_lambda, mask_weight=args.mask_weight,
            r1_gamma=args.r1_gamma, device=device, resume_path=args.resume,
            r1_every=args.r1_every, tb_writer=tb_writer,
        )

        final = gan_dir / "inpainting" / f"inpaint_gen_{args.epochs}.pth"
        torch.save(generator.state_dict(), str(final))
        logger.info("Final weights: %s", final)
        evaluate_and_report("inpainting", history, out_dir)

    if tb_writer is not None:
        tb_writer.close()

    logger.info("Full log saved to: %s", log_path)


if __name__ == "__main__":
    main()
