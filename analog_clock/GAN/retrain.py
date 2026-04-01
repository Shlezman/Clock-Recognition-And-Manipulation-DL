#!/usr/bin/env python3
"""
GAN retraining script with anti-vanishing-gradient training.

End-to-end workflow:
  1. Generate synthetic dataset (with "in the wild" realism effects)
  2. Train GAN with stabilised training (spectral-norm D, R1 gradient penalty,
     label smoothing, linear LR decay, D trained every step)
  3. Save weights to canonical path
  4. Run evaluation (val-set L1/PSNR + sample grid)

Supports both pipelines:
  --model sketch     →  Sketch-cGAN (Pix2Pix, 6-channel input)
  --model inpainting →  Inpainting GAN (4-channel input)

Example:
  python -m analog_clock.GAN.retrain --model sketch --epochs 100 --samples 20000
  python -m analog_clock.GAN.retrain --model inpainting --epochs 100 --samples 20000
"""

from __future__ import annotations

import argparse
import logging
import math
import os
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

    train_ds = SketchDataset(data_dir, "train", transform)
    val_ds = SketchDataset(data_dir, "val", transform)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

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

    logger.info("Training sketch-cGAN: %d epochs, %d samples, lr=%.5f", epochs, len(train_ds), lr)

    # Label smoothing: real=0.9
    REAL_LABEL = 0.9

    history: dict[str, list] = {"g_loss": [], "d_loss": [], "val_l1": [], "val_psnr": []}
    out_dir = Path(data_dir).parent

    for epoch in range(epochs):
        generator.train()
        discriminator.train()
        t0 = time.time()
        epoch_g, epoch_d = 0.0, 0.0
        n_batches = 0

        for src, skc, tgt in train_loader:
            src, skc, tgt = src.to(device), skc.to(device), tgt.to(device)
            bs = src.size(0)

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
            # Real
            tgt.requires_grad_(True)
            pred_real = discriminator(src, skc, tgt)
            loss_real = criterion_gan(pred_real, torch.full_like(pred_real, REAL_LABEL))
            # R1 penalty
            r1_pen = _r1_gradient_penalty(discriminator, [tgt], pred_real, gamma=r1_gamma)
            # Fake
            pred_fake_d = discriminator(src, skc, fake_tgt.detach())
            loss_fake = criterion_gan(pred_fake_d, torch.zeros_like(pred_fake_d))
            loss_d = 0.5 * (loss_real + loss_fake) + r1_pen
            loss_d.backward()
            opt_d.step()
            tgt.requires_grad_(False)

            epoch_g += loss_g.item()
            epoch_d += loss_d.item()
            n_batches += 1

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
        logger.info(
            "Epoch %3d/%d — G: %.4f  D: %.4f  val_L1: %.4f  val_PSNR: %.1f dB  lr: %.6f  (%.0fs)",
            epoch + 1, epochs, avg_g, avg_d, val_l1, val_psnr, sched_g.get_last_lr()[0], elapsed,
        )

        if (epoch + 1) % 10 == 0:
            ckpt = out_dir / f"generator_{epoch + 1}.pth"
            torch.save(generator.state_dict(), str(ckpt))
            # Save sample grid
            generator.eval()
            with torch.no_grad():
                sample_batch = next(iter(val_loader))
                s, sk, t = [x[:4].to(device) for x in sample_batch]
                fake_s = generator(torch.cat((s, sk), 1))
                _save_sample_grid([s, sk, fake_s, t], str(out_dir / f"samples_epoch_{epoch+1}.png"))

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

    train_ds = InpaintDataset(data_dir, "train", transform)
    val_ds = InpaintDataset(data_dir, "val", transform)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

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

    logger.info("Training inpainting GAN: %d epochs, %d samples, lr=%.5f", epochs, len(train_ds), lr)

    REAL_LABEL = 0.9
    history: dict[str, list] = {"g_loss": [], "d_loss": [], "val_l1": [], "val_psnr": []}
    out_dir = Path(data_dir).parent

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
            tgt.requires_grad_(True)
            pred_real = discriminator(gen_input.detach(), tgt)
            loss_real = criterion_gan(pred_real, torch.full_like(pred_real, REAL_LABEL))
            r1_pen = _r1_gradient_penalty(discriminator, [tgt], pred_real, gamma=r1_gamma)
            pred_fake_d = discriminator(gen_input.detach(), fake_clean.detach())
            loss_fake = criterion_gan(pred_fake_d, torch.zeros_like(pred_fake_d))
            loss_d = 0.5 * (loss_real + loss_fake) + r1_pen
            loss_d.backward()
            opt_d.step()
            tgt.requires_grad_(False)

            epoch_g += loss_g.item()
            epoch_d += loss_d.item()
            n_batches += 1

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
        logger.info(
            "Epoch %3d/%d — G: %.4f  D: %.4f  val_L1: %.4f  val_PSNR: %.1f dB  lr: %.6f  (%.0fs)",
            epoch + 1, epochs, avg_g, avg_d, val_l1, val_psnr, sched_g.get_last_lr()[0], elapsed,
        )

        if (epoch + 1) % 10 == 0:
            ckpt = out_dir / f"inpaint_gen_{epoch + 1}.pth"
            torch.save(generator.state_dict(), str(ckpt))
            generator.eval()
            with torch.no_grad():
                sample_batch = next(iter(val_loader))
                s, m, t = [x[:4].to(device) for x in sample_batch]
                fake_s = generator(torch.cat((s, m), 1))
                _save_sample_grid([s, fake_s, t], str(out_dir / f"inpaint_samples_{epoch+1}.png"))

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
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    device = torch.device(
        "mps" if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available()
        else "cpu"
    )
    logger.info("Device: %s", device)

    gan_dir = Path(__file__).resolve().parent
    out_dir = Path(args.data_dir)

    if args.model == "sketch":
        l1_lambda = args.l1_lambda if args.l1_lambda is not None else 1000.0
        cgan_dir = os.path.join(args.data_dir, "cgan")

        if not args.skip_datagen:
            logger.info("Generating %d sketch-cGAN samples…", args.samples)
            _generate_sketch_data(args.samples, args.data_dir)

        generator, history = train_sketch_cgan(
            data_dir=cgan_dir, epochs=args.epochs, batch_size=args.batch_size,
            lr=args.lr, l1_lambda=l1_lambda, r1_gamma=args.r1_gamma,
            device=device, resume_path=args.resume,
        )

        final = gan_dir / "sketch" / f"generator_{args.epochs}.pth"
        torch.save(generator.state_dict(), str(final))
        logger.info("Final weights: %s", final)
        evaluate_and_report("sketch_cgan", history, out_dir)

    elif args.model == "inpainting":
        l1_lambda = args.l1_lambda if args.l1_lambda is not None else 500.0
        inpaint_dir = os.path.join(args.data_dir, "inpainting")

        if not args.skip_datagen:
            logger.info("Generating %d inpainting samples…", args.samples)
            _generate_inpaint_data(args.samples, args.data_dir)

        generator, history = train_inpainting_gan(
            data_dir=inpaint_dir, epochs=args.epochs, batch_size=args.batch_size,
            lr=args.lr, l1_lambda=l1_lambda, mask_weight=args.mask_weight,
            r1_gamma=args.r1_gamma, device=device, resume_path=args.resume,
        )

        final = gan_dir / "inpainting" / f"inpaint_gen_{args.epochs}.pth"
        torch.save(generator.state_dict(), str(final))
        logger.info("Final weights: %s", final)
        evaluate_and_report("inpainting", history, out_dir)


if __name__ == "__main__":
    main()
