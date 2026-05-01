"""
ViT clock generation — end-to-end training script.

Usage (from project root):
    python analog_clock/vit_clock_gen/train.py
    python analog_clock/vit_clock_gen/train.py --epochs 50 --batch_size 8 --lr 2e-4
    python analog_clock/vit_clock_gen/train.py --resume analog_clock/vit_clock_gen/weights/vit_clock_gen_best.pth

The script:
  1. Reads dataset from  dataset/vit_clock_gen/{train,val}/{source,target}/
  2. Trains ViTClockGenerator with L1 + perceptual (VGG16) loss
  3. Saves best-val-loss checkpoint to analog_clock/vit_clock_gen/weights/vit_clock_gen_best.pth
  4. Saves epoch checkpoints every --save_every epochs
  5. Writes training_log.csv with per-epoch metrics
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from analog_clock.vit_clock_gen.model import ViTClockGenerator, VGGPerceptualLoss

logger = logging.getLogger(__name__)

IMAGE_SIZE = 224

_TRANSFORM = T.Compose([
    T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    T.ToTensor(),
    T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class ClockPairDataset(Dataset):
    """Paired (source, target) clock images with target time labels."""

    def __init__(self, split_dir: Path, transform: T.Compose = _TRANSFORM) -> None:
        self.src_dir = split_dir / "source"
        self.tgt_dir = split_dir / "target"
        self.transform = transform

        labels_path = split_dir / "labels.csv"
        if not labels_path.exists():
            raise FileNotFoundError(
                f"labels.csv not found in {split_dir}. "
                "Run dataset_generator.py first."
            )
        df = pd.read_csv(labels_path)
        self.records = df[["filename", "target_hh", "target_mm"]].values.tolist()

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        fname, tgt_hh, tgt_mm = self.records[idx]
        src = self.transform(Image.open(self.src_dir / f"{fname}.png").convert("RGB"))
        tgt = self.transform(Image.open(self.tgt_dir / f"{fname}.png").convert("RGB"))
        return src, tgt, torch.tensor(int(tgt_hh), dtype=torch.long), torch.tensor(int(tgt_mm), dtype=torch.long)


# ---------------------------------------------------------------------------
# Training utilities
# ---------------------------------------------------------------------------

def _run_epoch(
    model: ViTClockGenerator,
    loader: DataLoader,
    l1_loss: nn.L1Loss,
    perc_loss: VGGPerceptualLoss,
    lambda_perc: float,
    optimizer: AdamW | None,
    device: torch.device,
    scaler: torch.cuda.amp.GradScaler | None,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)

    total_l1 = 0.0
    total_perc = 0.0
    n = 0

    for src, tgt, hh, mm in tqdm(loader, leave=False, desc="train" if training else "val"):
        src  = src.to(device)
        tgt  = tgt.to(device)
        hh   = hh.to(device)
        mm   = mm.to(device)

        use_amp = scaler is not None
        with torch.autocast(device_type=device.type, enabled=use_amp):
            pred = model(src, hh, mm)
            loss_l1   = l1_loss(pred, tgt)
            loss_perc = perc_loss(pred, tgt)
            loss = loss_l1 + lambda_perc * loss_perc

        if training:
            optimizer.zero_grad(set_to_none=True)
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

        B = src.shape[0]
        total_l1   += loss_l1.item() * B
        total_perc += loss_perc.item() * B
        n += B

    return {"l1": total_l1 / n, "perc": total_perc / n, "total": (total_l1 + lambda_perc * total_perc) / n}


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train(args: argparse.Namespace) -> None:
    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    logger.info("Device: %s", device)

    dataset_dir = Path(args.dataset_dir)
    weights_dir = Path(args.weights_dir)
    weights_dir.mkdir(parents=True, exist_ok=True)

    train_ds = ClockPairDataset(dataset_dir / "train")
    val_ds   = ClockPairDataset(dataset_dir / "val")
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=(device.type == "cuda"),
    )

    model = ViTClockGenerator(vit_name=args.vit_name, pretrained=True).to(device)
    l1_loss   = nn.L1Loss()
    perc_loss = VGGPerceptualLoss().to(device)

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.01)

    start_epoch = 1
    best_val = float("inf")

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt.get("epoch", 0) + 1
        best_val = ckpt.get("best_val", float("inf"))
        logger.info("Resumed from epoch %d, best_val=%.4f", start_epoch - 1, best_val)

    use_amp = device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler() if use_amp else None

    log_path = weights_dir / "training_log.csv"
    log_fields = ["epoch", "train_l1", "train_perc", "train_total", "val_l1", "val_perc", "val_total", "lr"]
    with open(log_path, "w", newline="") as fh:
        csv.DictWriter(fh, fieldnames=log_fields).writeheader()

    for epoch in range(start_epoch, args.epochs + 1):
        train_metrics = _run_epoch(
            model, train_loader, l1_loss, perc_loss, args.lambda_perc,
            optimizer, device, scaler,
        )
        with torch.no_grad():
            val_metrics = _run_epoch(
                model, val_loader, l1_loss, perc_loss, args.lambda_perc,
                None, device, None,
            )
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        logger.info(
            "Epoch %d/%d  train_total=%.4f  val_total=%.4f  lr=%.2e",
            epoch, args.epochs, train_metrics["total"], val_metrics["total"], current_lr,
        )

        row = {
            "epoch": epoch,
            "train_l1": round(train_metrics["l1"], 5),
            "train_perc": round(train_metrics["perc"], 5),
            "train_total": round(train_metrics["total"], 5),
            "val_l1": round(val_metrics["l1"], 5),
            "val_perc": round(val_metrics["perc"], 5),
            "val_total": round(val_metrics["total"], 5),
            "lr": current_lr,
        }
        with open(log_path, "a", newline="") as fh:
            csv.DictWriter(fh, fieldnames=log_fields).writerow(row)

        def _save_checkpoint(path: Path) -> None:
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_val": best_val,
                    "vit_name": args.vit_name,
                },
                path,
            )

        if val_metrics["total"] < best_val:
            best_val = val_metrics["total"]
            _save_checkpoint(weights_dir / "vit_clock_gen_best.pth")
            logger.info("  -> New best val (%.4f) saved.", best_val)

        if epoch % args.save_every == 0:
            _save_checkpoint(weights_dir / f"vit_clock_gen_epoch_{epoch:03d}.pth")

    logger.info("Training complete. Best val loss: %.4f", best_val)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train ViT clock generation model")
    p.add_argument("--dataset_dir",  default="./dataset/vit_clock_gen",
                   help="Root of the generated dataset (default: ./dataset/vit_clock_gen)")
    p.add_argument("--weights_dir",  default="./analog_clock/vit_clock_gen/weights",
                   help="Directory for checkpoints (default: ./analog_clock/vit_clock_gen/weights)")
    p.add_argument("--vit_name",     default="vit_small_patch16_224",
                   choices=["vit_tiny_patch16_224", "vit_small_patch16_224", "vit_base_patch16_224"],
                   help="timm ViT variant (default: vit_small_patch16_224)")
    p.add_argument("--epochs",       type=int,   default=50,    help="Training epochs (default: 50)")
    p.add_argument("--batch_size",   type=int,   default=8,     help="Batch size (default: 8)")
    p.add_argument("--lr",           type=float, default=2e-4,  help="Peak learning rate (default: 2e-4)")
    p.add_argument("--lambda_perc",  type=float, default=0.1,   help="Perceptual loss weight (default: 0.1)")
    p.add_argument("--save_every",   type=int,   default=10,    help="Save checkpoint every N epochs")
    p.add_argument("--workers",      type=int,   default=4,     help="DataLoader worker processes")
    p.add_argument("--resume",       default=None,              help="Path to checkpoint to resume from")
    return p.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    train(_parse_args())
