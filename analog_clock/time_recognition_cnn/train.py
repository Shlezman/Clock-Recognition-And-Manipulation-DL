"""
Training script for ClockHandCNN.

Usage:
    python train.py --data_dir ./dataset --epochs 60 --batch_size 64

Expects the directory layout produced by dataset_generator.py:
    data_dir/
        train/images/  *.png
        val/images/    *.png
        train_labels.csv
        val_labels.csv
"""

import argparse
import csv
import math
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from model import ClockHandCNN


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class ClockMaskDataset(Dataset):
    """Loads binary masks + sin/cos labels from a generated dataset split."""

    def __init__(self, data_dir: str, split: str = "train"):
        self.img_dir = Path(data_dir) / split / "images"
        self.entries: list[dict] = []

        csv_path = Path(data_dir) / f"{split}_labels.csv"
        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.entries.append(row)

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.entries[idx]

        # Load grayscale mask and normalise to [0, 1]
        img = cv2.imread(str(self.img_dir / row["filename"]), cv2.IMREAD_GRAYSCALE)
        tensor = torch.from_numpy(img.astype(np.float32) / 255.0).unsqueeze(0)  # (1, H, W)

        label = torch.tensor([
            float(row["hour_sin"]),
            float(row["hour_cos"]),
            float(row["minute_sin"]),
            float(row["minute_cos"]),
        ], dtype=torch.float32)

        return tensor, label


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def angle_error_minutes(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Compute mean absolute time error in *minutes* for a batch.

    Both pred and target are (B, 4): [sin_h, cos_h, sin_m, cos_m].
    """
    # Recover angles in radians
    pred_h = torch.atan2(pred[:, 0], pred[:, 1])    # hour angle
    pred_m = torch.atan2(pred[:, 2], pred[:, 3])    # minute angle
    tgt_h = torch.atan2(target[:, 0], target[:, 1])
    tgt_m = torch.atan2(target[:, 2], target[:, 3])

    # Convert to minutes on a 12-hour clock (720 minutes total)
    pred_total = (pred_h % (2 * math.pi)) / (2 * math.pi) * 720.0 \
               + (pred_m % (2 * math.pi)) / (2 * math.pi) * 60.0  # redundant but intuitive
    tgt_total = (tgt_h % (2 * math.pi)) / (2 * math.pi) * 720.0 \
              + (tgt_m % (2 * math.pi)) / (2 * math.pi) * 60.0

    # Actually, measure hour and minute errors separately then combine.
    # Hour error (in hours, wrap-around at 12):
    h_pred = (pred_h % (2 * math.pi)) / (2 * math.pi) * 12.0
    h_tgt = (tgt_h % (2 * math.pi)) / (2 * math.pi) * 12.0
    h_diff = torch.abs(h_pred - h_tgt)
    h_diff = torch.min(h_diff, 12.0 - h_diff)  # wrap

    # Minute error (in minutes, wrap-around at 60):
    m_pred = (pred_m % (2 * math.pi)) / (2 * math.pi) * 60.0
    m_tgt = (tgt_m % (2 * math.pi)) / (2 * math.pi) * 60.0
    m_diff = torch.abs(m_pred - m_tgt)
    m_diff = torch.min(m_diff, 60.0 - m_diff)

    total_err_min = h_diff * 60.0 + m_diff  # total error in minutes
    return total_err_min.mean()


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Data
    train_ds = ClockMaskDataset(args.data_dir, "train")
    val_ds = ClockMaskDataset(args.data_dir, "val")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size,
                            shuffle=False, num_workers=0, pin_memory=True)

    print(f"Train: {len(train_ds)}  |  Val: {len(val_ds)}")

    # Model
    model = ClockHandCNN().to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        # --- Train ---
        model.train()
        running_loss = 0.0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)

            preds = model(imgs)
            loss = criterion(preds, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * imgs.size(0)

        train_loss = running_loss / len(train_ds)

        # --- Validate ---
        model.eval()
        val_loss_sum = 0.0
        val_err_sum = 0.0
        n_val = 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                preds = model(imgs)
                val_loss_sum += criterion(preds, labels).item() * imgs.size(0)
                val_err_sum += angle_error_minutes(preds, labels).item() * imgs.size(0)
                n_val += imgs.size(0)

        val_loss = val_loss_sum / n_val
        val_err = val_err_sum / n_val

        scheduler.step()

        print(
            f"[{epoch:3d}/{args.epochs}]  "
            f"train_loss={train_loss:.5f}  "
            f"val_loss={val_loss:.5f}  "
            f"val_err={val_err:.1f} min  "
            f"lr={scheduler.get_last_lr()[0]:.2e}"
        )

        # Checkpoint best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_path = Path(args.data_dir).parent / "clock_hand_cnn_best.pth"
            torch.save(model.state_dict(), save_path)
            print(f"  -> saved best model ({save_path})")

    print("Training complete.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ClockHandCNN")
    parser.add_argument("--data_dir", type=str, default="./dataset")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    train(args)
