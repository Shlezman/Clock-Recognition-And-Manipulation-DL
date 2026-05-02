# ViT Clock Generation

**Stage 5 of the full pipeline.** Given a source analog clock image and a target time `(HH, MM)`, generates a new clock image showing that time using a Vision Transformer encoder with a time-conditioned CNN decoder.

---

## Architecture

```
Source image (224×224)
        │
        ▼
  ViT-Small/16 encoder           ← pretrained on ImageNet (timm)
  196 patch tokens × 384 dim
        │
  Linear projection → 512 dim
  Reshape → (512, 14, 14)
        │
        │◄── Time embedding (HH, MM)
        │     sinusoidal → Linear(512,256) → SiLU → Linear(256,256)
        │
  Decoder block 1: (512→256, 14→28)   AdaIN conditioning
  Decoder block 2: (256→128, 28→56)   AdaIN conditioning
  Decoder block 3: (128→64,  56→112)  AdaIN conditioning
  Decoder block 4: ( 64→32, 112→224)  AdaIN conditioning
        │
  Conv 3×3 → Tanh
        │
        ▼
  Generated clock (224×224, RGB)
```

**Training loss:** `L1 + 0.1 × Perceptual (VGG16 relu_1_2 / relu_2_2 / relu_3_3)`

---

## Files

| File | Purpose |
|------|---------|
| `model.py` | `ViTClockGenerator` + `VGGPerceptualLoss` |
| `dataset_generator.py` | Procedural dataset generation |
| `train.py` | End-to-end training script |
| `weights/` | Saved checkpoints (`.gitkeep` only; populate by training) |

---

## Quick Start — Training Machine (GPU)

### Prerequisites

```bash
git clone <repo-url>
cd Clock-Recognition-And-Manipulation-DL

# CPU-only (testing)
pip install -e .

# CUDA GPU (recommended for training)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -e .
```

### Step 1 — Generate the dataset

```bash
# Default: 10 000 samples, 85/15 train/val split
python analog_clock/vit_clock_gen/dataset_generator.py

# Larger dataset for better quality (recommended)
python analog_clock/vit_clock_gen/dataset_generator.py --n_samples 20000

# Custom output directory
python analog_clock/vit_clock_gen/dataset_generator.py --n_samples 20000 --output_dir ./my_dataset
```

Output layout:
```
dataset/vit_clock_gen/
├── train/
│   ├── source/     # 224×224 PNG — source clock images
│   ├── target/     # 224×224 PNG — target clock images (same face, different time)
│   └── labels.csv  # filename, target_hh, target_mm
└── val/
    ├── source/
    ├── target/
    └── labels.csv
```

### Step 2 — Train

```bash
# Default: 50 epochs, batch size 8, vit_small_patch16_224
python analog_clock/vit_clock_gen/train.py

# Larger ViT (requires more VRAM)
python analog_clock/vit_clock_gen/train.py --vit_name vit_base_patch16_224 --epochs 100

# Resume from checkpoint
python analog_clock/vit_clock_gen/train.py \
    --resume analog_clock/vit_clock_gen/weights/vit_clock_gen_best.pth

# Full recommended run
python analog_clock/vit_clock_gen/train.py \
    --dataset_dir ./dataset/vit_clock_gen \
    --epochs 50 \
    --batch_size 8 \
    --lr 2e-4 \
    --save_every 10
```

All checkpoints are saved to `analog_clock/vit_clock_gen/weights/`:
- `vit_clock_gen_best.pth` — best validation loss (used by the pipeline)
- `vit_clock_gen_epoch_NNN.pth` — periodic checkpoints
- `training_log.csv` — per-epoch metrics

### Step 3 — Run the pipeline notebook

```bash
jupyter notebook full-pipeline.ipynb
```

The notebook auto-detects `analog_clock/vit_clock_gen/weights/vit_clock_gen_best.pth`. If the file exists, Stage 5 runs automatically. If not, it prints a helpful message.

---

## Training tips

| Setting | Recommendation |
|---------|----------------|
| GPU VRAM | 8 GB+ (batch 8, vit_small) · 16 GB+ (vit_base) |
| Dataset size | 10k samples → reasonable quality · 20k → better generalisation |
| Epochs | 50 minimum; 100 for best results |
| Quick sanity check | `--n_samples 500 --epochs 5 --batch_size 4` |

---

## Key design decisions

- **AdaIN conditioning:** each decoder block modulates its output with per-channel scale/shift predicted from the time embedding, giving the network fine-grained temporal control.
- **Sinusoidal time embedding:** encodes hour (12-hour cycle) and minute (60-unit cycle) separately with standard sinusoidal frequencies, avoiding discontinuities at the 12/0 boundary.
- **Perceptual loss:** VGG16 features penalise texture mismatch at three scales, producing sharper hands than L1 alone.
- **Pretrained ViT:** starting from ImageNet weights accelerates convergence and transfers texture/edge features useful for clock faces.
