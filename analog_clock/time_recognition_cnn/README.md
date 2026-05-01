# Time Recognition CNN

A lightweight CNN that reads the current time from a **binary clock-hand mask** (256x256, single channel).

## Problem

Given a binary image containing only the hour and minute hands of an analog clock, predict the displayed time. This is used in the full pipeline to determine the *original* time shown on a real clock photo (via YOLO-seg hand masks), so the animation can interpolate from the original time to the target time.

## Approach

### Sin/Cos Angle Encoding

Directly regressing hours (0-11) and minutes (0-59) creates a **discontinuity** at the wrap-around point (e.g. 11:59 vs 0:00 are numerically far apart but temporally adjacent). We avoid this by encoding each angle as a (sin, cos) pair:

```
hour_angle  = (hour % 12) * 30 + minute * 0.5   # degrees
minute_angle = minute * 6                         # degrees

Label = [sin(hour_angle), cos(hour_angle), sin(minute_angle), cos(minute_angle)]
```

Recovery at inference:

```python
hour_angle  = atan2(sin_h, cos_h)   # radians in [0, 2*pi)
minute_angle = atan2(sin_m, cos_m)

hours   = hour_angle   / (2*pi) * 12   # [0, 12)
minutes = minute_angle / (2*pi) * 60   # [0, 60)
```

### Architecture — ClockHandCNN

| Layer | Output Shape | Notes |
|-------|-------------|-------|
| Input | (B, 1, 256, 256) | Binary mask |
| ConvBlock 1 | (B, 32, 128, 128) | Conv3x3 + BN + ReLU + MaxPool |
| ConvBlock 2 | (B, 64, 64, 64) | |
| ConvBlock 3 | (B, 128, 32, 32) | |
| ConvBlock 4 | (B, 256, 16, 16) | |
| ConvBlock 5 | (B, 256, 8, 8) | |
| ConvBlock 6 | (B, 256, 4, 4) | |
| AdaptiveAvgPool | (B, 256) | Global average pooling |
| FC + ReLU + Dropout(0.3) | (B, 128) | |
| FC | (B, 4) | sin_h, cos_h, sin_m, cos_m |

Total parameters: ~1.1M

### Synthetic Dataset

The model is trained entirely on **synthetically generated masks** — no real photos needed for training. The generator (`dataset_generator.py`) produces diverse masks by randomising:

- **21 hand styles**: pointed, rectangle, modern, arrow, diamond, tapered, sword, lollipop, baton, leaf, pencil, dauphine, breguet, spade, cathedral, alpha, feuille, lance, plongeur, syringe, flamme
- **Hand dimensions**: length (50-95% of radius) and width (3-10% of radius) are randomised per sample
- **Random times**: uniform sampling over all 720 possible hour:minute combinations

Each sample is a 256x256 binary image (white hands on black background) with a CSV row containing the time and its sin/cos encoding.

Default: 15,000 training + 5,000 validation samples.

### Training

- **Loss**: MSE on the 4 sin/cos outputs
- **Optimiser**: AdamW (lr=1e-3, weight_decay=1e-4)
- **Scheduler**: Cosine annealing over 60 epochs
- **Batch size**: 64
- **Metric**: Mean absolute time error in minutes (accounts for wrap-around)

## Files

| File | Description |
|------|-------------|
| `model.py` | `ClockHandCNN` architecture definition |
| `dataset_generator.py` | Synthetic mask + label generator (21 hand styles) |
| `train.py` | Training script with CLI arguments |
| `time_recognition_cnn.ipynb` | Full notebook: generate data, train, visualise results, test on real images |
| `clock_hand_cnn_best.pth` | Best model weights (saved during training) |

## Usage

### Generate Dataset

```bash
python dataset_generator.py --n_samples 20000 --output_dir ./dataset
```

### Train

```bash
python train.py --data_dir ./dataset --epochs 60 --batch_size 64
```

### Inference (from Python)

```python
from analog_clock.time_recognition_cnn.model import ClockHandCNN
from analog_clock.pipeline_utils import load_time_recognition_cnn, recognize_time_from_masks

device = torch.device("mps")  # or "cuda" / "cpu"
model = load_time_recognition_cnn("clock_hand_cnn_best.pth", device)

# binary_mask: np.ndarray (H, W) with values 0/1
hours, minutes = recognize_time_from_masks(binary_mask, model, device)
print(f"Predicted time: {hours}:{minutes:02d}")
```

### Full Pipeline Integration

The CNN is integrated into `full-pipeline.ipynb` where it:
1. Takes YOLO-seg hand masks from the analog clock image
2. Predicts the original displayed time
3. Enables GIF animation from original time to target time (read from the digital clock)
