# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

End-to-end deep learning pipeline that reads time from a digital clock image and manipulates an analog clock image to display that time. Two main branches: **Digital** (time recognition) and **Analog** (image manipulation).

## Architecture

### Digital Clock Pipeline (`digital_clock/`)
1. **YOLO digit localization** (`yolo_detect_hh_mm/`) — YOLOv8 detects hour/minute bounding boxes. Includes a synthetic dataset creator (`dataset_creator/`) and a labeling/fine-tuning tool (`fine_tuning_framework.py`).
2. **CNN digit recognition** (`svhn_digit_recognition_cnn/`) — Custom 7-conv-layer CNN (`SVHNModel`) with 5 output heads (digits), trained on SVHN + synthetic 7-segment data. Model definition in `svhn_cnn_model.py`.

### Analog Clock Pipeline (`analog_clock/`)
1. **Clock localization** (`yolo_detect_clock/`) — YOLOv8 detects/crops the analog clock from a scene.
2. **Hand detection/segmentation** (`yolo_detetct_hands/`) — YOLOv8-Seg for instance segmentation of hour/minute/second hands. Includes a custom annotation platform (`tagging_platform/`).
3. **Two manipulation modes** (`GAN/`):
   - **Sketch-guided cGAN** (`sketch/`) — Pix2Pix-style `GeneratorUNet` (in_channels=6: image+sketch, 8-layer U-Net). Trained via `sketch-cGAN-model.ipynb`.
   - **Inpainting** (`inpainting/`) — `InpaintGenerator` U-Net (in_channels=4: image+mask). Removes hands, then algorithmically re-composites them at the target time. Trained via `inpainting_GAN_model.ipynb`.
4. **Sketch generation** (`analog_sketch_creator.py`) — Algorithmic function `draw_analog_clock(hh, mm)` that renders a minimalist binary clock sketch for a given time.

### Full Pipeline
`full-pipeline.ipynb` — Orchestrates the complete end-to-end flow from digital clock input to analog clock output.

## Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Run the full pipeline
jupyter notebook full-pipeline.ipynb

# Run individual training notebooks
jupyter notebook digital_clock/svhn_digit_recognition_cnn/svhn_cnn_model.ipynb
jupyter notebook digital_clock/yolo_detect_hh_mm/yolo.ipynb
jupyter notebook analog_clock/yolo_detect_clock/yolo.ipynb
jupyter notebook analog_clock/yolo_detetct_hands/yolo.ipynb
jupyter notebook analog_clock/GAN/sketch/sketch-cGAN-model.ipynb
jupyter notebook analog_clock/GAN/inpainting/inpainting_GAN_model.ipynb

# Generate synthetic datasets
python digital_clock/yolo_detect_hh_mm/dataset_creator/dataset_generator.py
python analog_clock/GAN/sketch/dataset_generator/main.py
python analog_clock/GAN/inpainting/dataset_generator/main.py

# YOLO labeling tools (interactive CV2 UI)
python analog_clock/yolo_detect_clock/fine_tuning_framework.py --input_dir ./images --output_dir ./dataset
python digital_clock/yolo_detect_hh_mm/fine_tuning_framework.py --input_dir ./images --output_dir ./dataset
```

## Key Details

- **Framework:** PyTorch + Ultralytics YOLOv8. No custom training loop abstraction — training code lives in Jupyter notebooks.
- **Pre-trained weights** are committed to the repo as `.pt` (YOLO) and `.pth` (PyTorch) files.
- **Image size:** GAN models expect 256x256 inputs. CNN expects 32x32 grayscale.
- **Dataset generators** use procedural rendering (PIL/matplotlib/OpenCV) to create synthetic training pairs — no external dataset downloads needed for YOLO or GAN training (SVHN is fetched via `datasets`/`kagglehub`).
- **Note:** The hands detection directory is misspelled as `yolo_detetct_hands` (not `detect`).
