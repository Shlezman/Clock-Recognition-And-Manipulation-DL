# TimeSync: Digital-to-Analog Clock Synchronization

> **An End-to-End Deep Learning Pipeline: From Digital Time Recognition to Generative Analog Clock Manipulation.**

This project implements a complex computer vision system that synchronizes an analog clock image to match a time read from a digital clock. The pipeline leverages state-of-the-art Deep Learning techniques, including **Object Detection (YOLO)**, **Instance Segmentation**, **CNN Classification/Regression**, and **Generative Adversarial Networks (cGAN/Inpainting)**.

---

## System Architecture

The project consists of two main branches: the **Digital Branch** (Time Recognition) and the **Analog Branch** (Image Manipulation).

### Part 1: Digital Clock Reader

The goal of this module is to extract the exact time (`HH:MM`) from an input image of a digital clock.

#### 1. Digit Localization (Detection)

* **Model:** YOLOv8 (You Only Look Once).
* **Dataset:** Trained on a custom **Synthetic Dataset** generated via Python scripts, followed by **Fine-Tuning** on real-world digital clock images.
* **Function:** Detects and extracts Bounding Boxes (BBox) for the Hour and Minute digits separately.

#### 2. Digit Recognition (Classification)

* **Model:** Custom CNN (Convolutional Neural Network).
* **Dataset:** Trained on the **SVHN** (Street View House Numbers) dataset + a custom **Synthetic 7-Segment** dataset.
* *Why?* The 7-segment data bridges the domain gap, allowing the model to generalize well to electronic displays commonly found in digital clocks.
* **Function:** Receives the BBoxes from the previous step, classifies the digits, and outputs the final time string.

---

### Part 2: Analog Clock Manipulation

The goal of this module is to modify an input image of an analog clock so that its hands display the time recognized in Part 1.

#### Pre-processing: Clock Localization

* **Model:** YOLOv8.
* **Function:** Detects the analog clock within a larger scene and performs a crop to focus the workspace.

#### Time Recognition from Hand Masks

* **Model:** ClockHandCNN -- a lightweight 6-block CNN trained on synthetic binary hand masks.
* **Encoding:** Sin/cos angle encoding avoids discontinuity at the 12/0 wrap-around boundary.
* **Function:** Predicts the *original* time displayed on the analog clock from YOLO-seg hand masks, enabling smooth GIF animation from the original time to the target time.
* **Details:** See [`analog_clock/time_recognition_cnn/README.md`](analog_clock/time_recognition_cnn/README.md).

The system supports **two distinct modes** for manipulating the clock face:

### Mode 1: Sketch-Guided Generation (Pix2Pix Style)

This approach uses a Conditional GAN to "redraw" the clock based on a structural guide.

1. **Sketch Generation:** An algorithmic function receives the target time and generates a minimalist binary sketch (black & white) of a clock face pointing to that time.
2. **Generative Model (cGAN):**
   * **Input:** The cropped analog clock image + The generated binary sketch.
   * **Output:** A newly generated, photorealistic image of the clock, where the hands are aligned according to the sketch.
3. **GIF Animation:** Generates an animated GIF showing the clock hands smoothly transitioning from the original time to the target time, with 2-minute intermediate steps.

### Mode 2: Segmentation & Inpainting (High-Fidelity)

This approach uses a "disassemble and reassemble" method to preserve the original background quality.

1. **Hand Segmentation:**
   * **Model:** **YOLOv8-Seg** (Instance Segmentation).
   * **Function:** Detects and segments the specific pixels of the hour, minute, and second hands.

2. **Mask Generation:** Creates a binary mask representing the exact area covered by the hands.
3. **Hand Removal (Inpainting):**
   * **Model:** **Inpainting GAN**.
   * **Input:** Original clock image + Binary Mask.
   * **Output:** A "Clean Plate" -- the clock background with the hands removed, where the hidden numbers/texture are reconstructed by the AI.

4. **Re-Composition (Improved):**
   * Extracts the style of the hands using the segmentation mask with **multi-scale feathered alpha blending** (two-pass Gaussian blur) for smooth, anti-aliased results.
   * Uses a **hybrid center-finding** algorithm (Hough circles + fitLine intersection + PCA intersection + dynamic centroid fallback) for sub-pixel accuracy.
   * Measures hand angles via **skeleton thinning + tip detection** for precise rotation.
   * Rotates the hands mathematically to the target time and blends them back onto the clean background.

5. **GIF Animation:** Generates an animated GIF showing the clock hands smoothly transitioning from the original time to the target time, with 2-minute intermediate steps.

---

## Pipeline Diagram

```
                    +----------------------+
                    |  Digital Clock Image  |
                    +-----------+----------+
                                |
                                v
                    +------------------------------------+
                    | Time Region Detection Model        |
                    | (Bounding Box Localization)        |
                    +-----------+------------------------+
                                |
                                v
                    +------------------------------------+
                    | Digit Recognition Model            |
                    +-----------+------------------------+
                                |
                                v
                    +----------------------+
                    | Extracted Time HH:MM |
                    +-----------+----------+
                                |
               +----------------+-------------- Y-SPLIT ----------------+
               v                                                        v

PATH 1: Sketch-Guided Generation               PATH 2: Segmentation & Inpainting

+------------------------------------+        +------------------------------+
| Time Recognition CNN               |        | Analog Clock Image (Input)   |
| (Recognise original time)          |        +---------------+--------------+
+------------+-----------------------+                        |
             |                                                v
             v                               +------------------------------+
+------------------------------------+       | Time Recognition CNN         |
| Generate Minimal Analog Sketch     |       | (Recognise original time)    |
+------------+-----------------------+       +---------------+--------------+
             |                                               |
             v                                               v
+------------------------------------+       +------------------------------+
| cGAN Model (spectral-norm D,       |       | Hand Segmentation (YOLO-Seg) |
|  R1 penalty, label smoothing)      |       +---------------+--------------+
+------------+-----------------------+                       |
             |                                               v
             v                               +------------------------------+
+------------------------------------+       | Hand Removal (Inpainting GAN |
| Animated GIF                       |       |  + spectral-norm + R1)       |
| (Original -> Target time)          |       +---------------+--------------+
+------------------------------------+                       |
                                                             v
                                              +------------------------------+
                                              | Recompose Hands at Target    |
                                              | Time (Multi-scale Feathered  |
                                              |  Alpha + Hybrid Center Find) |
                                              +---------------+--------------+
                                                             |
                                                             v
                                              +------------------------------+
                                              | Animated GIF                 |
                                              | (Original -> Target time)    |
                                              +------------------------------+
```

---

## Datasets

The project utilizes a hybrid of public benchmarks and custom synthetic data:

* **Synthetic Digital Clocks:** A generator script creating thousands of variations of digital displays for YOLO training.
* **Synthetic 7-Segment Digits:** Used to augment the CNN classifier.
* **SVHN:** Street View House Numbers dataset used for robust digit feature learning.
* **Synthetic Analog Clocks ("In the Wild"):** Procedurally generated with 23 hand styles, drop shadows, glass reflections, lighting gradients, frame shadows, aged yellowing, and specular highlights for realistic training data. Produces both `(Image, Sketch)` and `(Image, Mask)` pairs.
* **Synthetic Clock-Hand Masks:** 15k+ binary masks with 23 hand styles for training the time recognition CNN.

---

## Project Structure

```
+-- full-pipeline.ipynb                    # End-to-end orchestration notebook
+-- pyproject.toml                         # uv project configuration
+-- tests/                                 # pytest test suite (42 tests)
|   +-- test_pipeline_utils.py             # Time utils, center finding, blending, angle measurement
|   +-- test_shared_config.py              # Config hierarchy validation
|   +-- test_models.py                     # GAN forward pass + spectral norm checks
+-- digital_clock/
|   +-- yolo_detect_hh_mm/                 # YOLO digit localization
|   |   +-- dataset_creator/               # Synthetic dataset generator
|   |   +-- fine_tuning_framework.py       # Interactive labeling tool
|   +-- svhn_digit_recognition_cnn/        # CNN digit classifier
+-- analog_clock/
    +-- analog_sketch_creator.py           # Algorithmic sketch renderer
    +-- pipeline_utils.py                  # Pipeline utilities (center finding, time recognition,
    |                                      #   hand recomposition, GIF generation)
    +-- shared/                            # Shared module (deduplicated from GAN generators)
    |   +-- config.py                      # BaseConfig / SketchConfig / InpaintConfig (frozen dataclasses)
    |   +-- asset_manager.py               # Background texture downloader
    |   +-- procedural_generator.py        # 23 hand styles + "in the wild" effects
    |   +-- clock_renderer.py              # RGBA hand compositing
    |   +-- augmentations.py               # Albumentations pipeline (paired + masked modes)
    |   +-- sketch_generator.py            # Sketch wrapper (clean imports)
    +-- yolo_detect_clock/                 # YOLO clock localization
    +-- yolo_detetct_hands/                # YOLO-Seg hand segmentation
    |   +-- tagging_platform/              # Custom annotation platform
    +-- time_recognition_cnn/              # Time recognition from hand masks
    |   +-- model.py                       # ClockHandCNN architecture
    |   +-- dataset_generator.py           # Synthetic mask generator
    |   +-- train.py                       # Training script
    +-- GAN/
        +-- retrain.py                     # Full retraining script (datagen + train + eval)
        +-- sketch/                        # Sketch-guided cGAN (Pix2Pix)
        |   +-- generator_model.py         # GeneratorUNet + SketchDiscriminator (spectral norm)
        |   +-- dataset_generator/         # Sketch dataset generator (uses shared module)
        +-- inpainting/                    # Inpainting GAN
            +-- generator_model.py         # InpaintGenerator + InpaintDiscriminator (spectral norm)
            +-- dataset_generator/         # Inpainting dataset generator (uses shared module)
```

---

## Quick Start

```bash
# Install dependencies (using uv)
uv sync

# Or using pip
pip install -e .

# Run the full pipeline
jupyter notebook full-pipeline.ipynb

# Run tests
uv run pytest tests/ -v
```

### Retraining GAN Models

The retraining script handles the complete workflow: dataset generation, training with stabilised GAN training (spectral normalisation, R1 gradient penalty, label smoothing, linear LR decay), and evaluation with PSNR/L1 metrics.

```bash
# Retrain sketch-cGAN (generates 20k samples, trains 100 epochs)
python -m analog_clock.GAN.retrain --model sketch --epochs 100 --samples 20000

# Retrain inpainting GAN
python -m analog_clock.GAN.retrain --model inpainting --epochs 100 --samples 20000

# Fine-tune from existing weights (skip data generation)
python -m analog_clock.GAN.retrain --model sketch --resume analog_clock/GAN/sketch/generator_100.pth --skip-datagen --epochs 50

# Quick test run (small dataset)
python -m analog_clock.GAN.retrain --model sketch --samples 500 --epochs 10 --batch-size 4
```

**Output:**
- Checkpoints every 10 epochs: `dataset/generator_*.pth`
- Sample grids: `dataset/samples_epoch_*.png`
- Training log CSV: `dataset/sketch_cgan_training_log.csv`
- Final weights copied to canonical location: `analog_clock/GAN/sketch/generator_100.pth`

### Training Individual Models

```bash
# Digital clock models
jupyter notebook digital_clock/svhn_digit_recognition_cnn/svhn_cnn_model.ipynb
jupyter notebook digital_clock/yolo_detect_hh_mm/yolo.ipynb

# Analog clock models
jupyter notebook analog_clock/yolo_detect_clock/yolo.ipynb
jupyter notebook analog_clock/yolo_detetct_hands/yolo.ipynb

# Time recognition CNN
python analog_clock/time_recognition_cnn/train.py
```

### Generating Synthetic Datasets

```bash
python digital_clock/yolo_detect_hh_mm/dataset_creator/dataset_generator.py
python analog_clock/GAN/sketch/dataset_generator/dataset_generator.py
python analog_clock/GAN/inpainting/dataset_generator/dataset_generator.py
python analog_clock/time_recognition_cnn/dataset_generator.py --n_samples 20000
```

---

## GAN Training Stability

The GAN training incorporates several techniques to prevent vanishing gradients and mode collapse:

| Technique | Purpose |
|-----------|---------|
| **Spectral Normalization** on all D conv layers | Constrains Lipschitz constant, prevents D from becoming too confident |
| **R1 Gradient Penalty** (Mescheder 2018) | Penalises gradient magnitude on real samples |
| **Label Smoothing** (real=0.9) | Prevents D from outputting extreme values |
| **Linear LR Decay** (second half of training) | Gradual convergence |
| **LSGAN Loss** (MSE instead of BCE) | More stable gradient flow than standard GAN |
| **D trained every step** (not every other epoch) | Keeps D and G balanced |

---

## "In the Wild" Synthetic Data

The procedural clock generator produces realistic training data with:

- **23 hand polygon styles:** pointed, rectangle, modern, arrow, diamond, tapered, sword, lollipop, skeleton, baton, leaf, pencil, dauphine, breguet, spade, anchor, cathedral, alpha, feuille, lance, plongeur, syringe, flamme
- **Drop shadows** behind the clock (wall-mounted look)
- **Frame/bezel inner shadows** (depth effect)
- **Directional lighting gradients** across the face
- **Glass specular highlights** and **diagonal reflection stripes**
- **Aged/yellowed face tint** for vintage clocks
- **7 tick mark styles:** lines, thick lines, dots, squares, triangles, rings, minimal
- **Roman and Arabic numerals** with 8 OpenCV font variations
- **Solid and textured faces** with random colours
- **Albumentations augmentation:** perspective, affine, brightness/contrast, hue/saturation, noise, blur, motion blur, CLAHE, shadow overlays, compression artefacts

---

## Tech Stack

* **Language:** Python 3.12
* **Package Manager:** uv
* **Deep Learning Framework:** PyTorch
* **Detection & Segmentation:** Ultralytics YOLOv8
* **Image Processing:** OpenCV, PIL, Albumentations
* **Testing:** pytest (42 tests)
* **Visualization:** Matplotlib, NumPy

---

**Developed as part of a research project in deep learning.**
