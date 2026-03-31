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

#### Time Recognition from Hand Masks (NEW)

* **Model:** ClockHandCNN — a lightweight 6-block CNN trained on synthetic binary hand masks.
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
   * **Output:** A "Clean Plate" — the clock background with the hands removed, where the hidden numbers/texture are reconstructed by the AI.

4. **Re-Composition (Improved):**
   * Extracts the style of the hands using the segmentation mask with **feathered alpha blending** (Gaussian-blurred edges) for smooth, anti-aliased results.
   * Rotates the hands mathematically to the target time and blends them back onto the clean background.

5. **GIF Animation:** Generates an animated GIF showing the clock hands smoothly transitioning from the original time to the target time, with 2-minute intermediate steps.

---

## Pipeline Diagram

```
                    ┌──────────────────────┐
                    │  Digital Clock Image  │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌────────────────────────────────────┐
                    │ Time Region Detection Model        │
                    │ (Bounding Box Localization)        │
                    └──────────┬─────────────────────────┘
                               │
                               ▼
                    ┌────────────────────────────────────┐
                    │ Digit Recognition Model            │
                    └──────────┬─────────────────────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ Extracted Time HH:MM │
                    └──────────┬───────────┘
                               │
               ┌───────────────┴────────── Y-SPLIT ──────────────────┐
               ▼                                                     ▼

PATH 1: Sketch-Guided Generation               PATH 2: Segmentation & Inpainting

┌────────────────────────────────────┐        ┌──────────────────────────────┐
│ Time Recognition CNN               │        │ Analog Clock Image (Input)   │
│ (Recognise original time)          │        └──────────────┬───────────────┘
└─────────────┬──────────────────────┘                       │
              │                                              ▼
              ▼                               ┌──────────────────────────────┐
┌────────────────────────────────────┐        │ Time Recognition CNN         │
│ Generate Minimal Analog Sketch     │        │ (Recognise original time)    │
└─────────────┬──────────────────────┘        └──────────────┬───────────────┘
              │                                              │
              ▼                                              ▼
┌────────────────────────────────────┐        ┌──────────────────────────────┐
│ cGAN Model                         │        │ Hand Segmentation (YOLO-Seg) │
└─────────────┬──────────────────────┘        └──────────────┬───────────────┘
              │                                              │
              ▼                                              ▼
┌────────────────────────────────────┐        ┌──────────────────────────────┐
│ Animated GIF                       │        │ Hand Removal (Inpainting)    │
│ (Original → Target time)           │        └──────────────┬───────────────┘
└────────────────────────────────────┘                       │
                                                             ▼
                                              ┌──────────────────────────────┐
                                              │ Recompose Hands at Target    │
                                              │ Time (Feathered Alpha Blend) │
                                              └──────────────┬───────────────┘
                                                             │
                                                             ▼
                                              ┌──────────────────────────────┐
                                              │ Animated GIF                 │
                                              │ (Original → Target time)     │
                                              └──────────────────────────────┘
```

---

## Datasets

The project utilizes a hybrid of public benchmarks and custom synthetic data:

* **Synthetic Digital Clocks:** A generator script creating thousands of variations of digital displays for YOLO training.
* **Synthetic 7-Segment Digits:** Used to augment the CNN classifier.
* **SVHN:** Street View House Numbers dataset used for robust digit feature learning.
* **Synthetic Analog Clocks:** Pairs of `(Image, Sketch)` and `(Image, Mask)` generated to train the cGAN and Inpainting models.
* **Synthetic Clock-Hand Masks:** 15k+ binary masks with 21 hand styles for training the time recognition CNN.

---

## Project Structure

```
├── full-pipeline.ipynb                    # End-to-end orchestration notebook
├── digital_clock/
│   ├── yolo_detect_hh_mm/                 # YOLO digit localization
│   │   ├── dataset_creator/               # Synthetic dataset generator
│   │   └── fine_tuning_framework.py       # Interactive labeling tool
│   └── svhn_digit_recognition_cnn/        # CNN digit classifier
├── analog_clock/
│   ├── analog_sketch_creator.py           # Algorithmic sketch renderer
│   ├── pipeline_utils.py                  # Shared pipeline utilities (time recognition, GIF, blending)
│   ├── yolo_detect_clock/                 # YOLO clock localization
│   ├── yolo_detetct_hands/                # YOLO-Seg hand segmentation
│   │   └── tagging_platform/              # Custom annotation platform
│   ├── time_recognition_cnn/              # Time recognition from hand masks
│   │   ├── model.py                       # ClockHandCNN architecture
│   │   ├── dataset_generator.py           # Synthetic mask generator (21 styles)
│   │   ├── train.py                       # Training script
│   │   └── time_recognition_cnn.ipynb     # Training & evaluation notebook
│   └── GAN/
│       ├── sketch/                        # Sketch-guided cGAN (Pix2Pix)
│       └── inpainting/                    # Inpainting GAN + dataset generator
└── pyproject.toml                         # uv project configuration
```

---

## Quick Start

```bash
# Install dependencies (using uv)
uv sync

# Or using pip
pip install -r requirements.txt

# Run the full pipeline
jupyter notebook full-pipeline.ipynb
```

### Training Individual Models

```bash
# Digital clock models
jupyter notebook digital_clock/svhn_digit_recognition_cnn/svhn_cnn_model.ipynb
jupyter notebook digital_clock/yolo_detect_hh_mm/yolo.ipynb

# Analog clock models
jupyter notebook analog_clock/yolo_detect_clock/yolo.ipynb
jupyter notebook analog_clock/yolo_detetct_hands/yolo.ipynb
jupyter notebook analog_clock/GAN/sketch/sketch-cGAN-model.ipynb
jupyter notebook analog_clock/GAN/inpainting/inpainting_GAN_model.ipynb

# Time recognition CNN
jupyter notebook analog_clock/time_recognition_cnn/time_recognition_cnn.ipynb
```

### Generating Synthetic Datasets

```bash
python digital_clock/yolo_detect_hh_mm/dataset_creator/dataset_generator.py
python analog_clock/GAN/sketch/dataset_generator/main.py
python analog_clock/GAN/inpainting/dataset_generator/main.py
python analog_clock/time_recognition_cnn/dataset_generator.py --n_samples 20000
```

---

## Tech Stack

* **Language:** Python 3.12
* **Package Manager:** uv
* **Deep Learning Framework:** PyTorch
* **Detection & Segmentation:** Ultralytics YOLOv8
* **Image Processing:** OpenCV, PIL
* **Visualization:** Matplotlib, NumPy

---

**Developed as part of a research project in deep learning.**
