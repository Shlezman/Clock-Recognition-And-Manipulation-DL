# 🕰️ TimeSync: Digital-to-Analog Clock Synchronization

> **An End-to-End Deep Learning Pipeline: From Digital Time Recognition to Generative Analog Clock Manipulation.**

This project implements a complex computer vision system that synchronizes an analog clock image to match a time read from a digital clock. The pipeline leverages state-of-the-art Deep Learning techniques, including **Object Detection (YOLO)**, **Instance Segmentation**, **CNN Classification**, and **Generative Adversarial Networks (cGAN/Inpainting)**.

---

## 🚀 System Architecture

The project consists of two main branches: the **Digital Branch** (Time Recognition) and the **Analog Branch** (Image Manipulation).

### 🔢 Part 1: Digital Clock Reader

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

### ⌚ Part 2: Analog Clock Manipulation

The goal of this module is to modify an input image of an analog clock so that its hands display the time recognized in Part 1.

#### Pre-processing: Clock Localization

* **Model:** YOLOv8.
* **Function:** Detects the analog clock within a larger scene and performs a crop to focus the workspace.

The system supports **two distinct modes** for manipulating the clock face:

### 🛠️ Mode 1: Sketch-Guided Generation (Pix2Pix Style)

This approach uses a Conditional GAN to "redraw" the clock based on a structural guide.

1. **Sketch Generation:** An algorithmic function receives the target time and generates a minimalist binary sketch (black & white) of a clock face pointing to that time.
2. **Generative Model (cGAN):**
* **Input:** The cropped analog clock image + The generated binary sketch.
* **Output:** A newly generated, photorealistic image of the clock, where the hands are aligned according to the sketch.



### 🎨 Mode 2: Segmentation & Inpainting (High-Fidelity)

This approach uses a "disassemble and reassemble" method to preserve the original background quality.

1. **Hand Segmentation:**
* **Model:** **YOLOv8-Seg** (Instance Segmentation).
* **Function:** Detects and segments the specific pixels of the hour, minute, and second hands.


2. **Mask Generation:** Creates a binary mask representing the exact area covered by the hands.
3. **Hand Removal (Inpainting):**
* **Model:** **Inpainting GAN**.
* **Input:** Original clock image + Binary Mask.
* **Output:** A "Clean Plate" — the clock background with the hands removed, where the hidden numbers/texture are reconstructed by the AI.


4. **Re-Composition:**
* A geometric function extracts the style of the hands (using the segmentation mask), rotates them mathematically to the target time, and blends them back onto the clean background.



---

## 💾 Datasets

The project utilizes a hybrid of public benchmarks and custom synthetic data:

* **Synthetic Digital Clocks:** A generator script creating thousands of variations of digital displays for YOLO training.
* **Synthetic 7-Segment Digits:** Used to augment the CNN classifier.
* **SVHN:** Street View House Numbers dataset used for robust digit feature learning.
* **Synthetic Analog Clocks:** Pairs of `(Image, Sketch)` and `(Image, Mask)` generated to train the cGAN and Inpainting models.

---
```mathematica
┌──────────────────────┐
│  Digital Clock Image │
└─────────────┬────────┘
              │
              ▼
┌────────────────────────────────────┐
│ Time Region Detection Model        │
│ (Bounding Box Localization)        │
└─────────────┬──────────────────────┘
              │
              ▼
┌────────────────────────────────────┐
│ Digit Recognition Model            │
└─────────────┬──────────────────────┘
              │
              ▼
┌──────────────────────┐
│ Extracted Time HH:MM │
└─────────────┬────────┘
              │
              │
              │
      ┌───────┴─────────────── Y-SPLIT ─────────────────┐
      ▼                                                 ▼

PATH 1: Analog Generation & Alignment           PATH 2: Hand Editing Pipeline

┌────────────────────────────────────┐         ┌──────────────────────────────┐
│ Function: Generate Minimal         │         │ Analog Clock Image (Input)   │
│     Analog Sketch                  │         └──────────────┬───────────────┘
└─────────────┬──────────────────────┘                        │
              │                                               ▼
              ▼                                ┌──────────────────────────────┐
┌────────────────────────────────────┐         │ Hand Detection Model         │
│         cGAN Model                 │         └──────────────┬───────────────┘
└─────────────┬──────────────────────┘                        │
              │                                               ▼
              ▼                                ┌──────────────────────────────┐
┌────────────────────────────────────┐         │ Hand Removal Model           │
│ Adjusted Analog Clock Image        │         └──────────────┬───────────────┘
└────────────────────────────────────┘                        │
                                                              ▼
                                                ┌──────────────────────────────┐
                                                │ Function: Reposition Hands   │
                                                │ According to Extracted Time  │
                                                └──────────────┬───────────────┘
                                                              ▼
                                                ┌──────────────────────────────┐
                                                │ Reconstructed Analog Clock   │
                                                │ Image                        │
                                                └──────────────────────────────┘

```
---

## 🛠️ Tech Stack

* **Language:** Python 3.x
* **Deep Learning Framework:** PyTorch
* **Detection & Segmentation:** Ultralytics YOLOv8
* **Image Processing:** OpenCV, PIL
* **Visualization:** Matplotlib, NumPy


**Developed as part of a research project in deep learning.**