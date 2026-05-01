# Clock Hand Annotator — YOLO-Seg Dataset Builder

A desktop annotation tool for creating YOLO-Seg segmentation datasets
to train object detection models to find clock hands (hour, minute, second).

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

Python 3.9+ required.

### 2. Run the tool

```bash
python annotator.py
```

---

## Usage Guide

### Open images
Click **📂 Open Image Folder** and select a directory containing clock images.
Supported formats: JPG, JPEG, PNG, BMP, TIFF, WEBP.

### Select a class
Click a class button (or press **1**, **2**, **3**):
- 🔴 **1 — Hour Hand**
- 🔵 **2 — Minute Hand**
- 🟢 **3 — Second Hand**

### Draw a segmentation mask

**Polygon mode** (recommended, press **P**):
1. Click to place polygon vertices around the clock hand.
2. Double-click **or** click near the first vertex (yellow circle) to close the polygon.
3. Press **Enter** to close, **Escape** to cancel.

**Freehand mode** (press **F**):
1. Click and drag to paint around the clock hand.
2. Release the mouse to finish. The path is auto-simplified.

### Edit / erase
Press **E** or click **🧹 Eraser**, then click on any annotation to remove it.

### Navigate images
- **▶ Next** / **◀ Prev** buttons or press **D** / **A**
- Auto-save is enabled by default (toggle with checkbox).

### Save
- **💾 Save Annotation** (Ctrl+S) — saves to a sidecar `.ann.json` file next to the image.
- Auto-save fires whenever you switch images.

---

## Keyboard Shortcuts

| Key | Action |
|-----|--------|
| 1 / 2 / 3 | Select Hour / Minute / Second class |
| A / D | Previous / Next image |
| P | Polygon draw mode |
| F | Freehand draw mode |
| E | Toggle eraser |
| Enter | Close polygon |
| Escape | Cancel current drawing |
| Ctrl+S | Save annotation |
| Mouse wheel | Zoom in/out |
| Middle-click drag | Pan |

---

## Exporting the Dataset

1. Annotate your images (all three classes recommended per image).
2. Click **📦 Export Dataset…**.
3. Set the Train % (default 80%) and choose an output folder.
4. Click OK.

### Output structure

```
dataset/
├── images/
│   ├── train/   ← 80% of annotated images
│   └── val/     ← 20% of annotated images
├── labels/
│   ├── train/   ← YOLO .txt label files
│   └── val/
└── data.yaml    ← Training configuration
```

### YOLO-Seg label format

Each `.txt` file contains up to 3 lines (one per class instance):

```
<class_id> x1 y1 x2 y2 x3 y3 ... xn yn
```

- `class_id`: 0 = hour, 1 = minute, 2 = second
- All coordinates are **normalized** between 0.0 and 1.0
- Points form a closed polygon tracing the clock hand

Example:
```
0 0.421 0.312 0.435 0.298 0.450 0.310 0.448 0.325
1 0.500 0.150 0.510 0.160 0.508 0.500 0.495 0.500
```

### data.yaml (YOLOv8-compatible)

```yaml
path: /absolute/path/to/dataset
train: images/train
val:   images/val
nc:    3
names:
  0: hour
  1: minute
  2: second
```

---

## Training with YOLOv8

```bash
pip install ultralytics
yolo segment train model=yolov8n-seg.pt data=/path/to/dataset/data.yaml epochs=100 imgsz=640
```

---

## Annotation Tips

- Trace **only the hand itself**, not the clock body.
- Use ~8–15 polygon vertices for accuracy without excessive points.
- For thin hands, a slim elongated polygon is sufficient.
- The second hand is usually the thinnest — freehand mode may be easier.
- Annotate all three hands per image when visible for best model performance.

---

## Project Structure

```
clock_annotator/
├── annotator.py        ← Main UI (PyQt5 window + canvas)
├── annotation_logic.py ← Annotation persistence (JSON sidecars)
├── yolo_export.py      ← Dataset export & data.yaml generation
├── requirements.txt
└── README.md
```
