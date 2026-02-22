# Crop And Remap Hand Annotations

This utility creates a new dataset folder with:
- Cropped clock images (clock detected by `analog_clock_yolo_model.pt`)
- Remapped hand annotations (`.ann.json`) in the crop coordinate frame

It does not modify the original images or annotations.

## Input
- Source folder with image + same-stem sidecar:
  - `image.jpg` + `image.ann.json` (preferred), or
  - `image.jpg` + `image.json`

## Output
- Default output folder: `analog_clock/data/from_internet_cropped`
- Each saved pair:
  - `clock_crop_00000.jpg` (or png/webp/etc.)
  - `clock_crop_00000.ann.json`
- Metadata files:
  - `analog_clock/data/from_internet_cropped/_meta/mapping.csv`
  - `analog_clock/data/from_internet_cropped/_meta/summary.json`

## Run
From repo root:

```powershell
python analog_clock/yolo_detetct_hands/crop_with_clock_detector/crop_and_remap_annotations.py --overwrite
```

Optional flags:
- `--device 0` or `--device cpu`
- `--margin 0.15`
- `--conf 0.20`
- `--max-images 50`
- `--source-dir <path>`
- `--output-dir <path>`

