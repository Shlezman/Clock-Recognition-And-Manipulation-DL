"""
High-level pipeline runners for the full Clock-Recognition-And-Manipulation
notebook (full-pipeline.ipynb).

These wrappers hide the long technical glue from the notebook so it stays
short and reads like a presentation rather than a script.

Stages:
  1. Digital time recognition  (YOLO digit boxes + SVHN-style CNN)
  2. Sketch-cGAN clock manipulation
  3. Mask-guided inpainting clock manipulation
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from ultralytics import YOLO

from analog_clock.analog_sketch_creator import draw_analog_clock
from analog_clock.GAN.inpainting.generator_model import InpaintGenerator
from analog_clock.GAN.sketch.generator_model import GeneratorUNet
from analog_clock.pipeline_utils import (
    find_clock_center,
    get_angle_clockwise_from_12,
    load_time_recognition_cnn,
    recognize_time_from_masks,
    recompose_hand,
    time_to_degrees_cw,
)
from digital_clock.svhn_digit_recognition_cnn.svhn_cnn_model import SVHNModel


def _to_inference(module: torch.nn.Module) -> torch.nn.Module:
    """Switch to inference mode (equivalent to module.eval())."""
    module.train(False)
    return module


# ============================================================================
# Result dataclasses
# ============================================================================

@dataclass
class DigitalResult:
    image: Image.Image
    crops: Dict[str, Image.Image]
    hh: int
    mm: int
    sketch: np.ndarray


@dataclass
class SketchResult:
    original: Image.Image
    detected_crop: Image.Image
    target_sketch: Image.Image
    final: Image.Image
    crop_coords: Tuple[int, int, int, int]
    target_hh: int
    target_mm: int


@dataclass
class InpaintResult:
    img_cv: np.ndarray
    mask_display: np.ndarray
    clean_bg: np.ndarray
    result: np.ndarray
    center: Tuple[int, int]
    best_mask_h: np.ndarray
    best_mask_m: np.ndarray
    angle_h_orig: float
    angle_m_orig: float
    target_hh: int
    target_mm: int


# ============================================================================
# Stage 1 — Digital time recognition
# ============================================================================

_DIGIT_PREPROCESS = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
])


def _crop_bbox(image: Image.Image, bbox, pad_pct: float = 0.1) -> Image.Image:
    w, h = image.size
    x1, y1, x2, y2 = bbox
    pad_x = (x2 - x1) * pad_pct
    pad_y = (y2 - y1) * pad_pct
    return image.crop((
        max(0, int(x1 - pad_x)),
        max(0, int(y1 - pad_y)),
        min(w, int(x2 + pad_x)),
        min(h, int(y2 + pad_y)),
    ))


def _decode_digits(predictions) -> int:
    digits = []
    for head in predictions:
        idx = torch.argmax(head, dim=1).item()
        if idx != 10:
            digits.append(str(idx))
    return int("".join(digits)) if digits else 0


def load_digital_models(
    yolo_path: str,
    cnn_path: str,
    device: torch.device,
) -> Tuple[YOLO, SVHNModel]:
    yolo = YOLO(yolo_path)
    cnn = SVHNModel(num_channels=1).to(device)
    checkpoint = torch.load(cnn_path, map_location=device, weights_only=False)
    state = checkpoint.get("model_state_dict", checkpoint)
    cnn.load_state_dict(state)
    return yolo, _to_inference(cnn)


def recognize_digital_time(
    image_path: str,
    yolo: YOLO,
    cnn: SVHNModel,
    device: torch.device,
) -> DigitalResult:
    image = Image.open(image_path).convert("RGB")
    res = yolo(image, verbose=False)[0]
    names = res.names

    crops: Dict[str, Image.Image] = {}
    hh = mm = 0
    for box in res.boxes:
        label = names[int(box.cls[0])]
        crop = _crop_bbox(image, box.xyxy[0].tolist(), pad_pct=0.1)
        crops[label] = crop
        with torch.no_grad():
            tensor = _DIGIT_PREPROCESS(crop).unsqueeze(0).to(device)
            number = _decode_digits(cnn(tensor))
        if label in ("hours", "hh", "0"):
            hh = number
        elif label in ("minutes", "mm", "1"):
            mm = number

    sketch = draw_analog_clock(hh, mm, return_array=True)
    return DigitalResult(image=image, crops=crops, hh=hh, mm=mm, sketch=sketch)


def show_digital(result: DigitalResult) -> None:
    fig, ax = plt.subplots(1, 4, figsize=(15, 4.5))
    ax[0].imshow(result.image); ax[0].set_title("Digital Clock"); ax[0].axis("off")

    hh_crop = result.crops.get("hours", result.crops.get("hh"))
    mm_crop = result.crops.get("minutes", result.crops.get("mm"))
    if hh_crop is not None:
        ax[1].imshow(hh_crop); ax[1].set_title(f"Hours → {result.hh}")
    ax[1].axis("off")
    if mm_crop is not None:
        ax[2].imshow(mm_crop); ax[2].set_title(f"Minutes → {result.mm:02d}")
    ax[2].axis("off")

    ax[3].imshow(result.sketch); ax[3].set_title(f"Sketch  {result.hh}:{result.mm:02d}")
    ax[3].axis("off")
    plt.tight_layout()
    plt.show()


# ============================================================================
# Stage 2 — Sketch-cGAN
# ============================================================================

_SKETCH_TRANSFORM = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])


def load_sketch_models(
    yolo_clock_path: str,
    generator_path: str,
    device: torch.device,
) -> Tuple[YOLO, GeneratorUNet, transforms.Compose]:
    yolo = YOLO(yolo_clock_path)
    generator = GeneratorUNet(in_channels=6, out_channels=3).to(device)
    state = torch.load(generator_path, map_location=device, weights_only=False)
    generator.load_state_dict(state)
    return yolo, _to_inference(generator), _SKETCH_TRANSFORM


def _detect_clock_crop(
    image: Image.Image,
    yolo: YOLO,
    pad_pct: float = 0.1,
) -> Tuple[Image.Image, Tuple[int, int, int, int]]:
    res = yolo(image, verbose=False)
    if not res or len(res[0].boxes) == 0:
        return image, (0, 0, image.width, image.height)
    x1, y1, x2, y2 = res[0].boxes[0].xyxy[0].tolist()
    pad_w = (x2 - x1) * pad_pct
    pad_h = (y2 - y1) * pad_pct
    coords = (
        max(0, int(x1 - pad_w)),
        max(0, int(y1 - pad_h)),
        min(image.width, int(x2 + pad_w)),
        min(image.height, int(y2 + pad_h)),
    )
    return image.crop(coords), coords


def _denormalize(tensor: torch.Tensor) -> Image.Image:
    arr = tensor.cpu().detach().squeeze(0)
    arr = (arr * 0.5 + 0.5).permute(1, 2, 0).numpy().clip(0, 1)
    return Image.fromarray((arr * 255).astype(np.uint8))


def run_sketch_cgan(
    image_path: str,
    target_hh: int,
    target_mm: int,
    yolo: YOLO,
    generator: GeneratorUNet,
    transform: transforms.Compose,
    device: torch.device,
) -> SketchResult:
    original = Image.open(image_path).convert("RGB")
    crop, coords = _detect_clock_crop(original, yolo)

    src = transform(crop).unsqueeze(0).to(device)
    sketch_arr = draw_analog_clock(target_hh, target_mm, return_array=True)
    sketch_pil = Image.fromarray(sketch_arr).convert("RGB").resize((256, 256))
    skc = transform(sketch_pil).unsqueeze(0).to(device)

    with torch.no_grad():
        generated = generator(torch.cat((src, skc), dim=1))
    gen_pil = _denormalize(generated)

    x1, y1, x2, y2 = coords
    final = original.copy()
    final.paste(gen_pil.resize((x2 - x1, y2 - y1), Image.LANCZOS), (x1, y1))

    return SketchResult(
        original=original,
        detected_crop=crop,
        target_sketch=sketch_pil,
        final=final,
        crop_coords=coords,
        target_hh=target_hh,
        target_mm=target_mm,
    )


def show_sketch(result: SketchResult) -> None:
    fig, ax = plt.subplots(1, 4, figsize=(20, 5.5))
    ax[0].imshow(result.original); ax[0].set_title("Original"); ax[0].axis("off")
    ax[1].imshow(result.detected_crop); ax[1].set_title("Detected Clock"); ax[1].axis("off")
    ax[2].imshow(result.target_sketch)
    ax[2].set_title(f"Target Sketch  {result.target_hh}:{result.target_mm:02d}")
    ax[2].axis("off")
    ax[3].imshow(result.final); ax[3].set_title("Sketch-cGAN Result"); ax[3].axis("off")
    plt.tight_layout()
    plt.show()


# ============================================================================
# Stage 3 — Inpainting GAN
# ============================================================================

_INPAINT_NORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
])


@dataclass
class InpaintingModels:
    yolo_hands: YOLO
    igan: InpaintGenerator
    time_cnn: object
    transform_norm: transforms.Compose = field(default_factory=lambda: _INPAINT_NORM)


def load_inpainting_models(
    yolo_hands_path: str,
    igan_path: str,
    time_cnn_path: str,
    device: torch.device,
) -> InpaintingModels:
    yolo = YOLO(yolo_hands_path)
    igan = InpaintGenerator().to(device)
    igan.load_state_dict(torch.load(igan_path, map_location=device, weights_only=False))
    _to_inference(igan)
    time_cnn = load_time_recognition_cnn(time_cnn_path, device)
    return InpaintingModels(yolo_hands=yolo, igan=igan, time_cnn=time_cnn)


def recognize_analog_time(
    image_path: str,
    models: InpaintingModels,
    device: torch.device,
    img_size: int = 256,
) -> Tuple[int, int]:
    img = np.array(Image.open(image_path).convert("RGB").resize((img_size, img_size)))
    res = models.yolo_hands(img, conf=0.05, verbose=False)
    if res[0].masks is None:
        return 0, 0
    combined = np.zeros((img_size, img_size), dtype=np.float32)
    for m in res[0].masks.data.cpu().numpy():
        combined = np.maximum(combined, cv2.resize(m, (img_size, img_size), interpolation=cv2.INTER_NEAREST))
    return recognize_time_from_masks(combined, models.time_cnn, device)


def _filter_detections(detections, shape):
    if not detections:
        zero = np.zeros(shape, dtype=np.uint8)
        return zero, zero
    detections.sort(key=lambda x: x[1], reverse=True)
    best = detections[0][0]
    union = np.zeros(shape, dtype=np.uint8)
    for m, _ in detections:
        union = np.maximum(union, m)
    return best, union


def run_inpainting(
    image_path: str,
    target_hh: int,
    target_mm: int,
    models: InpaintingModels,
    device: torch.device,
) -> InpaintResult:
    img_pil = Image.open(image_path).convert("RGB")
    img_resized = img_pil.resize((256, 256))
    img_cv = np.array(img_resized)
    h, w, _ = img_cv.shape

    res = models.yolo_hands(img_cv, conf=0.05, verbose=False)[0]
    dets_h, dets_m, dets_s = [], [], []
    if res.masks is not None:
        masks_data = res.masks.data.cpu().numpy()
        classes = res.boxes.cls.cpu().numpy()
        confs = res.boxes.conf.cpu().numpy()
        for i, cls in enumerate(classes):
            m = cv2.resize(masks_data[i], (w, h), interpolation=cv2.INTER_NEAREST)
            target = (dets_h, dets_m, dets_s)[int(cls)] if int(cls) in (0, 1, 2) else None
            if target is not None:
                target.append((m, confs[i]))
    if not dets_m and dets_s:
        dets_m, dets_s = dets_s, []

    best_h, all_h = _filter_detections(dets_h, (h, w))
    best_m, all_m = _filter_detections(dets_m, (h, w))
    best_s, _ = _filter_detections(dets_s, (h, w))

    full_mask = (np.maximum(all_h, all_m) > 0.5).astype(np.float32)
    dilated_cgan = cv2.dilate(full_mask, np.ones((7, 7), np.uint8), iterations=1)
    dilated_inp = cv2.dilate(full_mask, np.ones((11, 11), np.uint8), iterations=1)

    img_t = models.transform_norm(img_resized).to(device)
    mask_t = ((torch.from_numpy(dilated_cgan).unsqueeze(0).to(device)) - 0.5) / 0.5
    inp = torch.cat([img_t, mask_t], dim=0).unsqueeze(0)
    with torch.no_grad():
        out = models.igan(inp).squeeze().permute(1, 2, 0).cpu().numpy()
    clean = ((out * 0.5 + 0.5) * 255).clip(0, 255).astype(np.uint8)
    clean = cv2.inpaint(clean, (dilated_inp * 255).astype(np.uint8), 3, cv2.INPAINT_TELEA)

    center = find_clock_center(img_cv, best_h, best_m, best_s)
    angle_h = get_angle_clockwise_from_12(best_h, center)
    angle_m = get_angle_clockwise_from_12(best_m, center)

    final = recompose_hand(
        best_h, img_cv, clean, center,
        angle_h, time_to_degrees_cw(target_hh, target_mm, "hour"),
        feather_radius=5,
    )
    final = recompose_hand(
        best_m, img_cv, final, center,
        angle_m, time_to_degrees_cw(target_hh, target_mm, "minute"),
        feather_radius=5,
    )
    cv2.circle(final, center, 4, (255, 0, 0), -1)

    return InpaintResult(
        img_cv=img_cv,
        mask_display=(dilated_cgan * 255).astype(np.uint8),
        clean_bg=clean,
        result=final,
        center=center,
        best_mask_h=best_h,
        best_mask_m=best_m,
        angle_h_orig=angle_h,
        angle_m_orig=angle_m,
        target_hh=target_hh,
        target_mm=target_mm,
    )


def show_inpainting(result: InpaintResult) -> None:
    fig, ax = plt.subplots(1, 4, figsize=(20, 5.5))
    panels = [
        (result.img_cv, "Original"),
        (result.mask_display, "Detected Hands"),
        (result.clean_bg, "Hands Removed"),
        (result.result, f"Inpaint Result  {result.target_hh}:{result.target_mm:02d}"),
    ]
    for axis, (img, title) in zip(ax, panels):
        cmap = "gray" if title == "Detected Hands" else None
        axis.imshow(img, cmap=cmap)
        axis.set_title(title); axis.axis("off")
    plt.tight_layout()
    plt.show()
