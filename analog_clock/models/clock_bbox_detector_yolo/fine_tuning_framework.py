"""
YOLO Bounding Box Labeling Tool for Analog Watch Detection
Usage: python label_tool.py --input_dir ./images --output_dir ./dataset

Controls:
- Left click and drag: Draw bounding box around watch
- 'd' key: Delete bounding box
- 'n' key or Right arrow: Next image (saves current)
- 'p' key or Left arrow: Previous image
- 'q' key or ESC: Quit and save
- 's' key: Save current image labels

Dataset Merging:
python label_tool.py --merge --new_dataset ./new_dataset --existing_dataset ./existing_dataset --output_dir ./merged_dataset --multiply 5
"""

import cv2
import os
import argparse
import yaml
from pathlib import Path
import shutil
import random
import numpy as np


class YOLOBboxLabeler:
    def __init__(self, input_dir, output_dir):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)

        # Create output directory structure
        self.images_dir = self.output_dir / 'images'
        self.labels_dir = self.output_dir / 'labels'
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.labels_dir.mkdir(parents=True, exist_ok=True)

        # Load images
        self.image_files = sorted([
            f for f in self.input_dir.glob('*')
            if f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']
        ])

        if not self.image_files:
            raise ValueError(f"No images found in {input_dir}")

        print(f"Found {len(self.image_files)} images")

        # State variables
        self.current_idx = 0
        self.label_color = (0, 255, 0)  # Green in BGR

        # Drawing state
        self.drawing = False
        self.start_point = None
        self.bbox = None  # Single bbox: (x_center, y_center, width, height)

        # Window name
        self.window_name = 'Analog Watch BBox Labeler'

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.start_point = (x, y)

        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing:
                self.end_point = (x, y)
                self.draw_display()

        elif event == cv2.EVENT_LBUTTONUP:
            if self.drawing and self.start_point:
                self.drawing = False
                self.end_point = (x, y)
                self.add_bbox()
                self.draw_display()

    def add_bbox(self):
        if not self.start_point or not self.end_point:
            return

        x1, y1 = self.start_point
        x2, y2 = self.end_point

        # Ensure x1 < x2 and y1 < y2
        x1, x2 = min(x1, x2), max(x1, x2)
        y1, y2 = min(y1, y2), max(y1, y2)

        # Convert to YOLO format (normalized)
        h, w = self.current_image.shape[:2]
        x_center = ((x1 + x2) / 2) / w
        y_center = ((y1 + y2) / 2) / h
        width = (x2 - x1) / w
        height = (y2 - y1) / h

        # Replace existing bbox (only one allowed)
        self.bbox = (x_center, y_center, width, height)
        print(f"Added watch bbox: center=({x_center:.3f}, {y_center:.3f}), "
              f"size=({width:.3f}, {height:.3f})")

    def draw_display(self):
        display = self.current_image.copy()
        h, w = display.shape[:2]

        # Draw existing bbox
        if self.bbox:
            x_c, y_c, bbox_w, bbox_h = self.bbox
            x1 = int((x_c - bbox_w / 2) * w)
            y1 = int((y_c - bbox_h / 2) * h)
            x2 = int((x_c + bbox_w / 2) * w)
            y2 = int((y_c + bbox_h / 2) * h)

            cv2.rectangle(display, (x1, y1), (x2, y2), self.label_color, 2)
            cv2.putText(display, 'analog_clock', (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.label_color, 2)

        # Draw current bbox being drawn
        if self.drawing and self.start_point and hasattr(self, 'end_point'):
            x1, y1 = self.start_point
            x2, y2 = self.end_point
            cv2.rectangle(display, (x1, y1), (x2, y2), self.label_color, 2)

        # Draw minimal status bar at bottom with semi-transparent background
        bar_height = 30
        overlay = display.copy()
        cv2.rectangle(overlay, (0, h - bar_height), (w, h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, display, 0.3, 0, display)

        bbox_status = "1 bbox" if self.bbox else "No bbox"
        status_text = f"[{self.current_idx + 1}/{len(self.image_files)}] {self.current_file.name} | {bbox_status}"
        cv2.putText(display, status_text, (10, h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        cv2.imshow(self.window_name, display)

    def load_image(self):
        self.current_file = self.image_files[self.current_idx]
        self.current_image = cv2.imread(str(self.current_file))

        if self.current_image is None:
            print(f"Error loading {self.current_file}")
            return False

        # Load existing label if it exists
        label_file = self.labels_dir / f"{self.current_file.stem}.txt"
        self.bbox = None

        if label_file.exists():
            with open(label_file, 'r') as f:
                line = f.readline().strip()
                if line:
                    parts = line.split()
                    if len(parts) == 5:
                        # Format: class x_center y_center width height
                        coords = [float(x) for x in parts[1:]]
                        self.bbox = tuple(coords)
                        print(f"Loaded existing bbox")

        return True

    def save_labels(self):
        if not self.bbox:
            print("No bbox to save - skipping")
            return

        # Save image as PNG
        output_image_path = self.images_dir / f"{self.current_file.stem}.png"
        cv2.imwrite(str(output_image_path), self.current_image)

        # Save label in YOLO format (class 0 for analog_clock)
        label_file = self.labels_dir / f"{self.current_file.stem}.txt"
        with open(label_file, 'w') as f:
            x_c, y_c, w, h = self.bbox
            f.write(f"0 {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}\n")

        print(f"Saved: {output_image_path.name} with 1 bbox")

    def create_dataset_yaml(self):
        yaml_content = {
            'train': 'images/train',
            'val': 'images/val',
            'nc': 1,
            'names': ['analog_clock']
        }

        yaml_file = self.output_dir / 'dataset.yaml'
        with open(yaml_file, 'w') as f:
            yaml.dump(yaml_content, f, default_flow_style=False)

        print(f"\nCreated dataset.yaml at {yaml_file}")

    def run(self):
        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)

        if not self.load_image():
            return

        self.draw_display()

        while True:
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q') or key == 27:  # q or ESC
                self.save_labels()
                break

            elif key == ord('d'):
                if self.bbox:
                    self.bbox = None
                    print("Deleted bbox")
                    self.draw_display()

            elif key == ord('s'):
                self.save_labels()

            elif key == ord('n') or key == 83:  # n or right arrow
                self.save_labels()
                if self.current_idx < len(self.image_files) - 1:
                    self.current_idx += 1
                    self.load_image()
                    self.draw_display()
                else:
                    print("Already at last image")

            elif key == ord('p') or key == 81:  # p or left arrow
                self.save_labels()
                if self.current_idx > 0:
                    self.current_idx -= 1
                    self.load_image()
                    self.draw_display()
                else:
                    print("Already at first image")

        cv2.destroyAllWindows()
        self.create_dataset_yaml()
        print(f"\nLabeling complete! Dataset saved to: {self.output_dir}")
        print(f"Total images processed: {len(list(self.images_dir.glob('*.png')))}")


def augment_image(image, bbox):
    """Apply random augmentation to image and adjust bbox accordingly"""
    h, w = image.shape[:2]
    x_c, y_c, bbox_w, bbox_h = bbox

    # Convert normalized coords to pixel coords
    x_center_px = x_c * w
    y_center_px = y_c * h
    width_px = bbox_w * w
    height_px = bbox_h * h

    # Random brightness adjustment
    brightness_factor = random.uniform(0.7, 1.3)
    image = np.clip(image * brightness_factor, 0, 255).astype(np.uint8)

    # Random horizontal flip
    if random.random() > 0.5:
        image = cv2.flip(image, 1)
        x_center_px = w - x_center_px

    # Random rotation (small angle)
    angle = random.uniform(-15, 15)
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    image = cv2.warpAffine(image, M, (w, h))

    # Transform bbox center
    center_point = np.array([x_center_px, y_center_px, 1])
    new_center = M @ center_point
    x_center_px, y_center_px = new_center

    # Normalize back
    x_c = x_center_px / w
    y_c = y_center_px / h

    # Ensure bbox stays within bounds
    x_c = max(bbox_w / 2, min(1 - bbox_w / 2, x_c))
    y_c = max(bbox_h / 2, min(1 - bbox_h / 2, y_c))

    return image, (x_c, y_c, bbox_w, bbox_h)


def merge_datasets(new_dataset_path, existing_dataset_path, output_path, multiply_factor=1):
    """
    Merge new dataset with existing dataset, multiplying new dataset images with augmentation

    Args:
        new_dataset_path: Path to the newly labeled dataset (small)
        existing_dataset_path: Path to the existing dataset (large)
        output_path: Path where merged dataset will be saved
        multiply_factor: How many times to multiply the new dataset (with augmentation)
    """
    new_dataset = Path(new_dataset_path)
    existing_dataset = Path(existing_dataset_path)
    output = Path(output_path)

    # Create output structure
    train_img_dir = output / 'images' / 'train'
    train_lbl_dir = output / 'labels' / 'train'
    val_img_dir = output / 'images' / 'val'
    val_lbl_dir = output / 'labels' / 'val'

    for dir_path in [train_img_dir, train_lbl_dir, val_img_dir, val_lbl_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 60}")
    print(f"MERGING DATASETS")
    print(f"{'=' * 60}")

    # Copy existing dataset first
    print("\n[1/3] Copying existing dataset...")
    existing_train_imgs = existing_dataset / 'images' / 'train'
    existing_train_lbls = existing_dataset / 'labels' / 'train'
    existing_val_imgs = existing_dataset / 'images' / 'val'
    existing_val_lbls = existing_dataset / 'labels' / 'val'

    existing_count = 0
    for split_img, split_lbl, out_img, out_lbl in [
        (existing_train_imgs, existing_train_lbls, train_img_dir, train_lbl_dir),
        (existing_val_imgs, existing_val_lbls, val_img_dir, val_lbl_dir)
    ]:
        if split_img.exists():
            for img_file in split_img.glob('*'):
                if img_file.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                    shutil.copy2(img_file, out_img / img_file.name)
                    lbl_file = split_lbl / f"{img_file.stem}.txt"
                    if lbl_file.exists():
                        shutil.copy2(lbl_file, out_lbl / lbl_file.name)
                        existing_count += 1

    print(f"   Copied {existing_count} images from existing dataset")

    # Process new dataset with multiplication and augmentation
    print(f"\n[2/3] Processing new dataset (multiplying by {multiply_factor}x with augmentation)...")
    new_images_dir = new_dataset / 'images'
    new_labels_dir = new_dataset / 'labels'

    new_image_files = list(new_images_dir.glob('*'))
    new_image_files = [f for f in new_image_files if f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']]

    new_count = 0
    augmented_count = 0

    for img_file in new_image_files:
        lbl_file = new_labels_dir / f"{img_file.stem}.txt"

        if not lbl_file.exists():
            print(f"   Warning: No label found for {img_file.name}, skipping")
            continue

        # Read bbox
        with open(lbl_file, 'r') as f:
            line = f.readline().strip()
            if not line:
                continue
            parts = line.split()
            bbox = tuple(float(x) for x in parts[1:])

        # Load image
        image = cv2.imread(str(img_file))
        if image is None:
            continue

        # Copy original
        original_name = f"new_{img_file.stem}_0{img_file.suffix}"
        cv2.imwrite(str(train_img_dir / original_name), image)
        with open(train_lbl_dir / f"new_{img_file.stem}_0.txt", 'w') as f:
            f.write(f"0 {bbox[0]:.6f} {bbox[1]:.6f} {bbox[2]:.6f} {bbox[3]:.6f}\n")
        new_count += 1

        # Create augmented versions
        for aug_idx in range(1, multiply_factor):
            aug_image, aug_bbox = augment_image(image.copy(), bbox)
            aug_name = f"new_{img_file.stem}_{aug_idx}{img_file.suffix}"

            cv2.imwrite(str(train_img_dir / aug_name), aug_image)
            with open(train_lbl_dir / f"new_{img_file.stem}_{aug_idx}.txt", 'w') as f:
                f.write(f"0 {aug_bbox[0]:.6f} {aug_bbox[1]:.6f} {aug_bbox[2]:.6f} {aug_bbox[3]:.6f}\n")
            augmented_count += 1

    print(f"   Added {new_count} original images from new dataset")
    print(f"   Created {augmented_count} augmented versions")

    # Create dataset.yaml
    print("\n[3/3] Creating dataset.yaml...")
    yaml_content = {
        'train': 'images/train',
        'val': 'images/val',
        'nc': 1,
        'names': ['analog_clock']
    }

    yaml_file = output / 'dataset.yaml'
    with open(yaml_file, 'w') as f:
        yaml.dump(yaml_content, f, default_flow_style=False)

    # Summary
    total_train = len(list(train_img_dir.glob('*')))
    total_val = len(list(val_img_dir.glob('*')))

    print(f"\n{'=' * 60}")
    print(f"MERGE COMPLETE!")
    print(f"{'=' * 60}")
    print(f"Existing dataset images: {existing_count}")
    print(f"New dataset images (original): {new_count}")
    print(f"New dataset images (augmented): {augmented_count}")
    print(f"\nTotal training images: {total_train}")
    print(f"Total validation images: {total_val}")
    print(f"Total images: {total_train + total_val}")
    print(f"\nMerged dataset saved to: {output}")
    print(f"{'=' * 60}\n")


def main():
    parser = argparse.ArgumentParser(description='YOLO BBox Labeling Tool for Analog Watch Detection')
    parser.add_argument('--input_dir', type=str,
                        help='Directory containing input images')
    parser.add_argument('--output_dir', type=str, default='./analog_watch_dataset',
                        help='Directory for output dataset')
    parser.add_argument('--merge', action='store_true',
                        help='Merge datasets instead of labeling')
    parser.add_argument('--new_dataset', type=str,
                        help='Path to new (small) dataset to merge')
    parser.add_argument('--existing_dataset', type=str,
                        help='Path to existing (large) dataset')
    parser.add_argument('--multiply', type=int, default=5,
                        help='Multiplication factor for new dataset (default: 5)')

    args = parser.parse_args()

    if args.merge:
        if not args.new_dataset or not args.existing_dataset:
            parser.error("--merge requires --new_dataset and --existing_dataset")
        merge_datasets(args.new_dataset, args.existing_dataset, args.output_dir, args.multiply)
    else:
        if not args.input_dir:
            parser.error("--input_dir is required for labeling mode")
        labeler = YOLOBboxLabeler(args.input_dir, args.output_dir)
        labeler.run()


if __name__ == '__main__':
    main()