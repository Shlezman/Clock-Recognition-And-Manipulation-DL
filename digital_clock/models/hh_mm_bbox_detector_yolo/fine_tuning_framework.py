"""
YOLO Bounding Box Labeling Tool for Digital Watch Time Detection
Usage: python label_tool.py --input_dir ./images --output_dir ./dataset

Controls:
- Left click and drag: Draw bounding box
- 'h' key: Switch to 'hours' label (red)
- 'm' key: Switch to 'minutes' label (green)
- 'd' key: Delete last bounding box
- 'n' key or Right arrow: Next image (saves current)
- 'p' key or Left arrow: Previous image
- 'q' key or ESC: Quit and save
- 's' key: Save current image labels
"""

import cv2
import os
import argparse
import yaml
from pathlib import Path
import shutil

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
        self.current_label = 0  # 0: hours, 1: minutes
        self.label_names = {0: 'hours', 1: 'minutes'}
        self.label_colors = {0: (0, 0, 255), 1: (0, 255, 0)}  # BGR: red, green

        # Drawing state
        self.drawing = False
        self.start_point = None
        self.bboxes = []  # List of (label, x_center, y_center, width, height)

        # Window name
        self.window_name = 'YOLO BBox Labeler'

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

        # Add bbox
        self.bboxes.append((self.current_label, x_center, y_center, width, height))
        print(f"Added bbox: {self.label_names[self.current_label]} - "
              f"center=({x_center:.3f}, {y_center:.3f}), "
              f"size=({width:.3f}, {height:.3f})")

    def draw_display(self):
        display = self.current_image.copy()
        h, w = display.shape[:2]

        # Draw existing bboxes
        for label, x_c, y_c, bbox_w, bbox_h in self.bboxes:
            x1 = int((x_c - bbox_w/2) * w)
            y1 = int((y_c - bbox_h/2) * h)
            x2 = int((x_c + bbox_w/2) * w)
            y2 = int((y_c + bbox_h/2) * h)

            color = self.label_colors[label]
            cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)
            cv2.putText(display, self.label_names[label], (x1, y1-5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Draw current bbox being drawn
        if self.drawing and self.start_point and hasattr(self, 'end_point'):
            x1, y1 = self.start_point
            x2, y2 = self.end_point
            color = self.label_colors[self.current_label]
            cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)

        # Draw minimal status bar at bottom with semi-transparent background
        bar_height = 30
        overlay = display.copy()
        cv2.rectangle(overlay, (0, h - bar_height), (w, h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, display, 0.3, 0, display)

        status_text = f"[{self.current_idx + 1}/{len(self.image_files)}] {self.current_file.name} | Label: {self.label_names[self.current_label]} | Bboxes: {len(self.bboxes)}"
        cv2.putText(display, status_text, (10, h - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        cv2.imshow(self.window_name, display)

    def load_image(self):
        self.current_file = self.image_files[self.current_idx]
        self.current_image = cv2.imread(str(self.current_file))

        if self.current_image is None:
            print(f"Error loading {self.current_file}")
            return False

        # Load existing labels if they exist
        label_file = self.labels_dir / f"{self.current_file.stem}.txt"
        self.bboxes = []

        if label_file.exists():
            with open(label_file, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        label = int(parts[0])
                        coords = [float(x) for x in parts[1:]]
                        self.bboxes.append((label, *coords))
            print(f"Loaded {len(self.bboxes)} existing bboxes")

        return True

    def save_labels(self):
        if not self.bboxes:
            print("No bboxes to save")
            return

        # Save image as PNG
        output_image_path = self.images_dir / f"{self.current_file.stem}.png"
        cv2.imwrite(str(output_image_path), self.current_image)

        # Save labels in YOLO format
        label_file = self.labels_dir / f"{self.current_file.stem}.txt"
        with open(label_file, 'w') as f:
            for label, x_c, y_c, w, h in self.bboxes:
                f.write(f"{label} {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}\n")

        print(f"Saved: {output_image_path.name} with {len(self.bboxes)} bboxes")

    def create_dataset_yaml(self):
        yaml_content = {
            'path': str(self.output_dir.absolute()),
            'train': 'images',
            'val': 'images',
            'nc': 2,
            'names': {0: 'hours', 1: 'minutes'}
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

            elif key == ord('h'):
                self.current_label = 0
                print(f"Switched to: {self.label_names[0]}")
                self.draw_display()

            elif key == ord('m'):
                self.current_label = 1
                print(f"Switched to: {self.label_names[1]}")
                self.draw_display()

            elif key == ord('d'):
                if self.bboxes:
                    removed = self.bboxes.pop()
                    print(f"Deleted bbox: {self.label_names[removed[0]]}")
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

def main():
    parser = argparse.ArgumentParser(description='YOLO BBox Labeling Tool for Watch Time Detection')
    parser.add_argument('--input_dir', type=str, required=True,
                       help='Directory containing input images')
    parser.add_argument('--output_dir', type=str, default='./raw_fine_tuning_dataset',
                       help='Directory for output dataset (default: ./raw_fine_tuning_dataset)')

    args = parser.parse_args()

    labeler = YOLOBboxLabeler(args.input_dir, args.output_dir)
    labeler.run()

if __name__ == '__main__':
    main()