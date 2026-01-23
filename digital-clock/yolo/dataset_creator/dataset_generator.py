import os
import random
import shutil
from font_manager import FontManager
from background_manager import BackgroundManager
from image_generator import ImageGenerator
from clock_renderer import ClockRenderer


class ClockDatasetGenerator:
    """Main class for generating synthetic clock dataset for YOLO"""

    def __init__(self, output_dir="clock_dataset"):
        self.output_dir = output_dir

        # Create temporary directories
        self.temp_images_dir = os.path.join(output_dir, "temp_images")
        self.temp_labels_dir = os.path.join(output_dir, "temp_labels")

        # Cleanup old temp directories
        if os.path.exists(self.temp_images_dir):
            shutil.rmtree(self.temp_images_dir)
        if os.path.exists(self.temp_labels_dir):
            shutil.rmtree(self.temp_labels_dir)

        os.makedirs(self.temp_images_dir, exist_ok=True)
        os.makedirs(self.temp_labels_dir, exist_ok=True)

        # Initialize managers
        print("Initializing components...")
        self.font_manager = FontManager()
        self.background_manager = BackgroundManager()
        self.image_generator = ImageGenerator(self.background_manager)
        self.clock_renderer = ClockRenderer(self.font_manager, self.image_generator)

        print("Dataset generator ready!")

    def polygon_to_yolo(self, bbox, img_w, img_h):
        """Convert bbox to YOLO OBB format (x1,y1,x2,y2,x3,y3,x4,y4 normalized)"""
        if not bbox:
            return None

        x1, y1, x2, y2 = bbox

        # Clamp to image boundaries
        x1 = max(0, min(img_w, x1))
        x2 = max(0, min(img_w, x2))
        y1 = max(0, min(img_h, y1))
        y2 = max(0, min(img_h, y2))

        # Check validity
        if x2 <= x1 or y2 <= y1:
            return None

        # Create polygon points (clockwise from top-left)
        pts = [x1, y1, x2, y1, x2, y2, x1, y2]

        # Normalize
        normalized = [
            pts[i] / img_w if i % 2 == 0 else pts[i] / img_h
            for i in range(len(pts))
        ]

        return normalized

    def generate_dataset(self, num_images=1000):
        """Generate complete dataset"""
        print(f"\nGenerating {num_images} clock images...")
        successful = 0

        for i in range(num_images):
            try:
                # Generate random time
                time_str = f"{random.randint(0, 23):02d}:{random.randint(0, 59):02d}"

                # Generate clock layer
                target_width = random.randint(500, 700)
                clock_layer, h_bbox, m_bbox = self.clock_renderer.generate_clock_layer(
                    time_str, target_width
                )

                # Composite final image
                final_img, h_bbox, m_bbox = self.image_generator.composite_final_image(
                    clock_layer, h_bbox, m_bbox
                )

                # Save image
                name = f"{i:06d}"
                img_path = os.path.join(self.temp_images_dir, f"{name}.jpg")
                final_img.save(img_path, quality=random.randint(85, 95))

                # Save YOLO labels
                img_w, img_h = final_img.size
                label_path = os.path.join(self.temp_labels_dir, f"{name}.txt")

                with open(label_path, 'w') as f:
                    # Hours bbox (class 0)
                    if h_bbox:
                        yolo_pts = self.polygon_to_yolo(h_bbox, img_w, img_h)
                        if yolo_pts:
                            f.write(f"0 {' '.join(map(str, yolo_pts))}\n")

                    # Minutes bbox (class 1)
                    if m_bbox:
                        yolo_pts = self.polygon_to_yolo(m_bbox, img_w, img_h)
                        if yolo_pts:
                            f.write(f"1 {' '.join(map(str, yolo_pts))}\n")

                successful += 1

                if (i + 1) % 100 == 0:
                    print(f"  Progress: {i + 1}/{num_images} images generated")

            except Exception as e:
                print(f"  Error generating image {i}: {e}")

        print(f"\nSuccessfully generated {successful}/{num_images} images")
        self.split_dataset()

    def split_dataset(self):
        """Split dataset into train/val sets"""
        print("\nSplitting dataset into train/val...")

        # Create directories
        for split in ['train', 'val']:
            os.makedirs(os.path.join(self.output_dir, split, 'images'), exist_ok=True)
            os.makedirs(os.path.join(self.output_dir, split, 'labels'), exist_ok=True)

        # Get all files
        files = [f for f in os.listdir(self.temp_images_dir) if f.endswith('.jpg')]
        random.shuffle(files)

        # 75/25 split
        split_idx = int(len(files) * 0.75)
        train_files = files[:split_idx]
        val_files = files[split_idx:]

        # Move files
        for f in train_files:
            self._move_file(f, 'train')

        for f in val_files:
            self._move_file(f, 'val')

        # Cleanup temp directories
        shutil.rmtree(self.temp_images_dir)
        shutil.rmtree(self.temp_labels_dir)

        # Create dataset.yaml
        yaml_content = f"""path: {os.path.abspath(self.output_dir)}
train: train/images
val: val/images

names:
  0: hours
  1: minutes

nc: 2
"""

        with open(os.path.join(self.output_dir, "dataset.yaml"), 'w') as f:
            f.write(yaml_content)

        print(f"  Train: {len(train_files)} images")
        print(f"  Val: {len(val_files)} images")
        print(f"\nDataset saved to: {os.path.abspath(self.output_dir)}")
        print(f"Config file: {os.path.join(self.output_dir, 'dataset.yaml')}")

    def integrate_real_dataset(self, real_dataset_path, replicate_factor=10):
        """
        Aggregates real data into the synthetic dataset (both train and val).
        Replicates each real image 'replicate_factor' times.
        """
        print(f"\nIntegrating real dataset from: {real_dataset_path}")

        # Process both train and val splits
        splits = ['train', 'val']

        total_added = 0

        for split in splits:
            real_img_dir = os.path.join(real_dataset_path, split, 'images')
            real_lbl_dir = os.path.join(real_dataset_path, split, 'labels')

            # Destination directories (synthetic set)
            dest_img_dir = os.path.join(self.output_dir, split, 'images')
            dest_lbl_dir = os.path.join(self.output_dir, split, 'labels')

            if not os.path.exists(real_img_dir):
                print(f"  Skipping {split} (folder not found: {real_img_dir})")
                continue

            if not os.path.exists(dest_img_dir):
                print(f"  Warning: Synthetic {split} folder not found, creating it...")
                os.makedirs(dest_img_dir, exist_ok=True)
                os.makedirs(dest_lbl_dir, exist_ok=True)

            # Get list of real images
            valid_exts = ['.jpg', '.jpeg', '.png', '.bmp']
            real_images = [f for f in os.listdir(real_img_dir) if os.path.splitext(f)[1].lower() in valid_exts]

            if not real_images:
                print(f"  No images found in {split} folder.")
                continue

            print(f"  Found {len(real_images)} real images in '{split}'. Replicating {replicate_factor}x...")

            count = 0
            for img_name in real_images:
                name_root, ext = os.path.splitext(img_name)
                # Find corresponding label (check txt)
                label_name = name_root + ".txt"

                src_img_path = os.path.join(real_img_dir, img_name)
                src_lbl_path = os.path.join(real_lbl_dir, label_name)

                # Check if label exists
                has_label = os.path.exists(src_lbl_path)
                if not has_label:
                    print(f"    Skipping {img_name} (missing label)")
                    continue

                for i in range(replicate_factor):
                    # Create unique name: real_split_ORIGINALNAME_00.jpg
                    new_name_root = f"real_{split}_{name_root}_{i:02d}"
                    new_img_name = f"{new_name_root}{ext}"
                    new_lbl_name = f"{new_name_root}.txt"

                    # Copy image
                    shutil.copy2(src_img_path, os.path.join(dest_img_dir, new_img_name))

                    # Copy label
                    shutil.copy2(src_lbl_path, os.path.join(dest_lbl_dir, new_lbl_name))

                    count += 1

            print(f"  > Added {count} images to {split} set")
            total_added += count

        print(f"  Total real images integrated: {total_added}")

    def _move_file(self, filename, split):
        """Move image and label files to split directory"""
        # Move image
        src_img = os.path.join(self.temp_images_dir, filename)
        dst_img = os.path.join(self.output_dir, split, 'images', filename)
        shutil.move(src_img, dst_img)

        # Move label if exists
        label_name = filename.replace('.jpg', '.txt')
        src_label = os.path.join(self.temp_labels_dir, label_name)
        if os.path.exists(src_label):
            dst_label = os.path.join(self.output_dir, split, 'labels', label_name)
            shutil.move(src_label, dst_label)


if __name__ == "__main__":
    generator = ClockDatasetGenerator()

    # 1. Generate synthetic data
    generator.generate_dataset(num_images=100)

    # 2. Integrate real data (UNCOMMENT AND SET PATH)
    # real_data_path = "path/to/your/real_dataset"
    # generator.integrate_real_dataset(real_data_path)

    print("\n✓ Dataset generation complete!")