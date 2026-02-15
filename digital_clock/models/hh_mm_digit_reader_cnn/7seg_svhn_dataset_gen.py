import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFilter
import random
import os
import h5py
import pandas as pd
import scipy.io as sio
import urllib.request
import tarfile
import pickle
from tqdm import tqdm


# ==========================================
# Part 1: Synthetic Data Generator
# ==========================================




class SevenSegmentGenerator:
    """Generate synthetic 7-segment display digits with augmentations"""

    # Define 7-segment encoding for digits 0-9
    SEGMENTS = {
        0: [1, 1, 1, 1, 1, 1, 0],
        1: [0, 1, 1, 0, 0, 0, 0],
        2: [1, 1, 0, 1, 1, 0, 1],
        3: [1, 1, 1, 1, 0, 0, 1],
        4: [0, 1, 1, 0, 0, 1, 1],
        5: [1, 0, 1, 1, 0, 1, 1],
        6: [1, 0, 1, 1, 1, 1, 1],
        7: [1, 1, 1, 0, 0, 0, 0],
        8: [1, 1, 1, 1, 1, 1, 1],
        9: [1, 1, 1, 1, 0, 1, 1]
    }

    def __init__(self, img_size=(32, 32)):
        self.img_size = img_size
        self.max_digits = 5

    def draw_segment(self, draw, points, color, width=2):
        draw.polygon(points, fill=color, outline=color)

    def get_segment_coords(self, digit, segment_width=3, size_variation=0.8,
                           x_offset=0, y_offset=0, scale_factor=1.0):
        w, h = self.img_size
        scale = min(w, h) * size_variation / 32 * scale_factor

        cx, cy = w // 2 + x_offset, h // 2 + y_offset
        seg_len = int(10 * scale)
        seg_w = int(segment_width * scale)
        gap = int(2 * scale)

        segments_coords = [
            [(cx - seg_len, cy - seg_len - gap), (cx + seg_len, cy - seg_len - gap),
             (cx + seg_len - seg_w, cy - seg_len - gap + seg_w),
             (cx - seg_len + seg_w, cy - seg_len - gap + seg_w)],
            [(cx + seg_len, cy - seg_len - gap), (cx + seg_len, cy - gap),
             (cx + seg_len - seg_w, cy - gap + seg_w),
             (cx + seg_len - seg_w, cy - seg_len - gap + seg_w)],
            [(cx + seg_len, cy + gap), (cx + seg_len, cy + seg_len + gap),
             (cx + seg_len - seg_w, cy + seg_len + gap - seg_w),
             (cx + seg_len - seg_w, cy + gap - seg_w)],
            [(cx - seg_len, cy + seg_len + gap), (cx + seg_len, cy + seg_len + gap),
             (cx + seg_len - seg_w, cy + seg_len + gap - seg_w),
             (cx - seg_len + seg_w, cy + seg_len + gap - seg_w)],
            [(cx - seg_len, cy + gap), (cx - seg_len, cy + seg_len + gap),
             (cx - seg_len + seg_w, cy + seg_len + gap - seg_w),
             (cx - seg_len + seg_w, cy + gap - seg_w)],
            [(cx - seg_len, cy - seg_len - gap), (cx - seg_len, cy - gap),
             (cx - seg_len + seg_w, cy - gap + seg_w),
             (cx - seg_len + seg_w, cy - seg_len - gap + seg_w)],
            [(cx - seg_len, cy), (cx + seg_len, cy),
             (cx + seg_len - seg_w, cy + seg_w),
             (cx - seg_len + seg_w, cy + seg_w)]
        ]

        segments = self.SEGMENTS[digit]
        active_coords = []
        for i, active in enumerate(segments):
            if active:
                active_coords.append(segments_coords[i])

        return active_coords

    def add_noise(self, img, noise_level=0.1):
        noise = np.random.normal(0, noise_level * 255, img.shape)
        noisy = np.clip(img + noise, 0, 255).astype(np.uint8)
        return noisy

    def add_texture(self, img):
        texture = np.random.randint(0, 50, img.shape, dtype=np.uint8)
        alpha = random.uniform(0.1, 0.3)
        textured = cv2.addWeighted(img, 1 - alpha, texture, alpha, 0)
        return textured

    def random_blur(self, img):
        if random.random() > 0.5:
            ksize = random.choice([3, 5])
            img = cv2.GaussianBlur(img, (ksize, ksize), 0)
        return img

    def random_brightness(self, img, factor_range=(0.5, 1.5)):
        factor = random.uniform(*factor_range)
        return np.clip(img * factor, 0, 255).astype(np.uint8)

    def random_rotation(self, img, max_angle=15):
        angle = random.uniform(-max_angle, max_angle)
        h, w = img.shape[:2]
        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        return cv2.warpAffine(img, M, (w, h), borderValue=(0, 0, 0))

    def add_perspective(self, img, max_shift=3):
        """Add subtle perspective/depth effect"""
        h, w = img.shape[:2]

        # Random perspective strength
        shift = random.randint(1, max_shift)

        # Source points (corners of the image)
        src_points = np.float32([
            [0, 0],
            [w - 1, 0],
            [w - 1, h - 1],
            [0, h - 1]
        ])

        # Destination points with slight perspective
        # Randomly choose perspective direction
        perspective_type = random.choice(['left', 'right', 'top', 'bottom'])

        if perspective_type == 'left':
            dst_points = np.float32([
                [shift, shift],
                [w - 1, 0],
                [w - 1, h - 1],
                [shift, h - 1 - shift]
            ])
        elif perspective_type == 'right':
            dst_points = np.float32([
                [0, 0],
                [w - 1 - shift, shift],
                [w - 1 - shift, h - 1 - shift],
                [0, h - 1]
            ])
        elif perspective_type == 'top':
            dst_points = np.float32([
                [shift, shift],
                [w - 1 - shift, shift],
                [w - 1, h - 1],
                [0, h - 1]
            ])
        else:  # bottom
            dst_points = np.float32([
                [0, 0],
                [w - 1, 0],
                [w - 1 - shift, h - 1 - shift],
                [shift, h - 1 - shift]
            ])

        # Calculate perspective transform
        matrix = cv2.getPerspectiveTransform(src_points, dst_points)

        # Apply perspective transform
        img = cv2.warpPerspective(img, matrix, (w, h), borderValue=(0, 0, 0))

        return img

    def generate_multi_digit(self, num_digits=None, apply_augmentations=True):
        """Generate an image with 1-4 digits"""
        if num_digits is None:
            num_digits = random.randint(1, self.max_digits)

        # Generate random number with specified number of digits
        if num_digits == 1:
            number = random.randint(0, 9)
        else:
            number = random.randint(10 ** (num_digits - 1), 10 ** num_digits - 1)

        digits = [int(d) for d in str(number)]

        # Random colors (same for all digits in the image)
        if random.random() > 0.3:
            colors = [
                (0, 255, 0),  # Green
                (255, 0, 0),  # Red
                (0, 100, 255),  # Orange
                (255, 255, 0),  # Cyan
                (255, 255, 255)  # White
            ]
            color = random.choice(colors)
        else:
            color = tuple(random.randint(100, 255) for _ in range(3))

        # Random background
        if random.random() > 0.5:
            bg_color = (0, 0, 0)
        else:
            bg_color = tuple(random.randint(0, 50) for _ in range(3))

        img = Image.new('RGB', self.img_size, bg_color)
        draw = ImageDraw.Draw(img)

        # Calculate spacing for multiple digits
        seg_width = random.randint(2, 4)

        # IMPROVED: Bigger font size and tighter spacing
        if num_digits == 1:
            size_var = random.uniform(0.75, 0.95)  # Increased from 0.6-0.9
            scale_factor = 1.0
            spacing = 0
        elif num_digits == 2:
            size_var = random.uniform(0.65, 0.8)  # Increased from 0.5-0.7
            scale_factor = 0.85  # Increased from 0.7
            spacing = 10  # Reduced from 12
        elif num_digits == 3:
            size_var = random.uniform(0.55, 0.7)  # Increased from 0.4-0.6
            scale_factor = 0.65  # Increased from 0.5
            spacing = 7  # Reduced from 9
        else:  # 4 digits
            size_var = random.uniform(0.45, 0.6)  # Increased from 0.35-0.5
            scale_factor = 0.55  # Increased from 0.4
            spacing = 5  # Reduced from 7

        # Calculate starting x position to center all digits
        total_width = spacing * (num_digits - 1)
        start_x = -total_width // 2

        # Draw each digit
        for i, digit in enumerate(digits):
            x_offset = start_x + i * spacing
            segment_coords = self.get_segment_coords(
                digit, seg_width, size_var,
                x_offset=x_offset, y_offset=0, scale_factor=scale_factor
            )
            for coords in segment_coords:
                self.draw_segment(draw, coords, color)

        img = np.array(img)

        if apply_augmentations:
            if random.random() > 0.3:
                img = self.add_noise(img, random.uniform(0.05, 0.15))

            if random.random() > 0.5:
                img = self.add_texture(img)

            if random.random() > 0.6:
                img = self.random_blur(img)

            if random.random() > 0.4:
                img = self.random_brightness(img, (0.6, 1.4))

            # IMPROVED: Much more frequent rotations with moderate angles
            if random.random() > 0.15:  # 85% of images get rotation
                img = self.random_rotation(img, max_angle=12)  # Slightly increased

            # NEW: Add perspective/depth variation (more frequent)
            if random.random() > 0.25:  # 75% of images get perspective
                img = self.add_perspective(img)

        return img, number, num_digits

    def generate_dataset(self, total_samples=10000, max_digits=5):
        """
        Generate dataset with labels in SVHN format
        Returns labels as array of shape (n_samples, max_digits) where:
        - First value is the length of the sequence
        - Remaining values are the digits (padded with 10 for unused positions)
        """
        images, labels = [], []
        print(f"Generating {total_samples} synthetic 7-segment images...")
        distribution = [1] * 4000 + [2] * 2500 + [3] * 2000 + [4] * 1500
        random.shuffle(distribution)

        for i in range(total_samples):
            num_digits = distribution[i] if i < len(distribution) else random.randint(1, 4)
            img, number, n_digits = self.generate_multi_digit(num_digits)
            images.append(img)

            # Convert to SVHN-style label format
            # [length, digit1, digit2, digit3, digit4, ...]
            digits = [int(d) for d in str(number)]
            label = digits + [10] * (max_digits - len(digits))
            labels.append(label[:max_digits])

        return np.array(images), np.array(labels)


# ==========================================
# Part 2: SVHN Raw Data Processing (Fixed for h5py > 3.0 & Pillow > 10)
# ==========================================

class DigitStructWrapper:
    """
    Wrapper for the H5PY digitStruct files from the SVHN dataset.
    Compatible with h5py >= 3.0 (no .value attribute).
    """

    def __init__(self, inf):
        self.inf = h5py.File(inf, 'r')
        self.digitStructName = self.inf['digitStruct']['name']
        self.digitStructBbox = self.inf['digitStruct']['bbox']

    def get_name(self, n):
        """Return the name of the n(th) digit struct"""
        # Get the reference
        ref = self.digitStructName[n][0]
        # Access the dataset using the reference
        name_arr = self.inf[ref][()]
        # Convert char array to string
        return ''.join([chr(int(c[0])) for c in name_arr])

    def get_attribute(self, attr):
        """Helper function for dealing with one vs. multiple bounding boxes"""
        # In h5py 3, we access data directly instead of .value
        if (len(attr) > 1):
            # If multiple items, it's an array of references
            attr = [self.inf[attr[j].item()][0][0] for j in range(len(attr))]
        else:
            # If single item, it is the value itself
            attr = [attr[0][0]]
        return attr

    def get_bbox(self, n):
        """Return a dict containing the data from the n(th) bbox"""
        bbox = {}
        # Get the reference to the bbox struct
        bb_ref = self.digitStructBbox[n].item()
        # Dereference to get the dataset
        bb = self.inf[bb_ref]

        # Helper to extract attributes safely
        def extract_val(key):
            attr = bb[key]
            # Check if it is a list of references or a single value
            if attr.shape[0] > 1:
                return [self.inf[attr[j].item()][0][0] for j in range(len(attr))]
            else:
                return [attr[0][0]]

        bbox['height'] = extract_val("height")
        bbox['label'] = extract_val("label")
        bbox['left'] = extract_val("left")
        bbox['top'] = extract_val("top")
        bbox['width'] = extract_val("width")

        return bbox

    def get_item(self, n):
        """Return the name and bounding boxes of a single image"""
        s = self.get_bbox(n)
        s['name'] = self.get_name(n)
        return s

    def unpack_all(self):
        """Unpack all items into a list of dictionaries"""
        result = []
        # Total number of images
        total = len(self.digitStructName)
        for i in tqdm(range(total), desc="Unpacking digitStruct"):
            item = self.get_item(i)

            # Combine into list of dicts per image
            boxes = []
            for j in range(len(item['label'])):
                boxes.append({
                    'label': item['label'][j],
                    'left': item['left'][j],
                    'top': item['top'][j],
                    'width': item['width'][j],
                    'height': item['height'][j]
                })
            result.append({'filename': item['name'], 'boxes': boxes})
        return result


class SVHNRawProcessor:
    """Handles downloading and processing of SVHN Format 1 (Full Images)"""

    def __init__(self, data_path='./svhn_raw_data', img_size=(32, 32)):
        self.data_path = data_path
        self.img_size = img_size
        os.makedirs(data_path, exist_ok=True)

    def download_data(self):
        """Download SVHN Format 1 (Full Numbers)"""
        base_url = "http://ufldl.stanford.edu/housenumbers/"
        files = ['train.tar.gz', 'test.tar.gz', 'extra.tar.gz']

        for filename in files:
            filepath = os.path.join(self.data_path, filename)
            folder_name = filename.split('.')[0]
            folder_path = os.path.join(self.data_path, folder_name)

            if os.path.exists(folder_path):
                print(f"{folder_name} already exists.")
                continue

            if not os.path.exists(filepath):
                print(f"Downloading {filename}...")
                try:
                    urllib.request.urlretrieve(base_url + filename, filepath)
                except Exception as e:
                    print(f"Error downloading: {e}")
                    continue

            print(f"Extracting {filename}...")
            with tarfile.open(filepath, 'r:gz') as tar:
                tar.extractall(path=self.data_path)

    def process_subset(self, subset_name):
        folder = os.path.join(self.data_path, subset_name)
        mat_file = os.path.join(folder, 'digitStruct.mat')

        if not os.path.exists(mat_file):
            print(f"Metadata file not found: {mat_file}")
            return None, None

        print(f"Parsing {mat_file}...")
        ds = DigitStructWrapper(mat_file)
        data = ds.unpack_all()

        print("Converting to DataFrame for BBox calculation...")
        records = []
        for item in data:
            filename = item['filename']
            for bbox in item['boxes']:
                records.append({
                    'filename': filename,
                    'label': bbox['label'],
                    'x0': bbox['left'],
                    'y0': bbox['top'],
                    'width': bbox['width'],
                    'height': bbox['height'],
                    'x1': bbox['left'] + bbox['width'],
                    'y1': bbox['top'] + bbox['height']
                })

        df = pd.DataFrame(records)

        agg_funcs = {
            'x0': 'min', 'y0': 'min', 'x1': 'max', 'y1': 'max',
            'label': lambda x: list(x)
        }
        df_grouped = df.groupby('filename').agg(agg_funcs).reset_index()

        print(f"Processing images in {subset_name}...")
        X_data = []
        y_data = []
        expansion_factor = 0.3

        for idx, row in tqdm(df_grouped.iterrows(), total=len(df_grouped), desc=f"Cropping {subset_name}"):
            img_path = os.path.join(folder, row['filename'])
            if not os.path.exists(img_path): continue

            try:
                img = Image.open(img_path)
            except:
                continue

            box_w = row['x1'] - row['x0']
            box_h = row['y1'] - row['y0']

            cx = row['x0'] + box_w / 2
            cy = row['y0'] + box_h / 2

            new_w = box_w * (1 + expansion_factor)
            new_h = box_h * (1 + expansion_factor)

            x0_new = max(0, cx - new_w / 2)
            y0_new = max(0, cy - new_h / 2)
            x1_new = min(img.width, cx + new_w / 2)
            y1_new = min(img.height, cy + new_h / 2)

            crop = img.crop((x0_new, y0_new, x1_new, y1_new))

            # --- FIX FOR PILLOW 10+: Use Image.LANCZOS instead of Image.ANTIALIAS ---
            crop = crop.resize(self.img_size, Image.LANCZOS)

            X_data.append(np.array(crop))

            lbls = row['label']
            # Map 10 -> 0 for logic, but since we pad with 10, let's treat digit 0 as 0.
            # SVHN has 10 for digit 0.
            clean_lbls = [int(l) if l != 10 else 0 for l in lbls]

            if len(clean_lbls) > 5:
                clean_lbls = clean_lbls[:5]
            else:
                clean_lbls += [10] * (5 - len(clean_lbls))

            y_data.append(clean_lbls)

        return np.array(X_data), np.array(y_data)


# ==========================================
# Part 3: Utilities & Main
# ==========================================

def combine_datasets(X1, y1, X2, y2, ratio=0.3):
    if len(X1) == 0: return X2, y2

    n_synthetic = int(len(X1) * ratio)
    if len(X2) > 0:
        indices = np.random.choice(len(X2), min(n_synthetic, len(X2)), replace=False)
        X2_sample = X2[indices]
        y2_sample = y2[indices]

        X_combined = np.concatenate([X1, X2_sample], axis=0)
        y_combined = np.concatenate([y1, y2_sample], axis=0)
    else:
        X_combined, y_combined = X1, y1

    idx = np.random.permutation(len(X_combined))
    return X_combined[idx], y_combined[idx]


def rgb2gray(images):
    return np.expand_dims(np.dot(images, [0.2990, 0.5870, 0.1140]), axis=3)


def save_h5(X_train, y_train, X_test, y_test, X_val, y_val, output_dir='data'):
    os.makedirs(output_dir, exist_ok=True)

    print("Saving RGB data...")
    with h5py.File(os.path.join(output_dir, 'SVHN_unified_rgb.h5'), 'w') as f:
        f.create_dataset('X_train', data=X_train)
        f.create_dataset('y_train', data=y_train)
        f.create_dataset('X_test', data=X_test)
        f.create_dataset('y_test', data=y_test)
        f.create_dataset('X_val', data=X_val)
        f.create_dataset('y_val', data=y_val)

    print("Processing and saving Normalized Grayscale data...")
    X_train_g = rgb2gray(X_train).astype(np.float32)
    X_test_g = rgb2gray(X_test).astype(np.float32)
    X_val_g = rgb2gray(X_val).astype(np.float32)

    mean = np.mean(X_train_g)
    std = np.std(X_train_g)

    X_train_norm = (X_train_g - mean) / std
    X_test_norm = (X_test_g - mean) / std
    X_val_norm = (X_val_g - mean) / std

    with h5py.File(os.path.join(output_dir, 'SVHN_unified_norm_gray.h5'), 'w') as f:
        f.create_dataset('X_train', data=X_train_norm)
        f.create_dataset('y_train', data=y_train)
        f.create_dataset('X_test', data=X_test_norm)
        f.create_dataset('y_test', data=y_test)
        f.create_dataset('X_val', data=X_val_norm)
        f.create_dataset('y_val', data=y_val)

    print(f"Saved. Train shape: {X_train.shape}, Labels shape: {y_train.shape}")


if __name__ == "__main__":
    print("=== Unified SVHN & Synthetic 7-Segment Dataset Generator ===")

    # 1. Generate Synthetic Data
    gen = SevenSegmentGenerator()
    print("\n[1] Generating synthetic data...")
    X_syn, y_syn = gen.generate_dataset(total_samples=35000)
    print(f"    Generated {len(X_syn)} synthetic images.")

    # 2. Process SVHN Raw Data
    print("\n[2] Processing SVHN Format 1 (Full Images)...")
    processor = SVHNRawProcessor()

    # Since you already have the files, you can comment this out to save time
    # processor.download_data()

    if not os.path.exists(os.path.join(processor.data_path, 'train/digitStruct.mat')):
        print(
            "WARNING: SVHN Raw data not found. Please ensure 'train', 'test', 'extra' folders are in './svhn_raw_data'.")
        X_train_svhn, y_train_svhn = np.array([]), np.array([])
        X_test_svhn, y_test_svhn = np.array([]), np.array([])
    else:
        X_train_svhn, y_train_svhn = processor.process_subset('train')
        X_test_svhn, y_test_svhn = processor.process_subset('test')

        # Optional: Process 'extra' for more data (Warning: It's very large)
        if os.path.exists(os.path.join(processor.data_path, 'extra/digitStruct.mat')):
            X_extra, y_extra = processor.process_subset('extra')
            X_train_svhn = np.concatenate([X_train_svhn, X_extra])
            y_train_svhn = np.concatenate([y_train_svhn, y_extra])

    # 3. Combine
    print("\n[3] Combining Real and Synthetic Data...")

    if len(X_train_svhn) == 0:
        print("Using only synthetic data (SVHN data missing).")
        split1 = int(0.7 * len(X_syn))
        split2 = int(0.85 * len(X_syn))
        X_train, y_train = X_syn[:split1], y_syn[:split1]
        X_val, y_val = X_syn[split1:split2], y_syn[split1:split2]
        X_test, y_test = X_syn[split2:], y_syn[split2:]
    else:
        X_train, y_train = combine_datasets(X_train_svhn, y_train_svhn, X_syn, y_syn, ratio=0.3)
        X_test, y_test = combine_datasets(X_test_svhn, y_test_svhn, X_syn, y_syn, ratio=0.1)

        val_split = int(0.9 * len(X_train))
        X_val = X_train[val_split:]
        y_val = y_train[val_split:]
        X_train = X_train[:val_split]
        y_train = y_train[:val_split]

    # 4. Save
    print("\n[4] Saving H5 files...")
    save_h5(X_train, y_train, X_test, y_test, X_val, y_val)

    print("\nDone! Labels are sequences of length 5 (10=padding, 0-9=digits).")
    print("Files saved in ./data/")