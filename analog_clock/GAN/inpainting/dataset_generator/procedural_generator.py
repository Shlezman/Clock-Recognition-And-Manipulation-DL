# ==============================================================================
# FILE: procedural_generator.py
# Handles automatic asset downloading and procedural generation
# Fixed: Removed 'mercedes' style to prevent >3 hands artifact
# ==============================================================================

import cv2
import numpy as np
import requests
import random
import os
import subprocess
from pathlib import Path
from tqdm import tqdm
from config import Config


class AssetManager:
    """Manages downloading of backgrounds"""

    def __init__(self):
        self.root_dir = Path(Config.BACKGROUNDS_DIR)
        self.root_dir.mkdir(parents=True, exist_ok=True)

    def prepare_assets(self):
        print(f"📂 Asset Directory: {self.root_dir}")
        if Config.DOWNLOAD_KAGGLE:
            self._check_and_download_backgrounds()

        images = self.get_all_images()
        print(f"✓ Current background count: {len(images)}")

        if len(images) < 10 and Config.DOWNLOAD_PICSUM:
            print("⚠ Backgrounds missing. Downloading fallback...")
            self._download_picsum(Config.NUM_TEXTURES_TO_DOWNLOAD)

    def _check_and_download_backgrounds(self):
        if not self.root_dir.exists() or not os.listdir(self.root_dir):
            print("Attempting download via Kaggle CLI...")
            try:
                subprocess.run([
                    "kaggle", "datasets", "download", "-d",
                    Config.KAGGLE_DATASET,
                    "-p", str(self.root_dir.resolve()), "--unzip"
                ], check=True)
                print("✓ Textures downloaded.")
            except Exception as e:
                print(f"⚠ CLI Download failed: {e}. Using fallback.")

    def _download_picsum(self, count):
        picsum_dir = self.root_dir / "picsum"
        picsum_dir.mkdir(exist_ok=True)
        for i in tqdm(range(count)):
            try:
                url = f"https://picsum.photos/1024/1024?random={i + 2000}"
                r = requests.get(url, timeout=5)
                if r.status_code == 200:
                    with open(picsum_dir / f"tex_{i}.jpg", 'wb') as f:
                        f.write(r.content)
            except:
                pass

    def get_all_images(self):
        files = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
            files.extend(self.root_dir.rglob(ext))
        return files


class ProceduralClockGenerator:
    """Generates clock faces with HIGH VARIETY - Supports Dynamic Canvas Sizes"""

    def __init__(self):
        self.hand_colors = [
            (0, 0, 0), (255, 255, 255), (40, 40, 40),
            (20, 20, 80), (80, 20, 20), (200, 150, 20)
        ]

        self.fonts = [
            cv2.FONT_HERSHEY_SIMPLEX, cv2.FONT_HERSHEY_PLAIN,
            cv2.FONT_HERSHEY_DUPLEX, cv2.FONT_HERSHEY_COMPLEX,
            cv2.FONT_HERSHEY_TRIPLEX, cv2.FONT_HERSHEY_COMPLEX_SMALL,
            cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, cv2.FONT_ITALIC
        ]

        self.roman_numerals = {
            1: "I", 2: "II", 3: "III", 4: "IV", 5: "V", 6: "VI",
            7: "VII", 8: "VIII", 9: "IX", 10: "X", 11: "XI", 12: "XII"
        }

    def create_clock_on_wall(self, wall_path: Path, face_path: Path,
                             center: tuple, radius: int, scene_size: tuple) -> np.ndarray:
        w, h = scene_size
        wall = cv2.imread(str(wall_path))
        if wall is None: wall = np.zeros((h, w, 3), np.uint8)
        wall = cv2.resize(wall, (w, h))

        d = radius * 2

        # Decide Face Type (Solid or Texture)
        if random.random() < Config.SOLID_FACE_PROB:
            face_tex = np.zeros((d, d, 3), dtype=np.uint8)
            # Logic for White Faces (approx 70% of solid faces -> 35% total)
            if random.random() < 0.70:
                val = random.randint(235, 255)
                color = (val, val, val)
            else:
                color = (random.randint(40, 240), random.randint(40, 240), random.randint(40, 240))
            face_tex[:] = color
        else:
            face_tex = cv2.imread(str(face_path))
            if face_tex is None:
                face_tex = np.full((d, d, 3), 200, dtype=np.uint8)
            else:
                face_tex = cv2.resize(face_tex, (d, d))

        mask = np.zeros((d, d), dtype=np.uint8)
        cv2.circle(mask, (radius, radius), radius, 255, -1, cv2.LINE_AA)

        face_circular = cv2.bitwise_and(face_tex, face_tex, mask=mask)
        self._decorate_face(face_circular, radius)

        x1 = center[0] - radius
        y1 = center[1] - radius
        x2 = x1 + d
        y2 = y1 + d

        if x1 < 0 or y1 < 0 or x2 > w or y2 > h:
            return wall

        roi = wall[y1:y2, x1:x2]
        mask_inv = cv2.bitwise_not(mask)
        img_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)
        img_fg = cv2.bitwise_and(face_circular, face_circular, mask=mask)
        dst = cv2.add(img_bg, img_fg)

        wall[y1:y2, x1:x2] = dst

        frame_color = (random.randint(20, 80), random.randint(20, 80), random.randint(20, 80))
        cv2.circle(wall, center, radius, frame_color, random.randint(2, 6), cv2.LINE_AA)

        return wall

    def _decorate_face(self, img, radius):
        center = (radius, radius)
        if np.mean(img) > 127:
            color = (random.randint(0, 60), random.randint(0, 60), random.randint(0, 60))
        else:
            color = (random.randint(200, 255), random.randint(200, 255), random.randint(200, 255))

        tick_style = random.choice(['lines', 'thick_lines', 'dots', 'squares', 'triangles', 'rings', 'minimal'])

        for i in range(60):
            angle = i * 6 * (np.pi / 180)
            is_hour = (i % 5 == 0)

            if tick_style == 'minimal' and not is_hour: continue

            r_out = radius - (radius * 0.03)

            if tick_style == 'dots':
                r_pos = r_out - 5
                x = int(center[0] + r_pos * np.cos(angle - np.pi / 2))
                y = int(center[1] + r_pos * np.sin(angle - np.pi / 2))
                size = 3 if is_hour else 1
                cv2.circle(img, (x, y), size, color, -1, cv2.LINE_AA)
            elif tick_style in ['squares', 'triangles'] and is_hour:
                r_pos = r_out - 8
                x = int(center[0] + r_pos * np.cos(angle - np.pi / 2))
                y = int(center[1] + r_pos * np.sin(angle - np.pi / 2))
                size = 4
                if tick_style == 'squares':
                    cv2.rectangle(img, (x - size, y - size), (x + size, y + size), color, -1)
                else:
                    cv2.circle(img, (x, y), size + 1, color, -1 if random.random() > 0.5 else 2, cv2.LINE_AA)
            elif tick_style == 'rings':
                if i == 0: cv2.circle(img, center, int(radius * 0.95), color, 1, cv2.LINE_AA)
                if is_hour:
                    r_in = radius * 0.85
                    x1 = int(center[0] + r_out * np.cos(angle - np.pi / 2))
                    y1 = int(center[1] + r_out * np.sin(angle - np.pi / 2))
                    x2 = int(center[0] + r_in * np.cos(angle - np.pi / 2))
                    y2 = int(center[1] + r_in * np.sin(angle - np.pi / 2))
                    cv2.line(img, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
            else:  # Lines
                length = radius * 0.12 if is_hour else radius * 0.04
                thickness = 2 if (is_hour or tick_style == 'thick_lines') else 1
                if is_hour and tick_style == 'thick_lines': thickness = 3
                r_in = r_out - length
                x1 = int(center[0] + r_out * np.cos(angle - np.pi / 2))
                y1 = int(center[1] + r_out * np.sin(angle - np.pi / 2))
                x2 = int(center[0] + r_in * np.cos(angle - np.pi / 2))
                y2 = int(center[1] + r_in * np.sin(angle - np.pi / 2))
                cv2.line(img, (x1, y1), (x2, y2), color, thickness, cv2.LINE_AA)

        if random.random() < Config.SHOW_NUMBERS_PROB:
            font = random.choice(self.fonts)
            use_roman = random.random() < 0.3
            font_scale = radius * random.uniform(0.003, 0.005)
            thickness = random.randint(1, 2)
            if font in [cv2.FONT_HERSHEY_COMPLEX_SMALL, cv2.FONT_HERSHEY_SCRIPT_SIMPLEX]:
                font_scale *= 1.5
            r_num = radius * 0.75

            for i in range(1, 13):
                angle = i * 30 * (np.pi / 180)
                text = self.roman_numerals[i] if use_roman else str(i)
                (w, h), base = cv2.getTextSize(text, font, font_scale, thickness)
                x = int(center[0] + r_num * np.cos(angle - np.pi / 2)) - w // 2
                y = int(center[1] + r_num * np.sin(angle - np.pi / 2)) + h // 2
                cv2.putText(img, text, (x, y), font, font_scale, color, thickness, cv2.LINE_AA)

    def generate_hand_set(self, center: tuple, radius: int, scene_size: tuple):
        """Generates hands layer including optional SECOND HAND"""

        color = random.choice(self.hand_colors)

        # REMOVED 'mercedes' and 'snowflake' to avoid the "6-hands" artifact
        style = random.choice([
            'pointed', 'rectangle', 'modern', 'arrow', 'diamond',
            'tapered', 'sword', 'lollipop', 'skeleton', 'baton',
            'leaf', 'pencil', 'dauphine', 'breguet', 'spade',
            'anchor', 'cathedral', 'alpha', 'feuille', 'lance',
            'plongeur', 'syringe', 'flamme'
        ])

        min_len_ratio = random.uniform(0.75, 0.95)
        minute_length = radius * min_len_ratio

        hour_len_ratio = random.uniform(0.50, 0.65)
        hour_length = radius * hour_len_ratio

        hour_width = radius * random.uniform(0.06, 0.09)
        minute_width = radius * random.uniform(0.03, 0.05)

        hour_hand = self._draw_single_hand(center, hour_length, hour_width, color, style, scene_size)
        minute_hand = self._draw_single_hand(center, minute_length, minute_width, color, style, scene_size)

        # --- Second Hand (50% Probability) ---
        second_hand = None
        if random.random() < Config.INCLUDE_SECOND_HAND_PROB:
            sec_colors = [(0, 0, 200), (0, 0, 0), (200, 0, 0)]  # Red/Black/Blue
            sec_color = random.choice(sec_colors)
            sec_length = radius * random.uniform(0.8, 0.95)
            sec_width = max(1, int(radius * 0.015))

            second_hand = self._draw_second_hand(center, sec_length, sec_width, sec_color, scene_size)

        # Must return 3 values!
        return hour_hand, minute_hand, second_hand

    def _draw_second_hand(self, center, length, width, color, scene_size):
        w, h = scene_size
        img = np.zeros((h, w, 4), dtype=np.uint8)
        cx, cy = center
        length = int(length)
        tail_length = int(length * 0.2)

        pts = np.array([
            [cx - width // 2, cy + tail_length],
            [cx - width // 2, cy - length],
            [cx + width // 2, cy - length],
            [cx + width // 2, cy + tail_length]
        ], dtype=np.int32)

        cv2.fillPoly(img, [pts], list(color) + [255], cv2.LINE_AA)
        cv2.circle(img, center, int(width * 3), list(color) + [255], -1, cv2.LINE_AA)

        return img

    def _draw_single_hand(self, center, length, width, color, style, scene_size):
        w, h = scene_size
        img = np.zeros((h, w, 4), dtype=np.uint8)
        cx, cy = center
        length = int(length)
        width = int(width)

        if style == 'pointed':
            pts = np.array([[cx, cy - length], [cx + width, cy], [cx, cy + width], [cx - width, cy]], dtype=np.int32)
        elif style == 'arrow':
            arrow_head = int(width * 2)
            pts = np.array([[cx, cy - length], [cx + arrow_head, cy - length + arrow_head],
                            [cx + width // 2, cy - length + arrow_head], [cx + width // 2, cy], [cx - width // 2, cy],
                            [cx - width // 2, cy - length + arrow_head], [cx - arrow_head, cy - length + arrow_head]],
                           dtype=np.int32)
        elif style == 'diamond':
            diamond_size = int(width * 1.5)
            pts = np.array([[cx, cy - length], [cx + diamond_size, cy - length + diamond_size],
                            [cx + width // 2, cy - length + diamond_size * 2], [cx + width // 2, cy],
                            [cx - width // 2, cy], [cx - width // 2, cy - length + diamond_size * 2],
                            [cx - diamond_size, cy - length + diamond_size]], dtype=np.int32)
        elif style == 'tapered':
            pts = np.array([[cx, cy - length], [cx + width // 4, int(cy - length * 0.7)], [cx + width // 2, cy],
                            [cx - width // 2, cy], [cx - width // 4, int(cy - length * 0.7)]], dtype=np.int32)
        elif style == 'sword':
            blade_width = width // 3
            pts = np.array([[cx, cy - length], [cx + blade_width, int(cy - length * 0.85)],
                            [cx + width // 2, int(cy - length * 0.85)], [cx + width // 2, cy], [cx - width // 2, cy],
                            [cx - width // 2, int(cy - length * 0.85)], [cx - blade_width, int(cy - length * 0.85)]],
                           dtype=np.int32)
        elif style == 'lollipop':
            pts = np.array([[cx - width // 2, cy], [cx - width // 2, cy - length + width * 2],
                            [cx + width // 2, cy - length + width * 2], [cx + width // 2, cy]], dtype=np.int32)
            cv2.fillPoly(img, [pts], list(color) + [255], cv2.LINE_AA)
            cv2.circle(img, (cx, cy - length + width), width, list(color) + [255], -1, cv2.LINE_AA)
            return img
        elif style == 'skeleton':
            outer_pts = np.array([[cx - width // 2, cy], [cx - width // 2, cy - length], [cx + width // 2, cy - length],
                                  [cx + width // 2, cy]], dtype=np.int32)
            inner_width = width // 3
            inner_pts = np.array([[cx - inner_width // 2, cy - width], [cx - inner_width // 2, cy - length + width],
                                  [cx + inner_width // 2, cy - length + width], [cx + inner_width // 2, cy - width]],
                                 dtype=np.int32)
            cv2.fillPoly(img, [outer_pts], list(color) + [255], cv2.LINE_AA)
            cv2.fillPoly(img, [inner_pts], [0, 0, 0, 0], cv2.LINE_AA)
            return img
        elif style == 'baton':
            pts = np.array([[cx - width // 3, cy], [cx - width // 3, cy - length], [cx + width // 3, cy - length],
                            [cx + width // 3, cy]], dtype=np.int32)
        elif style == 'leaf':
            pts = np.array(
                [[cx, cy - length], [cx + width, int(cy - length * 0.6)], [cx + width // 2, cy], [cx - width // 2, cy],
                 [cx - width, int(cy - length * 0.6)]], dtype=np.int32)
        elif style == 'pencil':
            pts = np.array([[cx, cy - length], [cx + width // 3, int(cy - length * 0.9)], [cx + width // 2, cy],
                            [cx - width // 2, cy], [cx - width // 3, int(cy - length * 0.9)]], dtype=np.int32)
        elif style == 'dauphine':
            pts = np.array(
                [[cx, cy - length], [cx + width // 3, int(cy - length * 0.7)], [cx + width, int(cy - length * 0.5)],
                 [cx + width // 2, cy], [cx - width // 2, cy], [cx - width, int(cy - length * 0.5)],
                 [cx - width // 3, int(cy - length * 0.7)]], dtype=np.int32)
        elif style == 'breguet':
            pts = np.array([[cx - width // 2, cy], [cx - width // 2, cy - length], [cx + width // 2, cy - length],
                            [cx + width // 2, cy]], dtype=np.int32)
            cv2.fillPoly(img, [pts], list(color) + [255], cv2.LINE_AA)
            hole_radius = int(width * 0.8)
            hole_y = cy - int(length * 0.75)
            cv2.circle(img, (cx, hole_y), hole_radius, [0, 0, 0, 0], -1, cv2.LINE_AA)
            return img
        elif style == 'spade':
            spade_width = int(width * 1.8)
            pts = np.array([[cx, cy - length], [cx + spade_width, int(cy - length * 0.8)],
                            [int(cx + spade_width * 0.7), int(cy - length * 0.65)],
                            [cx + width // 2, int(cy - length * 0.7)], [cx + width // 2, cy], [cx - width // 2, cy],
                            [cx - width // 2, int(cy - length * 0.7)],
                            [int(cx - spade_width * 0.7), int(cy - length * 0.65)],
                            [cx - spade_width, int(cy - length * 0.8)]], dtype=np.int32)
        elif style == 'anchor':
            pts = np.array([[cx - width // 3, cy], [cx - width // 3, int(cy - length * 0.85)],
                            [int(cx - width * 1.5), int(cy - length * 0.85)],
                            [int(cx - width * 1.5), int(cy - length * 0.7)], [cx - width // 3, int(cy - length * 0.7)],
                            [cx - width // 3, cy - length], [cx + width // 3, cy - length],
                            [cx + width // 3, int(cy - length * 0.7)], [int(cx + width * 1.5), int(cy - length * 0.7)],
                            [int(cx + width * 1.5), int(cy - length * 0.85)],
                            [cx + width // 3, int(cy - length * 0.85)], [cx + width // 3, cy]], dtype=np.int32)
        elif style == 'cathedral':
            pts = np.array([[cx, cy - length], [cx + width // 4, int(cy - length * 0.92)],
                            [int(cx + width * 0.6), int(cy - length * 0.8)], [cx + width // 2, int(cy - length * 0.65)],
                            [cx + width // 2, cy], [cx - width // 2, cy], [cx - width // 2, int(cy - length * 0.65)],
                            [int(cx - width * 0.6), int(cy - length * 0.8)],
                            [cx - width // 4, int(cy - length * 0.92)]], dtype=np.int32)
        elif style == 'alpha':
            pts = np.array([[cx, cy - length], [int(cx + width * 1.5), int(cy - length * 0.6)], [cx + width // 2, cy],
                            [cx - width // 2, cy], [int(cx - width * 1.5), int(cy - length * 0.6)]], dtype=np.int32)
        elif style == 'feuille':
            pts = np.array([[cx, cy - length], [int(cx + width * 0.8), int(cy - length * 0.5)], [cx + width // 3, cy],
                            [cx - width // 3, cy], [int(cx - width * 0.8), int(cy - length * 0.5)]], dtype=np.int32)
        elif style == 'lance':
            pts = np.array([[cx, cy - length], [int(cx + width * 0.6), int(cy - length * 0.9)],
                            [cx + width // 4, int(cy - length * 0.8)], [cx + width // 4, cy], [cx - width // 4, cy],
                            [cx - width // 4, int(cy - length * 0.8)], [int(cx - width * 0.6), int(cy - length * 0.9)]],
                           dtype=np.int32)
        elif style == 'plongeur':
            pts = np.array(
                [[int(cx - width * 0.7), cy], [int(cx - width * 0.7), int(cy - length * 0.9)], [cx, cy - length],
                 [int(cx + width * 0.7), int(cy - length * 0.9)], [int(cx + width * 0.7), cy]], dtype=np.int32)
        elif style == 'syringe':
            tip_width = width // 4
            pts = np.array([[cx, cy - length], [cx + tip_width, int(cy - length * 0.95)],
                            [cx + width // 2, int(cy - length * 0.9)], [cx + width // 2, cy], [cx - width // 2, cy],
                            [cx - width // 2, int(cy - length * 0.9)], [cx - tip_width, int(cy - length * 0.95)]],
                           dtype=np.int32)
        elif style == 'flamme':
            pts = np.array([[cx, cy - length], [int(cx + width * 0.3), int(cy - length * 0.9)],
                            [int(cx + width * 0.7), int(cy - length * 0.7)],
                            [int(cx + width * 0.5), int(cy - length * 0.5)], [cx + width // 2, cy],
                            [cx - width // 2, cy], [int(cx - width * 0.5), int(cy - length * 0.5)],
                            [int(cx - width * 0.7), int(cy - length * 0.7)],
                            [int(cx - width * 0.3), int(cy - length * 0.9)]], dtype=np.int32)
        elif style == 'modern':
            pts = np.array([[cx - width // 2, cy], [cx - width // 2, cy - length], [cx + width // 2, cy - length],
                            [cx + width // 2, cy]], dtype=np.int32)
        else:  # Default
            pts = np.array([[cx - width // 2, cy], [cx - width // 2, cy - length], [cx + width // 2, cy - length],
                            [cx + width // 2, cy]], dtype=np.int32)

        cv2.fillPoly(img, [pts], list(color) + [255], cv2.LINE_AA)
        return img