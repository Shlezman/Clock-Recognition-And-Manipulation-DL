import os
import random
import shutil
import glob
import math
import subprocess
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageEnhance, ImageOps


class ClockDatasetGenerator:
    def __init__(self, output_dir="clock_dataset", background_dir="backgrounds"):
        self.output_dir = output_dir
        self.background_dir = background_dir

        self.temp_images_dir = os.path.join(output_dir, "temp_images")
        self.temp_labels_dir = os.path.join(output_dir, "temp_labels")

        # Cleanup
        if os.path.exists(self.temp_images_dir): shutil.rmtree(self.temp_images_dir)
        if os.path.exists(self.temp_labels_dir): shutil.rmtree(self.temp_labels_dir)

        os.makedirs(self.temp_images_dir, exist_ok=True)
        os.makedirs(self.temp_labels_dir, exist_ok=True)
        os.makedirs(self.background_dir, exist_ok=True)

        self.display_formats = ['colon', '7segment', 'dot_separator', 'no_separator', 'dash_separator']

        self._check_and_download_backgrounds()

        self.real_bg_images = []
        if os.path.exists(self.background_dir):
            types = ['*.jpg', '*.png', '*.jpeg', '*.webp']
            for t in types:
                self.real_bg_images.extend(glob.glob(os.path.join(self.background_dir, "**", t), recursive=True))

        print(f"Status: Found {len(self.real_bg_images)} real background images.")

    def _check_and_download_backgrounds(self):
        if not os.listdir(self.background_dir):
            print("Attempting to download textures dataset via Kaggle...")
            try:
                subprocess.run(["kaggle", "datasets", "download", "-d", "roustoumabdelmoula/textures-dataset", "-p",
                                self.background_dir, "--unzip"], check=True)
            except:
                print("Warning: Could not download from Kaggle. Using synthetic backgrounds only.")

    def generate_random_color(self, dark=False, bright=False):
        if dark:
            return tuple(random.randint(0, 50) for _ in range(3))
        elif bright:
            colors = [(255, 0, 0), (0, 255, 0), (60, 160, 255), (255, 255, 255), (255, 140, 0), (0, 255, 255),
                      (255, 50, 50)]
            base = random.choice(colors)
            return tuple(min(255, max(0, c + random.randint(-40, 40))) for c in base)
        return tuple(random.randint(0, 255) for _ in range(3))

    def _load_font(self, size):
        # Expanded font list
        font_names = [
            # Common Digital/Monospace
            "arial.ttf", "consola.ttf", "cour.ttf", "lucon.ttf",
            "DejaVuSansMono-Bold.ttf", "OCRA.ttf",
            # System Fonts (Windows/Linux)
            "seguiemj.ttf", "segoeui.ttf", "tahoma.ttf", "verdana.ttf", "calibri.ttf",
            "Roboto-Bold.ttf", "LiberationSans-Bold.ttf", "FreeSansBold.ttf",
            "impact.ttf", "comic.ttf"  # Random styles
        ]

        # 1. Try explicit names
        for fn in font_names:
            try:
                return ImageFont.truetype(fn, size)
            except:
                continue

        # 2. System search
        search_paths = [
            "C:\\Windows\\Fonts", "/usr/share/fonts", "/usr/local/share/fonts",
            "/System/Library/Fonts", os.path.expanduser("~/.fonts")
        ]
        for path in search_paths:
            if os.path.exists(path):
                for root, dirs, files in os.walk(path):
                    for file in files:
                        if file.lower().endswith(".ttf"):
                            try:
                                return ImageFont.truetype(os.path.join(root, file), size)
                            except:
                                continue

        print("WARNING: Using default bitmap font (Tiny).")
        return ImageFont.load_default()

    def get_background(self, w, h):
        if self.real_bg_images and random.random() < 0.85:
            try:
                bg_path = random.choice(self.real_bg_images)
                with Image.open(bg_path) as bg:
                    if bg.mode != 'RGB': bg = bg.convert('RGB')
                    bw, bh = bg.size
                    if bw < w or bh < h:
                        scale = max(w / bw, h / bh) * random.uniform(1.0, 1.5)
                        bg = bg.resize((int(bw * scale), int(bh * scale)))
                        bw, bh = bg.size
                    x = random.randint(0, max(0, bw - w))
                    y = random.randint(0, max(0, bh - h))
                    crop = bg.crop((x, y, x + w, y + h))
                    return ImageEnhance.Brightness(crop).enhance(random.uniform(0.2, 0.6))
            except:
                pass
        color = (random.randint(0, 30), random.randint(0, 30), random.randint(0, 35))
        return Image.new('RGB', (w, h), color)

    # --- Geometric Transforms ---

    def find_coeffs(self, source_coords, target_coords):
        matrix = []
        for s, t in zip(source_coords, target_coords):
            matrix.append([t[0], t[1], 1, 0, 0, 0, -s[0] * t[0], -s[0] * t[1]])
            matrix.append([0, 0, 0, t[0], t[1], 1, -s[1] * t[0], -s[1] * t[1]])
        A = np.matrix(matrix, dtype=float)
        B = np.array(source_coords).reshape(8)
        res = np.dot(np.linalg.inv(A.T * A) * A.T, B)
        return np.array(res).reshape(8)

    def apply_perspective(self, img, h_bbox, m_bbox):
        """Applies 3D-like perspective transform to image and bboxes"""
        w, h = img.size

        # Define random perspective distortion
        # Max distortion factor (0.0 to 0.3)
        factor = random.uniform(0.05, 0.25)

        # Original corners
        orig_corners = [(0, 0), (w, 0), (w, h), (0, h)]

        # Distort corners
        dx = w * factor
        dy = h * factor

        # Choose a random perspective type: trapezoid (tilt) or skew
        mode = random.choice(['tilt_h', 'tilt_v', 'skew'])

        if mode == 'tilt_h':  # Like looking from side
            d = random.uniform(0, dy)
            new_corners = [(0, d), (w, 0), (w, h), (0, h - d)]  # Right side bigger
            if random.random() > 0.5:  # Flip: Left side bigger
                new_corners = [(0, 0), (w, d), (w, h - d), (0, h)]

        elif mode == 'tilt_v':  # Like looking from bottom/top
            d = random.uniform(0, dx)
            new_corners = [(d, 0), (w - d, 0), (w, h), (0, h)]  # Top narrower
            if random.random() > 0.5:  # Bottom narrower
                new_corners = [(0, 0), (w, 0), (w - d, h), (d, h)]

        else:  # Random skew
            new_corners = [
                (random.uniform(0, dx), random.uniform(0, dy)),
                (w - random.uniform(0, dx), random.uniform(0, dy)),
                (w - random.uniform(0, dx), h - random.uniform(0, dy)),
                (random.uniform(0, dx), h - random.uniform(0, dy))
            ]

        # Calculate coefficients for PIL
        coeffs = self.find_coeffs(orig_corners, new_corners)

        # Transform Image
        img_transformed = img.transform((w, h), Image.PERSPECTIVE, coeffs, Image.BICUBIC)

        # Transform BBoxes
        # To transform points forward, we need the inverse logic of PIL's backward mapping.
        # It's easier to compute the homography matrix from Source -> Target directly for points.

        # Matrix calculation for points (Source -> Target)
        pa = orig_corners
        pb = new_corners
        matrix = []
        for p1, p2 in zip(pa, pb):
            matrix.append([p1[0], p1[1], 1, 0, 0, 0, -p2[0] * p1[0], -p2[0] * p1[1]])
            matrix.append([0, 0, 0, p1[0], p1[1], 1, -p2[1] * p1[0], -p2[1] * p1[1]])

        A = np.matrix(matrix, dtype=float)
        B = np.array(pb).reshape(8)

        # Solve for projection matrix H
        # Note: This is slightly approximate for solving H directly with least squares for 4 points
        # But works well enough for bounding boxes.
        # A better way for exact point mapping is getting the H matrix.

        # Let's use a helper for point projection:
        def project_point(x, y, coeffs_inv):
            # PIL coeffs are for Target -> Source. We need Source -> Target.
            # We calculated coeffs above for PIL (T->S).
            # Let's re-calculate H for S->T specifically for the points.
            pass

            # Re-calc homography strictly for points S->T

        # We use a standard homography solver snippet
        src = np.array(orig_corners)
        dst = np.array(new_corners)

        # Find Homography H such that dst = H * src
        # Using a simple SVD based approach or OpenCV would be best, but trying to stick to numpy
        # A simple approximation: Just wrap the bbox around the transformed corners of the bbox

        def transform_point_via_coeffs(px, py, width, height, new_corners_list):
            # Bilinear interpolation approximation for the bounding box (faster/easier)
            # OR better: use projective geometry formula
            # Let's use the projective formula.
            # x' = (a*x + b*y + c) / (g*x + h*y + 1)
            # y' = (d*x + e*y + f) / (g*x + h*y + 1)
            # We need to solve for a..h.

            # Since we lack a robust solver in pure numpy without cv2, let's use the corners mapping
            # to estimate the new bbox.
            # We will rely on the fact that we know the 4 corners of the image transformed.
            # We can map the bbox relative to image dimensions.

            # Simple relative mapping (good for tilts, bad for strong perspective)
            # This is a heuristic. For high accuracy training, cv2.perspectiveTransform is preferred.

            # Heuristic: Interpolate between the new corners based on relative position
            rel_x = px / width
            rel_y = py / height

            # Top edge interpolation
            top_x = new_corners_list[0][0] + (new_corners_list[1][0] - new_corners_list[0][0]) * rel_x
            top_y = new_corners_list[0][1] + (new_corners_list[1][1] - new_corners_list[0][1]) * rel_x

            # Bottom edge interpolation
            bot_x = new_corners_list[3][0] + (new_corners_list[2][0] - new_corners_list[3][0]) * rel_x
            bot_y = new_corners_list[3][1] + (new_corners_list[2][1] - new_corners_list[3][1]) * rel_x

            # Final point
            final_x = top_x + (bot_x - top_x) * rel_y
            final_y = top_y + (bot_y - top_y) * rel_y
            return final_x, final_y

        def transform_bbox(bbox):
            if not bbox: return None
            x1, y1, x2, y2 = bbox
            pts = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
            new_pts = [transform_point_via_coeffs(p[0], p[1], w, h, new_corners) for p in pts]
            xs = [p[0] for p in new_pts]
            ys = [p[1] for p in new_pts]
            return (min(xs), min(ys), max(xs), max(ys))

        return img_transformed, transform_bbox(h_bbox), transform_bbox(m_bbox)

    # --- Drawing Helpers ---

    def add_text_noise(self, layer):
        """Adds noise/degradation specifically to the text layer alpha/rgb"""
        # 1. Pixel dropout (salt and pepper)
        arr = np.array(layer)
        mask = np.random.random(arr.shape[:2]) > 0.95
        arr[mask] = [0, 0, 0, 0]  # Make random pixels transparent

        # 2. Add noise to color
        noise = np.random.randint(-50, 50, arr.shape)
        arr = np.clip(arr.astype(int) + noise, 0, 255).astype(np.uint8)

        # Restore alpha where it was 0 (don't add noise to empty background)
        orig_alpha = np.array(layer)[:, :, 3]
        arr[:, :, 3] = np.where(orig_alpha == 0, 0, arr[:, :, 3])

        return Image.fromarray(arr)

    def add_random_artifacts(self, draw, w, h):
        """Draws distractors: dots, lines, squares between/near digits"""
        num_artifacts = random.randint(3, 10)
        color = self.generate_random_color(bright=True)

        for _ in range(num_artifacts):
            # Random shapes
            shape_type = random.choice(['dot', 'line', 'rect'])
            x = random.randint(0, w)
            y = random.randint(0, h)

            if shape_type == 'dot':
                r = random.randint(2, 5)
                draw.ellipse([x, y, x + r, y + r], fill=color)
            elif shape_type == 'line':
                lx = random.randint(5, 20)
                ly = random.randint(0, 5)
                draw.line([x, y, x + lx, y + ly], fill=color, width=random.randint(1, 3))
            elif shape_type == 'rect':
                s = random.randint(2, 8)
                draw.rectangle([x, y, x + s, y + s], fill=color)

    def draw_7segment_digit(self, draw, x, y, digit, sw, sh, color, t):
        segs = {'0': [1, 1, 1, 1, 1, 1, 0], '1': [0, 1, 1, 0, 0, 0, 0], '2': [1, 1, 0, 1, 1, 0, 1],
                '3': [1, 1, 1, 1, 0, 0, 1], '4': [0, 1, 1, 0, 0, 1, 1], '5': [1, 0, 1, 1, 0, 1, 1],
                '6': [1, 0, 1, 1, 1, 1, 1], '7': [1, 1, 1, 0, 0, 0, 0], '8': [1, 1, 1, 1, 1, 1, 1],
                '9': [1, 1, 1, 1, 0, 1, 1]}
        if digit not in segs: return
        a = segs[digit]
        g = t // 2
        coords = [
            [x + g, y, x + sw - g, y + t], [x + sw - t, y + g, x + sw, y + sh - g],
            [x + sw - t, y + sh + g, x + sw, y + 2 * sh - g],
            [x + g, y + 2 * sh - t, x + sw - g, y + 2 * sh], [x, y + sh + g, x + t, y + 2 * sh - g],
            [x, y + g, x + t, y + sh - g],
            [x + g, y + sh - t // 2, x + sw - g, y + sh + t // 2]
        ]
        for i, isActive in enumerate(a):
            if isActive: draw.rectangle(coords[i], fill=color)

    def _draw_colon(self, draw, xp, sy, sh, t, color):
        dot = int(t * 1.5);
        yo = sh // 2
        draw.ellipse([xp, sy + yo - dot, xp + dot, sy + yo], fill=color)
        draw.ellipse([xp, sy + sh + yo, xp + dot, sy + sh + yo + dot], fill=color)

    def generate_clock_layer(self, time_str, target_width):
        display_format = random.choice(self.display_formats)
        show_seconds = random.random() > 0.5
        seconds_str = f"{random.randint(0, 59):02d}" if show_seconds else ""
        text_color = self.generate_random_color(bright=True)

        hours, minutes = map(int, time_str.split(':'))

        canvas_w = int(target_width * 2.0)
        canvas_h = int(target_width * 1.0)
        img = Image.new('RGBA', (canvas_w, canvas_h), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)

        hours_bbox, minutes_bbox = None, None

        if display_format == '7segment':
            num_digits = 6 if show_seconds else 4
            estimated_sw = target_width / (num_digits * 1.5)
            sw = int(estimated_sw);
            sh = int(sw * 1.8);
            t = max(5, int(sw * 0.25))
            ds = int(sw * 1.5)

            total_content_w = (num_digits * ds) + (ds * 0.5)
            sx = (canvas_w - total_content_w) // 2
            sy = (canvas_h - sh * 2) // 2

            xp = sx;
            h_start = xp
            for d in f"{hours:02d}":
                self.draw_7segment_digit(draw, xp, sy, d, sw, sh, text_color, t)
                xp += ds
            hours_bbox = (h_start, sy, xp - ds + sw, sy + 2 * sh)

            self._draw_colon(draw, xp, sy, sh, t, text_color)

            # Random artifact near colon
            if random.random() > 0.7:
                self.add_random_artifacts(draw, int(xp + ds), int(sy + sh))

            xp += ds * 0.5;
            m_start = xp
            for d in f"{minutes:02d}":
                self.draw_7segment_digit(draw, xp, sy, d, sw, sh, text_color, t)
                xp += ds
            minutes_bbox = (m_start, sy, xp - ds + sw, sy + 2 * sh)

            if show_seconds:
                self._draw_colon(draw, xp, sy, sh, t, text_color)
                xp += ds * 0.5
                for d in seconds_str:
                    self.draw_7segment_digit(draw, xp, sy, d, sw, sh, text_color, t)
                    xp += ds

        else:
            # FONT
            num_chars = 8 if show_seconds else 5
            estimated_char_w = target_width / num_chars
            font_size = int(estimated_char_w * 1.6)
            font = self._load_font(font_size)

            ft = f"{hours:02d}:{minutes:02d}"
            if show_seconds: ft += f":{seconds_str}"
            if display_format == 'no_separator':
                ft = ft.replace(":", " ")
            elif display_format == 'dot_separator':
                ft = ft.replace(":", ".")
            elif display_format == 'dash_separator':
                ft = ft.replace(":", "-")

            bbox = draw.textbbox((0, 0), ft, font=font)
            text_w, text_h = bbox[2] - bbox[0], bbox[3] - bbox[1]
            x = (canvas_w - text_w) // 2
            y = (canvas_h - text_h) // 2

            draw.text((x, y), ft, fill=text_color, font=font)

            h_str = f"{hours:02d}"
            h_bb = draw.textbbox((x, y), h_str, font=font)
            hours_bbox = h_bb

            sep = ":" if ":" in ft else ("." if "." in ft else ("-" if "-" in ft else " "))
            prefix_w = draw.textlength(h_str + sep, font=font)
            m_w = draw.textlength(f"{minutes:02d}", font=font)
            minutes_bbox = (x + prefix_w, h_bb[1], x + prefix_w + m_w, h_bb[3])

            # Distractors between digits for Font mode
            if random.random() > 0.5:
                # Add random dots around the text
                self.add_random_artifacts(draw, canvas_w, canvas_h)

        # Apply specific text noise
        img = self.add_text_noise(img)

        # Crop
        bbox_union = img.getbbox()
        if bbox_union:
            pad = 20
            crop_box = (max(0, bbox_union[0] - pad), max(0, bbox_union[1] - pad),
                        min(canvas_w, bbox_union[2] + pad), min(canvas_h, bbox_union[3] + pad))
            img = img.crop(crop_box)
            ox, oy = crop_box[0], crop_box[1]
            if hours_bbox: hours_bbox = (hours_bbox[0] - ox, hours_bbox[1] - oy, hours_bbox[2] - ox, hours_bbox[3] - oy)
            if minutes_bbox: minutes_bbox = (
            minutes_bbox[0] - ox, minutes_bbox[1] - oy, minutes_bbox[2] - ox, minutes_bbox[3] - oy)

        return img, hours_bbox, minutes_bbox

    def generate_composite_image(self, time_str):
        bg_w, bg_h = random.randint(640, 800), random.randint(480, 640)
        bg_img = self.get_background(bg_w, bg_h)
        clock_target_width = int(bg_w * random.uniform(0.7, 0.95))

        clock_layer, h_bbox, m_bbox = self.generate_clock_layer(time_str, clock_target_width)

        # Apply Perspective Transform (Depth angles)
        if random.random() > 0.3:  # 70% chance of perspective
            clock_layer, h_bbox, m_bbox = self.apply_perspective(clock_layer, h_bbox, m_bbox)

        # Container for Glow/Shadow
        margin = 50
        full_w = clock_layer.width + margin * 2
        full_h = clock_layer.height + margin * 2
        container = Image.new('RGBA', (full_w, full_h), (0, 0, 0, 0))

        # Shadow
        shadow = clock_layer.copy()
        data = np.array(shadow);
        data[..., :3] = 0;
        shadow = Image.fromarray(data)
        shadow = shadow.filter(ImageFilter.GaussianBlur(radius=random.randint(4, 8)))

        # Glow
        glow = clock_layer.filter(ImageFilter.GaussianBlur(radius=random.randint(5, 12)))
        glow = ImageEnhance.Brightness(glow).enhance(1.5)

        cx, cy = margin, margin
        container.paste(shadow, (cx + 8, cy + 8), shadow)
        container.paste(glow, (cx, cy), glow)
        container.paste(clock_layer, (cx, cy), clock_layer)

        if h_bbox: h_bbox = (h_bbox[0] + cx, h_bbox[1] + cy, h_bbox[2] + cx, h_bbox[3] + cy)
        if m_bbox: m_bbox = (m_bbox[0] + cx, m_bbox[1] + cy, m_bbox[2] + cx, m_bbox[3] + cy)

        # Standard Rotation
        angle = random.uniform(-5, 5)
        rotated_container = container.rotate(angle, expand=True, resample=Image.BICUBIC)

        # Update BBoxes for rotation (Simplified)
        # Re-using the logic from before or just simple fitting
        # Since we applied perspective already, let's just shift center
        orig_center = (container.width / 2, container.height / 2)
        new_center_diff = (rotated_container.width / 2 - orig_center[0], rotated_container.height / 2 - orig_center[1])

        # Simple rotate point function
        def rot_p(p, a, c):
            rad = math.radians(a)
            ox, oy = c;
            px, py = p
            qx = ox + math.cos(rad) * (px - ox) - math.sin(rad) * (py - oy)
            qy = oy + math.sin(rad) * (px - ox) + math.cos(rad) * (py - oy)
            return qx, qy

        def rotate_bb(bbox):
            if not bbox: return None
            x1, y1, x2, y2 = bbox
            pts = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
            rot_pts = [rot_p(p, -angle, orig_center) for p in pts]
            xs = [p[0] for p in rot_pts];
            ys = [p[1] for p in rot_pts]
            return (min(xs) + new_center_diff[0], min(ys) + new_center_diff[1], max(xs) + new_center_diff[0],
                    max(ys) + new_center_diff[1])

        h_bbox = rotate_bb(h_bbox)
        m_bbox = rotate_bb(m_bbox)

        # Paste
        safe_x = max(0, bg_w - rotated_container.width)
        safe_y = max(0, bg_h - rotated_container.height)
        paste_x = random.randint(0, safe_x) if safe_x > 0 else (bg_w - rotated_container.width) // 2
        paste_y = random.randint(0, safe_y) if safe_y > 0 else (bg_h - rotated_container.height) // 2

        bg_img.paste(rotated_container, (paste_x, paste_y), rotated_container)

        if h_bbox: h_bbox = (h_bbox[0] + paste_x, h_bbox[1] + paste_y, h_bbox[2] + paste_x, h_bbox[3] + paste_y)
        if m_bbox: m_bbox = (m_bbox[0] + paste_x, m_bbox[1] + paste_y, m_bbox[2] + paste_x, m_bbox[3] + paste_y)

        # Final noise
        if random.random() > 0.6:
            bg_img = bg_img.filter(ImageFilter.GaussianBlur(radius=0.5))

        return bg_img, h_bbox, m_bbox

    def polygon_to_yolo(self, bbox, iw, ih):
        if not bbox: return None
        x1, y1, x2, y2 = bbox
        x1 = max(0, min(iw, x1));
        x2 = max(0, min(iw, x2))
        y1 = max(0, min(ih, y1));
        y2 = max(0, min(ih, y2))
        if x2 <= x1 or y2 <= y1: return None
        pts = [x1, y1, x2, y1, x2, y2, x1, y2]
        return [pts[i] / iw if i % 2 == 0 else pts[i] / ih for i in range(len(pts))]

    def generate_dataset(self, num_images=1000):
        print(f"Generating {num_images} enhanced images...")
        c = 0
        for i in range(num_images):
            try:
                time_str = f"{random.randint(0, 23):02d}:{random.randint(0, 59):02d}"
                img, hb, mb = self.generate_composite_image(time_str)
                name = f"{i:06d}"
                img_path = os.path.join(self.temp_images_dir, f"{name}.jpg")
                img.save(img_path, quality=random.randint(80, 95))
                iw, ih = img.size
                with open(os.path.join(self.temp_labels_dir, f"{name}.txt"), 'w') as f:
                    if hb:
                        y = self.polygon_to_yolo(hb, iw, ih)
                        if y: f.write(f"0 {' '.join(map(str, y))}\n")
                    if mb:
                        y = self.polygon_to_yolo(mb, iw, ih)
                        if y: f.write(f"1 {' '.join(map(str, y))}\n")
                c += 1
                if (i + 1) % 100 == 0:
                    print(f"Generated {i + 1}/{num_images} images...")
            except Exception as e:
                print(f"Error {i}: {e}")
        self.split_dataset()

    def split_dataset(self):
        print("Splitting dataset...")
        for s in ['train', 'val']:
            os.makedirs(os.path.join(self.output_dir, s, 'images'), exist_ok=True)
            os.makedirs(os.path.join(self.output_dir, s, 'labels'), exist_ok=True)
        files = [f for f in os.listdir(self.temp_images_dir) if f.endswith('.jpg')]
        random.shuffle(files)
        idx = int(len(files) * 0.75)
        for f in files[:idx]:
            self._move(f, 'train')
        for f in files[idx:]:
            self._move(f, 'val')
        shutil.rmtree(self.temp_images_dir);
        shutil.rmtree(self.temp_labels_dir)
        with open(os.path.join(self.output_dir, "dataset.yaml"), 'w') as f:
            f.write(
                f"path: {os.path.abspath(self.output_dir)}\ntrain: train/images\nval: val/images\nnames:\n  0: hours\n  1: minutes\nnc: 2")
        print("Done.")

    def _move(self, f, split):
        shutil.move(os.path.join(self.temp_images_dir, f), os.path.join(self.output_dir, split, 'images', f))
        l = f.replace('.jpg', '.txt')
        if os.path.exists(os.path.join(self.temp_labels_dir, l)):
            shutil.move(os.path.join(self.temp_labels_dir, l), os.path.join(self.output_dir, split, 'labels', l))


if __name__ == "__main__":
    gen = ClockDatasetGenerator()
    gen.generate_dataset(15000)