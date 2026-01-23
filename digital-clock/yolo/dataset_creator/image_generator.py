import random
import math
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageEnhance


class ImageGenerator:
    """Handles image composition, transformations, and effects"""

    def __init__(self, background_manager):
        self.background_manager = background_manager

    def find_coeffs(self, source_coords, target_coords):
        """Calculate perspective transform coefficients"""
        matrix = []
        for s, t in zip(source_coords, target_coords):
            matrix.append([t[0], t[1], 1, 0, 0, 0, -s[0] * t[0], -s[0] * t[1]])
            matrix.append([0, 0, 0, t[0], t[1], 1, -s[1] * t[0], -s[1] * t[1]])
        A = np.matrix(matrix, dtype=float)
        B = np.array(source_coords).reshape(8)
        res = np.dot(np.linalg.inv(A.T * A) * A.T, B)
        return np.array(res).reshape(8)

    def apply_perspective(self, img, h_bbox, m_bbox):
        """Apply 3D perspective transform to image and bounding boxes"""
        w, h = img.size

        # Random perspective distortion
        factor = random.uniform(0.05, 0.20)

        orig_corners = [(0, 0), (w, 0), (w, h), (0, h)]

        dx = w * factor
        dy = h * factor

        # Choose perspective type
        mode = random.choice(['tilt_h', 'tilt_v', 'slight_skew'])

        if mode == 'tilt_h':
            d = random.uniform(0, dy)
            if random.random() > 0.5:
                new_corners = [(0, d), (w, 0), (w, h), (0, h - d)]
            else:
                new_corners = [(0, 0), (w, d), (w, h - d), (0, h)]

        elif mode == 'tilt_v':
            d = random.uniform(0, dx * 0.5)
            if random.random() > 0.5:
                new_corners = [(d, 0), (w - d, 0), (w, h), (0, h)]
            else:
                new_corners = [(0, 0), (w, 0), (w - d, h), (d, h)]

        else:  # slight_skew
            new_corners = [
                (random.uniform(0, dx * 0.3), random.uniform(0, dy * 0.3)),
                (w - random.uniform(0, dx * 0.3), random.uniform(0, dy * 0.3)),
                (w - random.uniform(0, dx * 0.3), h - random.uniform(0, dy * 0.3)),
                (random.uniform(0, dx * 0.3), h - random.uniform(0, dy * 0.3))
            ]

        coeffs = self.find_coeffs(orig_corners, new_corners)
        img_transformed = img.transform((w, h), Image.PERSPECTIVE, coeffs, Image.BICUBIC)

        def transform_point(px, py, width, height, new_corners_list):
            """Transform point using bilinear interpolation"""
            rel_x = px / width
            rel_y = py / height

            top_x = new_corners_list[0][0] + (new_corners_list[1][0] - new_corners_list[0][0]) * rel_x
            top_y = new_corners_list[0][1] + (new_corners_list[1][1] - new_corners_list[0][1]) * rel_x

            bot_x = new_corners_list[3][0] + (new_corners_list[2][0] - new_corners_list[3][0]) * rel_x
            bot_y = new_corners_list[3][1] + (new_corners_list[2][1] - new_corners_list[3][1]) * rel_x

            final_x = top_x + (bot_x - top_x) * rel_y
            final_y = top_y + (bot_y - top_y) * rel_y
            return final_x, final_y

        def transform_bbox(bbox):
            if not bbox:
                return None
            x1, y1, x2, y2 = bbox
            pts = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
            new_pts = [transform_point(p[0], p[1], w, h, new_corners) for p in pts]
            xs = [p[0] for p in new_pts]
            ys = [p[1] for p in new_pts]
            return (min(xs), min(ys), max(xs), max(ys))

        return img_transformed, transform_bbox(h_bbox), transform_bbox(m_bbox)

    def add_text_noise(self, layer):
        """Add aggressive noise and degradation to text layer"""
        arr = np.array(layer)

        # 1. Pixel dropout (salt and pepper) - MORE AGGRESSIVE
        dropout_rate = random.uniform(0.02, 0.08)  # 2-8% pixel dropout
        mask = np.random.random(arr.shape[:2]) > (1 - dropout_rate)
        arr[mask] = [0, 0, 0, 0]

        # 2. Random bright/dark pixels (like dust or damage)
        noise_pixels = np.random.random(arr.shape[:2]) > 0.95
        for i in range(arr.shape[0]):
            for j in range(arr.shape[1]):
                if noise_pixels[i, j] and arr[i, j, 3] > 0:
                    if random.random() > 0.5:
                        arr[i, j, :3] = 255  # Bright pixel
                    else:
                        arr[i, j, :3] = 0  # Dark pixel

        # 3. Add color noise
        noise_amount = random.randint(20, 50)
        noise = np.random.randint(-noise_amount, noise_amount, arr.shape)
        arr = np.clip(arr.astype(int) + noise, 0, 255).astype(np.uint8)

        # 4. Edge degradation (like worn display)
        if random.random() > 0.5:
            orig_alpha = arr[:, :, 3].copy()
            edge_mask = np.zeros_like(orig_alpha)

            # Erode edges slightly
            from scipy import ndimage
            try:
                edge_mask = ndimage.binary_erosion(orig_alpha > 0, iterations=1)
                erosion_mask = np.random.random(arr.shape[:2]) > 0.8
                arr[:, :, 3] = np.where(erosion_mask & ~edge_mask, 0, arr[:, :, 3])
            except:
                pass

        # Preserve original alpha channel for transparent areas
        orig_alpha = np.array(layer)[:, :, 3]
        arr[:, :, 3] = np.where(orig_alpha == 0, 0, arr[:, :, 3])

        return Image.fromarray(arr)

    def add_random_artifacts(self, draw, area, color, num_artifacts=None):
        """Add MORE random visual artifacts (dots, lines, scratches)"""
        x_off, y_off, w, h = area

        if num_artifacts is None:
            num_artifacts = random.randint(8, 20)

        for _ in range(num_artifacts):
            shape_type = random.choice(['dot', 'line', 'rect', 'scratch', 'cluster'])

            # Generate coords relative to the area
            x = random.randint(int(x_off), int(x_off + w))
            y = random.randint(int(y_off), int(y_off + h))

            # Vary color slightly
            artifact_color = tuple(min(255, max(0, c + random.randint(-50, 50))) for c in color)

            if shape_type == 'dot':
                r = random.randint(1, 4)
                draw.ellipse([x, y, x + r, y + r], fill=artifact_color)

            elif shape_type == 'line':
                lx = random.randint(5, 20)
                ly = random.randint(-5, 5)
                draw.line([x, y, x + lx, y + ly], fill=artifact_color, width=random.randint(1, 3))

            elif shape_type == 'rect':
                s = random.randint(2, 8)
                draw.rectangle([x, y, x + s, y + s], fill=artifact_color)

            elif shape_type == 'scratch':
                # Long thin line (like a scratch)
                angle = random.uniform(0, 2 * 3.14159)
                length = random.randint(10, 40)
                ex = x + int(length * math.cos(angle))
                ey = y + int(length * math.sin(angle))
                draw.line([x, y, ex, ey], fill=artifact_color, width=1)

            elif shape_type == 'cluster':
                # Multiple small dots clustered together
                for _ in range(random.randint(3, 8)):
                    dx = random.randint(-8, 8)
                    dy = random.randint(-8, 8)
                    r = random.randint(1, 2)
                    draw.ellipse([x + dx, y + dy, x + dx + r, y + dy + r], fill=artifact_color)

    def add_noise_around_digits(self, img, h_bbox, m_bbox):
        """Add noise specifically AROUND the digit regions"""
        draw = ImageDraw.Draw(img)
        w, h = img.size

        # Get a color from the image
        arr = np.array(img)
        non_zero = arr[arr[:, :, 3] > 0]
        if len(non_zero) > 0:
            avg_color = tuple(int(x) for x in non_zero[:, :3].mean(axis=0))
        else:
            avg_color = (255, 255, 255)

        margin = 15

        def draw_bbox_noise(bbox):
            if not bbox: return
            x1, y1, x2, y2 = [int(val) for val in bbox]

            # Top area
            self.add_random_artifacts(
                draw, (x1, y1 - margin, x2 - x1, margin),
                avg_color, num_artifacts=random.randint(5, 12)
            )
            # Bottom area
            self.add_random_artifacts(
                draw, (x1, y2, x2 - x1, margin),
                avg_color, num_artifacts=random.randint(5, 12)
            )
            # Left area
            self.add_random_artifacts(
                draw, (x1 - margin, y1, margin, y2 - y1),
                avg_color, num_artifacts=random.randint(5, 12)
            )
            # Right area
            self.add_random_artifacts(
                draw, (x2, y1, margin, y2 - y1),
                avg_color, num_artifacts=random.randint(5, 12)
            )

        # Add noise around hours and minutes
        if h_bbox: draw_bbox_noise(h_bbox)
        if m_bbox: draw_bbox_noise(m_bbox)

        return img

    def add_glare(self, img):
        """Add screen glare/reflection effect"""
        if random.random() > 0.6:  # 40% chance of no glare
            return img

        glare_layer = Image.new('RGBA', img.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(glare_layer)
        w, h = img.size

        # 1. Diagonal glossy streak (specular reflection)
        if random.random() > 0.5:
            # Create a diagonal polygon
            width = random.randint(int(w * 0.1), int(w * 0.4))
            alpha = random.randint(15, 50)

            # Top-left start
            start_x = random.randint(-w, w)
            # Angle slope
            slope = random.uniform(0.5, 2.0)

            # Calculate points for a wide diagonal band
            points = [
                (start_x, 0),
                (start_x + width, 0),
                (start_x + width + h / slope, h),
                (start_x + h / slope, h)
            ]

            draw.polygon(points, fill=(255, 255, 255, alpha))

        # 2. Spot reflection (like a light source)
        else:
            x = random.randint(0, w)
            y = random.randint(0, h)
            r = random.randint(int(min(w, h) * 0.2), int(min(w, h) * 0.5))
            alpha = random.randint(20, 60)

            # Draw blurred circle
            draw.ellipse([x - r, y - r, x + r, y + r], fill=(255, 255, 255, alpha))

        # Blur the glare to make it soft and realistic
        glare_layer = glare_layer.filter(ImageFilter.GaussianBlur(radius=random.randint(5, 15)))

        # Composite
        return Image.alpha_composite(img, glare_layer)

    def apply_glow_shadow(self, clock_layer):
        """Apply glow and shadow effects to clock layer"""
        margin = 50
        full_w = clock_layer.width + margin * 2
        full_h = clock_layer.height + margin * 2
        container = Image.new('RGBA', (full_w, full_h), (0, 0, 0, 0))

        # Shadow
        shadow = clock_layer.copy()
        data = np.array(shadow)
        data[..., :3] = 0
        shadow = Image.fromarray(data)
        shadow = shadow.filter(ImageFilter.GaussianBlur(radius=random.randint(3, 6)))

        # Glow
        glow = clock_layer.filter(ImageFilter.GaussianBlur(radius=random.randint(4, 10)))
        glow = ImageEnhance.Brightness(glow).enhance(1.3)

        cx, cy = margin, margin
        container.paste(shadow, (cx + 6, cy + 6), shadow)
        container.paste(glow, (cx, cy), glow)
        container.paste(clock_layer, (cx, cy), clock_layer)

        return container, (cx, cy)

    def rotate_image_and_bbox(self, img, h_bbox, m_bbox):
        """Rotate image and update bounding boxes"""
        angle = random.uniform(-5, 5)
        orig_w, orig_h = img.size
        rotated = img.rotate(angle, expand=True, resample=Image.BICUBIC)

        orig_center = (orig_w / 2, orig_h / 2)
        new_center = (rotated.width / 2, rotated.height / 2)
        center_diff = (new_center[0] - orig_center[0], new_center[1] - orig_center[1])

        def rot_point(p, a, c):
            rad = math.radians(a)
            ox, oy = c
            px, py = p
            qx = ox + math.cos(rad) * (px - ox) - math.sin(rad) * (py - oy)
            qy = oy + math.sin(rad) * (px - ox) + math.cos(rad) * (py - oy)
            return qx, qy

        def rotate_bbox(bbox):
            if not bbox:
                return None
            x1, y1, x2, y2 = bbox
            pts = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
            rot_pts = [rot_point(p, -angle, orig_center) for p in pts]
            xs = [p[0] for p in rot_pts]
            ys = [p[1] for p in rot_pts]
            return (
                min(xs) + center_diff[0],
                min(ys) + center_diff[1],
                max(xs) + center_diff[0],
                max(ys) + center_diff[1]
            )

        return rotated, rotate_bbox(h_bbox), rotate_bbox(m_bbox)

    def composite_final_image(self, clock_layer, h_bbox, m_bbox):
        """Create final composite image with background"""
        bg_w, bg_h = random.randint(640, 800), random.randint(480, 640)
        bg_img = self.background_manager.get_background(bg_w, bg_h)

        # Add noise around digits BEFORE transforms
        clock_layer = self.add_noise_around_digits(clock_layer, h_bbox, m_bbox)

        # Apply perspective (70% of the time)
        if random.random() > 0.3:
            clock_layer, h_bbox, m_bbox = self.apply_perspective(clock_layer, h_bbox, m_bbox)

        # Apply glow and shadow
        container, (cx, cy) = self.apply_glow_shadow(clock_layer)

        if h_bbox:
            h_bbox = (h_bbox[0] + cx, h_bbox[1] + cy, h_bbox[2] + cx, h_bbox[3] + cy)
        if m_bbox:
            m_bbox = (m_bbox[0] + cx, m_bbox[1] + cy, m_bbox[2] + cx, m_bbox[3] + cy)

        # Add glare (New!)
        container = self.add_glare(container)

        # Rotate
        rotated_container, h_bbox, m_bbox = self.rotate_image_and_bbox(container, h_bbox, m_bbox)

        # Paste on background
        safe_x = max(0, bg_w - rotated_container.width)
        safe_y = max(0, bg_h - rotated_container.height)
        paste_x = random.randint(0, safe_x) if safe_x > 0 else (bg_w - rotated_container.width) // 2
        paste_y = random.randint(0, safe_y) if safe_y > 0 else (bg_h - rotated_container.height) // 2

        bg_img.paste(rotated_container, (paste_x, paste_y), rotated_container)

        if h_bbox:
            h_bbox = (h_bbox[0] + paste_x, h_bbox[1] + paste_y,
                      h_bbox[2] + paste_x, h_bbox[3] + paste_y)
        if m_bbox:
            m_bbox = (m_bbox[0] + paste_x, m_bbox[1] + paste_y,
                      m_bbox[2] + paste_x, m_bbox[3] + paste_y)

        # Final blur (sometimes)
        if random.random() > 0.7:
            bg_img = bg_img.filter(ImageFilter.GaussianBlur(radius=0.5))

        return bg_img, h_bbox, m_bbox