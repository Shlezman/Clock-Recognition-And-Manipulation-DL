import random
from PIL import Image, ImageDraw


class ClockRenderer:
    """Renders different clock display formats"""
    
    def __init__(self, font_manager, image_generator):
        self.font_manager = font_manager
        self.image_generator = image_generator
        
        # Display formats - removed binary clock
        self.display_formats = [
            'colon', '7segment', 'dot_separator', 
            'no_separator', 'dash_separator'
        ]
    
    def generate_random_color(self, dark=False, bright=False):
        """Generate random colors"""
        if dark:
            return tuple(random.randint(0, 50) for _ in range(3))
        elif bright:
            colors = [
                (255, 0, 0), (0, 255, 0), (60, 160, 255), 
                (255, 255, 255), (255, 140, 0), (0, 255, 255),
                (255, 50, 50), (50, 255, 150)
            ]
            base = random.choice(colors)
            return tuple(min(255, max(0, c + random.randint(-40, 40))) for c in base)
        return tuple(random.randint(0, 255) for _ in range(3))
    
    def draw_7segment_digit(self, draw, x, y, digit, sw, sh, color, t):
        """Draw a single 7-segment digit"""
        segments = {
            '0': [1, 1, 1, 1, 1, 1, 0],
            '1': [0, 1, 1, 0, 0, 0, 0],
            '2': [1, 1, 0, 1, 1, 0, 1],
            '3': [1, 1, 1, 1, 0, 0, 1],
            '4': [0, 1, 1, 0, 0, 1, 1],
            '5': [1, 0, 1, 1, 0, 1, 1],
            '6': [1, 0, 1, 1, 1, 1, 1],
            '7': [1, 1, 1, 0, 0, 0, 0],
            '8': [1, 1, 1, 1, 1, 1, 1],
            '9': [1, 1, 1, 1, 0, 1, 1]
        }
        
        if digit not in segments:
            return
        
        active = segments[digit]
        gap = t // 2
        
        # Segment coordinates: top, top-right, bottom-right, bottom, bottom-left, top-left, middle
        coords = [
            [x + gap, y, x + sw - gap, y + t],  # top
            [x + sw - t, y + gap, x + sw, y + sh - gap],  # top-right
            [x + sw - t, y + sh + gap, x + sw, y + 2 * sh - gap],  # bottom-right
            [x + gap, y + 2 * sh - t, x + sw - gap, y + 2 * sh],  # bottom
            [x, y + sh + gap, x + t, y + 2 * sh - gap],  # bottom-left
            [x, y + gap, x + t, y + sh - gap],  # top-left
            [x + gap, y + sh - t // 2, x + sw - gap, y + sh + t // 2]  # middle
        ]
        
        for i, is_active in enumerate(active):
            if is_active:
                draw.rectangle(coords[i], fill=color)
    
    def draw_colon(self, draw, xp, sy, sh, t, color):
        """Draw colon separator"""
        dot = int(t * 1.5)
        yo = sh // 2
        draw.ellipse([xp, sy + yo - dot, xp + dot, sy + yo], fill=color)
        draw.ellipse([xp, sy + sh + yo, xp + dot, sy + sh + yo + dot], fill=color)
    
    def generate_clock_layer(self, time_str, target_width):
        """Generate a clock display layer with bounding boxes"""
        display_format = random.choice(self.display_formats)
        
        # Decide if showing seconds (30% chance)
        show_seconds = random.random() > 0.7
        seconds_str = f"{random.randint(0, 59):02d}" if show_seconds else ""
        
        text_color = self.generate_random_color(bright=True)
        hours, minutes = map(int, time_str.split(':'))
        
        canvas_w = int(target_width * 2.0)
        canvas_h = int(target_width * 1.0)
        img = Image.new('RGBA', (canvas_w, canvas_h), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        
        hours_bbox, minutes_bbox = None, None
        
        if display_format == '7segment':
            hours_bbox, minutes_bbox = self._render_7segment(
                draw, hours, minutes, seconds_str, show_seconds,
                canvas_w, canvas_h, target_width, text_color
            )
        else:
            hours_bbox, minutes_bbox = self._render_font_based(
                draw, hours, minutes, seconds_str, show_seconds,
                canvas_w, canvas_h, target_width, text_color, display_format
            )
        
        # Add text noise
        img = self.image_generator.add_text_noise(img)
        
        # Crop to content
        bbox_union = img.getbbox()
        if bbox_union:
            pad = 20
            crop_box = (
                max(0, bbox_union[0] - pad),
                max(0, bbox_union[1] - pad),
                min(canvas_w, bbox_union[2] + pad),
                min(canvas_h, bbox_union[3] + pad)
            )
            img = img.crop(crop_box)
            ox, oy = crop_box[0], crop_box[1]
            
            if hours_bbox:
                hours_bbox = (
                    hours_bbox[0] - ox, hours_bbox[1] - oy,
                    hours_bbox[2] - ox, hours_bbox[3] - oy
                )
            if minutes_bbox:
                minutes_bbox = (
                    minutes_bbox[0] - ox, minutes_bbox[1] - oy,
                    minutes_bbox[2] - ox, minutes_bbox[3] - oy
                )
        
        return img, hours_bbox, minutes_bbox
    
    def _render_7segment(self, draw, hours, minutes, seconds_str, show_seconds,
                        canvas_w, canvas_h, target_width, text_color):
        """Render 7-segment display"""
        num_digits = 6 if show_seconds else 4
        estimated_sw = target_width / (num_digits * 1.5)
        sw = int(estimated_sw)
        sh = int(sw * 1.8)
        t = max(5, int(sw * 0.25))
        ds = int(sw * 1.5)
        
        total_content_w = (num_digits * ds) + (ds * 0.5 if show_seconds else 0)
        sx = (canvas_w - total_content_w) // 2
        sy = (canvas_h - sh * 2) // 2
        
        # Draw hours
        xp = sx
        h_start = xp
        for d in f"{hours:02d}":
            self.draw_7segment_digit(draw, xp, sy, d, sw, sh, text_color, t)
            xp += ds
        hours_bbox = (h_start, sy, xp - ds + sw, sy + 2 * sh)
        
        # Draw colon
        self.draw_colon(draw, xp, sy, sh, t, text_color)
        
        # Random artifact near colon
        if random.random() > 0.8:
            self.image_generator.add_random_artifacts(
                draw, (int(xp), int(sy), int(ds), int(sh)), text_color
            )
        
        xp += ds * 0.5
        
        # Draw minutes
        m_start = xp
        for d in f"{minutes:02d}":
            self.draw_7segment_digit(draw, xp, sy, d, sw, sh, text_color, t)
            xp += ds
        minutes_bbox = (m_start, sy, xp - ds + sw, sy + 2 * sh)
        
        # Draw seconds (no bbox)
        if show_seconds:
            self.draw_colon(draw, xp, sy, sh, t, text_color)
            xp += ds * 0.5
            for d in seconds_str:
                self.draw_7segment_digit(draw, xp, sy, d, sw, sh, text_color, t)
                xp += ds
        
        return hours_bbox, minutes_bbox
    
    def _render_font_based(self, draw, hours, minutes, seconds_str, show_seconds,
                           canvas_w, canvas_h, target_width, text_color, display_format):
        """Render font-based display"""
        num_chars = 8 if show_seconds else 5
        estimated_char_w = target_width / num_chars
        font_size = int(estimated_char_w * 1.6)
        
        # Use 7-segment font (30% chance)
        if random.random() < 0.3:
            font = self.font_manager.get_7segment_font(font_size)
        else:
            font = self.font_manager.get_random_font(font_size)
        
        # Format time string
        time_fmt = f"{hours:02d}:{minutes:02d}"
        if show_seconds:
            time_fmt += f":{seconds_str}"
        
        if display_format == 'no_separator':
            time_fmt = time_fmt.replace(":", " ")
        elif display_format == 'dot_separator':
            time_fmt = time_fmt.replace(":", ".")
        elif display_format == 'dash_separator':
            time_fmt = time_fmt.replace(":", "-")
        
        # Draw text
        bbox = draw.textbbox((0, 0), time_fmt, font=font)
        text_w, text_h = bbox[2] - bbox[0], bbox[3] - bbox[1]
        x = (canvas_w - text_w) // 2
        y = (canvas_h - text_h) // 2
        
        draw.text((x, y), time_fmt, fill=text_color, font=font)
        
        # Calculate bboxes
        h_str = f"{hours:02d}"
        h_bb = draw.textbbox((x, y), h_str, font=font)
        hours_bbox = h_bb
        
        # Determine separator
        sep = ":" if ":" in time_fmt else ("." if "." in time_fmt else ("-" if "-" in time_fmt else " "))
        prefix_w = draw.textlength(h_str + sep, font=font)
        m_w = draw.textlength(f"{minutes:02d}", font=font)
        minutes_bbox = (x + prefix_w, h_bb[1], x + prefix_w + m_w, h_bb[3])
        
        # Add artifacts
        if random.random() > 0.6:
            self.image_generator.add_random_artifacts(
                draw, (0, 0, canvas_w, canvas_h), text_color
            )
        
        return hours_bbox, minutes_bbox