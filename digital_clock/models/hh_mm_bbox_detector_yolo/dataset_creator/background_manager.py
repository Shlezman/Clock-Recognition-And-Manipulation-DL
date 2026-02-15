import os
import random
import glob
import subprocess
from PIL import Image, ImageEnhance


class BackgroundManager:
    """Manages background images for clock dataset"""
    
    def __init__(self, background_dir="backgrounds"):
        self.background_dir = background_dir
        os.makedirs(background_dir, exist_ok=True)
        
        self._check_and_download_backgrounds()
        
        self.real_bg_images = []
        if os.path.exists(self.background_dir):
            types = ['*.jpg', '*.png', '*.jpeg', '*.webp']
            for t in types:
                self.real_bg_images.extend(
                    glob.glob(os.path.join(self.background_dir, "**", t), recursive=True)
                )
        
        print(f"Found {len(self.real_bg_images)} background images")
    
    def _check_and_download_backgrounds(self):
        """Attempt to download background textures"""
        if not os.listdir(self.background_dir):
            print("Attempting to download textures dataset...")
            try:
                subprocess.run([
                    "kaggle", "datasets", "download", "-d",
                    "roustoumabdelmoula/textures-dataset",
                    "-p", self.background_dir, "--unzip"
                ], check=True)
                print("Textures downloaded successfully")
            except Exception as e:
                print(f"Could not download textures: {e}")
                print("Using synthetic backgrounds only")
    
    def get_background(self, w, h):
        """Get a background image (real or synthetic)"""
        if self.real_bg_images and random.random() < 0.85:
            try:
                bg_path = random.choice(self.real_bg_images)
                with Image.open(bg_path) as bg:
                    if bg.mode != 'RGB':
                        bg = bg.convert('RGB')
                    
                    bw, bh = bg.size
                    
                    # Scale if needed
                    if bw < w or bh < h:
                        scale = max(w / bw, h / bh) * random.uniform(1.0, 1.5)
                        bg = bg.resize((int(bw * scale), int(bh * scale)))
                        bw, bh = bg.size
                    
                    # Random crop
                    x = random.randint(0, max(0, bw - w))
                    y = random.randint(0, max(0, bh - h))
                    crop = bg.crop((x, y, x + w, y + h))
                    
                    # Darken for better contrast
                    return ImageEnhance.Brightness(crop).enhance(
                        random.uniform(0.2, 0.6)
                    )
            except Exception as e:
                print(f"Error loading background: {e}")
        
        # Fallback: solid dark color
        color = (
            random.randint(0, 30),
            random.randint(0, 30),
            random.randint(0, 35)
        )
        return Image.new('RGB', (w, h), color)