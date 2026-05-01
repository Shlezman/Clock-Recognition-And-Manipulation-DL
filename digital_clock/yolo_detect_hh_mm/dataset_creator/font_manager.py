import os
import requests
import zipfile
import platform
from PIL import ImageFont, Image, ImageDraw


class FontManager:
    """Manages font loading and downloading for clock dataset generation"""

    def __init__(self, fonts_dir="fonts"):
        self.fonts_dir = fonts_dir
        os.makedirs(fonts_dir, exist_ok=True)

        self.available_fonts = []
        self.seven_segment_fonts = []

        # Download essential fonts first
        self._download_google_fonts()
        self._download_dseg_fonts()

        # Then check what we have
        self._check_existing_fonts()
        self._find_system_fonts()

        print(f"✓ Found {len(self.available_fonts)} usable fonts")
        print(f"✓ Found {len(self.seven_segment_fonts)} 7-segment fonts")

        if len(self.available_fonts) == 0:
            print("ERROR: No fonts available! Using fallback.")

    def _test_font(self, font_path, size=48):
        """Test if a font can render digits correctly (no boxes)"""
        try:
            font = ImageFont.truetype(font_path, size)

            # Create test image
            test_img = Image.new('RGB', (300, 80), 'white')
            draw = ImageDraw.Draw(test_img)
            test_text = "0123456789:"
            draw.text((10, 10), test_text, fill='black', font=font)

            # Check if it actually drew something (not just boxes)
            pixels = list(test_img.getdata())
            black_pixels = sum(1 for p in pixels if p[0] < 200)

            # If there are enough black pixels, the font works
            if black_pixels > 100:
                return True

            return False
        except Exception as e:
            return False

    def _download_google_fonts(self):
        """Download reliable Google Fonts"""
        fonts_to_download = {
            'Roboto': 'https://github.com/google/roboto/releases/download/v2.138/roboto-unhinted.zip',
            'RobotoMono': 'https://github.com/googlefonts/RobotoMono/releases/download/v3.000/RobotoMono-VariableFont_wght.ttf',
            'Orbitron': 'https://github.com/theleagueof/orbitron/releases/download/2.0/orbitron-2.0.zip',
        }

        for font_name, url in fonts_to_download.items():
            try:
                if any(font_name.lower() in f.lower() for f in os.listdir(self.fonts_dir) if f.endswith('.ttf')):
                    continue

                print(f"Downloading {font_name}...")
                response = requests.get(url, timeout=30)

                if response.status_code == 200:
                    if url.endswith('.zip'):
                        zip_path = os.path.join(self.fonts_dir, f"{font_name}.zip")
                        with open(zip_path, 'wb') as f:
                            f.write(response.content)

                        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                            zip_ref.extractall(os.path.join(self.fonts_dir, font_name))

                        os.remove(zip_path)
                    else:
                        font_path = os.path.join(self.fonts_dir, f"{font_name}.ttf")
                        with open(font_path, 'wb') as f:
                            f.write(response.content)

            except Exception as e:
                print(f"Could not download {font_name}: {e}")

    def _download_dseg_fonts(self):
        """Download DSEG 7-segment fonts"""
        if any('DSEG' in f for f in os.listdir(self.fonts_dir) if f.endswith('.ttf')):
            return

        try:
            print("Downloading DSEG 7-segment fonts...")
            url = "https://github.com/keshikan/DSEG/releases/download/v0.46/DSEG_v046.zip"
            response = requests.get(url, timeout=30)

            if response.status_code == 200:
                dseg_zip = os.path.join(self.fonts_dir, "DSEG.zip")
                with open(dseg_zip, 'wb') as f:
                    f.write(response.content)

                with zipfile.ZipFile(dseg_zip, 'r') as zip_ref:
                    zip_ref.extractall(self.fonts_dir)

                os.remove(dseg_zip)
                print("✓ DSEG fonts downloaded")
        except Exception as e:
            print(f"Could not download DSEG fonts: {e}")

    def _check_existing_fonts(self):
        """Check fonts in the fonts directory"""
        if not os.path.exists(self.fonts_dir):
            return

        for root, dirs, files in os.walk(self.fonts_dir):
            for file in files:
                if file.endswith('.ttf') or file.endswith('.otf'):
                    font_path = os.path.join(root, file)
                    if self._test_font(font_path):
                        self.available_fonts.append(font_path)

                        # Check if it's a 7-segment font
                        file_lower = file.lower()
                        if 'dseg' in file_lower or 'digital' in file_lower or '7segment' in file_lower:
                            self.seven_segment_fonts.append(font_path)

    def _find_system_fonts(self):
        """Find working system fonts"""
        system = platform.system()

        # Platform-specific font paths
        if system == 'Windows':
            font_paths = ['C:\\Windows\\Fonts']
        elif system == 'Darwin':  # macOS
            font_paths = ['/Library/Fonts', '/System/Library/Fonts', os.path.expanduser('~/Library/Fonts')]
        else:  # Linux
            font_paths = ['/usr/share/fonts', '/usr/local/share/fonts', os.path.expanduser('~/.fonts')]

        # Reliable font names that should work
        priority_fonts = [
            # Monospace fonts (best for digits)
            'courier', 'consola', 'monaco', 'menlo', 'dejavusansmono',
            'ubuntumono', 'robotomono', 'liberationmono', 'droidmono',
            'sourcecodepro', 'inconsolata', 'firamono',

            # Sans-serif fonts (also good)
            'arial', 'helvetica', 'verdana', 'tahoma', 'segoeui',
            'roboto', 'ubuntu', 'opensans', 'liberation', 'dejavu',
            'noto', 'oxygen', 'cantarell', 'freesans', 'droid'
        ]

        for path in font_paths:
            if not os.path.exists(path):
                continue

            try:
                for root, dirs, files in os.walk(path):
                    for file in files:
                        if not (file.lower().endswith('.ttf') or file.lower().endswith('.otf')):
                            continue

                        file_lower = file.lower()

                        # Check if it's a priority font
                        if any(pf in file_lower for pf in priority_fonts):
                            # Prefer bold and regular variants
                            if 'bold' in file_lower or 'regular' in file_lower or 'medium' in file_lower:
                                font_path = os.path.join(root, file)

                                # Avoid duplicates
                                if font_path in self.available_fonts:
                                    continue

                                if self._test_font(font_path):
                                    self.available_fonts.append(font_path)

                                    # Stop if we have enough fonts
                                    if len(self.available_fonts) >= 20:
                                        return
            except (PermissionError, OSError):
                continue

    def get_random_font(self, size):
        """Get a random working font"""
        if not self.available_fonts:
            print("WARNING: No fonts available, using default")
            return ImageFont.load_default()

        import random
        font_path = random.choice(self.available_fonts)

        try:
            return ImageFont.truetype(font_path, size)
        except Exception as e:
            print(f"Error loading font {font_path}: {e}")
            # Try another font
            if len(self.available_fonts) > 1:
                other_fonts = [f for f in self.available_fonts if f != font_path]
                if other_fonts:
                    try:
                        return ImageFont.truetype(random.choice(other_fonts), size)
                    except:
                        pass
            return ImageFont.load_default()

    def get_7segment_font(self, size):
        """Get a 7-segment style font specifically"""
        import random

        # First try dedicated 7-segment fonts
        if self.seven_segment_fonts:
            font_path = random.choice(self.seven_segment_fonts)
            try:
                return ImageFont.truetype(font_path, size)
            except:
                pass

        # Fallback to monospace fonts
        mono_fonts = [f for f in self.available_fonts
                      if 'mono' in f.lower() or 'consola' in f.lower() or 'courier' in f.lower()]

        if mono_fonts:
            try:
                return ImageFont.truetype(random.choice(mono_fonts), size)
            except:
                pass

        # Last resort: any available font
        return self.get_random_font(size)