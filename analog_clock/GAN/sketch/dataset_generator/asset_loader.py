# ==============================================================================
# FILE 5: asset_loader.py
# Asset loading and management (NO SECOND HAND)
# ==============================================================================

import cv2
from pathlib import Path
from typing import List, Dict
import random


class AssetLoader:
    """Loads and manages dataset assets (backgrounds, clocks, hands - NO SECOND)"""

    def __init__(
            self,
            backgrounds_dir: str,
            clean_clocks_dir: str,
            hands_dir: str
    ):
        self.backgrounds_dir = Path(backgrounds_dir)
        self.clean_clocks_dir = Path(clean_clocks_dir)
        self.hands_dir = Path(hands_dir)

        # Load asset paths
        self.background_paths = self._load_image_paths(self.backgrounds_dir)
        self.clock_paths = self._load_image_paths(self.clean_clocks_dir)
        self.hands = self._load_hand_assets()

        # Validation
        self._validate_assets()

    def _load_image_paths(self, directory: Path) -> List[Path]:
        """Load all image paths from directory"""
        if not directory.exists():
            print(f"⚠ Warning: Directory not found: {directory}")
            return []

        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        paths = []

        for ext in image_extensions:
            paths.extend(directory.glob(f"*{ext}"))
            paths.extend(directory.glob(f"*{ext.upper()}"))

        return sorted(paths)

    def _load_hand_assets(self) -> List[Dict[str, str]]:
        """
        Load hand assets (hour and minute only - NO SECOND HAND).
        Supports two structures:
        1. Subdirectories per style: hands_dir/style_1/{hour,minute}.png
        2. Flat structure: hands_dir/{style}_hour.png, {style}_minute.png
        """
        hand_styles = []

        # Try subdirectory structure first
        if self.hands_dir.exists():
            for style_dir in self.hands_dir.iterdir():
                if style_dir.is_dir():
                    hour_path = style_dir / "hour.png"
                    minute_path = style_dir / "minute.png"

                    if hour_path.exists() and minute_path.exists():
                        hand_styles.append({
                            'hour': str(hour_path),
                            'minute': str(minute_path),
                            'style_name': style_dir.name
                        })

        # Fallback: flat structure
        if not hand_styles and self.hands_dir.exists():
            hour_files = list(self.hands_dir.glob("*hour*.png"))
            minute_files = list(self.hands_dir.glob("*minute*.png"))

            # Match hour and minute files by name prefix
            for h_file in hour_files:
                prefix = h_file.stem.replace('_hour', '').replace('hour', '').replace('-hour', '')

                for m_file in minute_files:
                    m_prefix = m_file.stem.replace('_minute', '').replace('minute', '').replace('-minute', '')

                    if prefix == m_prefix:
                        hand_styles.append({
                            'hour': str(h_file),
                            'minute': str(m_file),
                            'style_name': prefix or h_file.stem
                        })
                        break

        return hand_styles

    def _validate_assets(self):
        """Validate that required assets are loaded"""
        print("\n" + "=" * 60)
        print("ASSET VALIDATION")
        print("=" * 60)

        if not self.background_paths and not self.clock_paths:
            print("❌ ERROR: No backgrounds or clock faces found!")
            print(f"   Checked: {self.backgrounds_dir} and {self.clean_clocks_dir}")
        else:
            print(f"✓ Backgrounds: {len(self.background_paths)}")
            print(f"✓ Clean clocks: {len(self.clock_paths)}")

        if not self.hands:
            print("❌ ERROR: No hand assets found!")
            print(f"   Checked: {self.hands_dir}")
            print("   Expected structure:")
            print("     Option 1: hands_dir/style_name/{hour,minute}.png")
            print("     Option 2: hands_dir/{name}_hour.png, {name}_minute.png")
        else:
            print(f"✓ Hand styles: {len(self.hands)} (hour + minute only)")
            for i, style in enumerate(self.hands[:3], 1):
                print(f"   {i}. {style['style_name']}")
            if len(self.hands) > 3:
                print(f"   ... and {len(self.hands) - 3} more")

        print("=" * 60 + "\n")

        if not self.hands or (not self.background_paths and not self.clock_paths):
            raise ValueError("Missing required assets. Please check paths.")

    def get_random_background(self, use_textures: bool = True) -> Path:
        """Get random background path"""
        if use_textures and self.background_paths:
            return random.choice(self.background_paths)
        elif self.clock_paths:
            return random.choice(self.clock_paths)
        else:
            raise ValueError("No background assets available")

    def get_random_hand_style(self) -> Dict[str, str]:
        """Get random hand style"""
        return random.choice(self.hands)