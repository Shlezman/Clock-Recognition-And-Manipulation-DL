# ==============================================================================
# FILE 4: sketch_generator.py
# Sketch generation using your analog_sketch_creator.py
# ==============================================================================

import cv2
import numpy as np
import sys
from pathlib import Path
from config import Config


class SketchGenerator:
    """Generates clock sketches using analog_sketch_creator.py"""

    def __init__(self, image_size: int = 256):
        self.image_size = image_size
        self.config = Config()
        self._setup_sketch_creator()

    def _setup_sketch_creator(self):
        """Setup integration with analog_sketch_creator.py"""
        sketch_path = Path(self.config.SKETCH_CREATOR_PATH)

        # Add to Python path
        if sketch_path not in sys.path:
            sys.path.insert(0, str(sketch_path))

        try:
            from analog_sketch_creator import draw_analog_clock
            self.draw_analog_clock_func = draw_analog_clock
            print("✓ Successfully imported analog_sketch_creator.py")
        except ImportError as e:
            raise ImportError(
                f"Could not import analog_sketch_creator.py from {sketch_path}\n"
                f"Error: {e}\n"
                f"Please ensure analog_sketch_creator.py is in the correct location."
            )

    def draw_analog_clock(self, hour: int, minute: int) -> np.ndarray:
        """
        Generate analog clock sketch using your analog_sketch_creator.py

        Args:
            hour: Hour (0-23)
            minute: Minute (0-59)

        Returns:
            Sketch image as numpy array (RGB)
        """
        # Use your sketch creator function
        sketch = self.draw_analog_clock_func(
            hh=hour,
            mm=minute,
            return_array=True
        )

        # Resize to target size if needed
        if sketch.shape[:2] != (self.image_size, self.image_size):
            sketch = cv2.resize(sketch, (self.image_size, self.image_size))

        # Ensure it's RGB (your function returns RGBA, we need RGB)
        if sketch.shape[2] == 4:
            sketch = cv2.cvtColor(sketch, cv2.COLOR_RGBA2RGB)
        elif sketch.ndim == 2:
            sketch = cv2.cvtColor(sketch, cv2.COLOR_GRAY2RGB)

        return sketch