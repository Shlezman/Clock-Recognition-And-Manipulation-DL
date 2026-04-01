"""
Sketch generator: wraps analog_sketch_creator.draw_analog_clock().

Uses proper package imports instead of sys.path manipulation.
"""

from __future__ import annotations

import cv2
import numpy as np

from analog_clock.analog_sketch_creator import draw_analog_clock


class SketchGenerator:
    """Generates clock sketches via the canonical analog_sketch_creator module."""

    def __init__(self, image_size: int = 256) -> None:
        self.image_size = image_size

    def draw_analog_clock(self, hour: int, minute: int) -> np.ndarray:
        """Return an RGB sketch of a clock showing *hour*:*minute*."""
        sketch = draw_analog_clock(hh=hour, mm=minute, return_array=True)

        if sketch.shape[:2] != (self.image_size, self.image_size):
            sketch = cv2.resize(sketch, (self.image_size, self.image_size))

        # Ensure RGB (the upstream function may return RGBA or grayscale)
        if sketch.ndim == 3 and sketch.shape[2] == 4:
            sketch = cv2.cvtColor(sketch, cv2.COLOR_RGBA2RGB)
        elif sketch.ndim == 2:
            sketch = cv2.cvtColor(sketch, cv2.COLOR_GRAY2RGB)

        return sketch
