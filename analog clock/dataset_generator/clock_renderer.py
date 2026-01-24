# ==============================================================================
# FILE 3: clock_renderer.py
# Clock rendering and hand composition logic
# ==============================================================================

import cv2
import numpy as np
import math
from typing import Tuple, Optional, Dict


class ClockRenderer:
    """Handles clock hand rendering and composition"""

    def __init__(self, image_size: int = 256):
        self.image_size = image_size

    def calculate_hand_angle(
            self,
            hour: int,
            minute: int,
            is_hour_hand: bool = True
    ) -> float:
        """
        Calculate hand angle in degrees.
        12 o'clock = 0°, clockwise is positive

        Args:
            hour: Hour value (0-23)
            minute: Minute value (0-59)
            is_hour_hand: True for hour hand, False for minute hand

        Returns:
            Angle in degrees
        """
        if is_hour_hand:
            # Hour hand: 30° per hour + 0.5° per minute
            angle = (hour % 12) * 30 + minute * 0.5
        else:
            # Minute hand: 6° per minute
            angle = minute * 6

        return angle

    def rotate_hand(
            self,
            hand_img: np.ndarray,
            angle: float,
            center: Tuple[int, int]
    ) -> np.ndarray:
        """
        Rotate hand image around center point.

        Args:
            hand_img: Hand image with alpha channel
            angle: Angle in degrees (clockwise from 12 o'clock)
            center: Rotation center (x, y)

        Returns:
            Rotated hand image
        """
        h, w = hand_img.shape[:2]

        # Adjust angle (OpenCV rotation is counter-clockwise, subtract 90 for 12 o'clock = 0)
        rotation_angle = -(angle - 90)

        # Get rotation matrix
        M = cv2.getRotationMatrix2D(center, rotation_angle, 1.0)

        # Rotate with alpha channel preservation
        rotated = cv2.warpAffine(
            hand_img,
            M,
            (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0, 0)
        )

        return rotated

    def overlay_image(
            self,
            background: np.ndarray,
            overlay: np.ndarray
    ) -> np.ndarray:
        """
        Overlay image with alpha channel onto background.

        Args:
            background: Background image (RGB or RGBA)
            overlay: Overlay image with alpha channel (RGBA)

        Returns:
            Composited image
        """
        if overlay.shape[2] == 4:
            # Extract alpha channel
            alpha = overlay[:, :, 3] / 255.0
            alpha = np.stack([alpha] * 3, axis=2)

            # Blend images
            overlay_rgb = overlay[:, :, :3]
            background_rgb = background[:, :, :3]

            result_rgb = (
                    overlay_rgb * alpha + background_rgb * (1 - alpha)
            ).astype(np.uint8)

            # Preserve background alpha if exists
            if background.shape[2] == 4:
                result = np.dstack([result_rgb, background[:, :, 3]])
            else:
                result = result_rgb
        else:
            result = overlay

        return result

    def composite_hands(
            self,
            background: np.ndarray,
            hand_style: Dict[str, str],
            hour: int,
            minute: int,
            center: Optional[Tuple[int, int]] = None,
            include_second: bool = False
    ) -> np.ndarray:
        """
        Composite clock hands onto background at specified time.

        Args:
            background: Background image (clock face)
            hand_style: Dictionary with paths to hand images
            hour: Hour (0-23)
            minute: Minute (0-59)
            center: Optional center point, defaults to image center
            include_second: Whether to include second hand

        Returns:
            Image with composited hands
        """
        if center is None:
            h, w = background.shape[:2]
            center = (w // 2, h // 2)

        # Convert background to RGBA if needed
        if background.shape[2] == 3:
            background = cv2.cvtColor(background, cv2.COLOR_BGR2BGRA)

        result = background.copy()

        # Load and composite hour hand
        hour_img = cv2.imread(hand_style['hour'], cv2.IMREAD_UNCHANGED)
        if hour_img is None:
            raise ValueError(f"Failed to load hour hand: {hand_style['hour']}")
        hour_img = cv2.resize(hour_img, (self.image_size, self.image_size))
        hour_angle = self.calculate_hand_angle(hour, minute, is_hour_hand=True)
        hour_rotated = self.rotate_hand(hour_img, hour_angle, center)
        result = self.overlay_image(result, hour_rotated)

        # Load and composite minute hand
        minute_img = cv2.imread(hand_style['minute'], cv2.IMREAD_UNCHANGED)
        if minute_img is None:
            raise ValueError(f"Failed to load minute hand: {hand_style['minute']}")
        minute_img = cv2.resize(minute_img, (self.image_size, self.image_size))
        minute_angle = self.calculate_hand_angle(hour, minute, is_hour_hand=False)
        minute_rotated = self.rotate_hand(minute_img, minute_angle, center)
        result = self.overlay_image(result, minute_rotated)

        # Optional: second hand
        if include_second and hand_style.get('second'):
            import random
            second_img = cv2.imread(hand_style['second'], cv2.IMREAD_UNCHANGED)
            if second_img is not None:
                second_img = cv2.resize(second_img, (self.image_size, self.image_size))
                second = random.randint(0, 59)
                second_angle = second * 6
                second_rotated = self.rotate_hand(second_img, second_angle, center)
                result = self.overlay_image(result, second_rotated)

        return result