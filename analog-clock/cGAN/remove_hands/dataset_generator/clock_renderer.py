# ==============================================================================
# FILE: clock_renderer.py
# ==============================================================================

import cv2
import numpy as np
from typing import Tuple


class ClockRenderer:
    def __init__(self, image_size: int = 256):
        self.image_size = image_size

    def rotate_hand(self, hand_img: np.ndarray, angle: float, center: Tuple[int, int]) -> np.ndarray:
        h, w = hand_img.shape[:2]
        # Rotation around the specific center point provided
        M = cv2.getRotationMatrix2D(center, -angle, 1.0)

        rotated = cv2.warpAffine(
            hand_img, M, (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0, 0)
        )
        return rotated

    def overlay_image(self, background: np.ndarray, overlay: np.ndarray) -> np.ndarray:
        if overlay.shape[2] != 4: return background
        alpha = overlay[:, :, 3] / 255.0
        alpha = np.stack([alpha] * 3, axis=2)

        overlay_rgb = overlay[:, :, :3]
        background_rgb = background[:, :, :3]
        result = (overlay_rgb * alpha + background_rgb * (1 - alpha)).astype(np.uint8)
        return result

    def composite_hands(self, background: np.ndarray,
                        hour_hand_img: np.ndarray,
                        minute_hand_img: np.ndarray,
                        hour: int, minute: int,
                        center: Tuple[int, int] = None,
                        second_hand_img: np.ndarray = None,
                        second: int = 0) -> np.ndarray:
        """
        Composite hands. Supports Hour, Minute AND optional Second hand.
        """
        if center is None:
            h, w = background.shape[:2]
            center = (w // 2, h // 2)

        result = background.copy()

        # Hour Hand
        angle_h = (hour % 12) * 30 + minute * 0.5
        rot_h = self.rotate_hand(hour_hand_img, angle_h, center)
        result = self.overlay_image(result, rot_h)

        # Minute Hand
        angle_m = minute * 6
        rot_m = self.rotate_hand(minute_hand_img, angle_m, center)
        result = self.overlay_image(result, rot_m)

        # Second Hand (Optional)
        if second_hand_img is not None:
            angle_s = second * 6
            rot_s = self.rotate_hand(second_hand_img, angle_s, center)
            result = self.overlay_image(result, rot_s)

        # Center Pin
        cv2.circle(result, center, 4, (20, 20, 20), -1, cv2.LINE_AA)

        return result