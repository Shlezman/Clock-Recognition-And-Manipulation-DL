"""
Clock renderer: rotate hands, alpha-blend overlays, composite full clocks.

Unified version supporting optional second hand (used by both pipelines).
"""

from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np


class ClockRenderer:
    """Stateless renderer — works on any canvas size."""

    # ------------------------------------------------------------------
    # Core primitives
    # ------------------------------------------------------------------

    @staticmethod
    def rotate_hand(
        hand_img: np.ndarray,
        angle_cw_deg: float,
        center: Tuple[int, int],
    ) -> np.ndarray:
        """Rotate an RGBA hand image *angle_cw_deg* degrees clockwise from 12."""
        h, w = hand_img.shape[:2]
        M = cv2.getRotationMatrix2D(center, -angle_cw_deg, 1.0)
        return cv2.warpAffine(
            hand_img, M, (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0, 0),
        )

    @staticmethod
    def overlay_rgba(
        background: np.ndarray,
        overlay: np.ndarray,
    ) -> np.ndarray:
        """Alpha-blend an RGBA overlay onto an RGB background (in-place safe)."""
        if overlay.ndim < 3 or overlay.shape[2] != 4:
            return background
        alpha = overlay[:, :, 3:4].astype(np.float32) / 255.0
        fg = overlay[:, :, :3].astype(np.float32)
        bg = background.astype(np.float32)
        blended = fg * alpha + bg * (1.0 - alpha)
        return np.clip(blended, 0, 255).astype(np.uint8)

    # ------------------------------------------------------------------
    # High-level compositing
    # ------------------------------------------------------------------

    def composite_hands(
        self,
        background: np.ndarray,
        hour_hand_img: np.ndarray,
        minute_hand_img: np.ndarray,
        hour: int,
        minute: int,
        center: Optional[Tuple[int, int]] = None,
        second_hand_img: Optional[np.ndarray] = None,
        second: int = 0,
    ) -> np.ndarray:
        """Composite hour + minute (+ optional second) hands onto *background*."""
        if center is None:
            h, w = background.shape[:2]
            center = (w // 2, h // 2)

        result = background.copy()

        # Hour
        angle_h = (hour % 12) * 30 + minute * 0.5
        result = self.overlay_rgba(result, self.rotate_hand(hour_hand_img, angle_h, center))

        # Minute
        angle_m = minute * 6
        result = self.overlay_rgba(result, self.rotate_hand(minute_hand_img, angle_m, center))

        # Second (optional)
        if second_hand_img is not None:
            angle_s = second * 6
            result = self.overlay_rgba(result, self.rotate_hand(second_hand_img, angle_s, center))

        # Center pin
        cv2.circle(result, center, 4, (20, 20, 20), -1, cv2.LINE_AA)
        return result
