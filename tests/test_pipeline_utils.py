"""Tests for analog_clock.pipeline_utils."""

from __future__ import annotations

import math

import cv2
import numpy as np
import pytest
import torch

from analog_clock.pipeline_utils import (
    blend_rgba_onto_rgb,
    decode_sincos,
    extract_hand_rgba,
    find_clock_center,
    generate_intermediate_times,
    get_angle_clockwise_from_12,
    recompose_hand,
    rotate_hand_rgba,
    time_to_degrees_cw,
)


# ============================================================================
# time_to_degrees_cw
# ============================================================================

class TestTimeToDegrees:
    def test_12_oclock_hour(self) -> None:
        assert time_to_degrees_cw(12, 0, "hour") == pytest.approx(0.0)
        assert time_to_degrees_cw(0, 0, "hour") == pytest.approx(0.0)

    def test_3_oclock_hour(self) -> None:
        assert time_to_degrees_cw(3, 0, "hour") == pytest.approx(90.0)

    def test_6_oclock_hour(self) -> None:
        assert time_to_degrees_cw(6, 0, "hour") == pytest.approx(180.0)

    def test_hour_with_minutes(self) -> None:
        # 3:30 → 90 + 15 = 105
        assert time_to_degrees_cw(3, 30, "hour") == pytest.approx(105.0)

    def test_minute_hand(self) -> None:
        assert time_to_degrees_cw(0, 0, "minute") == pytest.approx(0.0)
        assert time_to_degrees_cw(0, 15, "minute") == pytest.approx(90.0)
        assert time_to_degrees_cw(0, 30, "minute") == pytest.approx(180.0)
        assert time_to_degrees_cw(0, 45, "minute") == pytest.approx(270.0)

    def test_unknown_hand_type(self) -> None:
        assert time_to_degrees_cw(5, 30, "second") == 0.0


# ============================================================================
# generate_intermediate_times
# ============================================================================

class TestGenerateIntermediateTimes:
    def test_simple_forward(self) -> None:
        times = generate_intermediate_times(3, 0, 3, 10, step_minutes=5)
        assert times[0] == (3, 0)
        assert times[-1] == (3, 10)
        assert len(times) >= 3  # 0, 5, 10

    def test_includes_end_time(self) -> None:
        times = generate_intermediate_times(1, 0, 1, 7, step_minutes=3)
        assert times[-1] == (1, 7)

    def test_wrap_around(self) -> None:
        # 11:50 → 0:10 wraps past 12
        times = generate_intermediate_times(11, 50, 0, 10, step_minutes=5)
        assert times[0] == (11, 50)
        assert times[-1] == (0, 10)
        assert len(times) >= 4

    def test_single_step(self) -> None:
        times = generate_intermediate_times(6, 0, 6, 0, step_minutes=1)
        # Same start/end triggers wrap: goes 12 full hours
        assert times[0] == (6, 0)


# ============================================================================
# decode_sincos
# ============================================================================

class TestDecodeSincos:
    def test_12_oclock(self) -> None:
        # sin(0)=0, cos(0)=1 for both hour and minute
        pred = torch.tensor([[0.0, 1.0, 0.0, 1.0]])
        hours, minutes = decode_sincos(pred)
        assert hours.item() == pytest.approx(0.0, abs=0.5)
        assert minutes.item() == pytest.approx(0.0, abs=1.0)

    def test_3_oclock(self) -> None:
        # 3:00 → hour angle = π/2, minute angle = 0
        h_angle = math.pi / 2
        pred = torch.tensor([[math.sin(h_angle), math.cos(h_angle), 0.0, 1.0]])
        hours, _ = decode_sincos(pred)
        assert hours.item() == pytest.approx(3.0, abs=0.5)

    def test_6_30(self) -> None:
        # 6:30 → hour_angle = π, minute_angle = π
        h_angle = math.pi
        m_angle = math.pi
        pred = torch.tensor([[math.sin(h_angle), math.cos(h_angle), math.sin(m_angle), math.cos(m_angle)]])
        hours, minutes = decode_sincos(pred)
        assert hours.item() == pytest.approx(6.0, abs=0.5)
        assert minutes.item() == pytest.approx(30.0, abs=1.0)


# ============================================================================
# extract_hand_rgba / rotate / blend
# ============================================================================

def _make_hand_mask(h: int = 256, w: int = 256) -> np.ndarray:
    """Create a simple vertical line mask (simulating a hand at 12 o'clock)."""
    mask = np.zeros((h, w), dtype=np.uint8)
    cx = w // 2
    cv2.line(mask, (cx, h // 2), (cx, h // 4), 255, thickness=4)
    return mask


def _make_rgb(h: int = 256, w: int = 256) -> np.ndarray:
    return np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)


class TestExtractHandRGBA:
    def test_output_shape(self) -> None:
        mask = _make_hand_mask()
        img = _make_rgb()
        rgba = extract_hand_rgba(mask / 255.0, img, feather_radius=3)
        assert rgba.shape == (256, 256, 4)

    def test_alpha_nonzero_where_mask(self) -> None:
        mask = _make_hand_mask()
        img = _make_rgb()
        rgba = extract_hand_rgba(mask, img, feather_radius=3)
        # Pixels at the mask centre should have non-zero alpha
        assert rgba[192, 128, 3] > 0 or rgba[128, 128, 3] > 0


class TestRotateHandRGBA:
    def test_shape_preserved(self) -> None:
        hand = np.zeros((256, 256, 4), dtype=np.uint8)
        hand[100:150, 126:130] = [200, 200, 200, 255]
        rotated = rotate_hand_rgba(hand, 90.0, (128, 128))
        assert rotated.shape == hand.shape

    def test_zero_rotation_identity(self) -> None:
        hand = np.zeros((64, 64, 4), dtype=np.uint8)
        hand[10:30, 30:34] = [100, 100, 100, 255]
        rotated = rotate_hand_rgba(hand, 0.0, (32, 32))
        # Should be nearly identical (Lanczos may introduce tiny differences)
        diff = np.abs(rotated.astype(int) - hand.astype(int))
        assert diff.max() < 5


class TestBlendRGBA:
    def test_fully_opaque_overlay(self) -> None:
        bg = np.full((64, 64, 3), 100, dtype=np.uint8)
        overlay = np.full((64, 64, 4), 200, dtype=np.uint8)
        overlay[:, :, 3] = 255  # fully opaque
        result = blend_rgba_onto_rgb(bg, overlay)
        # Should be all 200 (overlay colour)
        assert np.allclose(result[:, :, :3], 200, atol=1)

    def test_fully_transparent_overlay(self) -> None:
        bg = np.full((64, 64, 3), 100, dtype=np.uint8)
        overlay = np.zeros((64, 64, 4), dtype=np.uint8)
        result = blend_rgba_onto_rgb(bg, overlay)
        assert np.allclose(result, 100, atol=1)


# ============================================================================
# recompose_hand
# ============================================================================

class TestRecomposeHand:
    def test_empty_mask_returns_background(self) -> None:
        bg = _make_rgb(64, 64)
        mask = np.zeros((64, 64), dtype=np.uint8)
        img = _make_rgb(64, 64)
        result = recompose_hand(mask, img, bg, (32, 32), 0.0, 90.0)
        np.testing.assert_array_equal(result, bg)

    def test_nonzero_mask_changes_background(self) -> None:
        bg = np.full((64, 64, 3), 50, dtype=np.uint8)
        mask = np.zeros((64, 64), dtype=np.uint8)
        cv2.line(mask, (32, 32), (32, 10), 255, 3)
        img = np.full((64, 64, 3), 200, dtype=np.uint8)
        result = recompose_hand(mask, img, bg, (32, 32), 0.0, 0.0)
        # Result should differ from plain bg where hand was blended
        assert not np.array_equal(result, bg)


# ============================================================================
# get_angle_clockwise_from_12
# ============================================================================

class TestGetAngle:
    def test_hand_pointing_up(self) -> None:
        """Vertical line from centre upward → ~0 degrees."""
        mask = np.zeros((256, 256), dtype=np.uint8)
        cv2.line(mask, (128, 128), (128, 20), 255, 3)
        angle = get_angle_clockwise_from_12(mask, (128, 128))
        assert angle == pytest.approx(0.0, abs=10)

    def test_hand_pointing_right(self) -> None:
        """Horizontal line from centre rightward → ~90 degrees."""
        mask = np.zeros((256, 256), dtype=np.uint8)
        cv2.line(mask, (128, 128), (240, 128), 255, 3)
        angle = get_angle_clockwise_from_12(mask, (128, 128))
        assert angle == pytest.approx(90.0, abs=10)

    def test_hand_pointing_down(self) -> None:
        """Vertical line downward → ~180 degrees."""
        mask = np.zeros((256, 256), dtype=np.uint8)
        cv2.line(mask, (128, 128), (128, 240), 255, 3)
        angle = get_angle_clockwise_from_12(mask, (128, 128))
        assert angle == pytest.approx(180.0, abs=10)

    def test_empty_mask(self) -> None:
        mask = np.zeros((64, 64), dtype=np.uint8)
        assert get_angle_clockwise_from_12(mask, (32, 32)) == 0.0


# ============================================================================
# find_clock_center
# ============================================================================

class TestFindClockCenter:
    def test_single_hand_fallback(self) -> None:
        """With only one hand mask, should still return a reasonable centre."""
        img = np.full((256, 256, 3), 128, dtype=np.uint8)
        mask_h = np.zeros((256, 256), dtype=np.uint8)
        cv2.line(mask_h, (128, 128), (128, 20), 255, 3)
        mask_m = np.zeros((256, 256), dtype=np.uint8)

        center = find_clock_center(img, mask_h, mask_m)
        # Should be near image centre (within 40% of width)
        assert abs(center[0] - 128) < 100
        assert abs(center[1] - 128) < 100

    def test_two_hands_intersection(self) -> None:
        """Two crossing hand masks should give a centre near their intersection."""
        img = np.full((256, 256, 3), 128, dtype=np.uint8)
        # Draw a circle so Hough might find it
        cv2.circle(img, (128, 128), 100, (200, 200, 200), 2)

        mask_h = np.zeros((256, 256), dtype=np.uint8)
        cv2.line(mask_h, (128, 128), (128, 30), 255, 3)  # up
        mask_m = np.zeros((256, 256), dtype=np.uint8)
        cv2.line(mask_m, (128, 128), (230, 128), 255, 3)  # right

        center = find_clock_center(img, mask_h, mask_m)
        assert abs(center[0] - 128) < 30
        assert abs(center[1] - 128) < 30
