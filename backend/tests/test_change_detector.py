"""Tests for the lightweight OCR refresh decision."""

import numpy as np

from api.schemas.response import ExtractionResult
from pipeline.change_detector import FrameChangeGuard, has_material_change


def test_change_detector_ignores_identical_frames() -> None:
    """Identical frames should not cause another OCR inference."""

    frame = np.zeros((80, 160), dtype=np.uint8)

    assert not has_material_change(frame, frame.copy(), 80, 2.0)


def test_change_detector_refreshes_for_visible_change() -> None:
    """A newly visible high-contrast text-sized area should refresh OCR."""

    previous = np.zeros((80, 160), dtype=np.uint8)
    current = previous.copy()
    current[20:50, 30:130] = 255

    assert has_material_change(previous, current, 80, 2.0)


def test_tiled_change_detector_finds_a_small_local_change() -> None:
    """Tiled mode should notice text changes hidden by the whole-frame mean."""

    previous = np.zeros((80, 160), dtype=np.uint8)
    current = previous.copy()
    current[20:24, 30:46] = 255

    assert not has_material_change(previous, current, 160, 2.0)
    assert has_material_change(
        previous,
        current,
        160,
        6.0,
        mode="tiled",
        tile_grid=(4, 4),
    )


def test_guard_catches_cumulative_change_from_confirmed_frame() -> None:
    guard = FrameChangeGuard({"thumbnail_width": 80, "change_threshold": 6.0})
    original = np.zeros((80, 160), dtype=np.uint8)
    assert guard.change_reason(original) == "initial"
    guard.confirm(original, [])
    assert guard.change_reason(original.copy()) is None
    assert guard.change_reason(np.full_like(original, 4)) is None
    assert guard.change_reason(np.full_like(original, 8)) == "scene_change"
    assert guard.change_reason(np.zeros((160, 320), dtype=np.uint8)) == "dimensions"


def test_guard_catches_small_change_in_native_text_region() -> None:
    original = np.zeros((1080, 1920, 3), dtype=np.uint8)
    changed = original.copy()
    changed[104:110, 104:110] = 255
    assert not has_material_change(original, changed, 160, 6.0)
    guard = FrameChangeGuard({"thumbnail_width": 160, "change_threshold": 6.0})
    guard.confirm(original, [ExtractionResult(
        frame_index=0, timestamp_sec=0.0, text="Long label", confidence=0.9,
        bbox_xyxy=[100, 100, 900, 132],
    )])
    assert guard.change_reason(changed) == "text_change"
