"""Tests for the lightweight OCR refresh decision."""

import numpy as np

from pipeline.change_detector import has_material_change


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
