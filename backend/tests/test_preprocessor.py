"""Tests for ordered OpenCV frame preprocessing."""

from pathlib import Path

import cv2
import numpy as np

from pipeline.preprocessor import load_frame, process_frame, process_image


def test_process_frame_resizes_and_returns_grayscale(tmp_path: Path) -> None:
    """Configured preprocessing should preserve aspect ratio and grayscale."""

    source = np.zeros((50, 100, 3), dtype=np.uint8)
    source[:, :, 1] = np.tile(np.arange(100, dtype=np.uint8), (50, 1))
    frame_path = tmp_path / "frame.jpg"
    assert cv2.imwrite(str(frame_path), source)
    config = {
        "resize_width": 200,
        "grayscale": True,
        "clahe_clip_limit": 2.0,
        "clahe_tile_grid": [8, 8],
        "denoise_strength": 0,
        "sharpen": True,
    }

    processed = process_frame(frame_path, config)

    assert processed.image.shape == (100, 200)
    assert processed.image.dtype == np.uint8
    assert (processed.source_width, processed.source_height) == (100, 50)


def test_process_frame_rejects_unreadable_input(tmp_path: Path) -> None:
    """Unreadable images should produce a descriptive validation error."""

    frame_path = tmp_path / "broken.jpg"
    frame_path.write_bytes(b"not an image")
    config = {
        "resize_width": 200,
        "grayscale": True,
        "clahe_clip_limit": 2.0,
        "clahe_tile_grid": [8, 8],
        "denoise_strength": 0,
        "sharpen": True,
    }

    try:
        process_frame(frame_path, config)
    except ValueError as exc:
        assert "could not read" in str(exc)
    else:
        raise AssertionError("Unreadable input did not raise ValueError")


def test_loaded_image_can_be_preprocessed_without_reading_twice(
    tmp_path: Path,
) -> None:
    """The change gate and OCR preprocessing should share one decoded image."""

    source = np.zeros((50, 100, 3), dtype=np.uint8)
    frame_path = tmp_path / "frame.jpg"
    assert cv2.imwrite(str(frame_path), source)
    config = {
        "resize_width": 50,
        "grayscale": True,
        "clahe_clip_limit": 2.0,
        "clahe_tile_grid": [8, 8],
        "denoise_strength": 0,
        "sharpen": False,
    }

    loaded = load_frame(frame_path)
    processed = process_image(loaded, config)

    assert loaded.shape == (50, 100, 3)
    assert processed.image.shape == (25, 50)
    assert (processed.source_width, processed.source_height) == (100, 50)
