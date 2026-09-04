"""Tests for perceptual-hash frame deduplication."""

from pathlib import Path

from PIL import Image, ImageDraw

from pipeline.deduplicator import filter_frames
from pipeline.frame_extractor import FramePath


def _pattern_image(path: Path, inverted: bool = False) -> None:
    """Create a deterministic high-contrast image pattern."""

    background = "white" if inverted else "black"
    foreground = "black" if inverted else "white"
    image = Image.new("RGB", (96, 96), background)
    drawing = ImageDraw.Draw(image)
    drawing.rectangle((8, 8, 44, 44), fill=foreground)
    drawing.ellipse((50, 50, 88, 88), fill=foreground)
    image.save(path)


def test_filter_frames_removes_identical_images(tmp_path: Path) -> None:
    """An identical second frame should be skipped at distance zero."""

    first = tmp_path / "first.png"
    duplicate = tmp_path / "duplicate.png"
    different = tmp_path / "different.png"
    _pattern_image(first)
    duplicate.write_bytes(first.read_bytes())
    _pattern_image(different, inverted=True)
    frames = [
        FramePath(first, 0, 0.0),
        FramePath(duplicate, 1, 1.0),
        FramePath(different, 2, 2.0),
    ]

    unique = filter_frames(frames, threshold=0)

    assert [frame.frame_index for frame in unique] == [0, 2]


def test_filter_frames_can_be_disabled(tmp_path: Path) -> None:
    """Disabled deduplication should preserve every input frame."""

    frame = FramePath(tmp_path / "unused.png", 0, 0.0)
    assert filter_frames([frame, frame], threshold=0, enabled=False) == [frame, frame]
