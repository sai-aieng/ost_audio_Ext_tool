"""Ordered OpenCV preprocessing for OCR input frames."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2

from utils.image_utils import (
    ImageArray,
    apply_clahe,
    denoise,
    resize_to_width,
    sharpen,
    to_grayscale,
)


@dataclass(frozen=True, slots=True)
class ProcessedFrame:
    """OCR-ready pixels together with the source frame dimensions."""

    image: ImageArray
    source_width: int
    source_height: int


def process_frame(frame_path: Path, config: dict[str, Any]) -> ProcessedFrame:
    """Apply preprocessing and retain dimensions for source-coordinate mapping."""

    image = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"OpenCV could not read frame: {frame_path.name}")
    source_height, source_width = image.shape[:2]

    processed = resize_to_width(image, int(config["resize_width"]))
    if bool(config["grayscale"]):
        processed = to_grayscale(processed)
    tile_values = config["clahe_tile_grid"]
    processed = apply_clahe(
        processed,
        float(config["clahe_clip_limit"]),
        (int(tile_values[0]), int(tile_values[1])),
    )
    processed = denoise(processed, int(config["denoise_strength"]))
    if bool(config["sharpen"]):
        processed = sharpen(processed)
    return ProcessedFrame(
        image=processed,
        source_width=source_width,
        source_height=source_height,
    )
