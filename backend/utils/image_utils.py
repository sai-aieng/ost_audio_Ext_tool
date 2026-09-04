"""Reusable OpenCV transformations for OCR-oriented preprocessing."""

from typing import Any

import cv2
import numpy as np
from numpy.typing import NDArray

ImageArray = NDArray[np.uint8]


def resize_to_width(image: ImageArray, width: int) -> ImageArray:
    """Resize an image to a target width while retaining its aspect ratio."""

    if width <= 0:
        raise ValueError("Resize width must be positive")
    current_height, current_width = image.shape[:2]
    if current_width == width:
        return image.copy()
    scale = width / current_width
    target_height = max(1, int(round(current_height * scale)))
    interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC
    return cv2.resize(image, (width, target_height), interpolation=interpolation)


def to_grayscale(image: ImageArray) -> ImageArray:
    """Convert BGR or BGRA input to grayscale, preserving grayscale input."""

    if image.ndim == 2:
        return image.copy()
    conversion = cv2.COLOR_BGRA2GRAY if image.shape[2] == 4 else cv2.COLOR_BGR2GRAY
    return cv2.cvtColor(image, conversion)


def apply_clahe(
    image: ImageArray,
    clip_limit: float,
    tile_grid: tuple[int, int],
) -> ImageArray:
    """Apply contrast-limited adaptive histogram equalization."""

    gray = to_grayscale(image)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid)
    return clahe.apply(gray)


def denoise(image: ImageArray, strength: int) -> ImageArray:
    """Denoise a grayscale or color image using OpenCV's non-local means."""

    if strength <= 0:
        return image.copy()
    if image.ndim == 2:
        return cv2.fastNlMeansDenoising(image, None, strength, 7, 21)
    return cv2.fastNlMeansDenoisingColored(image, None, strength, strength, 7, 21)


def sharpen(image: ImageArray) -> ImageArray:
    """Sharpen an image with a compact high-pass convolution kernel."""

    kernel: NDArray[Any] = np.array(
        [[0, -1, 0], [-1, 5, -1], [0, -1, 0]],
        dtype=np.float32,
    )
    return cv2.filter2D(image, -1, kernel)
