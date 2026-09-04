"""Fast, conservative visual-change checks used to avoid repeated OCR."""

import cv2

from utils.image_utils import ImageArray


def has_material_change(
    previous: ImageArray,
    current: ImageArray,
    thumbnail_width: int,
    threshold: float,
) -> bool:
    """Return whether two OCR-ready frames differ enough to refresh OCR.

    This operates on small grayscale thumbnails, so it is much cheaper than an
    OCR inference. A low threshold is deliberately conservative: uncertainty
    leads to another OCR pass rather than to skipping possible text changes.
    """

    if thumbnail_width <= 0:
        raise ValueError("Change-detector thumbnail width must be positive")
    if threshold < 0:
        raise ValueError("Change-detector threshold cannot be negative")
    previous_thumb = _thumbnail(previous, thumbnail_width)
    current_thumb = _thumbnail(current, thumbnail_width)
    return float(cv2.absdiff(previous_thumb, current_thumb).mean()) >= threshold


def _thumbnail(image: ImageArray, thumbnail_width: int) -> ImageArray:
    """Convert an OCR image to a compact grayscale thumbnail."""

    if image.ndim == 2:
        grayscale = image
    elif image.ndim == 3 and image.shape[2] == 1:
        grayscale = image[:, :, 0]
    elif image.ndim == 3 and image.shape[2] == 3:
        grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    elif image.ndim == 3 and image.shape[2] == 4:
        grayscale = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    else:
        raise ValueError("Change-detector input must be a valid image")
    height, width = grayscale.shape[:2]
    if height <= 0 or width <= 0:
        raise ValueError("Change-detector input cannot be empty")
    target_height = max(1, round(height * thumbnail_width / width))
    return cv2.resize(
        grayscale,
        (thumbnail_width, target_height),
        interpolation=cv2.INTER_AREA,
    )
