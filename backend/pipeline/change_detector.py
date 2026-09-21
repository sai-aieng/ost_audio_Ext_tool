"""Fast, conservative visual-change checks used to avoid repeated OCR."""

import cv2

from api.schemas.response import ExtractionResult
from utils.image_utils import ImageArray


def has_material_change(
    previous: ImageArray,
    current: ImageArray,
    thumbnail_width: int,
    threshold: float,
    mode: str = "mean",
    tile_grid: tuple[int, int] = (4, 4),
) -> bool:
    """Return whether two frames differ enough to refresh OCR.

    Mean mode preserves the accuracy profile's whole-frame behavior. Tiled mode
    notices localized text changes that would be diluted by the rest of a frame.
    """

    if thumbnail_width <= 0:
        raise ValueError("Change-detector thumbnail width must be positive")
    if threshold < 0:
        raise ValueError("Change-detector threshold cannot be negative")
    previous_thumb = _thumbnail(previous, thumbnail_width)
    current_thumb = _thumbnail(current, thumbnail_width)
    return _thumbnails_changed(previous_thumb, current_thumb, threshold, mode, tile_grid)


def _thumbnails_changed(
    previous_thumb: ImageArray,
    current_thumb: ImageArray,
    threshold: float,
    mode: str,
    tile_grid: tuple[int, int],
) -> bool:
    """Compare precomputed thumbnails without repeating reference conversion."""

    if previous_thumb.shape != current_thumb.shape:
        return True
    difference = cv2.absdiff(previous_thumb, current_thumb)
    if mode == "mean":
        return float(difference.mean()) >= threshold
    if mode == "tiled":
        return _maximum_tile_mean(difference, tile_grid) >= threshold
    raise ValueError(f"Unsupported change-detection mode: {mode}")


class FrameChangeGuard:
    """Compare each sample with the frame that supplied its current OCR results."""

    def __init__(self, config: dict) -> None:
        self.config = config
        self.reference: ImageArray | None = None
        self.reference_thumb: ImageArray | None = None
        self.rectangles: list[tuple[int, int, int, int]] = []

    def confirm(self, image: ImageArray, results: list[ExtractionResult]) -> None:
        """Anchor only after OCR or an exact decoded-image cache hit."""

        self.reference = image
        self.reference_thumb = _thumbnail(image, int(self.config["thumbnail_width"]))
        self.rectangles = []
        height, width = image.shape[:2]
        for result in results:
            rectangle = result.bbox_xyxy
            if rectangle is None and result.bbox:
                rectangle = [
                    min(p[0] for p in result.bbox), min(p[1] for p in result.bbox),
                    max(p[0] for p in result.bbox), max(p[1] for p in result.bbox),
                ]
            if rectangle is None or len(rectangle) != 4:
                continue
            # Padding includes strokes moving just outside the prior detection.
            x1, y1 = max(0, rectangle[0] - 4), max(0, rectangle[1] - 4)
            x2, y2 = min(width, rectangle[2] + 5), min(height, rectangle[3] + 5)
            if x2 > x1 and y2 > y1:
                self.rectangles.append((x1, y1, x2, y2))

    def change_reason(self, image: ImageArray) -> str | None:
        """Return a refresh reason, or None when reuse passes visual checks."""

        if self.reference is None or self.reference_thumb is None:
            return "initial"
        if image.shape != self.reference.shape:
            return "dimensions"
        current_thumb = _thumbnail(image, int(self.config["thumbnail_width"]))
        grid = self.config.get("tile_grid", [4, 4])
        if _thumbnails_changed(
            self.reference_thumb, current_thumb,
            float(self.config["change_threshold"]),
            str(self.config.get("change_detection_mode", "mean")),
            (int(grid[0]), int(grid[1])),
        ):
            return "scene_change"
        if self.config.get("text_region_checks", True):
            pixel_threshold = int(self.config.get("region_pixel_threshold", 20))
            fraction = float(self.config.get("region_changed_fraction", 0.02))
            for x1, y1, x2, y2 in self.rectangles:
                difference = cv2.absdiff(
                    self.reference[y1:y2, x1:x2], image[y1:y2, x1:x2]
                )
                if difference.ndim == 3:
                    difference = difference.max(axis=2)
                # Local 32px tiles keep a single changed character from being
                # diluted by the remainder of a long sentence/paragraph.
                for y in range(0, difference.shape[0], 32):
                    for x in range(0, difference.shape[1], 32):
                        tile = difference[y:y + 32, x:x + 32]
                        changed = int((tile >= pixel_threshold).sum())
                        if changed >= max(3, tile.size * fraction):
                            return "text_change"
        return None


def _maximum_tile_mean(
    difference: ImageArray,
    tile_grid: tuple[int, int],
) -> float:
    """Return the largest mean difference among an even grid of image tiles."""

    columns, rows = tile_grid
    if columns <= 0 or rows <= 0:
        raise ValueError("Change-detector tile dimensions must be positive")
    height, width = difference.shape[:2]
    maximum = 0.0
    for row in range(rows):
        y_start = row * height // rows
        y_end = (row + 1) * height // rows
        for column in range(columns):
            x_start = column * width // columns
            x_end = (column + 1) * width // columns
            tile = difference[y_start:y_end, x_start:x_end]
            if tile.size:
                maximum = max(maximum, float(tile.mean()))
    return maximum


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
