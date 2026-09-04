"""OCR text cleaning and short-window appearance-range grouping."""

from dataclasses import dataclass
from difflib import SequenceMatcher

from api.schemas.response import ExtractionResult
from utils.text_utils import clean_text


@dataclass
class _ActiveGroup:
    """The latest observation used to extend one appearance range."""

    result_index: int
    comparison_text: str
    last_seen_sec: float
    bbox_xyxy: tuple[int, int, int, int] | None


def _comparison_text(value: str) -> str:
    """Return case-insensitive text without insignificant trailing punctuation."""

    return clean_text(value).rstrip(".,!?;:…").strip().casefold()


def _rectangle(
    result: ExtractionResult,
    *,
    use_end: bool,
) -> tuple[int, int, int, int] | None:
    """Return a validated rectangle, deriving one from a polygon when needed."""

    rectangle = result.bbox_end_xyxy if use_end else result.bbox_xyxy
    polygon = result.bbox_end if use_end else result.bbox
    if rectangle is not None and len(rectangle) == 4:
        x_min, y_min, x_max, y_max = rectangle
        return x_min, y_min, x_max, y_max
    if not polygon:
        return None
    x_values = [point[0] for point in polygon if len(point) >= 2]
    y_values = [point[1] for point in polygon if len(point) >= 2]
    if not x_values or not y_values:
        return None
    return min(x_values), min(y_values), max(x_values), max(y_values)


def _text_similarity(left: str, right: str) -> float:
    """Measure OCR text similarity on a zero-to-one scale."""

    if left == right:
        return 1.0
    return SequenceMatcher(None, left, right, autojunk=False).ratio()


def _bbox_similarity(
    left: tuple[int, int, int, int] | None,
    right: tuple[int, int, int, int] | None,
) -> float:
    """Return rectangle intersection-over-union, treating two absent boxes alike."""

    if left is None or right is None:
        return 1.0 if left is None and right is None else 0.0

    left_x_min, left_y_min, left_x_max, left_y_max = left
    right_x_min, right_y_min, right_x_max, right_y_max = right
    intersection_width = max(
        0,
        min(left_x_max, right_x_max) - max(left_x_min, right_x_min),
    )
    intersection_height = max(
        0,
        min(left_y_max, right_y_max) - max(left_y_min, right_y_min),
    )
    intersection_area = intersection_width * intersection_height
    left_area = max(0, left_x_max - left_x_min) * max(
        0, left_y_max - left_y_min
    )
    right_area = max(0, right_x_max - right_x_min) * max(
        0, right_y_max - right_y_min
    )
    union_area = left_area + right_area - intersection_area
    if union_area == 0:
        return 1.0 if left == right else 0.0
    return intersection_area / union_area


def clean_results(
    results: list[ExtractionResult],
    maximum_gap_sec: float = 0.15,
    minimum_text_similarity: float = 0.85,
    minimum_bbox_iou: float = 0.50,
) -> list[ExtractionResult]:
    """Group similar nearby text observed no more than 0.15 seconds apart."""

    if maximum_gap_sec < 0:
        raise ValueError("Maximum time gap cannot be negative")
    if not 0 <= minimum_text_similarity <= 1:
        raise ValueError("Minimum text similarity must be between zero and one")
    if not 0 <= minimum_bbox_iou <= 1:
        raise ValueError("Minimum bounding-box IoU must be between zero and one")
    ordered = sorted(results, key=lambda item: (item.frame_index, item.timestamp_sec))
    cleaned_results: list[ExtractionResult] = []
    active_groups: list[_ActiveGroup] = []

    for result in ordered:
        cleaned_text = clean_text(result.text)
        if not cleaned_text:
            continue
        comparison_text = _comparison_text(cleaned_text)
        if not comparison_text:
            continue

        start_sec = (
            result.timestamp_start_sec
            if result.timestamp_start_sec is not None
            else result.timestamp_sec
        )
        end_sec = (
            result.timestamp_end_sec
            if result.timestamp_end_sec is not None
            else result.timestamp_sec
        )
        start_bbox = _rectangle(result, use_end=False)
        end_bbox = _rectangle(result, use_end=True) or start_bbox
        active_groups = [
            group
            for group in active_groups
            if start_sec - group.last_seen_sec <= maximum_gap_sec + 1e-9
        ]

        best_match: _ActiveGroup | None = None
        best_score = (-1.0, -1.0)
        for group in active_groups:
            text_score = _text_similarity(group.comparison_text, comparison_text)
            bbox_score = _bbox_similarity(group.bbox_xyxy, start_bbox)
            if (
                text_score >= minimum_text_similarity
                and bbox_score >= minimum_bbox_iou
                and (bbox_score, text_score) > best_score
            ):
                best_match = group
                best_score = (bbox_score, text_score)

        if best_match is not None:
            previous_index = best_match.result_index
            previous = cleaned_results[previous_index]
            cleaned_results[previous_index] = previous.model_copy(
                update={
                    "timestamp_end_sec": max(
                        previous.timestamp_end_sec
                        if previous.timestamp_end_sec is not None
                        else previous.timestamp_sec,
                        end_sec,
                    ),
                    "confidence": max(previous.confidence, result.confidence),
                    "bbox_end": result.bbox_end or result.bbox,
                    "bbox_end_xyxy": result.bbox_end_xyxy or result.bbox_xyxy,
                }
            )
            best_match.comparison_text = comparison_text
            best_match.last_seen_sec = end_sec
            best_match.bbox_xyxy = end_bbox
            continue

        cleaned_results.append(
            result.model_copy(
                update={
                    "text": cleaned_text,
                    "timestamp_start_sec": start_sec,
                    "timestamp_end_sec": end_sec,
                    "bbox_end": result.bbox_end or result.bbox,
                    "bbox_end_xyxy": result.bbox_end_xyxy or result.bbox_xyxy,
                }
            )
        )
        active_groups.append(
            _ActiveGroup(
                result_index=len(cleaned_results) - 1,
                comparison_text=comparison_text,
                last_seen_sec=end_sec,
                bbox_xyxy=end_bbox,
            )
        )
    return cleaned_results
