"""Single-instance PaddleOCR adapter and result normalization."""

import json
from collections.abc import Mapping
from threading import Lock
from typing import Any

import cv2

from api.schemas.response import ExtractionResult
from utils.image_utils import ImageArray

try:
    from paddleocr import PaddleOCR
except ImportError:
    PaddleOCR = None  # type: ignore[assignment]


class PaddleOCRService:
    """Lazily load one PaddleOCR model and reuse it for every frame."""

    def __init__(self) -> None:
        """Create an unloaded, thread-safe service."""

        self._engine: Any | None = None
        self._lock = Lock()
        self._settings: tuple[str, bool, bool, str, str] | None = None

    def get_engine(self, config: dict[str, Any]) -> Any:
        """Return the process-wide engine, initializing it exactly once."""

        settings = (
            str(config["lang"]),
            bool(config["use_gpu"]),
            bool(config["use_angle_cls"]),
            str(config["text_detection_model_name"]),
            str(config["text_recognition_model_name"]),
        )
        if self._engine is not None:
            if self._settings != settings:
                raise RuntimeError("OCR model settings cannot change after initialization")
            return self._engine
        with self._lock:
            if self._engine is None:
                if PaddleOCR is None:
                    raise RuntimeError(
                        "PaddleOCR is not installed; install backend/requirements.txt"
                    )
                self._engine = PaddleOCR(
                    lang=settings[0],
                    device="gpu:0" if settings[1] else "cpu",
                    use_doc_orientation_classify=False,
                    use_doc_unwarping=False,
                    use_textline_orientation=settings[2],
                    enable_mkldnn=False,
                    text_detection_model_name=settings[3],
                    text_recognition_model_name=settings[4],
                )
                self._settings = settings
        return self._engine

    def recognize(self, image: ImageArray, config: dict[str, Any]) -> list[Any]:
        """Run the shared engine against one preprocessed frame."""

        engine = self.get_engine(config)
        return list(
            engine.predict(
                _to_paddle_image(image),
                use_textline_orientation=bool(config["use_angle_cls"]),
                text_rec_score_thresh=float(config["confidence_threshold"]),
            )
        )


# The service is constructed at module scope so model loading occurs at most once
# per process and never once per video frame.
OCR_ENGINE = PaddleOCRService()


def _to_paddle_image(image: ImageArray) -> ImageArray:
    """Ensure PaddleOCR 3.x receives a three-channel BGR image."""

    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.ndim != 3:
        raise ValueError("OCR input must be a grayscale or color image")
    if image.shape[2] == 1:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.shape[2] == 3:
        return image
    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    raise ValueError("OCR input must have 1, 3, or 4 channels")


def _looks_like_detection(value: Any) -> bool:
    """Return whether a PaddleOCR value resembles one detection tuple."""

    return (
        isinstance(value, (list, tuple))
        and len(value) == 2
        and isinstance(value[1], (list, tuple))
        and len(value[1]) >= 2
    )


def _paddle_v3_detections(raw_result: list[Any]) -> list[Any] | None:
    """Convert PaddleOCR 3.x Result objects into the legacy detection shape."""

    detections: list[Any] = []
    found_v3_result = False
    for item in raw_result:
        payload: Any = item if isinstance(item, Mapping) else getattr(item, "json", None)
        if callable(payload):
            payload = payload()
        if isinstance(payload, str):
            try:
                payload = json.loads(payload)
            except json.JSONDecodeError:
                continue
        if not isinstance(payload, Mapping):
            continue
        result = payload.get("res", payload)
        if not isinstance(result, Mapping):
            continue
        texts = result.get("rec_texts")
        scores = result.get("rec_scores")
        polygons = result.get("rec_polys")
        if texts is None or scores is None or polygons is None:
            continue
        found_v3_result = True
        detections.extend(
            [polygon, (text, score)]
            for polygon, text, score in zip(polygons, texts, scores)
        )
    return detections if found_v3_result else None


def _rectangle_from_polygon(polygon: list[list[int]]) -> list[int]:
    """Return x-min, y-min, x-max, y-max for an OCR polygon."""

    x_values = [point[0] for point in polygon]
    y_values = [point[1] for point in polygon]
    return [min(x_values), min(y_values), max(x_values), max(y_values)]


def _map_to_source_coordinates(
    polygon: list[list[int]],
    image: ImageArray,
    source_size: tuple[int, int] | None,
) -> list[list[int]]:
    """Scale OCR-image coordinates back to original video-frame pixels."""

    if source_size is None:
        return polygon
    source_width, source_height = source_size
    ocr_height, ocr_width = image.shape[:2]
    if source_width <= 0 or source_height <= 0 or ocr_width <= 0 or ocr_height <= 0:
        raise ValueError("Source and OCR image dimensions must be positive")
    return [
        [
            int(round(point[0] * source_width / ocr_width)),
            int(round(point[1] * source_height / ocr_height)),
        ]
        for point in polygon
    ]


def run_ocr(
    image: ImageArray,
    frame_index: int,
    timestamp_sec: float,
    config: dict[str, Any],
    service: PaddleOCRService = OCR_ENGINE,
    source_size: tuple[int, int] | None = None,
) -> list[ExtractionResult]:
    """Recognize text and normalize qualifying PaddleOCR detections."""

    raw_result = service.recognize(image, config)
    if not raw_result or raw_result == [None]:
        return []

    detections = _paddle_v3_detections(raw_result)
    if detections is None:
        first_item = raw_result[0]
        detections = (
            first_item
            if isinstance(first_item, list)
            and (not first_item or _looks_like_detection(first_item[0]))
            else raw_result
        )
    threshold = float(config["confidence_threshold"])
    results: list[ExtractionResult] = []
    for detection in detections:
        if not _looks_like_detection(detection):
            continue
        bbox_value, recognition = detection
        text = str(recognition[0]).strip()
        confidence = float(recognition[1])
        if not text or confidence < threshold:
            continue
        ocr_bbox = [
            [int(round(float(point[0]))), int(round(float(point[1])))]
            for point in bbox_value
        ]
        bbox = _map_to_source_coordinates(ocr_bbox, image, source_size)
        bbox_xyxy = _rectangle_from_polygon(bbox)
        results.append(
            ExtractionResult(
                frame_index=frame_index,
                timestamp_sec=timestamp_sec,
                timestamp_start_sec=timestamp_sec,
                timestamp_end_sec=timestamp_sec,
                text=text,
                confidence=confidence,
                bbox=bbox,
                bbox_end=bbox,
                bbox_xyxy=bbox_xyxy,
                bbox_end_xyxy=bbox_xyxy,
                source_frame_width=source_size[0] if source_size is not None else None,
                source_frame_height=source_size[1] if source_size is not None else None,
            )
        )
    return results
