"""Tests for PaddleOCR result normalization and post-processing."""

from typing import Any

import numpy as np

from api.schemas.response import ExtractionResult
from pipeline.ocr_engine import PaddleOCRService, run_ocr
from pipeline.post_processor import clean_results


class FakeOCRService:
    """Deterministic OCR service returning high and low confidence hits."""

    def recognize(
        self,
        image: np.ndarray[Any, Any],
        config: dict[str, Any],
    ) -> list[Any]:
        """Return data in PaddleOCR's single-image result shape."""

        assert image.shape == (20, 20)
        return [
            [
                [
                    [[0.2, 1.7], [10.0, 1.0], [10.0, 8.0], [0.0, 8.0]],
                    ("  Hello world  ", 0.91),
                ],
                [
                    [[1, 1], [2, 1], [2, 2], [1, 2]],
                    ("discard me", 0.4),
                ],
            ]
        ]


class FakePaddleV3Result:
    """Minimal PaddleOCR 3.x result object exposing its JSON payload."""

    json = {
        "res": {
            "rec_texts": ["Paddle 3", "below threshold"],
            "rec_scores": np.array([0.97, 0.2]),
            "rec_polys": np.array(
                [
                    [[2, 3], [12, 3], [12, 9], [2, 9]],
                    [[1, 1], [4, 1], [4, 4], [1, 4]],
                ]
            ),
        }
    }


class FakePaddleV3Service:
    """Return the result-object shape used by PaddleOCR 3.x."""

    def recognize(
        self,
        image: np.ndarray[Any, Any],
        config: dict[str, Any],
    ) -> list[Any]:
        """Return one deterministic PaddleOCR 3.x result object."""

        assert image.shape == (20, 20)
        return [FakePaddleV3Result()]


def test_run_ocr_filters_confidence_and_normalizes_bbox() -> None:
    """Only qualifying OCR hits should become typed extraction results."""

    image = np.zeros((20, 20), dtype=np.uint8)
    results = run_ocr(
        image,
        frame_index=4,
        timestamp_sec=2.5,
        config={"confidence_threshold": 0.6},
        service=FakeOCRService(),
    )

    assert len(results) == 1
    assert results[0].text == "Hello world"
    assert results[0].bbox == [[0, 2], [10, 1], [10, 8], [0, 8]]
    assert results[0].timestamp_start_sec == 2.5
    assert results[0].timestamp_end_sec == 2.5
    assert results[0].bbox_xyxy == [0, 1, 10, 8]


def test_run_ocr_normalizes_paddleocr_v3_results() -> None:
    """PaddleOCR 3.x Result objects should retain text, score, and polygon."""

    image = np.zeros((20, 20), dtype=np.uint8)
    results = run_ocr(
        image,
        frame_index=5,
        timestamp_sec=3.0,
        config={"confidence_threshold": 0.6},
        service=FakePaddleV3Service(),
    )

    assert len(results) == 1
    assert results[0].text == "Paddle 3"
    assert results[0].confidence == 0.97
    assert results[0].bbox == [[2, 3], [12, 3], [12, 9], [2, 9]]
    assert results[0].bbox_end_xyxy == [2, 3, 12, 9]


def test_run_ocr_maps_coordinates_to_the_original_video_frame() -> None:
    """Coordinates should be returned in source-frame pixels after OCR resizing."""

    image = np.zeros((20, 20), dtype=np.uint8)
    results = run_ocr(
        image,
        frame_index=6,
        timestamp_sec=3.5,
        config={"confidence_threshold": 0.6},
        service=FakeOCRService(),
        source_size=(40, 60),
    )

    assert results[0].bbox == [[0, 5], [20, 3], [20, 24], [0, 24]]
    assert results[0].bbox_xyxy == [0, 3, 20, 24]
    assert results[0].source_frame_width == 40
    assert results[0].source_frame_height == 60


def test_clean_results_tracks_the_sampled_appearance_range() -> None:
    """Repeated nearby text becomes one result with its first and last timestamp."""

    results = [
        ExtractionResult(
            frame_index=0,
            timestamp_sec=0,
            text="  Alpha  ",
            confidence=0.9,
            bbox=None,
        ),
        ExtractionResult(
            frame_index=1,
            timestamp_sec=0.1,
            text="Alpha",
            confidence=0.95,
            bbox=None,
        ),
        ExtractionResult(
            frame_index=2,
            timestamp_sec=0.2,
            text="Alpha",
            confidence=0.96,
            bbox=None,
        ),
    ]

    cleaned = clean_results(results)

    assert [item.frame_index for item in cleaned] == [0]
    assert cleaned[0].timestamp_start_sec == 0
    assert cleaned[0].timestamp_end_sec == 0.2


def test_clean_results_keeps_identical_text_separate_after_a_large_gap() -> None:
    """Identical text must not be merged when more than 0.15 seconds apart."""

    results = [
        ExtractionResult(frame_index=0, timestamp_sec=0.0, text="Alpha", confidence=0.9),
        ExtractionResult(frame_index=1, timestamp_sec=0.151, text="Alpha", confidence=0.9),
    ]

    cleaned = clean_results(results)

    assert len(cleaned) == 2


def test_clean_results_merges_trailing_punctuation_variants() -> None:
    """A trailing period must not split one continuous sentence range."""

    results = [
        ExtractionResult(frame_index=0, timestamp_sec=4.0, text="Same sentence", confidence=0.9),
        ExtractionResult(frame_index=1, timestamp_sec=4.1, text="Same sentence", confidence=0.9),
        ExtractionResult(frame_index=2, timestamp_sec=4.2, text="Same sentence.", confidence=0.9),
        ExtractionResult(frame_index=3, timestamp_sec=4.3, text="Same sentence.", confidence=0.9),
    ]

    cleaned = clean_results(results)

    assert len(cleaned) == 1
    assert cleaned[0].timestamp_start_sec == 4.0
    assert cleaned[0].timestamp_end_sec == 4.3


def test_clean_results_merges_similar_text_and_coordinates_at_point_15_seconds() -> None:
    """Minor OCR and box variation at the inclusive time boundary is one item."""

    results = [
        ExtractionResult(
            frame_index=10,
            timestamp_sec=1.0,
            text="Loading dsta.",
            confidence=0.88,
            bbox_xyxy=[10, 10, 110, 30],
        ),
        ExtractionResult(
            frame_index=11,
            timestamp_sec=1.15,
            text="Loading data",
            confidence=0.94,
            bbox_xyxy=[12, 9, 112, 31],
        ),
    ]

    cleaned = clean_results(results)

    assert len(cleaned) == 1
    assert cleaned[0].text == "Loading data"
    assert cleaned[0].confidence == 0.94
    assert cleaned[0].timestamp_start_sec == 1.0
    assert cleaned[0].timestamp_end_sec == 1.15
    assert cleaned[0].bbox_end_xyxy == [12, 9, 112, 31]


def test_clean_results_keeps_similar_text_at_different_coordinates_separate() -> None:
    """Repeated wording in different screen areas represents distinct text."""

    results = [
        ExtractionResult(
            frame_index=0,
            timestamp_sec=2.0,
            text="Chapter one",
            confidence=0.9,
            bbox_xyxy=[10, 10, 110, 30],
        ),
        ExtractionResult(
            frame_index=1,
            timestamp_sec=2.1,
            text="Chapter 0ne",
            confidence=0.9,
            bbox_xyxy=[400, 300, 500, 320],
        ),
    ]

    assert len(clean_results(results)) == 2


def test_clean_results_keeps_different_text_at_same_coordinates_separate() -> None:
    """A changed message in the same screen area starts a new result."""

    results = [
        ExtractionResult(
            frame_index=0,
            timestamp_sec=3.0,
            text="Loading data",
            confidence=0.9,
            bbox_xyxy=[10, 10, 110, 30],
        ),
        ExtractionResult(
            frame_index=1,
            timestamp_sec=3.1,
            text="Process complete",
            confidence=0.9,
            bbox_xyxy=[10, 10, 110, 30],
        ),
    ]

    assert len(clean_results(results)) == 2


class FakePaddleEngine:
    """Capture the image passed to PaddleOCR's predict method."""

    def predict(self, image: np.ndarray[Any, Any], **kwargs: Any) -> list[Any]:
        """Assert that grayscale preprocessing becomes a BGR image."""

        assert image.shape == (20, 20, 3)
        assert kwargs["use_textline_orientation"] is False
        assert kwargs["text_rec_score_thresh"] == 0.6
        return []


def test_service_converts_grayscale_frame_to_bgr_for_paddleocr_v3() -> None:
    """PaddleOCR 3.x must not receive the preprocessor's two-dimensional frame."""

    service = PaddleOCRService()
    settings = (
        "en",
        False,
        False,
        "PP-OCRv6_small_det",
        "PP-OCRv6_small_rec",
        2,
    )
    service._engines[settings] = FakePaddleEngine()
    service.recognize(
        np.zeros((20, 20), dtype=np.uint8),
        {
            "lang": "en",
            "use_gpu": False,
            "use_angle_cls": False,
            "confidence_threshold": 0.6,
            "text_detection_model_name": "PP-OCRv6_small_det",
            "text_recognition_model_name": "PP-OCRv6_small_rec",
            "cpu_threads": 2,
        },
    )
