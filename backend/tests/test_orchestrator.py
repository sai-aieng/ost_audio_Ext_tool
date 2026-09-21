"""Tests for timestamp preservation when OCR is skipped on stable frames."""

from api.schemas.response import ExtractionResult
from pipeline.orchestrator import _apply_overrides, _carry_results


def test_carried_results_keep_the_current_sample_timestamp() -> None:
    """Stable-frame propagation must still create a 10 FPS timing observation."""

    original = ExtractionResult(
        frame_index=10,
        timestamp_sec=1.0,
        text="Stable label",
        confidence=0.9,
        bbox=[[2, 3], [12, 3], [12, 9], [2, 9]],
        bbox_xyxy=[2, 3, 12, 9],
    )

    carried = _carry_results([original], frame_index=11, timestamp_sec=1.1)

    assert len(carried) == 1
    assert carried[0].frame_index == 11
    assert carried[0].timestamp_sec == 1.1
    assert carried[0].timestamp_start_sec == 1.1
    assert carried[0].timestamp_end_sec == 1.1
    assert carried[0].bbox_end_xyxy == [2, 3, 12, 9]


def test_fast_cpu_profile_is_applied_without_mutating_base_config() -> None:
    """A per-job fast profile should override only its declared settings."""

    config = {
        "processing": {
            "default_mode": "accuracy",
            "profiles": {
                "accuracy": {},
                "fast_cpu": {
                    "ocr_acceleration": {"safety_check_seconds": 10.0},
                    "preprocessing": {
                        "resize_width": 640,
                        "denoise_strength": 0,
                    },
                    "ocr": {"use_angle_cls": False},
                },
            },
        },
        "frame_extraction": {"sample_rate_fps": 10},
        "ocr_acceleration": {"safety_check_seconds": 1.0},
        "preprocessing": {"resize_width": 960, "denoise_strength": 10},
        "ocr": {"use_angle_cls": True, "confidence_threshold": 0.6},
        "output": {"formats": ["json"]},
    }

    effective = _apply_overrides(
        config,
        {
            "processing_mode": "fast_cpu",
            "confidence_threshold": 0.75,
        },
    )

    assert effective["processing"]["selected_mode"] == "fast_cpu"
    assert effective["ocr_acceleration"]["safety_check_seconds"] == 10.0
    assert effective["preprocessing"]["resize_width"] == 640
    assert effective["preprocessing"]["denoise_strength"] == 0
    assert effective["ocr"]["use_angle_cls"] is False
    assert effective["ocr"]["confidence_threshold"] == 0.75
    assert config["preprocessing"]["resize_width"] == 960
