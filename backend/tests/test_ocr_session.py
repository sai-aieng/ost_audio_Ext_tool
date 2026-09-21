"""Per-job ONNX fallback must be visible and preserve quality settings."""

import numpy as np
import pytest

from api.schemas.request import ProcessOptions
from pipeline.ocr_engine import OCRSession
from pipeline.orchestrator import _apply_overrides


class FakeService:
    def __init__(self):
        self.calls = []

    def recognize(self, image, config):
        self.calls.append(config["inference_engine"])
        if config["inference_engine"] == "onnxruntime":
            raise RuntimeError("Simulated unavailable ONNX engine")
        return ["paddle-result"]


def test_fallback_happens_once_per_job_without_mutating_input():
    config = {"inference_engine": "onnxruntime", "use_angle_cls": True}
    service = FakeService()
    session = OCRSession(config, service)
    image = np.zeros((8, 8), dtype=np.uint8)
    assert session.recognize(image, config) == ["paddle-result"]
    assert session.recognize(image, config) == ["paddle-result"]
    assert service.calls == ["onnxruntime", "paddle", "paddle"]
    assert session.engine_name == "paddle"
    assert session.fallback_reason
    assert config["inference_engine"] == "onnxruntime"
    assert session.config["use_angle_cls"] is True


def test_fallback_can_be_disabled():
    config = {"inference_engine": "onnxruntime", "onnx_fallback_to_paddle": False}
    session = OCRSession(config, FakeService())
    with pytest.raises(RuntimeError, match="Simulated"):
        session.recognize(np.zeros((8, 8), dtype=np.uint8), config)
    assert session.engine_name == "onnxruntime"


def test_engine_override_preserves_quality_settings():
    config = {
        "processing": {"default_mode": "accuracy", "profiles": {"accuracy": {}}},
        "ocr": {"inference_engine": "paddle", "use_angle_cls": True},
        "preprocessing": {"resize_width": 960},
        "frame_extraction": {"sample_rate_fps": 10},
    }
    options = ProcessOptions(inference_engine="onnxruntime")
    effective = _apply_overrides(config, options.model_dump(exclude_none=True))
    assert effective["ocr"]["inference_engine"] == "onnxruntime"
    assert effective["ocr"]["use_angle_cls"] is True
    assert effective["preprocessing"] == config["preprocessing"]
    assert effective["frame_extraction"] == config["frame_extraction"]
    assert config["ocr"]["inference_engine"] == "paddle"


def test_invalid_engine_is_rejected():
    with pytest.raises(ValueError):
        ProcessOptions(inference_engine="unknown")
