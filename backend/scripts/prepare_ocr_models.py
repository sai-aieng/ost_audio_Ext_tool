"""Cache the configured CPU Paddle models during the container build."""

from pathlib import Path
import sys

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))
from utils.windows_runtime import prepare_native_runtime
prepare_native_runtime()

import yaml
from pipeline.ocr_engine import PaddleOCRService


def main():
    config = yaml.safe_load((BACKEND / "config.yaml").read_text(encoding="utf-8"))["ocr"]
    config.update(inference_engine="paddle", use_gpu=False, use_angle_cls=True)
    PaddleOCRService().get_engine(config)
    print("Paddle OCR models ready", flush=True)


if __name__ == "__main__":
    main()
