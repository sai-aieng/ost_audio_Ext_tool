"""Compare inference engines on identical JPEG samples without touching API jobs.

Run from backend with: python scripts/benchmark_ocr.py --help
Each output directory must be new. Reports include complete per-frame detections.
"""

import argparse
import json
import sys
from pathlib import Path
from time import perf_counter
from uuid import uuid4

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline.frame_extractor import extract_frames
from pipeline.ocr_engine import PaddleOCRService, _to_paddle_image, run_ocr
from pipeline.orchestrator import _apply_overrides
from pipeline.preprocessor import process_frame


class SelectedEngineService(PaddleOCRService):
    def __init__(self, engine):
        super().__init__()
        self.engine = engine

    def recognize(self, image, config):
        return list(self.engine.predict(
            _to_paddle_image(image),
            use_textline_orientation=bool(config["use_angle_cls"]),
            text_rec_score_thresh=float(config["confidence_threshold"]),
        ))


def compare_reports(reference, candidate):
    comparisons = []
    if [f["frame_index"] for f in reference["frames"]] != [
        f["frame_index"] for f in candidate["frames"]
    ]:
        raise ValueError("Reports must contain identical sampled frame indexes")
    for old, new in zip(reference["frames"], candidate["frames"]):
        left, right = old["results"], new["results"]
        same_text = [r["text"] for r in left] == [r["text"] for r in right]
        delta = None
        score_delta = None
        if same_text:
            delta = max((
                abs(a - b)
                for x, y in zip(left, right)
                for p, q in zip(x["bbox"], y["bbox"])
                for a, b in zip(p, q)
            ), default=0)
            score_delta = max((
                abs(x["confidence"] - y["confidence"]) for x, y in zip(left, right)
            ), default=0)
        comparisons.append({
            "frame_index": old["frame_index"], "same_text": same_text,
            "max_coordinate_delta_px": delta, "max_confidence_delta": score_delta,
        })
    return comparisons


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--video", type=Path)
    source.add_argument("--frames-dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--engine", choices=["paddle", "onnxruntime"], default="paddle")
    parser.add_argument("--model-root", type=Path)
    parser.add_argument("--sample-count", type=int, default=8)
    parser.add_argument("--reference-report", type=Path)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    if args.sample_count < 1 or args.threads < 1:
        parser.error("Sample count and threads must be positive")
    args.output.mkdir(parents=True, exist_ok=False)
    backend = Path(__file__).resolve().parents[1]
    config = _apply_overrides(
        yaml.safe_load((backend / "config.yaml").read_text(encoding="utf-8")),
        {"processing_mode": "fast_cpu"},
    )
    config["ocr"]["cpu_threads"] = args.threads
    fps = float(config["frame_extraction"]["sample_rate_fps"])
    started = perf_counter()
    if args.video:
        frames = extract_frames(args.video.resolve(), str(uuid4()), fps, args.output.resolve())
        paths = [frame.path for frame in frames]
    else:
        paths = sorted(args.frames_dir.glob("frame_*.jpg"))
    if not paths:
        raise ValueError("No sampled JPEG frames found")
    indices = np.linspace(0, len(paths) - 1, min(args.sample_count, len(paths)), dtype=int)
    report = {"engine": args.engine, "threads": args.threads,
              "frames_dir": str(paths[0].parent.resolve()),
              "preprocessing": config["preprocessing"], "frames": [],
              "extraction_seconds": perf_counter() - started}
    started = perf_counter()
    if args.engine == "paddle":
        service = PaddleOCRService()
        service.get_engine(config["ocr"])
    else:
        if args.model_root is None:
            parser.error("--model-root is required for ONNX")
        from paddleocr import PaddleOCR
        settings = config["ocr"]
        engine = PaddleOCR(
            device="cpu", engine="onnxruntime",
            engine_config={"intra_op_num_threads": args.threads,
                           "inter_op_num_threads": 1, "execution_mode": "sequential"},
            use_doc_orientation_classify=False, use_doc_unwarping=False,
            use_textline_orientation=False,
            text_detection_model_name=settings["text_detection_model_name"],
            text_recognition_model_name=settings["text_recognition_model_name"],
            text_detection_model_dir=str(args.model_root / settings["text_detection_model_name"]),
            text_recognition_model_dir=str(args.model_root / settings["text_recognition_model_name"]),
        )
        service = SelectedEngineService(engine)
    report["initialization_seconds"] = perf_counter() - started
    for position in indices:
        path = paths[int(position)]
        index = int(path.stem.removeprefix("frame_"))
        processed = process_frame(path, config["preprocessing"])
        started = perf_counter()
        results = run_ocr(
            processed.image, index, round(index / fps, 3), config["ocr"],
            service=service, source_size=(processed.source_width, processed.source_height),
        )
        elapsed = perf_counter() - started
        report["frames"].append({
            "frame_index": index, "inference_seconds": elapsed,
            "results": [r.model_dump(mode="json") for r in results],
        })
        print(f"engine={args.engine} frame={index} seconds={elapsed:.3f} texts={len(results)}",
              flush=True)
        (args.output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    report["inference_seconds"] = sum(f["inference_seconds"] for f in report["frames"])
    if args.reference_report:
        reference = json.loads(args.reference_report.read_text(encoding="utf-8"))
        report["comparison"] = compare_reports(reference, report)
        print(json.dumps(report["comparison"]), flush=True)
    (args.output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"TOTAL inference={report['inference_seconds']:.3f}s", flush=True)


if __name__ == "__main__":
    main()
