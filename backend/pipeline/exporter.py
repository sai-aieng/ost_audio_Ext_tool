"""JSON, CSV, and plain-text result exporters."""

import csv
import json
from pathlib import Path
from typing import Literal

from api.schemas.response import ExtractionResult
from utils.file_utils import ensure_directory, job_directory

OutputFormat = Literal["json", "csv", "txt"]
_SUPPORTED_FORMATS: set[str] = {"json", "csv", "txt"}


def _serialize(result: ExtractionResult) -> dict[str, object]:
    """Convert a result model to a JSON-serializable dictionary."""

    return result.model_dump()


def export_results(
    job_id: str,
    results: list[ExtractionResult],
    formats: list[str],
    output_root: Path,
) -> dict[str, str]:
    """Write every requested output format in one exporter invocation."""

    requested_formats = list(dict.fromkeys(formats))
    if not requested_formats:
        raise ValueError("At least one output format is required")
    invalid = set(requested_formats) - _SUPPORTED_FORMATS
    if invalid:
        raise ValueError(f"Unsupported output formats: {sorted(invalid)}")

    output_dir = ensure_directory(job_directory(output_root, job_id))
    serialized = [_serialize(result) for result in results]
    paths: dict[str, str] = {}

    if "json" in requested_formats:
        json_path = output_dir / "results.json"
        json_path.write_text(
            json.dumps(serialized, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        paths["json"] = str(json_path)

    if "csv" in requested_formats:
        csv_path = output_dir / "results.csv"
        with csv_path.open("w", encoding="utf-8", newline="") as csv_file:
            writer = csv.DictWriter(
                csv_file,
                fieldnames=[
                    "frame_index",
                    "timestamp_sec",
                    "timestamp_start_sec",
                    "timestamp_end_sec",
                    "text",
                    "confidence",
                    "bbox",
                    "bbox_end",
                    "bbox_xyxy",
                    "bbox_end_xyxy",
                    "source_frame_width",
                    "source_frame_height",
                ],
            )
            writer.writeheader()
            for row in serialized:
                csv_row = dict(row)
                for field in ("bbox", "bbox_end", "bbox_xyxy", "bbox_end_xyxy"):
                    csv_row[field] = json.dumps(csv_row[field], ensure_ascii=False)
                writer.writerow(csv_row)
        paths["csv"] = str(csv_path)

    if "txt" in requested_formats:
        txt_path = output_dir / "results.txt"
        lines = []
        for result in results:
            start_sec = result.timestamp_start_sec or result.timestamp_sec
            end_sec = result.timestamp_end_sec or result.timestamp_sec
            time_range = f"{start_sec:.2f}s - {end_sec:.2f}s"
            coordinates = json.dumps(result.bbox, ensure_ascii=False)
            lines.append(
                f"[{time_range}] {result.text} "
                f"(confidence: {result.confidence:.3f}; coordinates: {coordinates})"
            )
        txt_path.write_text("\n".join(lines), encoding="utf-8")
        paths["txt"] = str(txt_path)

    return paths
