"""Export timestamped Whisper transcript artifacts."""

import csv
import json
from pathlib import Path

from api.schemas.response import TranscriptSegment
from utils.file_utils import ensure_directory, job_directory


def export_transcript(
    job_id: str,
    segments: list[TranscriptSegment],
    output_root: Path,
) -> dict[str, str]:
    """Write JSON, CSV, and TXT transcript artifacts for an audio job."""

    output_dir = ensure_directory(job_directory(output_root, job_id))
    serialized = [segment.model_dump() for segment in segments]
    json_path = output_dir / "transcript.json"
    json_path.write_text(json.dumps(serialized, ensure_ascii=False, indent=2), encoding="utf-8")
    csv_path = output_dir / "transcript.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=["start_sec", "end_sec", "text"])
        writer.writeheader()
        writer.writerows(serialized)
    txt_path = output_dir / "transcript.txt"
    txt_path.write_text(
        "\n".join(f"[{item.start_sec:.2f}s - {item.end_sec:.2f}s] {item.text}" for item in segments),
        encoding="utf-8",
    )
    return {"json": str(json_path), "csv": str(csv_path), "txt": str(txt_path)}
