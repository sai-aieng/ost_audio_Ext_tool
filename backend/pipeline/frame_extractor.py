"""Configurable frame extraction implemented with ffmpeg-python."""

from dataclasses import dataclass
from pathlib import Path

import ffmpeg

from utils.file_utils import ensure_directory, job_directory, remove_directory_within


@dataclass(frozen=True, slots=True)
class FramePath:
    """An extracted frame path with its source index and timestamp."""

    path: Path
    frame_index: int
    timestamp_sec: float


def extract_frames(
    video_path: Path,
    job_id: str,
    sample_rate_fps: float,
    temp_root: Path,
) -> list[FramePath]:
    """Extract JPEG frames at the requested sample rate."""

    if sample_rate_fps <= 0:
        raise ValueError("Frame sample rate must be positive")
    job_temp_dir = job_directory(temp_root, job_id)
    frames_dir = job_temp_dir / "frames"
    if frames_dir.exists():
        remove_directory_within(frames_dir, job_temp_dir)
    ensure_directory(frames_dir)
    output_pattern = frames_dir / "frame_%08d.jpg"

    try:
        (
            ffmpeg.input(str(video_path))
            .filter("fps", fps=sample_rate_fps)
            .output(str(output_pattern), start_number=0, qscale=2)
            .overwrite_output()
            .run(capture_stdout=True, capture_stderr=True)
        )
    except ffmpeg.Error as exc:
        stderr = exc.stderr.decode("utf-8", errors="replace") if exc.stderr else str(exc)
        raise RuntimeError(f"ffmpeg frame extraction failed: {stderr.strip()}") from exc

    paths = sorted(frames_dir.glob("frame_*.jpg"))
    if not paths:
        raise RuntimeError("ffmpeg completed without producing any frames")
    return [
        FramePath(
            path=path,
            frame_index=index,
            timestamp_sec=round(index / sample_rate_fps, 3),
        )
        for index, path in enumerate(paths)
    ]
