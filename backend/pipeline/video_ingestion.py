"""Video validation and ffprobe metadata extraction."""

from pathlib import Path
from typing import Any, TypedDict

import ffmpeg


class VideoMetadata(TypedDict):
    """Normalized metadata for an uploaded video."""

    duration_sec: float
    fps: float
    width: int
    height: int
    resolution: str


def _parse_frame_rate(rate: str) -> float:
    """Parse an ffprobe frame-rate fraction into frames per second."""

    numerator, separator, denominator = rate.partition("/")
    if not separator:
        return float(numerator)
    denominator_value = float(denominator)
    return float(numerator) / denominator_value if denominator_value else 0.0


def validate_video(
    video_path: Path,
    allowed_extensions: list[str] | None = None,
) -> VideoMetadata:
    """Validate a video file and return duration, FPS, and resolution."""

    if not video_path.is_file():
        raise ValueError(f"Video file does not exist: {video_path.name}")
    if video_path.stat().st_size == 0:
        raise ValueError("Uploaded video is empty")
    normalized_extensions = {
        extension.lower() for extension in (allowed_extensions or [video_path.suffix])
    }
    if video_path.suffix.lower() not in normalized_extensions:
        raise ValueError(f"Unsupported video extension: {video_path.suffix}")

    try:
        probe: dict[str, Any] = ffmpeg.probe(str(video_path))
    except ffmpeg.Error as exc:
        stderr = exc.stderr.decode("utf-8", errors="replace") if exc.stderr else str(exc)
        raise ValueError(f"ffprobe could not read the video: {stderr.strip()}") from exc

    video_stream = next(
        (stream for stream in probe.get("streams", []) if stream.get("codec_type") == "video"),
        None,
    )
    if video_stream is None:
        raise ValueError("The uploaded file contains no video stream")

    width = int(video_stream.get("width", 0))
    height = int(video_stream.get("height", 0))
    duration_value = video_stream.get("duration") or probe.get("format", {}).get("duration")
    frame_rate = video_stream.get("avg_frame_rate") or video_stream.get("r_frame_rate") or "0"
    duration = float(duration_value or 0)
    fps = _parse_frame_rate(str(frame_rate))
    if width <= 0 or height <= 0 or duration <= 0 or fps <= 0:
        raise ValueError("Video metadata is incomplete or invalid")

    return VideoMetadata(
        duration_sec=round(duration, 3),
        fps=round(fps, 3),
        width=width,
        height=height,
        resolution=f"{width}x{height}",
    )
