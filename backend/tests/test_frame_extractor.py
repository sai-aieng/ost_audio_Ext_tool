"""Tests for ffmpeg-python frame extraction orchestration."""

from pathlib import Path

import pytest

from pipeline import frame_extractor


class FakeFFmpegStream:
    """Small fluent ffmpeg-python test double."""

    def __init__(self) -> None:
        """Initialize without an output pattern."""

        self.output_pattern: str | None = None

    def filter(self, name: str, **kwargs: float) -> "FakeFFmpegStream":
        """Accept the configured FPS filter."""

        assert name == "fps"
        assert kwargs["fps"] == 2.0
        return self

    def output(self, pattern: str, **kwargs: int) -> "FakeFFmpegStream":
        """Capture the output pattern and numbering options."""

        assert kwargs["start_number"] == 0
        self.output_pattern = pattern
        return self

    def overwrite_output(self) -> "FakeFFmpegStream":
        """Support the fluent overwrite call."""

        return self

    def run(self, **kwargs: bool) -> tuple[bytes, bytes]:
        """Materialize three files as if ffmpeg extracted them."""

        assert kwargs == {"capture_stdout": True, "capture_stderr": True}
        assert self.output_pattern is not None
        for index in range(3):
            Path(self.output_pattern.replace("%08d", f"{index:08d}")).write_bytes(
                b"frame"
            )
        return b"", b""


def test_extract_frames_builds_timestamped_frame_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The extractor should sort files and calculate sample timestamps."""

    video_path = tmp_path / "video.mp4"
    video_path.write_bytes(b"video")
    fake_stream = FakeFFmpegStream()
    monkeypatch.setattr(frame_extractor.ffmpeg, "input", lambda _: fake_stream)

    frames = frame_extractor.extract_frames(
        video_path,
        "11111111-1111-1111-1111-111111111111",
        2.0,
        tmp_path,
    )

    assert [frame.frame_index for frame in frames] == [0, 1, 2]
    assert [frame.timestamp_sec for frame in frames] == [0.0, 0.5, 1.0]
    assert all(frame.path.is_file() for frame in frames)


def test_extract_frames_rejects_non_positive_sample_rate(tmp_path: Path) -> None:
    """Invalid sample rates should fail before invoking ffmpeg."""

    with pytest.raises(ValueError, match="positive"):
        frame_extractor.extract_frames(
            tmp_path / "video.mp4",
            "11111111-1111-1111-1111-111111111111",
            0,
            tmp_path,
        )
