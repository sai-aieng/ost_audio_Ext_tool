"""Audio terminal timing uses its own clock and does not affect OCR jobs."""

import pytest

from pipeline import audio_pipeline


@pytest.mark.parametrize("fail", [False, True])
def test_audio_records_independent_terminal_duration(monkeypatch, tmp_path, fail):
    updates = []
    ticks = iter([100.0, 112.3456])
    monkeypatch.setattr(audio_pipeline, "perf_counter", lambda: next(ticks))
    monkeypatch.setattr(audio_pipeline, "get_job", lambda job_id: {"video_path": "audio.mp4"})
    monkeypatch.setattr(audio_pipeline, "update_job",
                        lambda job_id, **changes: updates.append((job_id, changes)))

    def transcribe(*args):
        if fail:
            raise RuntimeError("Audio error")
        return "en", []

    monkeypatch.setattr(audio_pipeline.WHISPER_SERVICE, "transcribe", transcribe)
    monkeypatch.setattr(audio_pipeline, "export_transcript", lambda *args: {})
    monkeypatch.setattr(audio_pipeline, "set_transcript", lambda *args: None)
    audio_pipeline.run_audio_pipeline(
        "audio-job", {"audio_transcription": {}, "upload": {"output_dir": str(tmp_path)}}, tmp_path,
    )
    assert all(job_id == "audio-job" for job_id, _ in updates)
    assert updates[0][1]["started_at"] is not None
    assert updates[0][1]["processing_duration_sec"] is None
    final = updates[-1][1]
    assert final["status"] == ("failed" if fail else "completed")
    assert final["completed_at"] is not None
    assert final["processing_duration_sec"] == 12.346
