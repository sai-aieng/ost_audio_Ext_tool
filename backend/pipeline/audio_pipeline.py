"""Independent Whisper audio-transcription pipeline."""

from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from time import perf_counter
from typing import Any

from api.schemas.response import TranscriptSegment
from pipeline.audio_exporter import export_transcript
from pipeline.faster_whisper_service import FASTER_WHISPER_SERVICE
from storage.job_store import get_job, set_transcript, update_job
from utils.file_utils import resolve_configured_path
from utils.logger import get_logger

LOGGER = get_logger("audio")

class WhisperService:
    """Lazily load and reuse the configured local Whisper model."""

    def __init__(self) -> None:
        self._model: Any | None = None
        self._settings: tuple[str, bool] | None = None
        self._lock = Lock()

    def transcribe(self, video_path: Path, config: dict[str, Any]) -> tuple[str | None, list[TranscriptSegment]]:
        """Transcribe a video audio stream and return Whisper's segment times."""

        settings = (str(config["model"]), bool(config["use_gpu"]))
        with self._lock:
            if self._model is None:
                import whisper

                self._model = whisper.load_model(settings[0], device="cuda" if settings[1] else "cpu")
                self._settings = settings
            elif self._settings != settings:
                raise RuntimeError("Whisper model settings cannot change after initialization")
            model = self._model
        options: dict[str, Any] = {"task": "transcribe", "verbose": False, "fp16": settings[1]}
        if config.get("language"):
            options["language"] = str(config["language"])
        payload = model.transcribe(str(video_path), **options)
        segments = [
            TranscriptSegment(start_sec=float(item["start"]), end_sec=float(item["end"]), text=str(item["text"]).strip())
            for item in payload.get("segments", [])
            if str(item.get("text", "")).strip()
        ]
        return payload.get("language"), segments


WHISPER_SERVICE = WhisperService()


def run_audio_pipeline(job_id: str, config: dict[str, Any], base_dir: Path) -> None:
    """Run audio transcription without affecting the independent OCR job."""

    started_at = perf_counter()
    started_at_utc = datetime.now(timezone.utc)
    try:
        record = get_job(job_id)
        if record is None:
            raise KeyError(f"Unknown audio job: {job_id}")
        audio_config = config["audio_transcription"]
        engine = str(audio_config.get("engine", "whisper"))
        services = {"whisper": WHISPER_SERVICE, "faster_whisper": FASTER_WHISPER_SERVICE}
        if engine not in services:
            raise ValueError(f"Unsupported audio engine: {engine}")
        update_job(
            job_id, status="processing", progress_pct=5.0, error=None,
            started_at=started_at_utc, completed_at=None, processing_duration_sec=None,
            requested_inference_engine=engine, inference_engine=engine,
        )
        LOGGER.info("job=%s step=audio_transcribe_start engine=%s model=%s", job_id, engine, audio_config.get("model"))
        language, segments = services[engine].transcribe(Path(record["video_path"]), audio_config)
        output_root = resolve_configured_path(base_dir, str(config["upload"]["output_dir"]))
        paths = export_transcript(job_id, segments, output_root)
        set_transcript(job_id, segments, paths, language)
        elapsed_seconds = round(perf_counter() - started_at, 3)
        update_job(
            job_id, status="completed", progress_pct=100.0,
            completed_at=datetime.now(timezone.utc), processing_duration_sec=elapsed_seconds,
        )
        LOGGER.info("job=%s step=audio_complete segments=%d duration=%.3fs", job_id, len(segments), elapsed_seconds)
    except Exception as exc:
        LOGGER.exception("job=%s audio pipeline failed", job_id)
        if get_job(job_id) is not None:
            update_job(
                job_id, status="failed", error=str(exc),
                started_at=started_at_utc, completed_at=datetime.now(timezone.utc),
                processing_duration_sec=round(perf_counter() - started_at, 3),
            )
