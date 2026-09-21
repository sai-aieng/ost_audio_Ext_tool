"""Faster-Whisper adapter preserving the existing timestamped transcript schema."""

from pathlib import Path
from threading import Lock
from time import perf_counter
from typing import Any

from api.schemas.response import TranscriptSegment
from utils.logger import get_logger

LOGGER = get_logger("audio")
BACKEND = Path(__file__).resolve().parents[1]


class FasterWhisperService:
    def __init__(self):
        self._models: dict[tuple, Any] = {}
        self._lock = Lock()

    def get_model(self, config):
        model_name = str(config["model"])
        root = Path(config.get("faster_model_root", "models/faster-whisper"))
        if not root.is_absolute():
            root = BACKEND / root
        model_path = root / model_name
        device = "cuda" if config.get("use_gpu", False) else "cpu"
        compute_type = str(config.get("compute_type", "float32"))
        threads = int(config.get("cpu_threads", 2))
        if threads < 1:
            raise ValueError("Audio cpu_threads must be positive")
        key = (str(model_path.resolve()), device, compute_type, threads)
        with self._lock:
            model = self._models.get(key)
            if model is None:
                if not (model_path / "model.bin").is_file():
                    raise FileNotFoundError(
                        f"Faster-Whisper model missing: {model_path}. "
                        "Run scripts/prepare_audio_model.py first."
                    )
                # Lazy import: original Whisper/PyTorch is not needed on this path.
                from faster_whisper import WhisperModel

                started = perf_counter()
                LOGGER.info("step=audio_model_load_start engine=faster_whisper model=%s", model_name)
                model = WhisperModel(
                    str(model_path), device=device, compute_type=compute_type,
                    cpu_threads=threads, num_workers=1, local_files_only=True,
                )
                self._models[key] = model
                LOGGER.info(
                    "step=audio_model_loaded engine=faster_whisper model=%s compute_type=%s duration=%.3fs",
                    model_name, compute_type, perf_counter() - started,
                )
        return model

    def transcribe(self, video_path: Path, config: dict) -> tuple[str | None, list[TranscriptSegment]]:
        model = self.get_model(config)
        started = perf_counter()
        # The existing direct Whisper API used greedy decoding and one sampling
        # candidate, not the beam_size=5 / best_of=5 CLI defaults.
        generated, info = model.transcribe(
            str(video_path), language=config.get("language") or None,
            task="transcribe", beam_size=1, best_of=1,
            temperature=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
            compression_ratio_threshold=2.4, log_prob_threshold=-1.0,
            no_speech_threshold=0.6, condition_on_previous_text=True,
            without_timestamps=False, word_timestamps=False, vad_filter=False,
        )
        # Transcription is lazy; consume it before stopping the timer or exporting.
        segments = [
            TranscriptSegment(start_sec=float(item.start), end_sec=float(item.end),
                              text=str(item.text).strip())
            for item in generated if str(item.text).strip()
        ]
        LOGGER.info(
            "step=audio_transcribed engine=faster_whisper segments=%d duration=%.3fs",
            len(segments), perf_counter() - started,
        )
        return info.language, segments


FASTER_WHISPER_SERVICE = FasterWhisperService()
