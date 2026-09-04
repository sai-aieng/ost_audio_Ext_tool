"""Synchronous end-to-end pipeline orchestration for BackgroundTasks."""

from copy import deepcopy
from datetime import datetime, timezone
from math import ceil
from pathlib import Path
from time import perf_counter
from typing import Any

from api.schemas.response import ExtractionResult
from pipeline.change_detector import has_material_change
from pipeline.deduplicator import filter_frames
from pipeline.exporter import export_results
from pipeline.frame_extractor import extract_frames
from pipeline.ocr_engine import run_ocr
from pipeline.post_processor import clean_results
from pipeline.preprocessor import process_frame
from pipeline.video_ingestion import validate_video
from storage.job_store import get_job, set_results, update_job
from utils.file_utils import (
    cleanup_frame_directories,
    resolve_configured_path,
)
from utils.logger import get_logger

LOGGER = get_logger("orchestrator")


def _apply_overrides(
    config: dict[str, Any],
    overrides: dict[str, Any],
) -> dict[str, Any]:
    """Return an isolated config with supported per-job overrides applied."""

    effective = deepcopy(config)
    if overrides.get("sample_rate_fps") is not None:
        effective["frame_extraction"]["sample_rate_fps"] = float(
            overrides["sample_rate_fps"]
        )
    if overrides.get("confidence_threshold") is not None:
        effective["ocr"]["confidence_threshold"] = float(
            overrides["confidence_threshold"]
        )
    if overrides.get("output_formats") is not None:
        effective["output"]["formats"] = list(overrides["output_formats"])
    return effective


def _carry_results(
    results: list[ExtractionResult],
    frame_index: int,
    timestamp_sec: float,
) -> list[ExtractionResult]:
    """Attach confirmed unchanged text to the current timestamp sample."""

    return [
        result.model_copy(
            update={
                "frame_index": frame_index,
                "timestamp_sec": timestamp_sec,
                "timestamp_start_sec": timestamp_sec,
                "timestamp_end_sec": timestamp_sec,
                "bbox_end": result.bbox,
                "bbox_end_xyxy": result.bbox_xyxy,
            }
        )
        for result in results
    ]


def run_pipeline(
    job_id: str,
    config_overrides: dict[str, Any],
    config: dict[str, Any],
    base_dir: Path,
) -> None:
    """Run ingestion through export synchronously and update job state."""

    started_at = perf_counter()
    try:
        record = get_job(job_id)
        if record is None:
            raise KeyError(f"Unknown job: {job_id}")
        effective = _apply_overrides(config, config_overrides)
        temp_root = resolve_configured_path(
            base_dir, str(effective["upload"]["temp_dir"])
        )
        output_root = resolve_configured_path(
            base_dir, str(effective["upload"]["output_dir"])
        )
        video_path = Path(record["video_path"])
        update_job(
            job_id,
            status="processing",
            error=None,
            completed_at=None,
            progress_pct=1.0,
        )

        step_started = perf_counter()
        metadata = validate_video(
            video_path,
            list(effective["upload"]["allowed_extensions"]),
        )
        update_job(job_id, metadata=metadata, progress_pct=5.0)
        LOGGER.info(
            "job=%s step=validate duration=%.3fs video_duration=%.3fs",
            job_id,
            perf_counter() - step_started,
            metadata["duration_sec"],
        )

        step_started = perf_counter()
        frames = extract_frames(
            video_path,
            job_id,
            float(effective["frame_extraction"]["sample_rate_fps"]),
            temp_root,
        )
        update_job(
            job_id,
            frames_extracted=len(frames),
            progress_pct=20.0,
        )
        LOGGER.info(
            "job=%s step=extract frames=%d duration=%.3fs",
            job_id,
            len(frames),
            perf_counter() - step_started,
        )

        step_started = perf_counter()
        unique_frames = filter_frames(
            frames,
            int(effective["deduplication"]["hash_threshold"]),
            bool(effective["deduplication"]["enabled"]),
        )
        update_job(
            job_id,
            frames_deduplicated=len(unique_frames),
            progress_pct=30.0,
        )
        LOGGER.info(
            "job=%s step=deduplicate input=%d unique=%d duration=%.3fs",
            job_id,
            len(frames),
            len(unique_frames),
            perf_counter() - step_started,
        )

        raw_results: list[ExtractionResult] = []
        step_started = perf_counter()
        total_unique = len(unique_frames)
        acceleration = effective["ocr_acceleration"]
        acceleration_enabled = bool(acceleration["enabled"])
        safety_check_frames = max(
            1,
            ceil(
                float(acceleration["safety_check_seconds"])
                * float(effective["frame_extraction"]["sample_rate_fps"])
            ),
        )
        previous_image = None
        active_results: list[ExtractionResult] = []
        last_ocr_frame_index: int | None = None
        ocr_frames_processed = 0
        for processed_count, frame in enumerate(unique_frames, start=1):
            processed_frame = process_frame(frame.path, effective["preprocessing"])
            refresh_for_change = previous_image is None or has_material_change(
                previous_image,
                processed_frame.image,
                int(acceleration["thumbnail_width"]),
                float(acceleration["change_threshold"]),
            )
            refresh_for_safety = (
                last_ocr_frame_index is None
                or frame.frame_index - last_ocr_frame_index >= safety_check_frames
            )
            should_run_ocr = (
                not acceleration_enabled or refresh_for_change or refresh_for_safety
            )
            if should_run_ocr:
                frame_results = run_ocr(
                    processed_frame.image,
                    frame.frame_index,
                    frame.timestamp_sec,
                    effective["ocr"],
                    source_size=(
                        processed_frame.source_width,
                        processed_frame.source_height,
                    ),
                )
                active_results = frame_results
                last_ocr_frame_index = frame.frame_index
                ocr_frames_processed += 1
            else:
                frame_results = _carry_results(
                    active_results,
                    frame.frame_index,
                    frame.timestamp_sec,
                )
            raw_results.extend(frame_results)
            previous_image = processed_frame.image
            progress = 30.0 + (55.0 * processed_count / max(total_unique, 1))
            update_job(
                job_id,
                frames_processed=ocr_frames_processed,
                texts_found=len(raw_results),
                progress_pct=round(progress, 2),
            )
        LOGGER.info(
            "job=%s step=ocr samples=%d ocr_frames=%d detections=%d duration=%.3fs",
            job_id,
            total_unique,
            ocr_frames_processed,
            len(raw_results),
            perf_counter() - step_started,
        )

        step_started = perf_counter()
        cleaned_results = clean_results(raw_results)
        update_job(
            job_id,
            texts_found=len(cleaned_results),
            progress_pct=90.0,
        )
        LOGGER.info(
            "job=%s step=postprocess input=%d output=%d duration=%.3fs",
            job_id,
            len(raw_results),
            len(cleaned_results),
            perf_counter() - step_started,
        )

        step_started = perf_counter()
        output_paths = export_results(
            job_id,
            cleaned_results,
            list(effective["output"]["formats"]),
            output_root,
        )
        set_results(job_id, cleaned_results, output_paths)
        LOGGER.info(
            "job=%s step=export formats=%s duration=%.3fs",
            job_id,
            sorted(output_paths),
            perf_counter() - step_started,
        )

        job_temp_dir = temp_root / job_id
        try:
            cleanup_frame_directories(job_temp_dir)
        except (OSError, ValueError):
            LOGGER.warning("job=%s frame cleanup failed", job_id, exc_info=True)

        update_job(
            job_id,
            status="completed",
            progress_pct=100.0,
            completed_at=datetime.now(timezone.utc),
        )
        LOGGER.info(
            "job=%s step=complete total_duration=%.3fs",
            job_id,
            perf_counter() - started_at,
        )
    except Exception as exc:
        LOGGER.exception("job=%s pipeline failed", job_id)
        if get_job(job_id) is not None:
            update_job(
                job_id,
                status="failed",
                error=str(exc),
                completed_at=datetime.now(timezone.utc),
            )
