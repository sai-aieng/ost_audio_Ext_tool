"""Synchronous end-to-end pipeline orchestration for BackgroundTasks."""

from copy import deepcopy
from collections import Counter
from datetime import datetime, timezone
from math import ceil
from pathlib import Path
from time import perf_counter
from typing import Any

from api.schemas.response import ExtractionResult
from pipeline.change_detector import FrameChangeGuard
from pipeline.frame_cache import ExactFrameCache
from pipeline.deduplicator import filter_frames
from pipeline.exporter import export_results
from pipeline.frame_extractor import extract_frames
from pipeline.ocr_engine import OCRSession, run_ocr
from pipeline.post_processor import clean_results
from pipeline.preprocessor import load_frame, process_image
from pipeline.video_ingestion import validate_video
from storage.job_store import get_job, set_results, update_job
from utils.file_utils import (
    cleanup_frame_directories,
    resolve_configured_path,
)
from utils.logger import get_logger

LOGGER = get_logger("orchestrator")


def _merge_config(target: dict[str, Any], overlay: dict[str, Any]) -> None:
    """Recursively apply a processing-profile overlay to a configuration."""

    for key, value in overlay.items():
        existing = target.get(key)
        if isinstance(existing, dict) and isinstance(value, dict):
            _merge_config(existing, value)
        else:
            target[key] = deepcopy(value)


def _apply_overrides(
    config: dict[str, Any],
    overrides: dict[str, Any],
) -> dict[str, Any]:
    """Return an isolated config with supported per-job overrides applied."""

    effective = deepcopy(config)
    processing = effective["processing"]
    mode = str(
        overrides.get("processing_mode") or processing.get("default_mode", "accuracy")
    )
    profiles = processing["profiles"]
    if mode not in profiles:
        raise ValueError(f"Unsupported processing mode: {mode}")
    profile = profiles[mode]
    if not isinstance(profile, dict):
        raise ValueError(f"Invalid processing profile: {mode}")
    _merge_config(effective, profile)
    processing["selected_mode"] = mode
    if overrides.get("inference_engine") is not None:
        effective["ocr"]["inference_engine"] = str(overrides["inference_engine"])
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
    """Retime reused detections; visually skipped samples remain estimates."""

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
    started_at_utc = datetime.now(timezone.utc)
    try:
        record = get_job(job_id)
        if record is None:
            raise KeyError(f"Unknown job: {job_id}")
        effective = _apply_overrides(config, config_overrides)
        ocr_session = OCRSession(effective["ocr"])
        LOGGER.info(
            "job=%s step=configure mode=%s",
            job_id,
            effective["processing"]["selected_mode"],
        )
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
            started_at=started_at_utc,
            processing_duration_sec=None,
            progress_pct=1.0,
            requested_inference_engine=ocr_session.engine_name,
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
        change_guard = FrameChangeGuard(acceleration)
        frame_cache = ExactFrameCache(
            int(acceleration.get("exact_cache_size", 32)) if acceleration_enabled else 0
        )
        active_results: list[ExtractionResult] = []
        last_ocr_frame_index: int | None = None
        ocr_frames_processed = 0
        cache_hits = 0
        carried_frames = 0
        inference_seconds = 0.0
        preprocessing_seconds = 0.0
        refresh_reasons: Counter[str] = Counter()
        for processed_count, frame in enumerate(unique_frames, start=1):
            source_image = load_frame(frame.path)
            change_reason = change_guard.change_reason(source_image) if acceleration_enabled else "disabled"
            refresh_for_change = change_reason is not None
            refresh_for_safety = (
                last_ocr_frame_index is None
                or frame.frame_index - last_ocr_frame_index >= safety_check_frames
            )
            should_run_ocr = (
                not acceleration_enabled or refresh_for_change or refresh_for_safety
            )
            if should_run_ocr:
                refresh_reason = change_reason or "safety"
                refresh_reasons[refresh_reason] += 1
                LOGGER.debug(
                    "job=%s step=ocr_refresh frame=%d timestamp=%.3f reason=%s",
                    job_id, frame.frame_index, frame.timestamp_sec, refresh_reason,
                )
                cache_key = frame_cache.key(source_image) if frame_cache.capacity else None
                cached = frame_cache.get(cache_key) if cache_key is not None else None
                if cached is not None:
                    frame_results = _carry_results(cached, frame.frame_index, frame.timestamp_sec)
                    cache_hits += 1
                else:
                    stage_started = perf_counter()
                    processed_frame = process_image(source_image, effective["preprocessing"])
                    preprocessing_seconds += perf_counter() - stage_started
                    stage_started = perf_counter()
                    frame_results = run_ocr(
                        processed_frame.image, frame.frame_index, frame.timestamp_sec,
                        effective["ocr"],
                        service=ocr_session,
                        source_size=(processed_frame.source_width, processed_frame.source_height),
                    )
                    inference_seconds += perf_counter() - stage_started
                    ocr_frames_processed += 1
                    if cache_key is not None:
                        frame_cache.put(cache_key, frame_results)
                if acceleration_enabled:
                    change_guard.confirm(source_image, frame_results)
                active_results = frame_results
                last_ocr_frame_index = frame.frame_index
            else:
                carried_frames += 1
                frame_results = _carry_results(
                    active_results,
                    frame.frame_index,
                    frame.timestamp_sec,
                )
            raw_results.extend(frame_results)
            progress = 30.0 + (55.0 * processed_count / max(total_unique, 1))
            update_job(
                job_id,
                frames_processed=ocr_frames_processed,
                texts_found=len(raw_results),
                progress_pct=round(progress, 2),
                inference_engine=ocr_session.engine_name,
                inference_fallback_reason=ocr_session.fallback_reason,
            )
        LOGGER.info(
            "job=%s step=ocr_engine requested=%s effective=%s refresh_reasons=%s",
            job_id, effective["ocr"].get("inference_engine", "paddle"),
            ocr_session.engine_name, dict(refresh_reasons),
        )
        LOGGER.info(
            "job=%s step=ocr_breakdown cache_hits=%d carried_frames=%d preprocessing=%.3fs inference=%.3fs",
            job_id, cache_hits, carried_frames, preprocessing_seconds, inference_seconds,
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

        elapsed_seconds = round(perf_counter() - started_at, 3)
        update_job(
            job_id,
            status="completed",
            progress_pct=100.0,
            completed_at=datetime.now(timezone.utc),
            processing_duration_sec=elapsed_seconds,
        )
        LOGGER.info(
            "job=%s step=complete total_duration=%.3fs",
            job_id,
            elapsed_seconds,
        )
    except Exception as exc:
        LOGGER.exception("job=%s pipeline failed", job_id)
        if get_job(job_id) is not None:
            update_job(
                job_id,
                status="failed",
                error=str(exc),
                completed_at=datetime.now(timezone.utc),
                started_at=started_at_utc,
                processing_duration_sec=round(perf_counter() - started_at, 3),
            )
