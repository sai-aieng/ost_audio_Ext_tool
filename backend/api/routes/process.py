"""Pipeline start and job status endpoints."""

from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any
from uuid import uuid4

from fastapi import APIRouter, BackgroundTasks, Body, Depends, HTTPException, status

from api.dependencies import get_base_dir, get_config
from api.schemas.request import ProcessOptions
from api.schemas.response import JobStatus, ProcessResponse
from pipeline.audio_pipeline import run_audio_pipeline
from pipeline.orchestrator import run_pipeline
from storage.job_store import create_job, get_job, to_job_status, update_job

router = APIRouter(tags=["processing"])


def run_parallel_pipelines(
    ocr_job_id: str,
    audio_job_id: str | None,
    overrides: dict[str, Any],
    config: dict[str, Any],
    base_dir: Path,
) -> None:
    """Execute the independent OCR and audio jobs concurrently."""

    with ThreadPoolExecutor(max_workers=2, thread_name_prefix="video-analysis") as pool:
        futures = [pool.submit(run_pipeline, ocr_job_id, overrides, deepcopy(config), base_dir)]
        if audio_job_id is not None:
            futures.append(pool.submit(run_audio_pipeline, audio_job_id, deepcopy(config), base_dir))
        for future in futures:
            future.result()


@router.post("/process/{job_id}", response_model=ProcessResponse)
async def start_processing(
    job_id: str,
    background_tasks: BackgroundTasks,
    options: ProcessOptions | None = Body(default=None),
    config: dict[str, Any] = Depends(get_config),
    base_dir: Path = Depends(get_base_dir),
) -> ProcessResponse:
    """Queue the synchronous OCR pipeline as a FastAPI background task."""

    record = get_job(job_id)
    if record is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Job not found",
        )
    if record["status"] in {"processing", "completed"}:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Job is already {record['status']}",
        )

    overrides = options.model_dump(exclude_none=True) if options else {}
    audio_job_id: str | None = None
    if bool(config["audio_transcription"]["enabled"]):
        audio_job_id = str(uuid4())
        create_job(
            audio_job_id,
            record["filename"],
            record["video_path"],
            record["metadata"],
            task_type="audio",
            parent_job_id=job_id,
        )
    update_job(
        job_id,
        status="processing",
        error=None,
        progress_pct=1.0,
        audio_job_id=audio_job_id,
    )
    background_tasks.add_task(
        run_parallel_pipelines,
        job_id,
        audio_job_id,
        overrides,
        deepcopy(config),
        base_dir,
    )
    return ProcessResponse(job_id=job_id, status="processing", audio_job_id=audio_job_id)


@router.get("/status/{job_id}", response_model=JobStatus)
async def get_job_status(job_id: str) -> JobStatus:
    """Return current counters, progress, and lifecycle state."""

    record = get_job(job_id)
    if record is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Job not found",
        )
    return to_job_status(record)
