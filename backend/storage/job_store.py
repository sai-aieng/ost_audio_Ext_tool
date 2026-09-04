"""In-memory job registry for single-worker development deployments."""

from copy import deepcopy
from datetime import datetime, timezone
from threading import RLock
from typing import Any, Literal, TypedDict

from api.schemas.response import ExtractionResult, JobStatus, TranscriptSegment

JobState = Literal["uploaded", "processing", "completed", "failed"]


class JobRecord(TypedDict, total=False):
    """Internal shape stored for one uploaded video job."""

    job_id: str
    filename: str
    video_path: str
    metadata: dict[str, Any]
    status: JobState
    frames_extracted: int
    frames_deduplicated: int
    frames_processed: int
    texts_found: int
    progress_pct: float
    error: str | None
    created_at: datetime
    completed_at: datetime | None
    results: list[dict[str, Any]]
    output_paths: dict[str, str]
    task_type: str
    parent_job_id: str | None
    audio_job_id: str | None
    transcript: list[dict[str, Any]]
    language: str | None


# This process-local registry serves the requested single-worker v1. Replace it
# with Redis or a database before running multiple workers or multiple replicas.
_JOBS: dict[str, JobRecord] = {}
_LOCK = RLock()


def create_job(
    job_id: str,
    filename: str,
    video_path: str,
    metadata: dict[str, Any],
    task_type: str = "ocr",
    parent_job_id: str | None = None,
) -> JobRecord:
    """Create and return a new uploaded job record."""

    record = JobRecord(
        job_id=job_id,
        filename=filename,
        video_path=video_path,
        metadata=deepcopy(metadata),
        status="uploaded",
        frames_extracted=0,
        frames_deduplicated=0,
        frames_processed=0,
        texts_found=0,
        progress_pct=0.0,
        error=None,
        created_at=datetime.now(timezone.utc),
        completed_at=None,
        results=[],
        output_paths={},
        task_type=task_type,
        parent_job_id=parent_job_id,
        audio_job_id=None,
        transcript=[],
        language=None,
    )
    with _LOCK:
        _JOBS[job_id] = record
        return deepcopy(record)


def get_job(job_id: str) -> JobRecord | None:
    """Return a defensive copy of a job, or None when it is unknown."""

    with _LOCK:
        record = _JOBS.get(job_id)
        return deepcopy(record) if record is not None else None


def update_job(job_id: str, **changes: Any) -> JobRecord:
    """Atomically update fields on an existing job and return a copy."""

    with _LOCK:
        if job_id not in _JOBS:
            raise KeyError(f"Unknown job: {job_id}")
        _JOBS[job_id].update(changes)
        return deepcopy(_JOBS[job_id])


def delete_job(job_id: str) -> JobRecord | None:
    """Remove a job from the registry and return the removed record."""

    with _LOCK:
        record = _JOBS.pop(job_id, None)
        return deepcopy(record) if record is not None else None


def to_job_status(record: JobRecord) -> JobStatus:
    """Convert an internal job record to its public status model."""

    return JobStatus(
        job_id=record["job_id"],
        status=record["status"],
        frames_extracted=record.get("frames_extracted", 0),
        frames_deduplicated=record.get("frames_deduplicated", 0),
        frames_processed=record.get("frames_processed", 0),
        texts_found=record.get("texts_found", 0),
        progress_pct=record.get("progress_pct", 0.0),
        error=record.get("error"),
        created_at=record["created_at"],
        completed_at=record.get("completed_at"),
    )


def set_results(
    job_id: str,
    results: list[ExtractionResult],
    output_paths: dict[str, str],
) -> JobRecord:
    """Persist serialized results and generated artifact paths."""

    return update_job(
        job_id,
        results=[result.model_dump() for result in results],
        output_paths=output_paths,
        texts_found=len(results),
    )


def set_transcript(
    job_id: str,
    segments: list[TranscriptSegment],
    output_paths: dict[str, str],
    language: str | None,
) -> JobRecord:
    """Persist audio transcript segments and their export paths."""

    return update_job(
        job_id,
        transcript=[segment.model_dump() for segment in segments],
        output_paths=output_paths,
        language=language,
        texts_found=len(segments),
    )


def clear_jobs() -> None:
    """Clear all in-memory jobs for controlled test isolation."""

    with _LOCK:
        _JOBS.clear()
