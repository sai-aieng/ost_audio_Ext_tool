"""Result retrieval, download, and job cleanup endpoints."""

import json
from pathlib import Path
from typing import Any, Literal

from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.responses import FileResponse

from api.dependencies import get_base_dir, get_config
from api.schemas.response import (
    DeleteResponse,
    DownloadLinkResponse,
    ExtractionResult,
    TranscriptResponse,
    TranscriptSegment,
)
from storage.job_store import delete_job, get_job
from utils.file_utils import job_directory, remove_directory_within, resolve_configured_path

router = APIRouter(tags=["results"])
ResultFormat = Literal["json", "csv", "txt"]
_MEDIA_TYPES = {
    "json": "application/json",
    "csv": "text/csv",
    "txt": "text/plain",
}


@router.get("/saved-extractions")
async def list_saved_extractions(
    config: dict[str, Any] = Depends(get_config),
    base_dir: Path = Depends(get_base_dir),
) -> list[dict[str, Any]]:
    """List already-exported OCR runs without starting any processing."""

    output_root = resolve_configured_path(
        base_dir, str(config["upload"]["output_dir"])
    )
    if not output_root.exists():
        return []
    runs = []
    for directory in output_root.iterdir():
        result_path = directory / "results.json"
        if not directory.is_dir() or not result_path.is_file():
            continue
        try:
            payload = json.loads(result_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(payload, list):
            runs.append(
                {
                    "job_id": directory.name,
                    "completed_at": result_path.stat().st_mtime,
                    "result_count": len(payload),
                }
            )
    return sorted(runs, key=lambda item: item["completed_at"], reverse=True)


@router.get("/saved-extractions/{job_id}", response_model=list[ExtractionResult])
async def get_saved_extraction(
    job_id: str,
    config: dict[str, Any] = Depends(get_config),
    base_dir: Path = Depends(get_base_dir),
) -> list[ExtractionResult]:
    """Read one existing OCR export without rerunning the video pipeline."""

    output_root = resolve_configured_path(
        base_dir, str(config["upload"]["output_dir"])
    )
    try:
        result_path = job_directory(output_root, job_id) / "results.json"
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid job identifier") from exc
    if not result_path.is_file():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Saved extraction not found")
    try:
        payload = json.loads(result_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Saved extraction could not be read") from exc
    return [ExtractionResult.model_validate(item) for item in payload]


@router.get("/audio/results/{job_id}", response_model=TranscriptResponse)
async def get_audio_results(job_id: str) -> TranscriptResponse:
    """Return the transcript from an independently completed Whisper job."""

    record = _require_completed_job(job_id)
    if record.get("task_type") != "audio":
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Audio job not found")
    return TranscriptResponse(
        job_id=job_id,
        language=record.get("language"),
        segments=[TranscriptSegment.model_validate(item) for item in record.get("transcript", [])],
    )


def _require_completed_job(job_id: str) -> dict[str, Any]:
    """Return a completed job or raise an appropriate HTTP error."""

    record = get_job(job_id)
    if record is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Job not found",
        )
    if record["status"] != "completed":
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Results are unavailable while job is {record['status']}",
        )
    return record


@router.get(
    "/results/{job_id}",
    response_model=list[ExtractionResult] | DownloadLinkResponse,
)
async def get_results(
    job_id: str,
    result_format: ResultFormat | None = Query(default=None, alias="format"),
) -> list[ExtractionResult] | DownloadLinkResponse:
    """Return JSON results, or a download URL when a format is requested."""

    record = _require_completed_job(job_id)
    if result_format is not None:
        if result_format not in record.get("output_paths", {}):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"{result_format.upper()} output was not generated",
            )
        return DownloadLinkResponse(
            job_id=job_id,
            format=result_format,
            download_url=(
                f"/api/v1/results/{job_id}/download?format={result_format}"
            ),
        )
    return [ExtractionResult.model_validate(item) for item in record.get("results", [])]


@router.get("/results/{job_id}/download", response_class=FileResponse)
async def download_results(
    job_id: str,
    result_format: ResultFormat = Query(alias="format"),
) -> FileResponse:
    """Stream one generated result artifact as an attachment."""

    record = _require_completed_job(job_id)
    configured_path = record.get("output_paths", {}).get(result_format)
    if configured_path is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"{result_format.upper()} output was not generated",
        )
    path = Path(configured_path)
    if not path.is_file():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Result artifact is missing",
        )
    return FileResponse(
        path,
        media_type=_MEDIA_TYPES[result_format],
        filename=f"{job_id}-results.{result_format}",
    )


@router.delete("/jobs/{job_id}", response_model=DeleteResponse)
async def remove_job(
    job_id: str,
    config: dict[str, Any] = Depends(get_config),
    base_dir: Path = Depends(get_base_dir),
) -> DeleteResponse:
    """Delete a non-running job and all of its stored files."""

    record = get_job(job_id)
    if record is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Job not found",
        )
    if record["status"] == "processing":
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="A processing job cannot be deleted",
        )

    temp_root = resolve_configured_path(base_dir, str(config["upload"]["temp_dir"]))
    output_root = resolve_configured_path(
        base_dir, str(config["upload"]["output_dir"])
    )
    try:
        remove_directory_within(temp_root / job_id, temp_root)
        remove_directory_within(output_root / job_id, output_root)
    except (OSError, ValueError) as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Job files could not be removed",
        ) from exc
    delete_job(job_id)
    return DeleteResponse(job_id=job_id, status="deleted")
