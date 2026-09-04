"""Video upload endpoint."""

from pathlib import Path
from typing import Any
from uuid import uuid4

import aiofiles
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status

from api.dependencies import get_base_dir, get_config
from api.schemas.response import UploadResponse
from pipeline.video_ingestion import validate_video
from storage.job_store import create_job
from utils.file_utils import (
    ensure_directory,
    job_directory,
    remove_directory_within,
    resolve_configured_path,
)

router = APIRouter(tags=["uploads"])
_CHUNK_SIZE = 1024 * 1024


@router.post(
    "/upload",
    response_model=UploadResponse,
    status_code=status.HTTP_201_CREATED,
)
async def upload_video(
    file: UploadFile = File(...),
    config: dict[str, Any] = Depends(get_config),
    base_dir: Path = Depends(get_base_dir),
) -> UploadResponse:
    """Validate, persist, probe, and register an uploaded video."""

    filename = Path(file.filename or "").name
    extension = Path(filename).suffix.lower()
    upload_config = config["upload"]
    allowed_extensions = {
        str(value).lower() for value in upload_config["allowed_extensions"]
    }
    if extension not in allowed_extensions:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported video extension: {extension or 'none'}",
        )

    max_bytes = int(upload_config["max_file_size_mb"]) * 1024 * 1024
    temp_root = ensure_directory(
        resolve_configured_path(base_dir, str(upload_config["temp_dir"]))
    )
    job_id = str(uuid4())
    job_dir = ensure_directory(job_directory(temp_root, job_id))
    video_path = job_dir / f"input_video{extension}"
    bytes_written = 0

    try:
        async with aiofiles.open(video_path, "wb") as destination:
            while chunk := await file.read(_CHUNK_SIZE):
                bytes_written += len(chunk)
                if bytes_written > max_bytes:
                    raise HTTPException(
                        status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                        detail=(
                            f"Video exceeds the {upload_config['max_file_size_mb']} MB "
                            "upload limit"
                        ),
                    )
                await destination.write(chunk)
        metadata = validate_video(video_path, list(allowed_extensions))
    except HTTPException:
        remove_directory_within(job_dir, temp_root)
        raise
    except (OSError, ValueError) as exc:
        remove_directory_within(job_dir, temp_root)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc
    finally:
        await file.close()

    create_job(job_id, filename, str(video_path), metadata)
    return UploadResponse(
        job_id=job_id,
        filename=filename,
        duration_sec=metadata["duration_sec"],
        fps=metadata["fps"],
        resolution=metadata["resolution"],
        status="uploaded",
    )
