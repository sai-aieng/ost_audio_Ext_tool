"""Standalone presenter-face validation API with no frontend/OCR integration."""

import json
from pathlib import Path
from typing import Any
from uuid import uuid4

import aiofiles
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse

from api.dependencies import get_base_dir, get_config
from faces import service
from utils.file_utils import remove_directory_within, resolve_configured_path

router = APIRouter(prefix="/faces", tags=["presenter faces (standalone validation)"])


def roots(config, base_dir):
    return (resolve_configured_path(base_dir, config["upload"]["temp_dir"]) / "faces",
            resolve_configured_path(base_dir, config["upload"]["output_dir"]) / "faces")


def output_directory(job_id, config, base_dir):
    _, output_root = roots(config, base_dir)
    try:
        return service.job_directory(output_root, job_id)
    except ValueError as exc:
        raise HTTPException(400, "Invalid face job ID") from exc


def status_for(output):
    try:
        return service.read_status(output)
    except FileNotFoundError as exc:
        raise HTTPException(404, "Face job not found") from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise HTTPException(500, "Face job status could not be read") from exc


@router.post("/extract", status_code=202)
async def extract_faces(
    file: UploadFile = File(...),
    sample_rate_fps: float = Form(default=2, ge=0.5, le=5),
    max_presenters: int = Form(default=3, ge=1, le=10),
    config: dict[str, Any] = Depends(get_config),
    base_dir: Path = Depends(get_base_dir),
):
    """Upload a video for faces only. Does not start OCR or audio.

    Body movement selects likely presenters, not confirmed narrators.
    Track IDs do not identify a person across scene cuts.
    """
    task_config = service.load_face_config()
    task_config.update(sample_rate_fps=sample_rate_fps, max_presenters=max_presenters)
    try:
        service.verify_setup(task_config)
    except RuntimeError as exc:
        await file.close()
        raise HTTPException(503, str(exc)) from exc
    filename = Path(file.filename or "").name
    extension = Path(filename).suffix.lower()
    if extension not in config["upload"]["allowed_extensions"]:
        await file.close()
        raise HTTPException(400, "Unsupported video extension")
    if not service.reserve_slot():
        await file.close()
        raise HTTPException(429, "Face queue is full. Retry after an existing face job finishes.")
    job_id = str(uuid4())
    temp_root, output_root = roots(config, base_dir)
    temporary = service.job_directory(temp_root, job_id)
    output = service.job_directory(output_root, job_id)
    submitted = False
    try:
        temporary.mkdir(parents=True)
        output.mkdir(parents=True)
        video_path = temporary / ("input_video" + extension)
        size = 0
        limit = int(config["upload"]["max_file_size_mb"]) * 1024 * 1024
        async with aiofiles.open(video_path, "wb") as destination:
            while chunk := await file.read(1024 * 1024):
                size += len(chunk)
                if size > limit:
                    raise HTTPException(413, "Video exceeds the configured upload limit")
                await destination.write(chunk)
        if size == 0:
            raise HTTPException(400, "Uploaded video is empty")
        # Video decoding/probing happens in the isolated face worker; malformed
        # files become failed jobs without native libraries entering the API.
        record = service.submit(job_id, video_path, output, task_config, filename)
        submitted = True
        return {
            "face_job_id": job_id, "status": record["status"],
            "status_url": f"/api/v1/faces/status/{job_id}",
            "results_url": f"/api/v1/faces/results/{job_id}",
        }
    finally:
        await file.close()
        if not submitted:
            service.release_slot()
            remove_directory_within(temporary, temp_root)
            remove_directory_within(output, output_root)


@router.get("/status/{face_job_id}")
def face_status(face_job_id: str, config: dict = Depends(get_config), base_dir: Path = Depends(get_base_dir)):
    """Independent face progress and timer; queue wait is excluded from runtime."""
    return status_for(output_directory(face_job_id, config, base_dir))


def completed_output(face_job_id, config, base_dir):
    output = output_directory(face_job_id, config, base_dir)
    record = status_for(output)
    if record["status"] != "completed":
        raise HTTPException(409, f"Face results unavailable while job is {record['status']}")
    return output


@router.get("/results/{face_job_id}")
def face_results(face_job_id: str, config: dict = Depends(get_config), base_dir: Path = Depends(get_base_dir)):
    output = completed_output(face_job_id, config, base_dir)
    try:
        result = json.loads((output / "faces.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise HTTPException(500, "Face results could not be read") from exc
    for key in ("faces", "narrators", "static_faces", "presenters"):
        for face in result.get(key, []):
            face["image_url"] = f"/api/v1/faces/images/{face_job_id}/{face['track_id']}"
    result["download_url"] = f"/api/v1/faces/download/{face_job_id}"
    return result


@router.get("/images/{face_job_id}/{track_id}")
def face_image(face_job_id: str, track_id: str, config: dict = Depends(get_config), base_dir: Path = Depends(get_base_dir)):
    result = face_results(face_job_id, config, base_dir)
    selected = next((p for p in result.get("faces", result["presenters"]) if p["track_id"] == track_id), None)
    if selected is None:
        raise HTTPException(404, "Face crop not found")
    output = output_directory(face_job_id, config, base_dir)
    # Construct filenames only from allowlisted result track IDs, not user paths.
    if not track_id.startswith("track-") or not track_id[6:].isdigit():
        raise HTTPException(400, "Invalid track ID")
    path = (output / selected.get("face_image", f"{track_id}.jpg")).resolve()
    try:
        path.relative_to(output.resolve())
    except ValueError as exc:
        raise HTTPException(400, "Invalid face crop path") from exc
    if not path.is_file():
        raise HTTPException(404, "Face crop is missing")
    return FileResponse(path, media_type="image/jpeg", filename=path.name)


@router.get("/download/{face_job_id}")
def download_faces(face_job_id: str, config: dict = Depends(get_config), base_dir: Path = Depends(get_base_dir)):
    output = completed_output(face_job_id, config, base_dir)
    path = output / "faces.json"
    if not path.is_file():
        raise HTTPException(404, "Face results are missing")
    return FileResponse(path, media_type="application/json", filename=f"{face_job_id}-faces.json")


@router.delete("/jobs/{face_job_id}")
def delete_faces(face_job_id: str, config: dict = Depends(get_config), base_dir: Path = Depends(get_base_dir)):
    output = output_directory(face_job_id, config, base_dir)
    record = status_for(output)
    if record["status"] in {"queued", "processing"}:
        raise HTTPException(409, "Wait for face processing to finish before deleting")
    temp_root, output_root = roots(config, base_dir)
    remove_directory_within(service.job_directory(temp_root, face_job_id), temp_root)
    remove_directory_within(output, output_root)
    return {"face_job_id": face_job_id, "status": "deleted"}
