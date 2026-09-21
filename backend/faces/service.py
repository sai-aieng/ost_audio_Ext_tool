"""Face-only jobs and subprocess lifecycle; never starts OCR or audio."""

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from datetime import datetime, timezone
import importlib.util
import json
import logging
import os
from pathlib import Path
import subprocess
import sys
from threading import BoundedSemaphore, RLock
from time import perf_counter
from uuid import UUID
from faces.json_io import write_json

BACKEND = Path(__file__).resolve().parents[1]
LOGGER = logging.getLogger("video_ocr.faces")
_POOL = None
_SLOTS = BoundedSemaphore(4)
_LOCK = RLock()
_ACTIVE = set()
_PROCESSES = {}
_FUTURES = {}
_STOPPING = False


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def load_face_config():
    return json.loads((BACKEND / "face_config.json").read_text(encoding="utf-8"))


def verify_setup(config):
    if importlib.util.find_spec("mediapipe") is None:
        raise RuntimeError("MediaPipe is missing. Install backend/requirements.txt in the current environment.")
    root = Path(config["model_root"])
    if not root.is_absolute():
        root = BACKEND / root
    for filename in ("pose_landmarker_lite.task", "blaze_face_short_range.tflite"):
        if not (root / filename).is_file():
            raise RuntimeError("Face models are missing. Run backend/scripts/prepare_face_models.py first.")


def job_directory(root, job_id):
    try:
        canonical = str(UUID(job_id))
    except (ValueError, AttributeError) as exc:
        raise ValueError("Invalid face job ID") from exc
    if canonical != job_id:
        raise ValueError("Invalid face job ID")
    path = (root / canonical).resolve()
    path.relative_to(root.resolve())
    return path


def reserve_slot():
    return _SLOTS.acquire(blocking=False)


def release_slot():
    _SLOTS.release()


def submit(job_id, video, output, config, filename):
    global _POOL, _STOPPING
    record = {
        "face_job_id": job_id, "status": "queued", "stage": "Waiting for face worker",
        "filename": filename, "created_at": utc_now(), "started_at": None,
        "completed_at": None, "processing_duration_sec": None,
        "progress_pct": 0, "frames_processed": 0, "faces_found": 0, "error": None,
    }
    write_json(output / "status.json", record)
    write_json(output / "config.json", config)
    try:
        with _LOCK:
            if _POOL is None:
                _POOL = ThreadPoolExecutor(max_workers=1, thread_name_prefix="face-validation")
                _STOPPING = False
            if _STOPPING:
                raise RuntimeError("Face service is shutting down")
            _ACTIVE.add(job_id)
            future = _POOL.submit(_run, job_id, video, output, deepcopy(record), config)
            _FUTURES[job_id] = future
            future.add_done_callback(lambda done: _finish_future(done, job_id, output, record))
    except Exception:
        with _LOCK:
            _ACTIVE.discard(job_id)
        raise
    return record


def _finish_future(future, job_id, output, record):
    with _LOCK:
        _FUTURES.pop(job_id, None)
    if future.cancelled():
        record = dict(record, status="failed", stage="Interrupted", completed_at=utc_now(),
                      error="Backend stopped while this face job was queued.")
        try:
            write_json(output / "status.json", record)
        finally:
            with _LOCK:
                _ACTIVE.discard(job_id)
            release_slot()


def stop_process(process):
    """Stop only a child owned by this face service, including its venv launcher."""
    if process.poll() is not None:
        return
    if os.name == "nt":
        subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                       creationflags=subprocess.CREATE_NO_WINDOW, timeout=10, check=False)
    else:
        process.kill()
    process.wait(timeout=10)


def _run(job_id, video, output, record, config):
    started = perf_counter()
    record.update(status="processing", stage="Starting face worker", started_at=utc_now())
    process = None
    try:
        write_json(output / "status.json", record)
        environment = os.environ.copy()
        environment.update(OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2",
                           MKL_NUM_THREADS="2", PYTHONUNBUFFERED="1")
        flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
        with (output / "worker.log").open("wb") as log:
            with _LOCK:
                if _STOPPING:
                    raise RuntimeError("Backend stopped before the face worker started")
                process = subprocess.Popen(
                    [sys.executable, "-B", str(BACKEND / "faces" / "worker.py"),
                     "--video", str(video), "--output", str(output),
                     "--config", str(output / "config.json")],
                    cwd=str(BACKEND), env=environment, stdout=log, stderr=subprocess.STDOUT,
                    creationflags=flags,
                )
                _PROCESSES[job_id] = process
            try:
                code = process.wait(timeout=config["worker_timeout_seconds"])
            except subprocess.TimeoutExpired:
                stop_process(process)
                raise RuntimeError("Face extraction exceeded its configured time limit.")
        if code != 0:
            # The detailed native/Python traceback stays in this job's worker.log.
            raise RuntimeError(f"Face worker failed (exit code {code}). Check the face job worker.log.")
        payload = json.loads((output / "faces.json").read_text(encoding="utf-8"))
        payload["face_job_id"] = job_id
        write_json(output / "faces.json", payload)
        record.update(status="completed", stage="Completed", progress_pct=100,
                      frames_processed=payload["frames_sampled"],
                      faces_found=len(payload.get("faces", payload["presenters"])),
                      output_folder=payload.get("output_folder"),
                      narrator_faces_found=len(payload.get("narrators", payload["presenters"])),
                      static_faces_found=len(payload.get("static_faces", [])))
    except Exception as exc:
        LOGGER.exception("face_job=%s failed", job_id)
        record.update(status="failed", stage="Failed", error=str(exc))
    finally:
        if process is not None and process.poll() is None:
            try:
                stop_process(process)
            except Exception:
                LOGGER.exception("face_job=%s child cleanup failed", job_id)
        record.update(completed_at=utc_now(), processing_duration_sec=round(perf_counter() - started, 3))
        try:
            write_json(output / "status.json", record)
            LOGGER.info("face_job=%s status=%s duration=%.3fs", job_id, record["status"], record["processing_duration_sec"])
        finally:
            with _LOCK:
                _ACTIVE.discard(job_id)
                _PROCESSES.pop(job_id, None)
            release_slot()


def read_status(output):
    path = output / "status.json"
    if not path.is_file():
        raise FileNotFoundError("Face job not found")
    record = json.loads(path.read_text(encoding="utf-8"))
    if record["status"] in {"queued", "processing"}:
        with _LOCK:
            alive = record["face_job_id"] in _ACTIVE
        if not alive:
            # Completion can race this read: recheck the persisted terminal
            # status before deciding that a previously running job was lost.
            latest = json.loads(path.read_text(encoding="utf-8"))
            if latest["status"] in {"completed", "failed"}:
                return latest
            record.update(status="failed", stage="Interrupted",
                          error="Backend restarted before this face job finished. Submit the video again.")
        progress_path = output / "progress.json"
        if alive and progress_path.is_file():
            try:
                progress = json.loads(progress_path.read_text(encoding="utf-8"))
                record.update({key: progress[key] for key in
                               ("stage", "progress_pct", "frames_processed", "tracks_detected", "video_timestamp_sec")
                               if key in progress})
            except (OSError, json.JSONDecodeError):
                pass
    return record


def shutdown():
    # Called before server shutdown; leave OCR/audio lifecycle untouched.
    global _POOL, _STOPPING
    with _LOCK:
        _STOPPING = True
        pool = _POOL
        processes = list(_PROCESSES.values())
    if pool is not None:
        pool.shutdown(wait=False, cancel_futures=True)
    for process in processes:
        try:
            stop_process(process)
        except Exception:
            LOGGER.exception("Could not stop a face worker during shutdown")
    if pool is not None:
        pool.shutdown(wait=True)
    with _LOCK:
        _POOL = None
