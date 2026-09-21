"""Standalone face routes: no OCR/audio or model calls in these tests."""

from uuid import uuid4

from fastapi import FastAPI
import httpx
import pytest

from api.routes.faces import router
from faces import service


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture
def app(tmp_path, monkeypatch):
    app = FastAPI()
    app.state.base_dir = tmp_path
    app.state.config = {"upload": {
        "temp_dir": "temp", "output_dir": "output",
        "max_file_size_mb": 1, "allowed_extensions": [".mp4"],
    }}
    app.include_router(router, prefix="/api/v1")
    monkeypatch.setattr(service, "verify_setup", lambda config: None)
    return app


def completed_job(app):
    job_id = str(uuid4())
    output = app.state.base_dir / "output" / "faces" / job_id
    output.mkdir(parents=True)
    service.write_json(output / "status.json", {"face_job_id": job_id, "status": "completed"})
    service.write_json(output / "faces.json", {"face_job_id": job_id, "presenters": [
        {"track_id": "track-0001", "face_image": "track-0001.jpg", "bbox_xyxy": [1, 2, 3, 4]}
    ]})
    (output / "track-0001.jpg").write_bytes(b"test-jpeg")
    return job_id, output


@pytest.mark.anyio
async def test_upload_only_submits_face_job(app, monkeypatch):
    captured = []
    def fake_submit(job_id, video, output, config, filename):
        captured.append((job_id, video, config))
        record = {"face_job_id": job_id, "status": "completed"}
        service.write_json(output / "status.json", record)
        service.release_slot()
        return record
    monkeypatch.setattr(service, "submit", fake_submit)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post("/api/v1/faces/extract",
            files={"file": ("video.mp4", b"fake-video", "video/mp4")},
            data={"sample_rate_fps": "2", "max_presenters": "1"})
        assert response.status_code == 202
        assert len(captured) == 1
        assert captured[0][1].read_bytes() == b"fake-video"
        assert captured[0][2]["max_presenters"] == 1
        assert (await client.get(response.json()["status_url"])).status_code == 200


@pytest.mark.anyio
async def test_results_image_download_and_delete_are_face_only(app):
    job_id, output = completed_job(app)
    unrelated = app.state.base_dir / "output" / "keep.txt"
    unrelated.write_text("keep")
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        result = await client.get(f"/api/v1/faces/results/{job_id}")
        assert result.status_code == 200
        assert result.json()["presenters"][0]["image_url"].endswith("track-0001")
        image = await client.get(result.json()["presenters"][0]["image_url"])
        assert image.content == b"test-jpeg"
        assert (await client.get(f"/api/v1/faces/images/{job_id}/track-9999")).status_code == 404
        assert (await client.get(f"/api/v1/faces/download/{job_id}")).status_code == 200
        assert (await client.delete(f"/api/v1/faces/jobs/{job_id}")).status_code == 200
    assert not output.exists()
    assert unrelated.read_text() == "keep"


@pytest.mark.anyio
async def test_invalid_missing_and_running_jobs(app):
    job_id, output = completed_job(app)
    service.write_json(output / "status.json", {"face_job_id": job_id, "status": "processing"})
    with service._LOCK:
        service._ACTIVE.add(job_id)
    try:
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            assert (await client.get("/api/v1/faces/status/not-a-uuid")).status_code == 400
            assert (await client.get(f"/api/v1/faces/status/{uuid4()}")).status_code == 404
            assert (await client.get(f"/api/v1/faces/results/{job_id}")).status_code == 409
            assert (await client.delete(f"/api/v1/faces/jobs/{job_id}")).status_code == 409
    finally:
        service._ACTIVE.discard(job_id)


@pytest.mark.anyio
async def test_bad_uploads_and_queue_backpressure(app, monkeypatch):
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        path = "/api/v1/faces/extract"
        assert (await client.post(path, files={"file": ("x.txt", b"x")})).status_code == 400
        assert (await client.post(path, files={"file": ("x.mp4", b"")})).status_code == 400
        assert (await client.post(path, files={"file": ("x.mp4", b"x" * (1024 * 1024 + 1))})).status_code == 413
        assert (await client.post(path, files={"file": ("x.mp4", b"x")},
                                  data={"sample_rate_fps": 0})).status_code == 422
        monkeypatch.setattr(service, "reserve_slot", lambda: False)
        assert (await client.post(path, files={"file": ("x.mp4", b"x")})).status_code == 429


def test_restart_marks_unfinished_job_failed(tmp_path):
    service.write_json(tmp_path / "status.json", {"face_job_id": str(uuid4()), "status": "processing"})
    assert service.read_status(tmp_path)["status"] == "failed"


@pytest.mark.anyio
async def test_nested_static_image_is_served(app):
    job_id, output = completed_job(app)
    relative = "output_2026-09-16_12-00-00_000000_UTC/static/track-0002.jpg"
    path = output / relative
    path.parent.mkdir(parents=True)
    path.write_bytes(b"static-jpeg")
    service.write_json(output / "faces.json", {
        "presenters": [], "faces": [{"track_id": "track-0002", "face_image": relative}],
        "static_faces": [{"track_id": "track-0002", "face_image": relative}],
    })
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        result = await client.get(f"/api/v1/faces/results/{job_id}")
        url = result.json()["static_faces"][0]["image_url"]
        assert (await client.get(url)).content == b"static-jpeg"
        service.write_json(output / "faces.json", {
            "presenters": [], "faces": [{"track_id": "track-0002", "face_image": "../outside.jpg"}],
        })
        assert (await client.get(url)).status_code == 400


def test_native_worker_failure_becomes_failed_job(tmp_path, monkeypatch):
    class FailedProcess:
        pid = 123
        def wait(self, timeout=None):
            return -1073741819
        def poll(self):
            return -1073741819
    monkeypatch.setattr(service.subprocess, "Popen", lambda *args, **kwargs: FailedProcess())
    monkeypatch.setattr(service, "release_slot", lambda: None)
    monkeypatch.setattr(service, "_STOPPING", False)
    job_id = str(uuid4())
    record = {"face_job_id": job_id, "status": "queued"}
    service._run(job_id, tmp_path / "input.mp4", tmp_path, record, {"worker_timeout_seconds": 2})
    status = service.read_status(tmp_path)
    assert status["status"] == "failed"
    assert "-1073741819" in status["error"]
    assert status["processing_duration_sec"] >= 0
