"""Direct-ASGI integration tests for the FastAPI endpoints."""

from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

import httpx
import pytest

from api.schemas.response import ExtractionResult
from main import create_app, load_config
from pipeline.exporter import export_results
from storage.job_store import clear_jobs, set_results, update_job


@pytest.fixture
def anyio_backend() -> str:
    """Limit AnyIO tests to asyncio, which is available with FastAPI."""

    return "asyncio"


@pytest.fixture(autouse=True)
def isolated_job_store() -> Iterator[None]:
    """Prevent process-local jobs from leaking between API tests."""

    clear_jobs()
    yield
    clear_jobs()


def _test_config(tmp_path: Path) -> dict:
    """Return default config redirected to pytest temporary directories."""

    config = deepcopy(load_config())
    config["upload"]["temp_dir"] = str(tmp_path / "temp")
    config["upload"]["output_dir"] = str(tmp_path / "output")
    config["logging"]["log_file"] = str(tmp_path / "logs" / "test.log")
    config["audio_transcription"]["enabled"] = False
    return config


@pytest.mark.anyio
async def test_upload_process_results_download_and_delete(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exercise the complete API lifecycle without a live HTTP server."""

    config = _test_config(tmp_path)
    app = create_app(config=config, base_dir=tmp_path)

    def fake_validate(*args: object, **kwargs: object) -> dict:
        """Return stable metadata without invoking ffprobe."""

        return {
            "duration_sec": 5.0,
            "fps": 30.0,
            "width": 1280,
            "height": 720,
            "resolution": "1280x720",
        }

    def fake_pipeline(
        job_id: str,
        overrides: dict,
        task_config: dict,
        base_dir: Path,
    ) -> None:
        """Complete a job and export one deterministic result."""

        result = ExtractionResult(
            frame_index=0,
            timestamp_sec=0.0,
            text="Test result",
            confidence=0.99,
            bbox=[[0, 0], [10, 0], [10, 10], [0, 10]],
        )
        paths = export_results(
            job_id,
            [result],
            ["json", "csv", "txt"],
            Path(task_config["upload"]["output_dir"]),
        )
        set_results(job_id, [result], paths)
        update_job(
            job_id,
            status="completed",
            frames_extracted=1,
            frames_deduplicated=1,
            frames_processed=1,
            progress_pct=100.0,
            completed_at=datetime.now(timezone.utc),
        )

    monkeypatch.setattr("api.routes.upload.validate_video", fake_validate)
    monkeypatch.setattr("api.routes.process.run_pipeline", fake_pipeline)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://test",
    ) as client:
        upload_response = await client.post(
            "/api/v1/upload",
            files={"file": ("clip.mp4", b"fake-video", "video/mp4")},
        )
        assert upload_response.status_code == 201
        job_id = upload_response.json()["job_id"]

        process_response = await client.post(
            f"/api/v1/process/{job_id}",
            json={"sample_rate_fps": 1, "output_formats": ["json", "csv", "txt"]},
        )
        assert process_response.status_code == 200

        status_response = await client.get(f"/api/v1/status/{job_id}")
        assert status_response.json()["status"] == "completed"

        results_response = await client.get(f"/api/v1/results/{job_id}")
        assert results_response.status_code == 200
        assert results_response.json()[0]["text"] == "Test result"

        link_response = await client.get(
            f"/api/v1/results/{job_id}",
            params={"format": "csv"},
        )
        assert link_response.json()["download_url"].endswith("format=csv")

        download_response = await client.get(
            f"/api/v1/results/{job_id}/download",
            params={"format": "txt"},
        )
        assert download_response.status_code == 200
        assert "Test result" in download_response.text

        delete_response = await client.delete(f"/api/v1/jobs/{job_id}")
        assert delete_response.status_code == 200
        assert delete_response.json()["status"] == "deleted"


@pytest.mark.anyio
async def test_upload_rejects_unsupported_extension(tmp_path: Path) -> None:
    """Extension validation should fail before writing the upload."""

    app = create_app(config=_test_config(tmp_path), base_dir=tmp_path)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://test",
    ) as client:
        response = await client.post(
            "/api/v1/upload",
            files={"file": ("notes.txt", b"not-video", "text/plain")},
        )

    assert response.status_code == 400
    assert "Unsupported video extension" in response.json()["detail"]
