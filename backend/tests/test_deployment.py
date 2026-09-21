"""Check deployed UI/API routing and runtime storage configuration."""

import httpx
import pytest
from main import create_app, load_config


@pytest.fixture
def anyio_backend():
    return "asyncio"


def test_data_directory_environment_override(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    config = load_config()
    assert config["upload"]["temp_dir"] == str(tmp_path / "temp")
    assert config["upload"]["output_dir"] == str(tmp_path / "output")
    assert config["logging"]["log_file"] == str(tmp_path / "logs" / "pipeline.log")


@pytest.mark.anyio
async def test_ui_mount_preserves_health_api_and_missing_asset_errors(tmp_path, monkeypatch):
    frontend = tmp_path / "frontend"
    frontend.mkdir()
    (frontend / "index.html").write_text("<h1>Video tool</h1>", encoding="utf-8")
    monkeypatch.setenv("FRONTEND_DIST", str(frontend))
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    app = create_app()
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        page = await client.get("/")
        assert page.status_code == 200
        assert "Video tool" in page.text
        assert (await client.get("/health")).json() == {"status": "ok"}
        assert (await client.get("/api/v1/saved-extractions")).json() == []
        assert (await client.get("/api/v1/status/missing")).status_code == 404
        assert (await client.get("/assets/missing.js")).status_code == 404
        assert (await client.get("/config.yaml")).status_code == 404
