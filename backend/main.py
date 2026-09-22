"""FastAPI application entry point for the standalone backend."""

from utils.windows_runtime import prepare_native_runtime

# Must precede imports of Paddle, CTranslate2, and other native dependencies.
prepare_native_runtime()

from contextlib import asynccontextmanager
from copy import deepcopy
import os
from pathlib import Path
from typing import Any, AsyncIterator

import uvicorn
import yaml
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from api.routes.process import router as process_router
from api.routes.results import router as results_router
from api.routes.upload import router as upload_router
from api.routes.faces import router as faces_router
from api.routes.gemini import router as gemini_router
from faces.service import shutdown as shutdown_faces
from utils.file_utils import ensure_directory, resolve_configured_path
from utils.logger import configure_logging

BASE_DIR = Path(__file__).resolve().parent
CONFIG_PATH = BASE_DIR / "config.yaml"


def load_config(config_path: Path = CONFIG_PATH) -> dict[str, Any]:
    """Load and validate the top-level YAML configuration mapping."""

    with config_path.open("r", encoding="utf-8") as config_file:
        loaded = yaml.safe_load(config_file)
    if not isinstance(loaded, dict):
        raise ValueError("config.yaml must contain a mapping")
    data_dir = os.environ.get("DATA_DIR")
    if data_dir:
        data_root = Path(data_dir).resolve()
        loaded["upload"]["temp_dir"] = str(data_root / "temp")
        loaded["upload"]["output_dir"] = str(data_root / "output")
        loaded["logging"]["log_file"] = str(data_root / "logs" / "pipeline.log")
    return loaded


def create_app(
    config: dict[str, Any] | None = None,
    base_dir: Path = BASE_DIR,
) -> FastAPI:
    """Build an independently deployable FastAPI application."""

    app_config = deepcopy(config) if config is not None else load_config()
    resolved_base_dir = base_dir.resolve()

    @asynccontextmanager
    async def lifespan(application: FastAPI) -> AsyncIterator[None]:
        """Initialize runtime directories and logging around app lifetime."""

        upload_config = application.state.config["upload"]
        ensure_directory(
            resolve_configured_path(
                application.state.base_dir,
                str(upload_config["temp_dir"]),
            )
        )
        ensure_directory(
            resolve_configured_path(
                application.state.base_dir,
                str(upload_config["output_dir"]),
            )
        )
        logger = configure_logging(
            application.state.config,
            application.state.base_dir,
        )
        logger.info("Video OCR backend started")
        try:
            yield
        finally:
            shutdown_faces()
        logger.info("Video OCR backend stopped")

    application = FastAPI(
        title="Video OCR Pipeline API",
        version="1.0.0",
        lifespan=lifespan,
    )
    application.state.config = app_config
    application.state.base_dir = resolved_base_dir
    application.add_middleware(
        CORSMiddleware,
        allow_origins=[origin.strip() for origin in os.environ.get(
            "CORS_ORIGINS", "http://localhost:5174,http://127.0.0.1:5174"
        ).split(",") if origin.strip()],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    application.include_router(upload_router, prefix="/api/v1")
    application.include_router(process_router, prefix="/api/v1")
    application.include_router(results_router, prefix="/api/v1")
    application.include_router(faces_router, prefix="/api/v1")
    application.include_router(gemini_router, prefix="/api/v1")

    @application.get("/health", tags=["system"])
    async def health_check() -> dict[str, str]:
        """Return a lightweight service readiness response."""

        return {"status": "ok"}

    frontend_dist = os.environ.get("FRONTEND_DIST")
    if frontend_dist:
        # Register last so API, health and docs routes keep their own responses.
        application.mount("/", StaticFiles(directory=frontend_dist, html=True), name="frontend")

    return application


app = create_app()


if __name__ == "__main__":
    server_config = app.state.config["server"]
    uvicorn.run(
        "main:app",
        host=str(server_config["host"]),
        port=int(server_config["port"]),
        reload=bool(server_config["reload"]),
    )
