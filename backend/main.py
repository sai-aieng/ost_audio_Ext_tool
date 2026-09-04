"""FastAPI application entry point for the standalone backend."""

from contextlib import asynccontextmanager
from copy import deepcopy
from pathlib import Path
from typing import Any, AsyncIterator

import uvicorn
import yaml
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes.process import router as process_router
from api.routes.results import router as results_router
from api.routes.upload import router as upload_router
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
        yield
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
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    application.include_router(upload_router, prefix="/api/v1")
    application.include_router(process_router, prefix="/api/v1")
    application.include_router(results_router, prefix="/api/v1")

    @application.get("/health", tags=["system"])
    async def health_check() -> dict[str, str]:
        """Return a lightweight service readiness response."""

        return {"status": "ok"}

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
