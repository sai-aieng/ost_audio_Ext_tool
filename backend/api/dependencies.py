"""Shared FastAPI dependencies backed by application state."""

from pathlib import Path
from typing import Any

from fastapi import Request


def get_config(request: Request) -> dict[str, Any]:
    """Return the configuration attached during application creation."""

    return request.app.state.config


def get_base_dir(request: Request) -> Path:
    """Return the backend base directory attached to the application."""

    return request.app.state.base_dir
