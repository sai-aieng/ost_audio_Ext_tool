"""Filesystem helpers with workspace-bound cleanup safeguards."""

import shutil
from pathlib import Path


def ensure_directory(path: Path) -> Path:
    """Create a directory and all missing parents, then return it."""

    path.mkdir(parents=True, exist_ok=True)
    return path


def resolve_configured_path(base_dir: Path, configured_path: str) -> Path:
    """Resolve a configured relative path beneath the backend directory."""

    candidate = Path(configured_path)
    return candidate.resolve() if candidate.is_absolute() else (base_dir / candidate).resolve()


def job_directory(base_dir: Path, job_id: str) -> Path:
    """Return the validated directory reserved for a job."""

    if not job_id or any(character not in "0123456789abcdef-" for character in job_id.lower()):
        raise ValueError("Invalid job identifier")
    return base_dir / job_id


def remove_directory_within(path: Path, allowed_parent: Path) -> None:
    """Remove a directory only when it resolves beneath the allowed parent."""

    resolved_parent = allowed_parent.resolve()
    resolved_path = path.resolve()
    if resolved_path == resolved_parent:
        raise ValueError("Refusing to remove the configured root directory")
    try:
        resolved_path.relative_to(resolved_parent)
    except ValueError as exc:
        raise ValueError(f"Unsafe cleanup path: {resolved_path}") from exc
    if resolved_path.exists():
        shutil.rmtree(resolved_path)


def cleanup_frame_directories(job_temp_dir: Path) -> None:
    """Remove extracted and processed frame folders while retaining the upload."""

    for child_name in ("frames", "processed"):
        remove_directory_within(job_temp_dir / child_name, job_temp_dir)
