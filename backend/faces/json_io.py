"""Atomic face-job metadata writes with bounded retries for Windows locks."""

import json
import logging
import os
from pathlib import Path
import tempfile
import time

LOGGER = logging.getLogger("video_ocr.faces")
RETRY_DELAYS = (0.05, 0.1, 0.2, 0.4)


def write_json(path, payload):
    """Required writes raise on failure; never truncate the previous JSON."""
    path = Path(path)
    content = json.dumps(payload, indent=2, allow_nan=False)
    temporary = None
    try:
        for attempt in range(len(RETRY_DELAYS) + 1):
            try:
                if temporary is None:
                    with tempfile.NamedTemporaryFile(
                        mode="w", encoding="utf-8", dir=path.parent,
                        prefix=path.name + ".", suffix=".partial", delete=False,
                    ) as stream:
                        temporary = Path(stream.name)
                        stream.write(content)
                os.replace(temporary, path)
                return
            except OSError as exc:
                locked = isinstance(exc, PermissionError) or getattr(exc, "winerror", None) in (5, 32, 33)
                if not locked or attempt == len(RETRY_DELAYS):
                    raise
                # A failed temporary-file write must be recreated, not published.
                if temporary is not None and temporary.exists():
                    try:
                        temporary.unlink()
                    except OSError:
                        raise
                temporary = None
                time.sleep(RETRY_DELAYS[attempt])
    finally:
        if temporary is not None:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                LOGGER.warning("Could not remove temporary face metadata file %s", temporary)


def write_progress(path, payload):
    """Progress is advisory: a filesystem failure must not stop extraction."""
    try:
        write_json(path, payload)
        return True
    except OSError as exc:
        LOGGER.warning("Skipping face progress update for %s; extraction continues: %s", path, exc)
        return False
