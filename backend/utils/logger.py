"""Central rotating-file and console logging configuration."""

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any

from utils.file_utils import ensure_directory, resolve_configured_path

_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"


def configure_logging(config: dict[str, Any], base_dir: Path) -> logging.Logger:
    """Configure the application logger from the logging config section."""

    logging_config = config["logging"]
    logger = logging.getLogger("video_ocr")
    level = getattr(logging, str(logging_config["level"]).upper(), logging.INFO)
    logger.setLevel(level)
    logger.propagate = False

    for handler in list(logger.handlers):
        handler.close()
        logger.removeHandler(handler)

    formatter = logging.Formatter(_FORMAT)
    console = logging.StreamHandler()
    console.setLevel(level)
    console.setFormatter(formatter)
    logger.addHandler(console)

    log_path = resolve_configured_path(base_dir, str(logging_config["log_file"]))
    ensure_directory(log_path.parent)
    rotating_file = RotatingFileHandler(
        log_path,
        maxBytes=int(logging_config["max_bytes"]),
        backupCount=int(logging_config["backup_count"]),
        encoding="utf-8",
    )
    rotating_file.setLevel(level)
    rotating_file.setFormatter(formatter)
    logger.addHandler(rotating_file)
    return logger


def get_logger(name: str) -> logging.Logger:
    """Return a child logger within the configured application namespace."""

    return logging.getLogger(f"video_ocr.{name}")
