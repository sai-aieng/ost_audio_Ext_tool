"""Text normalization and line-level deduplication helpers."""

import re
import unicodedata
from collections.abc import Iterable

_WHITESPACE = re.compile(r"\s+")
_NOISE = re.compile(r"[^\w\s.,!?;:'\"()\-/%&@#+]", flags=re.UNICODE)


def clean_text(value: str) -> str:
    """Normalize Unicode, strip OCR noise, and collapse whitespace."""

    normalized = unicodedata.normalize("NFKC", value)
    without_noise = _NOISE.sub("", normalized)
    return _WHITESPACE.sub(" ", without_noise).strip()


def deduplicate_lines(lines: Iterable[str]) -> list[str]:
    """Clean lines and retain the first occurrence of each non-empty value."""

    unique: list[str] = []
    seen: set[str] = set()
    for line in lines:
        cleaned = clean_text(line)
        if cleaned and cleaned not in seen:
            seen.add(cleaned)
            unique.append(cleaned)
    return unique
