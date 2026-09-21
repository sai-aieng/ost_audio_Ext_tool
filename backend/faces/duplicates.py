"""Conservative per-extraction photo deduplication, not identity recognition."""

from hashlib import sha256
from io import BytesIO

import imagehash
import numpy as np
from PIL import Image, UnidentifiedImageError


def signature(data):
    digest = sha256(data).digest()
    try:
        with Image.open(BytesIO(data)) as source:
            rgb = source.convert("RGB")
            pixels = np.asarray(rgb.resize((64, 64), Image.Resampling.LANCZOS), dtype=float)
            detail = float(np.asarray(rgb.convert("L"), dtype=float).std())
            return digest, rgb.width / rgb.height, imagehash.phash(rgb), pixels, detail
    except (UnidentifiedImageError, OSError, ValueError):
        # Invalid images can only be matched byte-for-byte.
        return digest, None, None, None, 0


def matches(left, right):
    if left[0] == right[0]:
        return True
    if left[1] is None or right[1] is None or min(left[4], right[4]) < 10:
        return False
    if abs(left[1] / right[1] - 1) > 0.10 or left[2] - right[2] > 4:
        return False
    # Hash alone can confuse different faces; require close RGB pixels too.
    return float(np.sqrt(np.mean((left[3] - right[3]) ** 2))) <= 10


def unique_tracks(tracks, config):
    """Prefer narrator evidence, then crop quality; compare only to kept photos."""
    ordered = sorted(tracks, key=lambda t: (
        bool(t.eligible(config)), t.best["quality_score"], t.visible_seconds,
    ), reverse=True)
    kept, signatures, duplicates = [], [], {}
    for track in ordered:
        candidate = signature(track.best_jpeg)
        for existing, previous in zip(kept, signatures):
            if matches(candidate, previous):
                duplicates[existing.track_id].append(track.track_id)
                break
        else:
            kept.append(track)
            signatures.append(candidate)
            duplicates[track.track_id] = []
    return kept, duplicates
