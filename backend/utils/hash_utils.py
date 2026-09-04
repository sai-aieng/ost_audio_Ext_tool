"""Perceptual hashing helpers used for frame deduplication."""

from pathlib import Path

import imagehash
from PIL import Image


def compute_phash(image_path: Path) -> imagehash.ImageHash:
    """Compute a perceptual hash for an image file."""

    with Image.open(image_path) as image:
        return imagehash.phash(image.convert("RGB"))


def hash_distance(
    first_hash: imagehash.ImageHash,
    second_hash: imagehash.ImageHash,
) -> int:
    """Return the Hamming distance between two perceptual hashes."""

    return int(first_hash - second_hash)


def are_near_duplicates(
    first_hash: imagehash.ImageHash,
    second_hash: imagehash.ImageHash,
    threshold: int,
) -> bool:
    """Return whether two hashes are within the configured distance."""

    if threshold < 0:
        raise ValueError("Hash threshold cannot be negative")
    return hash_distance(first_hash, second_hash) <= threshold
