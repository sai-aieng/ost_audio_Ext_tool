"""Perceptual-hash based near-duplicate frame filtering."""

from pipeline.frame_extractor import FramePath
from utils.hash_utils import are_near_duplicates, compute_phash


def filter_frames(
    frames: list[FramePath],
    threshold: int,
    enabled: bool = True,
) -> list[FramePath]:
    """Keep frames whose pHash differs from every previously accepted frame."""

    if not enabled or len(frames) < 2:
        return list(frames)
    if threshold < 0:
        raise ValueError("Hash threshold cannot be negative")

    accepted: list[FramePath] = []
    accepted_hashes = []
    for frame in frames:
        current_hash = compute_phash(frame.path)
        if any(
            are_near_duplicates(current_hash, previous_hash, threshold)
            for previous_hash in accepted_hashes
        ):
            continue
        accepted.append(frame)
        accepted_hashes.append(current_hash)
    return accepted
