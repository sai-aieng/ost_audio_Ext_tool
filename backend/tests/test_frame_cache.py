"""Exact-image reuse must not mix pixels, geometry, or mutable results."""

import numpy as np

from api.schemas.response import ExtractionResult
from pipeline.frame_cache import ExactFrameCache
from pipeline.orchestrator import _carry_results


def test_cache_rejects_changed_pixels_and_dimensions() -> None:
    cache = ExactFrameCache(2)
    image = np.zeros((8, 16), dtype=np.uint8)
    key = cache.key(image)
    cache.put(key, [])
    assert cache.get(cache.key(image.copy())) == []
    changed = image.copy()
    changed[0, 0] = 1
    assert cache.get(cache.key(changed)) is None
    assert cache.get(cache.key(image.reshape(16, 8))) is None


def test_cache_snapshots_are_isolated_and_retimed() -> None:
    cache = ExactFrameCache(2)
    key = cache.key(np.zeros((8, 16), dtype=np.uint8))
    result = ExtractionResult(
        frame_index=0, timestamp_sec=0.0, text="Label", confidence=0.9,
        bbox=[[1, 1], [10, 1], [10, 5], [1, 5]],
    )
    cache.put(key, [result])
    result.bbox[0][0] = 7
    cached = cache.get(key)
    assert cached is not None
    assert cached[0].bbox[0][0] == 1
    carried = _carry_results(cached, 21, 2.1)
    assert carried[0].timestamp_start_sec == 2.1
    assert carried[0].timestamp_end_sec == 2.1
    carried[0].bbox[0][0] = 8
    assert cache.get(key)[0].bbox[0][0] == 1
    assert cache.get(key)[0].timestamp_sec == 0.0


def test_cache_evicts_least_recently_used_and_can_be_disabled() -> None:
    cache = ExactFrameCache(2)
    keys = [cache.key(np.full((4, 4), value, dtype=np.uint8)) for value in range(3)]
    cache.put(keys[0], [])
    cache.put(keys[1], [])
    assert cache.get(keys[0]) == []
    cache.put(keys[2], [])
    assert cache.get(keys[1]) is None
    assert cache.get(keys[0]) == []
    disabled = ExactFrameCache(0)
    disabled.put(keys[0], [])
    assert disabled.get(keys[0]) is None
