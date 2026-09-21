"""Bounded per-job OCR reuse for byte-identical decoded source images."""

from collections import OrderedDict
from hashlib import blake2b

from api.schemas.response import ExtractionResult
from utils.image_utils import ImageArray


class ExactFrameCache:
    """Cache result snapshots only; no full video frames or global job state."""

    def __init__(self, capacity: int = 32) -> None:
        self.capacity = max(0, capacity)
        self.entries: OrderedDict[tuple, list[ExtractionResult]] = OrderedDict()

    @staticmethod
    def key(image: ImageArray) -> tuple:
        # Include dimensions so differently shaped images cannot share geometry.
        pixels = memoryview(image).cast("B") if image.flags.c_contiguous else image.tobytes()
        return image.shape, image.dtype.str, blake2b(pixels, digest_size=32).digest()

    def get(self, key: tuple) -> list[ExtractionResult] | None:
        results = self.entries.get(key)
        if results is not None:
            self.entries.move_to_end(key)
            return [result.model_copy(deep=True) for result in results]
        return None

    def put(self, key: tuple, results: list[ExtractionResult]) -> None:
        if self.capacity == 0:
            return
        self.entries[key] = [result.model_copy(deep=True) for result in results]
        self.entries.move_to_end(key)
        while len(self.entries) > self.capacity:
            self.entries.popitem(last=False)
