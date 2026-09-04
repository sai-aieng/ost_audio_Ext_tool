"""Tests for timestamp preservation when OCR is skipped on stable frames."""

from api.schemas.response import ExtractionResult
from pipeline.orchestrator import _carry_results


def test_carried_results_keep_the_current_sample_timestamp() -> None:
    """Stable-frame propagation must still create a 10 FPS timing observation."""

    original = ExtractionResult(
        frame_index=10,
        timestamp_sec=1.0,
        text="Stable label",
        confidence=0.9,
        bbox=[[2, 3], [12, 3], [12, 9], [2, 9]],
        bbox_xyxy=[2, 3, 12, 9],
    )

    carried = _carry_results([original], frame_index=11, timestamp_sec=1.1)

    assert len(carried) == 1
    assert carried[0].frame_index == 11
    assert carried[0].timestamp_sec == 1.1
    assert carried[0].timestamp_start_sec == 1.1
    assert carried[0].timestamp_end_sec == 1.1
    assert carried[0].bbox_end_xyxy == [2, 3, 12, 9]
