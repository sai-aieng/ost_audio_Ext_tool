"""Timing fields are additive and retain the server-measured duration."""

from datetime import datetime, timedelta, timezone

from storage.job_store import to_job_status


def test_job_status_exposes_processing_times():
    start = datetime(2026, 9, 7, 14, 0, tzinfo=timezone.utc)
    status = to_job_status({
        "job_id": "timed-job", "status": "completed",
        "created_at": start - timedelta(minutes=2), "started_at": start,
        "completed_at": start + timedelta(seconds=1083.766),
        "processing_duration_sec": 1083.766,
    })
    assert status.started_at == start
    assert status.processing_duration_sec == 1083.766
    assert status.completed_at == start + timedelta(seconds=1083.766)


def test_older_records_do_not_invent_processing_duration():
    status = to_job_status({
        "job_id": "old-job", "status": "uploaded",
        "created_at": datetime.now(timezone.utc),
    })
    assert status.started_at is None
    assert status.processing_duration_sec is None
