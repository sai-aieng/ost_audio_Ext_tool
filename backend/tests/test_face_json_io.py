import errno
import json
from unittest.mock import patch

import pytest

from faces import json_io


def test_temporary_lock_retries_and_publishes_complete_json(tmp_path):
    target = tmp_path / "progress.json"
    target.write_text('{"old": true}', encoding="utf-8")
    replace = json_io.os.replace
    attempts = []

    def locked_once(source, destination):
        attempts.append(source)
        assert json.loads(target.read_text()) == {"old": True}
        if len(attempts) == 1:
            raise PermissionError(13, "locked")
        replace(source, destination)

    with patch.object(json_io.os, "replace", side_effect=locked_once), patch.object(json_io.time, "sleep") as sleep:
        json_io.write_json(target, {"new": True})
    assert len(attempts) == 2
    sleep.assert_called_once_with(0.05)
    assert json.loads(target.read_text()) == {"new": True}
    assert not list(tmp_path.glob("*.partial"))


def test_persistent_progress_lock_is_nonfatal_and_next_update_recovers(tmp_path, caplog):
    target = tmp_path / "progress.json"
    target.write_text('{"frames": 1}', encoding="utf-8")
    with patch.object(json_io.os, "replace", side_effect=PermissionError(13, "locked")) as replace, patch.object(json_io.time, "sleep"):
        assert json_io.write_progress(target, {"frames": 2}) is False
        assert replace.call_count == 5
    assert json.loads(target.read_text()) == {"frames": 1}
    assert "extraction continues" in caplog.text
    assert json_io.write_progress(target, {"frames": 3}) is True
    assert json.loads(target.read_text()) == {"frames": 3}
    assert not list(tmp_path.glob("*.partial"))


def test_final_results_failure_is_not_hidden(tmp_path):
    with patch.object(json_io.os, "replace", side_effect=PermissionError(13, "locked")) as replace, patch.object(json_io.time, "sleep"):
        with pytest.raises(PermissionError):
            json_io.write_json(tmp_path / "faces.json", {"faces": []})
        assert replace.call_count == 5
    assert not (tmp_path / "faces.json").exists()
    assert not list(tmp_path.glob("*.partial"))


def test_other_io_errors_are_not_retried(tmp_path):
    with patch.object(json_io.os, "replace", side_effect=OSError(errno.ENOSPC, "disk full")), patch.object(json_io.time, "sleep") as sleep:
        assert json_io.write_progress(tmp_path / "progress.json", {}) is False
        with pytest.raises(OSError):
            json_io.write_json(tmp_path / "faces.json", {})
        sleep.assert_not_called()


def test_invalid_payload_is_not_silently_ignored(tmp_path):
    with pytest.raises(ValueError):
        json_io.write_progress(tmp_path / "progress.json", {"value": float("nan")})
