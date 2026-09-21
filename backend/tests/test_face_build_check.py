"""Build validation must catch missing libraries without executing models."""

from types import SimpleNamespace
import pytest
from scripts import check_face_runtime as checker


@pytest.mark.parametrize("returncode,output,passes", [
    (0, "libEGL.so.1 => /lib/libEGL.so.1", True),
    (0, "libEGL.so.1 => not found", False),
    (1, "ldd failed", False),
])
def test_linker_validation(tmp_path, monkeypatch, returncode, output, passes):
    library = tmp_path / "libmediapipe.so"
    library.write_bytes(b"placeholder")
    monkeypatch.setattr(checker, "distribution", lambda name: SimpleNamespace(locate_file=lambda path: library))
    calls = []
    def run(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(returncode=returncode, stdout=output, stderr="")
    monkeypatch.setattr(checker.subprocess, "run", run)
    if passes:
        checker.check_libraries()
    else:
        with pytest.raises(RuntimeError, match="dependencies are missing"):
            checker.check_libraries()
    assert calls == [["ldd", str(library)]]


def test_missing_mediapipe_library_fails(tmp_path, monkeypatch):
    monkeypatch.setattr(checker, "distribution", lambda name: SimpleNamespace(
        locate_file=lambda path: tmp_path / "missing.so"))
    with pytest.raises(FileNotFoundError):
        checker.check_libraries()
