"""Regression coverage for early, non-destructive Windows runtime selection."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from utils import windows_runtime as runtime


def setup_windows(monkeypatch, loaded=None, version=(14, 50, 35719, 0)):
    monkeypatch.setattr(runtime, "sys", SimpleNamespace(platform="win32"))
    monkeypatch.setattr(runtime, "_loaded_runtime_path", lambda: loaded)
    monkeypatch.setattr(runtime, "_system_directory", lambda: Path("C:/Windows/System32"))
    monkeypatch.setattr(runtime, "_file_version", lambda path: version)
    monkeypatch.setattr(runtime, "_RUNTIME_HANDLES", [])


def test_non_windows_is_unchanged(monkeypatch):
    monkeypatch.setattr(runtime, "sys", SimpleNamespace(platform="linux"))
    assert runtime.prepare_native_runtime() is None


def test_loads_system_runtime_with_restricted_dependency_search(monkeypatch):
    setup_windows(monkeypatch)
    calls = []
    monkeypatch.setattr(runtime.ctypes, "WinDLL", lambda path, **kwargs: calls.append((path, kwargs)), raising=False)
    path = runtime.prepare_native_runtime()
    assert Path(path) == Path("C:/Windows/System32/msvcp140.dll")
    assert calls == [(path, {"winmode": 0x800})]
    assert len(runtime._RUNTIME_HANDLES) == 1


def test_rejects_already_loaded_old_runtime(monkeypatch):
    setup_windows(monkeypatch, Path("anaconda3/msvcp140.dll"), (14, 27, 29016, 0))
    with pytest.raises(RuntimeError, match="Incompatible Microsoft runtime"):
        runtime.prepare_native_runtime()


def test_reuses_compatible_loaded_runtime(monkeypatch):
    setup_windows(monkeypatch, Path("runtime/msvcp140.dll"))
    assert runtime.prepare_native_runtime() == str(Path("runtime/msvcp140.dll"))
    assert runtime._RUNTIME_HANDLES == []


def test_missing_system_runtime_is_actionable(monkeypatch):
    setup_windows(monkeypatch)
    def missing(path):
        raise OSError("missing DLL")
    monkeypatch.setattr(runtime, "_file_version", missing)
    with pytest.raises(RuntimeError, match="Visual C\\+\\+ x64"):
        runtime.prepare_native_runtime()
