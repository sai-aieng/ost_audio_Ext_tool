"""Select the installed Microsoft runtime before native ML libraries load."""

import ctypes
import logging
import sys
from pathlib import Path

_RUNTIME_HANDLES = []
_MIN_VERSION = (14, 40)


def _system_directory():
    buffer = ctypes.create_unicode_buffer(32768)
    if not ctypes.windll.kernel32.GetSystemDirectoryW(buffer, len(buffer)):
        raise ctypes.WinError()
    return Path(buffer.value)


def _loaded_runtime_path():
    kernel = ctypes.windll.kernel32
    kernel.GetModuleHandleW.restype = ctypes.c_void_p
    handle = kernel.GetModuleHandleW("msvcp140.dll")
    if not handle:
        return None
    buffer = ctypes.create_unicode_buffer(32768)
    if not kernel.GetModuleFileNameW(ctypes.c_void_p(handle), buffer, len(buffer)):
        raise ctypes.WinError()
    return Path(buffer.value)


def _file_version(path):
    version = ctypes.windll.version
    size = version.GetFileVersionInfoSizeW(str(path), None)
    if not size:
        raise ctypes.WinError()
    data = ctypes.create_string_buffer(size)
    if not version.GetFileVersionInfoW(str(path), 0, size, data):
        raise ctypes.WinError()
    pointer = ctypes.c_void_p()
    length = ctypes.c_uint()
    if not version.VerQueryValueW(data, chr(92), ctypes.byref(pointer), ctypes.byref(length)):
        raise ctypes.WinError()
    fields = ctypes.cast(pointer, ctypes.POINTER(ctypes.c_uint32))
    return (fields[2] >> 16, fields[2] & 0xFFFF, fields[3] >> 16, fields[3] & 0xFFFF)


def prepare_native_runtime():
    """Avoid old Anaconda MSVCP140 crashing CTranslate2 with 0xC0000005.

    No DLL files or machine settings are changed. An already loaded old runtime
    cannot safely be replaced in-process, so fail with an actionable error.
    """
    if sys.platform != "win32":
        return None
    loaded = _loaded_runtime_path()
    path = loaded or (_system_directory() / "msvcp140.dll")
    try:
        version = _file_version(path)
    except OSError as exc:
        raise RuntimeError("Install the current Microsoft Visual C++ x64 Redistributable and restart the backend.") from exc
    if version[:2] < _MIN_VERSION:
        raise RuntimeError(
            f"Incompatible Microsoft runtime {path} ({'.'.join(map(str, version))}). "
            "Restart through main:app before importing ML libraries. If this persists, "
            "update the Microsoft Visual C++ x64 Redistributable / Python environment."
        )
    if loaded is None:
        # Absolute system path, with dependencies restricted to System32.
        _RUNTIME_HANDLES.append(ctypes.WinDLL(str(path), winmode=0x00000800))
    logging.getLogger("video_ocr.runtime").info("Microsoft runtime: %s version=%s", path, version)
    return str(path)
