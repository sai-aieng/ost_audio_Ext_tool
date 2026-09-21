"""Check Linux linking at build time; optionally load models on a runtime host."""

import argparse
from importlib.metadata import distribution
import json
from pathlib import Path
import subprocess
import sys

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))


def check_libraries():
    # ldd resolves dependencies without invoking MediaPipe's native constructors.
    library = Path(distribution("mediapipe").locate_file("mediapipe/tasks/c/libmediapipe.so"))
    if not library.is_file():
        raise FileNotFoundError(f"MediaPipe shared library missing: {library}")
    result = subprocess.run(["ldd", str(library)], capture_output=True,
                            text=True, timeout=30, check=False)
    output = result.stdout + result.stderr
    print(output, flush=True)
    if result.returncode or "not found" in output:
        raise RuntimeError("MediaPipe shared-library dependencies are missing; see ldd output above.")
    print("MediaPipe library dependencies resolved; model execution is deferred to runtime.", flush=True)


def check_models():
    from faces.worker import open_models

    config = json.loads((BACKEND / "face_config.json").read_text(encoding="utf-8"))
    root = Path(config["model_root"])
    if not root.is_absolute():
        root = BACKEND / root
    pose, face = open_models(root, config)
    try:
        print("MediaPipe CPU models and native libraries loaded successfully", flush=True)
    finally:
        face.close()
        pose.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--libraries-only", action="store_true")
    args = parser.parse_args()
    if args.libraries_only:
        check_libraries()
    else:
        check_models()
