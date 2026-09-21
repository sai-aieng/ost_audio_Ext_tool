"""Fail image builds if MediaPipe native libraries or face models cannot load."""

import json
from pathlib import Path
import sys

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))

from faces.worker import open_models


def main():
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
    main()
