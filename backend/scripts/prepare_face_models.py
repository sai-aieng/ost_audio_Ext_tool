"""Download pinned official MediaPipe models; record SHA256 provenance."""

import hashlib
import json
from pathlib import Path
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1] / "models" / "faces"
MODELS = {
    "pose_landmarker_lite.task": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task",
    "blaze_face_short_range.tflite": "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite",
}


def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    manifest_path = ROOT / "source.json"
    old = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    manifest = {}
    for filename, url in MODELS.items():
        path = ROOT / filename
        if path.exists() and filename in old:
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if digest != old[filename]["sha256"]:
                raise RuntimeError(f"Model checksum mismatch: {path}")
        else:
            with urlopen(url, timeout=90) as response:
                data = response.read()
            if len(data) < 10000:
                raise RuntimeError(f"Invalid model download: {url}")
            partial = path.with_suffix(path.suffix + ".partial")
            partial.write_bytes(data)
            partial.replace(path)
            digest = hashlib.sha256(data).hexdigest()
        manifest[filename] = {"url": url, "sha256": digest, "bytes": path.stat().st_size}
        print(f"Ready: {path}", flush=True)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
