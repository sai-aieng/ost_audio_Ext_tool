"""Download the selected converted Whisper model without running transcription."""

import json
from pathlib import Path

import yaml
from huggingface_hub import HfApi, snapshot_download


def main():
    backend = Path(__file__).resolve().parents[1]
    config = yaml.safe_load((backend / "config.yaml").read_text(encoding="utf-8"))["audio_transcription"]
    model = str(config["model"])
    if model not in {"tiny", "tiny.en", "base", "base.en", "small", "small.en", "medium", "medium.en"}:
        raise ValueError("This setup script supports standard tiny/base/small/medium Whisper models")
    root = Path(config.get("faster_model_root", "models/faster-whisper"))
    if not root.is_absolute():
        root = backend / root
    destination = root / model
    repo = f"Systran/faster-whisper-{model}"
    manifest = destination / "source.json"
    if manifest.is_file():
        revision = json.loads(manifest.read_text(encoding="utf-8"))["revision"]
    else:
        revision = HfApi().model_info(repo).sha
    print(f"Preparing {repo} revision={revision}; no transcription", flush=True)
    snapshot_download(
        repo_id=repo, revision=revision, local_dir=str(destination),
        allow_patterns=["model.bin", "config.json", "tokenizer.json", "vocabulary.*",
                        "preprocessor_config.json"],
    )
    for filename in ("model.bin", "config.json", "tokenizer.json"):
        if not (destination / filename).is_file():
            raise FileNotFoundError(f"Model download incomplete: {filename}")
    manifest.write_text(json.dumps({"repository": repo, "revision": revision}, indent=2),
                        encoding="utf-8")
    print(f"Model ready at {destination}", flush=True)


if __name__ == "__main__":
    main()
