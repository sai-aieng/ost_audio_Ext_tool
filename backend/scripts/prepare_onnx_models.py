"""Prepare official ONNX exports, or convert cached weights; never runs inference."""

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.request import urlopen

import yaml

BACKEND = Path(__file__).resolve().parents[1]
OFFICIAL_MODELS = {
    "PP-OCRv6_small_det": (
        "28fe5895c24fd108c19eb3e8479f4ab385fbfc62",
        "d73e0058b7a8086bbd57f3d10b8bcd4ff95363f67e06e2762b5e814fe9c9410e",
    ),
    "PP-OCRv6_small_rec": (
        "b8f84f0b80c529de40b4fbb3544b84fa7233a513",
        "5435fd747c9e0efe15a96d0b378d5bd157e9492ed8fd80edf08f30d02fa24634",
    ),
    "PP-LCNet_x1_0_textline_ori": (
        "7fdcf3cf7061163eda7183b224aa334bd33068f7",
        "38aa97cd4be591e0ad304e659f07ba30d946f27a63315433f6659c69c8778345",
    ),
}


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def check_config(name, directory, source_root):
    downloaded = yaml.safe_load((directory / "inference.yml").read_text(encoding="utf-8"))
    if downloaded.get("Global", {}).get("model_name") != name:
        raise ValueError(f"Model identity mismatch: {name}")
    source = source_root / name / "inference.yml"
    if source.is_file():
        original = yaml.safe_load(source.read_text(encoding="utf-8"))
        for section in ("PreProcess", "PostProcess"):
            if downloaded.get(section) != original.get(section):
                raise ValueError(f"Official export changes {section}: {name}")


def download_official(name, source_root, output_root):
    revision, expected_hash = OFFICIAL_MODELS[name]
    destination = output_root / name
    if destination.exists():
        check_config(name, destination, source_root)
        if digest(destination / "inference.onnx") != expected_hash:
            raise ValueError(f"Existing model checksum differs; not overwriting {destination}")
        print(f"Already prepared: {name}", flush=True)
        return
    repo = f"PaddlePaddle/{name}_onnx"
    with TemporaryDirectory(prefix="download-", dir=output_root) as working:
        staging = Path(working)
        for filename in ("inference.onnx", "inference.yml"):
            url = f"https://huggingface.co/{repo}/resolve/{revision}/{filename}"
            print(f"Downloading official {name}/{filename}", flush=True)
            with urlopen(url, timeout=60) as response, (staging / filename).open("wb") as target:
                shutil.copyfileobj(response, target)
        if digest(staging / "inference.onnx") != expected_hash:
            raise ValueError(f"Official model checksum mismatch: {name}")
        check_config(name, staging, source_root)
        (staging / "source.json").write_text(json.dumps({
            "repository": repo, "revision": revision, "sha256": expected_hash,
            "note": "Official export; numerical equivalence requires manual comparison.",
        }, indent=2), encoding="utf-8")
        ready = staging / "ready"
        ready.mkdir()
        for artifact in ("inference.onnx", "inference.yml", "source.json"):
            shutil.move(str(staging / artifact), str(ready / artifact))
        ready.rename(destination)
    print(f"Prepared and checksum verified: {name}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path,
                        default=Path.home() / ".paddlex" / "official_models")
    parser.add_argument("--output-root", type=Path, default=BACKEND / "models" / "onnx")
    parser.add_argument("--convert-cached", action="store_true",
                        help="Use local Paddle2ONNX instead (requires a compatible converter)")
    args = parser.parse_args()
    config = yaml.safe_load((BACKEND / "config.yaml").read_text(encoding="utf-8"))
    names = [
        config["ocr"]["text_detection_model_name"],
        config["ocr"]["text_recognition_model_name"],
        "PP-LCNet_x1_0_textline_ori",
    ]
    args.output_root.mkdir(parents=True, exist_ok=True)
    if not args.convert_cached:
        for name in dict.fromkeys(names):
            if name not in OFFICIAL_MODELS:
                raise ValueError(f"No pinned official export for {name}; use --convert-cached")
            download_official(name, args.source_root, args.output_root)
        return
    for name in dict.fromkeys(names):
        source = args.source_root / name
        files = ["inference.json", "inference.pdiparams", "inference.yml"]
        for filename in files:
            if not (source / filename).is_file():
                raise FileNotFoundError(f"Missing cached model file: {source / filename}")
        manifest = {"model": name, "precision": "fp32",
                    "sources": {filename: digest(source / filename) for filename in files}}
        destination = args.output_root / name
        if destination.exists():
            saved = destination / "conversion.json"
            if (saved.is_file() and (destination / "inference.onnx").is_file()
                    and (destination / "inference.yml").is_file()
                    and json.loads(saved.read_text(encoding="utf-8")) == manifest):
                print(f"Already converted: {name}", flush=True)
                continue
            raise FileExistsError(f"Refusing to replace existing model directory: {destination}")
        # Publish only complete conversions; originals are always left untouched.
        with TemporaryDirectory(prefix="convert-", dir=args.output_root) as working:
            staging = Path(working)
            print(f"Converting {name} to FP32 ONNX (no inference)", flush=True)
            subprocess.run([
                sys.executable, "-m", "paddle2onnx.command",
                "--model_dir", str(source), "--model_filename", "inference.json",
                "--params_filename", "inference.pdiparams",
                "--save_file", str(staging / "inference.onnx"),
                "--opset_version", "17", "--optimize_tool", "None",
            ], check=True)
            shutil.copy2(source / "inference.yml", staging / "inference.yml")
            (staging / "conversion.json").write_text(
                json.dumps(manifest, indent=2), encoding="utf-8")
            # Rename a complete folder into place on the same filesystem.
            ready = staging / "ready"
            ready.mkdir()
            for artifact in ("inference.onnx", "inference.yml", "conversion.json"):
                shutil.move(str(staging / artifact), str(ready / artifact))
            ready.rename(destination)
        print(f"Prepared: {destination}", flush=True)


if __name__ == "__main__":
    main()
