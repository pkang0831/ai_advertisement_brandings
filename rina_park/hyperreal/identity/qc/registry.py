"""Load and verify the pinned local MediaPipe task registry."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


QC_DIR = Path(__file__).resolve().parent
REGISTRY_PATH = QC_DIR / "model_registry.v1.json"
MODEL_DIR = QC_DIR / "models"


def load_model_registry(path: Path = REGISTRY_PATH) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verified_model_paths(
    registry_path: Path = REGISTRY_PATH,
    model_dir: Path = MODEL_DIR,
) -> dict[str, Path]:
    """Return paths only after size and SHA-256 match the pinned registry."""
    registry = load_model_registry(registry_path)
    paths: dict[str, Path] = {}
    failures: list[str] = []
    for name, record in registry["models"].items():
        path = model_dir / record["filename"]
        if not path.is_file():
            failures.append(f"{name}:missing:{path}")
            continue
        if path.stat().st_size != record["size_bytes"]:
            failures.append(f"{name}:size_mismatch")
            continue
        if file_sha256(path) != record["sha256"]:
            failures.append(f"{name}:sha256_mismatch")
            continue
        paths[name] = path
    if failures:
        raise RuntimeError("QC model registry verification failed: " + ";".join(failures))
    return paths
