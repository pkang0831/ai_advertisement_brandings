#!/usr/bin/env python3
"""Fail-closed offline readiness for the pinned Z-Image Base MLX snapshot."""

from __future__ import annotations

import argparse
import contextlib
import fcntl
import gc
import hashlib
import json
import os
import resource
import socket
import time
from importlib.metadata import version
from pathlib import Path
from typing import Iterator

ROOT = Path(__file__).resolve().parents[4]
RINA = ROOT / "rina_park"
REGISTRY_PATH = RINA / "models/registry_z_image.yml"
MODEL_REPO = "mlx-community/Z-Image-bf16"
MODEL_REVISION = "450b7ef8d94fe4cda355af9215ff55db4c6db194"
BASE_MODEL_REPO = "Tongyi-MAI/Z-Image"
BASE_MODEL_REVISION = "04cc4abb7c5069926f75c9bfde9ef43d49423021"
MODEL_PATH = (
    RINA / "models/z_image/mlx-community-Z-Image-bf16-450b7ef8"
)
GPU_LOCK_PATH = Path("/tmp/rina-hyperreal-mlx-gpu.lock")
OFFLINE_ENV = {
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
    "HF_DATASETS_OFFLINE": "1",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def load_registry(path: Path = REGISTRY_PATH) -> dict[str, object]:
    # JSON is strict YAML and keeps readiness dependency-free.
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload["source"]["repo_id"] != MODEL_REPO:
        raise RuntimeError("registry source repo does not match pinned runner")
    if payload["source"]["revision"] != MODEL_REVISION:
        raise RuntimeError("registry source revision does not match pinned runner")
    if payload["base_model"]["repo_id"] != BASE_MODEL_REPO:
        raise RuntimeError("registry base model does not match pinned runner")
    if payload["base_model"]["revision"] != BASE_MODEL_REVISION:
        raise RuntimeError("registry base revision does not match pinned runner")
    if payload["license"]["weights_spdx"] != "Apache-2.0":
        raise RuntimeError("weights are not registered as Apache-2.0")
    if payload["runtime"]["code_spdx"] != "MIT":
        raise RuntimeError("runtime code license does not match pinned mlx-gen")
    return payload


def verify_artifacts(
    registry: dict[str, object],
    model_path: Path = MODEL_PATH,
    verify_hashes: bool = True,
) -> dict[str, object]:
    if not model_path.is_dir():
        raise RuntimeError(f"model directory missing: {model_path}")
    registered = registry["artifacts"]
    actual = {
        item.relative_to(model_path).as_posix(): item
        for item in model_path.rglob("*")
        if item.is_file() and ".cache" not in item.relative_to(model_path).parts
    }
    if set(actual) != set(registered):
        missing = sorted(set(registered) - set(actual))
        extra = sorted(set(actual) - set(registered))
        raise RuntimeError(
            f"artifact inventory mismatch; missing={missing}, extra={extra}"
        )
    size_bytes = 0
    for relative_path, path in actual.items():
        expected = registered[relative_path]
        size = path.stat().st_size
        size_bytes += size
        if size != expected["size_bytes"]:
            raise RuntimeError(f"size mismatch: {relative_path}")
        if verify_hashes and sha256_file(path) != expected["sha256"]:
            raise RuntimeError(f"SHA-256 mismatch: {relative_path}")
    if size_bytes != registry["size_bytes"]:
        raise RuntimeError("total model size does not match registry")
    return {
        "file_count": len(actual),
        "size_bytes": size_bytes,
        "hashes_verified": verify_hashes,
    }


@contextlib.contextmanager
def no_network() -> Iterator[None]:
    old_env = {key: os.environ.get(key) for key in OFFLINE_ENV}
    original_connect = socket.socket.connect
    original_create_connection = socket.create_connection

    def blocked(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("network access is forbidden during Z-Image inference")

    os.environ.update(OFFLINE_ENV)
    socket.socket.connect = blocked  # type: ignore[method-assign]
    socket.create_connection = blocked  # type: ignore[assignment]
    try:
        yield
    finally:
        socket.socket.connect = original_connect  # type: ignore[method-assign]
        socket.create_connection = original_create_connection
        for key, value in old_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


@contextlib.contextmanager
def sequential_gpu() -> Iterator[None]:
    with GPU_LOCK_PATH.open("w", encoding="utf-8") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def memory_snapshot() -> dict[str, float]:
    import mlx.core as mx
    import psutil

    return {
        "mlx_active_gib": round(mx.get_active_memory() / 1024**3, 3),
        "mlx_peak_gib": round(mx.get_peak_memory() / 1024**3, 3),
        "process_rss_gib": round(psutil.Process().memory_info().rss / 1024**3, 3),
        "process_max_rss_gib": round(
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**3, 3
        ),
    }


def config_readiness(verify_hashes: bool = True) -> dict[str, object]:
    registry = load_registry()
    artifacts = verify_artifacts(registry, verify_hashes=verify_hashes)
    with no_network():
        import mlx.core as mx
        from mflux.models.common.config import ModelConfig
        from mflux.models.z_image import ZImage

        model_config = ModelConfig.z_image()
    return {
        "status": "config_ready",
        "ready_for_generation": False,
        "blocker": "shared qwen_image_2512/z_image Phase-0 adapter smoke required",
        "model_repo": MODEL_REPO,
        "model_revision": MODEL_REVISION,
        "base_model_repo": BASE_MODEL_REPO,
        "base_model_revision": BASE_MODEL_REVISION,
        "model_path": str(MODEL_PATH),
        "license": registry["license"],
        "precision": registry["precision"],
        "artifacts": artifacts,
        "runtime": {
            "python": os.sys.version.split()[0],
            "mlx_gen": version("mlx-gen"),
            "mlx": version("mlx"),
            "huggingface_hub": version("huggingface-hub"),
            "code_license": registry["runtime"]["code_spdx"],
        },
        "device": str(mx.default_device()),
        "model_config": model_config.model_name,
        "runner_class": f"{ZImage.__module__}.{ZImage.__name__}",
        "network_at_inference": "blocked",
        "gpu_execution": "sequential_file_lock",
    }


def load_readiness(verify_hashes: bool = True) -> dict[str, object]:
    import mlx.core as mx
    from mflux.models.common.config import ModelConfig
    from mflux.models.z_image import ZImage

    report = config_readiness(verify_hashes=verify_hashes)
    if mx.default_device() != mx.gpu:
        raise RuntimeError(f"MLX GPU required, got {mx.default_device()}")
    mx.clear_cache()
    mx.reset_peak_memory()
    started = time.monotonic()
    model = None
    try:
        with sequential_gpu(), no_network():
            model = ZImage(
                model_config=ModelConfig.z_image(),
                model_path=str(MODEL_PATH),
            )
            # MLX is lazy: evaluate every parameter to prove all BF16 shards load.
            mx.eval(model.parameters())
            mx.synchronize()
            report.update(
                {
                    "status": "load_ready",
                    "load_seconds": round(time.monotonic() - started, 3),
                    "load_memory": memory_snapshot(),
                }
            )
    finally:
        if model is not None:
            del model
        mx.clear_cache()
        gc.collect()
        report["cleanup_memory"] = memory_snapshot()
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("config", "load"), default="config")
    parser.add_argument("--skip-hashes", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    check = load_readiness if args.mode == "load" else config_readiness
    print(json.dumps(check(verify_hashes=not args.skip_hashes), indent=2))


if __name__ == "__main__":
    main()
