"""Read-only readiness probes for the exact official 2604 checkpoint."""

from __future__ import annotations

import hashlib
import importlib
import importlib.metadata
import json
import platform
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

REGISTRY_PATH = (
    Path(__file__).resolve().parents[3] / "models" / "registry_hidream_o1.yml"
)
EXPECTED_REPO = "HiDream-ai/HiDream-O1-Image-Dev-2604"
EXPECTED_REVISION = "b6acc2fe452b3120430620dc4354fa442ee081ea"
EXPECTED_ARCHITECTURE = "Qwen3VLForConditionalGeneration"


@dataclass(frozen=True)
class Check:
    name: str
    passed: bool
    detail: str


@dataclass(frozen=True)
class ReadinessReport:
    status: str
    can_download: bool
    can_load: bool
    can_generate: bool
    model_path: str
    free_disk_bytes: int
    duplicate_cache_paths: tuple[str, ...]
    checks: tuple[Check, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True)


def load_registry(path: Path = REGISTRY_PATH) -> dict[str, Any]:
    """The fragment is JSON-form YAML, parsed without an optional YAML package."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    component = payload["component"]
    if component["repo_id"] != EXPECTED_REPO:
        raise ValueError("unexpected HiDream repository; substitution refused")
    if component["revision"] != EXPECTED_REVISION:
        raise ValueError("unexpected HiDream revision; substitution refused")
    return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact_checks(model_path: Path, files: dict[str, Any]) -> list[Check]:
    if not model_path.exists():
        return [Check("artifact_present", False, "pinned artifact is not downloaded")]

    checks: list[Check] = []
    for relative_path, expected in sorted(files.items()):
        candidate = model_path / relative_path
        if not candidate.is_file():
            checks.append(Check(f"file:{relative_path}", False, "missing"))
            continue
        if candidate.stat().st_size != expected["size_bytes"]:
            checks.append(Check(f"file:{relative_path}", False, "size mismatch"))
            continue
        actual = _sha256(candidate)
        checks.append(
            Check(
                f"file:{relative_path}",
                actual == expected["sha256"],
                "sha256 verified" if actual == expected["sha256"] else "sha256 mismatch",
            )
        )
    return checks


def _import_check(module: str) -> Check:
    try:
        imported = importlib.import_module(module)
    except Exception as exc:  # pragma: no cover - environment-specific
        return Check(f"import:{module}", False, f"{type(exc).__name__}: {exc}")
    version = getattr(imported, "__version__", "unknown")
    return Check(f"import:{module}", True, f"version {version}")


def _transformers_architecture_check() -> Check:
    try:
        transformers = importlib.import_module("transformers")
        architecture = getattr(transformers, EXPECTED_ARCHITECTURE)
    except Exception as exc:
        return Check(
            "config_architecture_import",
            False,
            f"{EXPECTED_ARCHITECTURE} unavailable: {type(exc).__name__}: {exc}",
        )
    return Check(
        "config_architecture_import",
        architecture.__name__ == EXPECTED_ARCHITECTURE,
        f"resolved {architecture.__module__}.{architecture.__name__}",
    )


def _official_dependency_check() -> Check:
    try:
        version = importlib.metadata.version("transformers")
    except importlib.metadata.PackageNotFoundError:
        version = "not installed"
    return Check(
        "official_dependency_pin",
        version == "4.57.1",
        f"installed transformers {version}; official code pins 4.57.1",
    )


def inspect_readiness(registry_path: Path = REGISTRY_PATH) -> ReadinessReport:
    registry = load_registry(registry_path)
    component = registry["component"]
    models_root = registry_path.parent
    model_path = models_root / component["local_path"]
    free_disk = shutil.disk_usage(models_root).free
    hf_cache = (
        Path.home()
        / ".cache"
        / "huggingface"
        / "hub"
        / "models--HiDream-ai--HiDream-O1-Image-Dev-2604"
    )
    duplicates = tuple(
        str(path)
        for path in (model_path, hf_cache)
        if path.exists()
    )

    checks = [
        Check("repo_pin", component["repo_id"] == EXPECTED_REPO, EXPECTED_REPO),
        Check(
            "revision_pin",
            component["revision"] == EXPECTED_REVISION,
            EXPECTED_REVISION,
        ),
        Check(
            "license",
            component["license"] == "MIT" and component["code_license"] == "MIT",
            "official model card and code repository declare MIT",
        ),
        Check(
            "external_text_encoders",
            not component["text_encoder_licenses"]["required_external_encoders"],
            "none; optional Gemma prompt refiner is excluded",
        ),
        Check(
            "disk_capacity",
            free_disk >= component["size_bytes"] * 2,
            f"{free_disk} bytes free; two-artifact safety floor",
        ),
        Check(
            "apple_runtime_exact_official",
            False,
            "no verified MPS/MLX path for the exact official 2604 shards",
        ),
        _import_check("torch"),
        _import_check("transformers"),
        _transformers_architecture_check(),
        _official_dependency_check(),
        Check(
            "official_model_load",
            False,
            "not attempted: official code requires CUDA and artifact is absent",
        ),
        Check(
            "host",
            platform.machine() == "arm64",
            f"{platform.platform()} ({platform.machine()})",
        ),
    ]
    checks.extend(_artifact_checks(model_path, component["files"]))

    # Platform approval is an independent hard gate even if files later appear.
    can_download = bool(
        component["download_approved"]
        and component["approved_platform"]
        and checks[4].passed
    )
    artifacts_valid = all(
        check.passed for check in checks if check.name.startswith("file:")
    ) and any(check.name.startswith("file:") for check in checks)
    can_load = bool(
        component["approved_platform"]
        and artifacts_valid
        and next(c for c in checks if c.name == "apple_runtime_exact_official").passed
    )
    return ReadinessReport(
        status=(
            "ready_for_model_load"
            if can_load
            else "blocked_apple_exact_official_path_unverified"
        ),
        can_download=can_download,
        can_load=can_load,
        can_generate=False,
        model_path=str(model_path),
        free_disk_bytes=free_disk,
        duplicate_cache_paths=duplicates,
        checks=tuple(checks),
    )
