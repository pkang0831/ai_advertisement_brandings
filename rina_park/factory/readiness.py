"""License, identity, ControlNet and video capability gates."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any, Iterable, Mapping

REQUIRED_COMPONENT_TYPES = {
    "checkpoint", "lora", "vae", "clip_encoder", "identity_adapter",
    "face_embedding", "controlnet", "preprocessor", "upscaler",
    "interpolation", "video_model", "video_runtime",
}
INSIGHTFACE_NONCOMMERCIAL_FAMILIES = {
    "antelope", "antelopev2", "buffalo", "buffalo_l", "buffalo_m",
    "buffalo_s", "buffalo_sc",
}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_directory(path: Path) -> str:
    """Hash model snapshots without Hugging Face's resumable-download metadata."""
    digest = hashlib.sha256()
    files = sorted(
        candidate
        for candidate in path.rglob("*")
        if candidate.is_file() and ".cache" not in candidate.relative_to(path).parts
    )
    for candidate in files:
        digest.update(candidate.relative_to(path).as_posix().encode())
        digest.update(b"\0")
        digest.update(bytes.fromhex(_sha256_file(candidate)))
    return digest.hexdigest()


def load_registry(path: str | Path) -> dict[str, Any]:
    # registry.yml is intentionally JSON-compatible YAML so the gate has no dependency.
    data = json.loads(Path(path).read_text())
    validate_registry(data)
    return data


def validate_registry(registry: Mapping[str, Any]) -> None:
    components = registry.get("components")
    if not isinstance(components, list):
        raise ValueError("registry components must be a list")
    types = {item.get("type") for item in components}
    missing = REQUIRED_COMPONENT_TYPES - types
    if missing:
        raise ValueError(f"registry missing component types: {sorted(missing)}")
    required = {
        "id", "type", "source_url", "revision", "sha256", "license",
        "commercial_use", "attribution", "reviewed_at", "approved_platform",
        "local_path",
    }
    for item in components:
        absent = required - set(item)
        if absent:
            raise ValueError(f"{item.get('id', '<unknown>')} missing {sorted(absent)}")


def license_gate(component: Mapping[str, Any], platform: bool = True) -> list[str]:
    reasons: list[str] = []
    if platform and component.get("commercial_use") is not True:
        reasons.append("commercial use is not explicitly true")
    if platform and component.get("approved_platform") is not True:
        reasons.append("platform approval is false")
    if component.get("revision") in {None, "", "UNPINNED"}:
        reasons.append("revision is not pinned")
    digest = component.get("sha256")
    if not isinstance(digest, str) or len(digest) != 64:
        reasons.append("sha256 is not pinned")
    if not component.get("license") or component.get("license") == "unknown":
        reasons.append("license is unknown")
    return reasons


def verify_local_file(component: Mapping[str, Any], model_root: str | Path) -> list[str]:
    reasons: list[str] = []
    relative = component.get("local_path")
    if not relative:
        return ["local path is not registered"]
    root = Path(model_root).resolve()
    candidate = root / relative
    if candidate.is_symlink():
        return ["model path is a symlink"]
    try:
        path = candidate.resolve(strict=True)
        path.relative_to(root)
    except (FileNotFoundError, ValueError):
        return ["model path missing or outside model root"]
    actual_digest = _sha256_directory(path) if path.is_dir() else _sha256_file(path)
    if actual_digest != component.get("sha256"):
        reasons.append("local model hash mismatch")
    return reasons


def identity_lock_readiness(
    components: Iterable[Mapping[str, Any]], model_root: str | Path,
    benchmark_passed: int, benchmark_total: int = 12,
) -> dict[str, Any]:
    identity = [
        item for item in components
        if item.get("type") in {"identity_adapter", "face_embedding"}
    ]
    reasons: list[str] = []
    if not identity:
        reasons.append("no identity component registered")
    for item in identity:
        provider = str(item.get("provider", "")).lower()
        family = str(item.get("weight_family", "")).lower()
        if provider == "insightface" and family in INSIGHTFACE_NONCOMMERCIAL_FAMILIES:
            reasons.append(
                f"{item['id']}: InsightFace-provided {family} pretrained weights "
                "are rejected for commercial platform use"
            )
            continue
        reasons.extend(f"{item['id']}: {reason}" for reason in license_gate(item))
        reasons.extend(
            f"{item['id']}: {reason}" for reason in verify_local_file(item, model_root)
        )
    if benchmark_total != 12 or benchmark_passed != 12:
        reasons.append("identity benchmark must pass exactly 12/12 scenes")
    return {"ready": not reasons, "reasons": reasons}


def controlnet_readiness(
    components: Iterable[Mapping[str, Any]], model_root: str | Path,
    benchmark_results: Mapping[str, tuple[int, int]],
) -> dict[str, Any]:
    reasons: list[str] = []
    controls = {
        str(item.get("control_type")): item
        for item in components
        if item.get("type") in {"controlnet", "preprocessor"}
    }
    for required in ("pose", "depth"):
        item = controls.get(required)
        if item is None:
            reasons.append(f"{required} component is missing")
            continue
        reasons.extend(f"{item['id']}: {reason}" for reason in license_gate(item))
        reasons.extend(
            f"{item['id']}: {reason}" for reason in verify_local_file(item, model_root)
        )
        if benchmark_results.get(required) != (12, 12):
            reasons.append(f"{required} benchmark must pass 12/12")
    return {"ready": not reasons, "reasons": reasons}


def video_readiness(
    components: Iterable[Mapping[str, Any]], architecture: str,
    e2e_clips_passed: int, frame_benchmark_passed: bool,
    control_video_format: str | None = None,
    mps_compatible: bool = False,
) -> dict[str, Any]:
    reasons: list[str] = []
    selected = [
        item for item in components
        if item.get("type") in {"video_model", "video_runtime"}
        and (
            item.get("architecture") == architecture
            or item.get("runtime") == "mps"
        )
    ]
    if len(selected) < 2:
        reasons.append("exact video model/runtime tuple is incomplete")
    for item in selected:
        reasons.extend(f"{item['id']}: {reason}" for reason in license_gate(item))
    if e2e_clips_passed != 3:
        reasons.append("three E2E clips must pass")
    if not frame_benchmark_passed:
        reasons.append("81-frame benchmark has not passed")
    if architecture == "vace":
        if not control_video_format:
            reasons.append("VACE control-video format is not pinned")
        if not mps_compatible:
            reasons.append("VACE MPS compatibility has not passed")
    return {
        "ready": not reasons,
        "reasons": reasons,
        "fallback": "ffmpeg_v0" if reasons else architecture,
        "ffmpeg_available": shutil.which("ffmpeg") is not None,
    }
