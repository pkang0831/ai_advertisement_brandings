"""Deterministic finishing fallback with no model or learned image operation."""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

from .contract import (
    PROHIBITED_OUTPUT_PARTS,
    RestorationContractError,
    sha256_file,
)
from .fallback_validation import validate_fallback
from .runner import _atomic_json, _exclusive_lock
from .validation import LandmarkProvider, LpipsProvider


FALLBACK_API_VERSION = "phase3-non-generative-finishing-v1"
ORIGINAL_METADATA = "original_metadata_only"
LANCZOS_15 = "lanczos_1.5x"
LANCZOS_20 = "lanczos_2x"
CONSERVATIVE_15 = "conservative_1.5x"
CONSERVATIVE_20 = "conservative_2x"
ALL_PRESETS = (
    ORIGINAL_METADATA,
    LANCZOS_15,
    LANCZOS_20,
    CONSERVATIVE_15,
    CONSERVATIVE_20,
)
PIXEL_PRESETS = {
    LANCZOS_15: 1.5,
    LANCZOS_20: 2.0,
    CONSERVATIVE_15: 1.5,
    CONSERVATIVE_20: 2.0,
}
MATURE_PATH_MARKERS = ("mature", "nsfw", "adult_only")


@dataclass(frozen=True)
class NonGenerativeFinishingRequest:
    asset_id: str
    source_path: Path
    source_sha256: str
    metadata_path: Path
    preset: str = ORIGINAL_METADATA
    output_path: Path | None = None
    source_review_status: str = "rejected"
    source_media_class: str = "unknown"
    anatomy_gate_passed: bool = False
    critical_rejections: tuple[str, ...] = ()
    human_finishing_authorized: bool = False


def _path_has_marker(path: Path, markers: tuple[str, ...]) -> bool:
    lowered = "/".join(part.lower() for part in path.parts)
    return any(marker in lowered for marker in markers)


def _validate_isolated_path(path: Path, suffixes: set[str]) -> None:
    lowered = {part.lower() for part in path.parts}
    if lowered.intersection(PROHIBITED_OUTPUT_PARTS):
        raise RestorationContractError("fallback_path_targets_publish_tree")
    if path.suffix.lower() not in suffixes:
        raise RestorationContractError("fallback_path_suffix_invalid")


def validate_fallback_request(request: NonGenerativeFinishingRequest) -> None:
    if request.preset not in ALL_PRESETS:
        raise RestorationContractError("unknown_non_generative_preset")
    if request.source_review_status != "approved_for_finishing":
        raise RestorationContractError("only_approved_for_finishing_source_allowed")
    if request.source_media_class != "standard_sfw":
        raise RestorationContractError("mature_or_unknown_media_cannot_enter_finishing")
    if _path_has_marker(request.source_path, MATURE_PATH_MARKERS):
        raise RestorationContractError("mature_path_cannot_enter_finishing")
    if not request.anatomy_gate_passed:
        raise RestorationContractError("finishing_cannot_repair_anatomy")
    if request.critical_rejections:
        raise RestorationContractError("critically_rejected_source_cannot_be_finished")
    if not request.human_finishing_authorized:
        raise RestorationContractError("explicit_human_finishing_authorization_required")
    if not request.source_path.is_file():
        raise RestorationContractError("finishing_source_missing")
    if sha256_file(request.source_path) != request.source_sha256:
        raise RestorationContractError("finishing_source_sha256_mismatch")
    _validate_isolated_path(request.metadata_path, {".json"})
    if request.metadata_path.exists():
        raise RestorationContractError("finishing_metadata_already_exists")
    if request.preset == ORIGINAL_METADATA:
        if request.output_path is not None:
            raise RestorationContractError("metadata_only_preset_must_not_write_pixels")
    else:
        if request.output_path is None:
            raise RestorationContractError("pixel_preset_requires_output_path")
        _validate_isolated_path(request.output_path, {".png", ".tif", ".tiff"})
        if request.output_path.exists():
            raise RestorationContractError("finishing_output_already_exists")
        if request.output_path.resolve() == request.source_path.resolve():
            raise RestorationContractError("source_must_never_be_overwritten")


def _lanczos(source: Image.Image, scale: float) -> Image.Image:
    width = max(1, round(source.width * scale))
    height = max(1, round(source.height * scale))
    return source.convert("RGB").resize((width, height), Image.Resampling.LANCZOS)


def _bounded_local_contrast(image: Image.Image) -> Image.Image:
    rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    clahe = cv2.createCLAHE(clipLimit=1.05, tileGridSize=(8, 8))
    adjusted = lab.copy()
    adjusted[:, :, 0] = clahe.apply(lab[:, :, 0])
    corrected = cv2.cvtColor(adjusted, cv2.COLOR_LAB2RGB)
    blended = np.clip(
        rgb.astype(np.float32) * 0.97 + corrected.astype(np.float32) * 0.03,
        0,
        255,
    ).astype(np.uint8)
    return Image.fromarray(blended, mode="RGB")


def apply_preset(source_path: Path, preset: str) -> Image.Image | None:
    """Apply only deterministic classical pixel operations."""
    if preset == ORIGINAL_METADATA:
        return None
    scale = PIXEL_PRESETS[preset]
    with Image.open(source_path) as source:
        output = _lanczos(source, scale)
    if preset in {LANCZOS_15, LANCZOS_20}:
        return output
    output = output.filter(ImageFilter.UnsharpMask(radius=0.6, percent=10, threshold=5))
    output = _bounded_local_contrast(output)
    output = ImageEnhance.Color(output).enhance(1.005)
    return output


def run_non_generative_finishing(
    request: NonGenerativeFinishingRequest,
    *,
    landmark_provider: LandmarkProvider | None,
    lpips_provider: LpipsProvider | None,
    readiness_report: dict[str, object],
) -> dict[str, object]:
    validate_fallback_request(request)
    if landmark_provider is None or lpips_provider is None:
        raise RuntimeError("fallback_validation_providers_required")
    if readiness_report.get("fallback_execution_ready") is not True:
        raise RuntimeError("non_generative_fallback_readiness_gate_is_closed")
    fallback = readiness_report.get("fallback", {})
    enabled = fallback.get("enabled_presets", []) if isinstance(fallback, dict) else []
    if request.preset not in enabled:
        raise RuntimeError("non_generative_preset_not_enabled")

    started = time.perf_counter()
    metadata: dict[str, object] = {
        "schema_version": "1.0.0",
        "api_version": FALLBACK_API_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "asset_id": request.asset_id,
        "source_path": str(request.source_path),
        "source_sha256": request.source_sha256,
        "preset": request.preset,
        "source_review_status": request.source_review_status,
        "source_media_class": request.source_media_class,
        "anatomy_gate_passed": request.anatomy_gate_passed,
        "generative_model_used": False,
        "learned_image_operation_used": False,
        "publication_allowed": False,
        "automatic_promotion_allowed": False,
    }
    output = None
    try:
        with _exclusive_lock():
            output = apply_preset(request.source_path, request.preset)
            if output is not None:
                assert request.output_path is not None
                request.output_path.parent.mkdir(parents=True, exist_ok=True)
                output.save(request.output_path, format="PNG", compress_level=9)
            validation = validate_fallback(
                request.source_path,
                request.output_path,
                preset=request.preset,
                landmark_provider=landmark_provider,
                lpips_provider=lpips_provider,
            )
            metadata["validation"] = validation
            if validation["passed"] is not True:
                if request.output_path is not None:
                    request.output_path.unlink(missing_ok=True)
                metadata["status"] = "rejected_and_pixels_deleted"
                metadata["output_sha256"] = None
            elif request.preset == ORIGINAL_METADATA:
                metadata["status"] = "metadata_only_source_preserved"
                metadata["output_sha256"] = None
            else:
                assert request.output_path is not None
                metadata["status"] = "pending_100_percent_human_review"
                metadata["output_sha256"] = sha256_file(request.output_path)
    finally:
        if output is not None:
            output.close()
        metadata["wall_seconds"] = round(time.perf_counter() - started, 6)
        _atomic_json(request.metadata_path, metadata)
    return metadata
