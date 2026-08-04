"""Strict, non-promoting contract for conservative SeedVR2 finishing."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

from PIL import Image


FINISHING_API_VERSION = "phase3-seedvr2-still-v1"
ALLOWED_SCALES = (1.5, 2.0)
MIN_STRENGTH = 0.15
MAX_STRENGTH = 0.45
DEFAULT_STRENGTH = 0.30
DEFAULT_TILE_SIZE = 768
DEFAULT_TILE_OVERLAP = 128
DEFAULT_CACHE_LIMIT_GIB = 8.0
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
PROHIBITED_OUTPUT_PARTS = {
    "approved_exports",
    "calendar",
    "instagram",
    "patreon",
    "publisher",
    "staging",
}


class RestorationContractError(ValueError):
    """Raised before loading the restoration model."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class RestorationRequest:
    asset_id: str
    source_path: Path
    source_sha256: str
    output_path: Path
    scale: float
    seed: int = 42
    strength: float = DEFAULT_STRENGTH
    tile_size: int = DEFAULT_TILE_SIZE
    tile_overlap: int = DEFAULT_TILE_OVERLAP
    mlx_cache_limit_gib: float = DEFAULT_CACHE_LIMIT_GIB
    source_review_status: str = "rejected"
    anatomy_gate_passed: bool = False
    critical_rejections: tuple[str, ...] = ()
    human_finishing_authorized: bool = False


def _validate_output_is_isolated(path: Path) -> None:
    lowered = {part.lower() for part in path.parts}
    prohibited = sorted(lowered.intersection(PROHIBITED_OUTPUT_PARTS))
    if prohibited:
        raise RestorationContractError(
            f"output_path_must_not_target_publish_or_approval_tree:{','.join(prohibited)}"
        )
    if path.suffix.lower() not in {".png", ".tif", ".tiff"}:
        raise RestorationContractError("lossless_output_format_required")


def validate_request(request: RestorationRequest) -> None:
    """Reject unsafe input before model construction or output creation."""
    if not request.asset_id.strip():
        raise RestorationContractError("asset_id_required")
    if request.source_review_status != "approved_for_finishing":
        raise RestorationContractError("only_approved_for_finishing_source_allowed")
    if not request.anatomy_gate_passed:
        raise RestorationContractError("restoration_cannot_repair_anatomy")
    if request.critical_rejections:
        raise RestorationContractError("critically_rejected_source_cannot_be_restored")
    if not request.human_finishing_authorized:
        raise RestorationContractError("explicit_human_finishing_authorization_required")
    if request.scale not in ALLOWED_SCALES:
        raise RestorationContractError("scale_must_be_1.5x_or_2x")
    if not MIN_STRENGTH <= request.strength <= MAX_STRENGTH:
        raise RestorationContractError("strength_outside_conservative_range")
    if request.seed < 0:
        raise RestorationContractError("seed_must_be_non_negative")
    if request.tile_size < 256 or request.tile_size % 16:
        raise RestorationContractError("tile_size_must_be_multiple_of_16_and_at_least_256")
    if request.tile_overlap < 32 or request.tile_overlap >= request.tile_size // 2:
        raise RestorationContractError("tile_overlap_outside_safe_range")
    if request.tile_overlap % 16:
        raise RestorationContractError("tile_overlap_must_be_multiple_of_16")
    if not 1.0 <= request.mlx_cache_limit_gib <= 8.0:
        raise RestorationContractError("mlx_cache_limit_must_be_between_1_and_8_gib")
    if not request.source_path.is_file():
        raise RestorationContractError(f"source_missing:{request.source_path}")
    if not SHA256_RE.fullmatch(request.source_sha256):
        raise RestorationContractError("source_sha256_invalid")
    if sha256_file(request.source_path) != request.source_sha256:
        raise RestorationContractError("source_sha256_mismatch")
    with Image.open(request.source_path) as source:
        if source.width < 64 or source.height < 64:
            raise RestorationContractError("source_dimensions_too_small")
    _validate_output_is_isolated(request.output_path)
    if request.output_path.exists():
        raise RestorationContractError("output_already_exists_no_overwrite")
    if request.output_path.resolve() == request.source_path.resolve():
        raise RestorationContractError("source_must_never_be_overwritten")
