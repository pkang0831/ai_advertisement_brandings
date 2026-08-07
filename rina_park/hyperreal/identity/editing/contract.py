"""Strict request contract for scene-preserving Qwen identity edits."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

from PIL import Image


EDIT_API_VERSION = "qwen-identity-region-edit-v1"
PRESERVATION_THRESHOLD = 0.02
MODEL_REPO = "AbstractFramework/qwen-image-edit-2511-8bit"
MODEL_REVISION = "5f35885dfe4061e57c87df2c03123e5e8124edfa"


class EditContractError(ValueError):
    """Raised before model loading when an edit request violates the lane."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class IdentityEditRequest:
    pilot_id: str
    seed: int
    width: int
    height: int
    scene_path: Path
    scene_sha256: str
    master_path: Path
    master_sha256: str
    angle_reference_path: Path
    angle_reference_sha256: str
    mask_path: Path
    mask_sha256: str
    prompt: str
    phase1_qc_ready: bool
    lpips_provider_ready: bool

    @property
    def ordered_image_paths(self) -> tuple[Path, Path, Path]:
        """Picture 1 is always the scene; Pictures 2–3 are identity-only references."""
        return self.scene_path, self.master_path, self.angle_reference_path


def build_identity_prompt(*, angle_instruction: str) -> str:
    return (
        "Picture 1 is the complete realistic scene and the immutable composition source. "
        "Picture 2 is the approved master identity reference. Picture 3 is an internally "
        "approved matching-angle Qwen identity reference. Apply the face and hair identity "
        "from Pictures 2 and 3 only inside the supplied face-and-hair mask in Picture 1. "
        f"{angle_instruction.strip()} "
        "Preserve Picture 1 exactly outside that region: pose, body geometry, hands, wardrobe, "
        "props, background, lighting and shadows, reflections, lens perspective, depth of field, "
        "motion blur, sensor noise, color response, compression, and pixel structure. Do not "
        "expand a headshot into a body or scene. Do not copy the master/reference background, "
        "wardrobe, neck, shoulders, pose, or camera. Replace the source scene face; do not blend, "
        "retain, or leak its identity. Produce no text or watermark."
    )


def _validate_hash(path: Path, expected: str, label: str) -> None:
    if len(expected) != 64:
        raise EditContractError(f"{label}_sha256_invalid")
    if not path.is_file():
        raise EditContractError(f"{label}_missing:{path}")
    actual = sha256_file(path)
    if actual != expected:
        raise EditContractError(f"{label}_sha256_mismatch:{actual}")


def validate_request(request: IdentityEditRequest) -> None:
    """Validate every non-generative gate; fail on any ambiguity."""
    if request.seed < 0:
        raise EditContractError("seed_must_be_non_negative")
    if request.width <= 0 or request.height <= 0:
        raise EditContractError("output_dimensions_invalid")
    if len(request.ordered_image_paths) != 3:
        raise EditContractError("exactly_three_images_required")
    if request.ordered_image_paths != (
        request.scene_path,
        request.master_path,
        request.angle_reference_path,
    ):
        raise EditContractError("input_order_must_be_scene_master_angle_reference")
    if len(set(request.ordered_image_paths)) != 3:
        raise EditContractError("scene_and_identity_inputs_must_be_distinct")
    for path, expected, label in (
        (request.scene_path, request.scene_sha256, "scene"),
        (request.master_path, request.master_sha256, "master"),
        (request.angle_reference_path, request.angle_reference_sha256, "angle_reference"),
        (request.mask_path, request.mask_sha256, "mask"),
    ):
        _validate_hash(path, expected, label)
    with Image.open(request.scene_path) as scene:
        if scene.size != (request.width, request.height):
            raise EditContractError(
                f"requested_dimensions_must_equal_scene:{scene.size}!={(request.width, request.height)}"
            )
    required_prompt_phrases = (
        "Picture 1 is the complete realistic scene",
        "Picture 2 is the approved master identity reference",
        "Picture 3 is an internally approved matching-angle Qwen identity reference",
        "only inside the supplied face-and-hair mask",
        "Preserve Picture 1 exactly outside that region",
        "Do not expand a headshot into a body or scene",
        "do not blend, retain, or leak its identity",
    )
    if any(phrase not in request.prompt for phrase in required_prompt_phrases):
        raise EditContractError("prompt_missing_required_identity_region_contract")
    if not request.phase1_qc_ready:
        raise EditContractError("phase1_qc_runtime_not_ready")
    if not request.lpips_provider_ready:
        raise EditContractError("mask_exterior_lpips_provider_not_ready")
