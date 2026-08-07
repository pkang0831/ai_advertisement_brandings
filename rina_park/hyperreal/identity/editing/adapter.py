"""Non-generating adapter boundary for the future authorized pilot."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

from .contract import IdentityEditRequest, validate_request
from .readiness import require_ready


class QwenIdentityRegionEditAdapter:
    """Prepare the only allowed call shape; this module never loads the model."""

    api_version = "qwen-identity-region-edit-v1"
    model_id = "qwen-image-edit-2511-q8"
    fallback_allowed = False
    network_allowed = False
    training_allowed = False

    def prepare_call(self, request: IdentityEditRequest) -> dict[str, Any]:
        require_ready()
        validate_request(request)
        scene, master, angle = request.ordered_image_paths
        return {
            "seed": request.seed,
            "prompt": request.prompt,
            "image_path": str(scene),
            "image_paths": [str(scene), str(master), str(angle)],
            "mask_path": str(request.mask_path),
            "canvas_policy": "source-aspect",
            "width": request.width,
            "height": request.height,
            "num_inference_steps": 40,
            "guidance": 4.0,
            "negative_prompt": " ",
            "request_provenance": asdict(request),
        }

    def generate(self, request: IdentityEditRequest) -> None:
        self.prepare_call(request)
        raise RuntimeError(
            "generation_not_implemented_or_authorized; validate the combined MLX route, "
            "memory smoke, masks, LPIPS, and Phase-1 QC first"
        )


ADAPTER = QwenIdentityRegionEditAdapter()
