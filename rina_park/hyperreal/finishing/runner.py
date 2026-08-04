"""Offline, serialized SeedVR2 still-image runner.

Importing this module does not load MLX or model weights. Execution remains
separately gated by readiness and explicit human authorization.
"""

from __future__ import annotations

import fcntl
import gc
import json
import os
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

from PIL import Image

from .contract import FINISHING_API_VERSION, RestorationRequest, sha256_file, validate_request
from .validation import LandmarkProvider, LpipsProvider, validate_restoration


ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = (
    ROOT
    / "models/seedvr2/AbstractFramework-seedvr2-3b-8bit-6e6b162d"
)
REGISTRY_PATH = ROOT / "models/registry_seedvr2.yml"
LOCK_PATH = ROOT / "private/locks/seedvr2-still.lock"


@contextmanager
def _exclusive_lock() -> Iterator[None]:
    LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOCK_PATH.open("w", encoding="utf-8") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("another_unified_memory_job_is_running") from error
        handle.write(f"{os.getpid()}\n")
        handle.flush()
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _offline_environment() -> None:
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"


def _atomic_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _blend_conservatively(
    source_path: Path,
    restored: Image.Image,
    *,
    strength: float,
) -> Image.Image:
    with Image.open(source_path) as source:
        baseline = source.convert("RGB").resize(restored.size, Image.Resampling.LANCZOS)
    return Image.blend(baseline, restored.convert("RGB"), strength)


def run_restoration(
    request: RestorationRequest,
    *,
    landmark_provider: LandmarkProvider | None,
    lpips_provider: LpipsProvider | None,
    readiness_report: dict[str, object],
) -> dict[str, object]:
    """Run one private candidate and never promote it automatically."""
    validate_request(request)
    if landmark_provider is None or lpips_provider is None:
        raise RuntimeError("identity_and_lpips_validators_required_before_inference")
    if readiness_report.get("production_execution_ready") is not True:
        raise RuntimeError("seedvr2_production_readiness_gate_is_closed")
    allowed_scales = readiness_report.get("calibration", {}).get(
        "enabled_scales",
        [],
    )
    if f"{request.scale:g}x" not in allowed_scales:
        raise RuntimeError("requested_scale_is_not_calibration_enabled")
    if not MODEL_PATH.is_dir():
        raise RuntimeError("pinned_seedvr2_model_missing")

    request.output_path.parent.mkdir(parents=True, exist_ok=True)
    sidecar_path = request.output_path.with_suffix(".restoration.json")
    started = time.perf_counter()
    metadata: dict[str, object] = {
        "schema_version": "1.0.0",
        "api_version": FINISHING_API_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "asset_id": request.asset_id,
        "source_path": str(request.source_path),
        "source_sha256": request.source_sha256,
        "output_path": str(request.output_path),
        "model_path": str(MODEL_PATH),
        "registry_path": str(REGISTRY_PATH),
        "scale": request.scale,
        "seed": request.seed,
        "conservative_blend_strength": request.strength,
        "tile_size": request.tile_size,
        "tile_overlap": request.tile_overlap,
        "mlx_cache_limit_gib": request.mlx_cache_limit_gib,
        "source_review_status": request.source_review_status,
        "anatomy_gate_passed": request.anatomy_gate_passed,
        "publication_allowed": False,
        "automatic_promotion_allowed": False,
        "status": "running_private_validation",
    }

    model = None
    try:
        with _exclusive_lock():
            _offline_environment()
            import mlx.core as mx
            from mflux.models.common.config.model_config import ModelConfig
            from mflux.models.common.vae.tiling_config import TilingConfig
            from mflux.models.seedvr2.variants.upscale.seedvr2 import SeedVR2
            from mflux.utils.scale_factor import ScaleFactor

            mx.set_cache_limit(int(request.mlx_cache_limit_gib * 1024**3))
            mx.reset_peak_memory()
            model = SeedVR2(
                model_path=str(MODEL_PATH),
                model_config=ModelConfig.seedvr2_3b(),
            )
            model.tiling_config = TilingConfig(
                vae_encode_tile_size=request.tile_size,
                vae_encode_tile_overlap=request.tile_overlap,
                vae_decode_tiles_per_dim=8,
            )
            result = model.generate_image(
                seed=request.seed,
                image_path=request.source_path,
                resolution=ScaleFactor(request.scale),
                softness=0.0,
                color_correction_mode="wavelet",
            )
            conservative = _blend_conservatively(
                request.source_path,
                result.image,
                strength=request.strength,
            )
            conservative.save(request.output_path, format="PNG", compress_level=9)
            validation = validate_restoration(
                request.source_path,
                request.output_path,
                landmark_provider=landmark_provider,
                lpips_provider=lpips_provider,
            )
            metadata["validation"] = validation
            metadata["mlx_peak_gib"] = round(mx.get_peak_memory() / 1024**3, 3)
            if validation["passed"] is not True:
                request.output_path.unlink(missing_ok=True)
                metadata["status"] = "rejected_and_image_deleted"
                metadata["output_sha256"] = None
            else:
                metadata["status"] = "pending_100_percent_human_review"
                metadata["output_sha256"] = sha256_file(request.output_path)
    except Exception as error:
        request.output_path.unlink(missing_ok=True)
        metadata["status"] = "failed_and_image_deleted"
        metadata["error_type"] = type(error).__name__
        metadata["error"] = str(error)
        raise
    finally:
        metadata["wall_seconds"] = round(time.perf_counter() - started, 3)
        if model is not None:
            del model
        try:
            import mlx.core as mx

            mx.clear_cache()
            metadata["mlx_active_gib_after_cleanup"] = round(
                mx.get_active_memory() / 1024**3,
                3,
            )
        except (ImportError, RuntimeError):
            metadata["mlx_active_gib_after_cleanup"] = None
        gc.collect()
        _atomic_json(sidecar_path, metadata)
    return metadata
