"""Read-only readiness probe for the identity-region edit pilot."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import inspect
import json
from pathlib import Path
from typing import Any

from .contract import MODEL_REPO, MODEL_REVISION, sha256_file


ROOT = Path(__file__).resolve().parents[4]
RINA = ROOT / "rina_park"
PLAN_PATH = Path(__file__).with_name("pilot_plan.v1.json")
MODEL_PATH = RINA / "models/qwen_image_edit/AbstractFramework-qwen-image-edit-2511-8bit-5f35885d"
MODEL_TREE_SHA256 = "6cd4bfa9e260e72b6febb2f94876f402bfa140b9c107fcf3763eba729a3334fd"
MODEL_SIZE_BYTES = 30_343_385_206


def _hash_directory(path: Path) -> tuple[str, int, int]:
    digest = hashlib.sha256()
    files = sorted(
        candidate
        for candidate in path.rglob("*")
        if candidate.is_file() and ".cache" not in candidate.relative_to(path).parts
    )
    for candidate in files:
        digest.update(candidate.relative_to(path).as_posix().encode())
        digest.update(b"\0")
        digest.update(bytes.fromhex(sha256_file(candidate)))
    return digest.hexdigest(), len(files), sum(candidate.stat().st_size for candidate in files)


def _model_probe(*, verify_integrity: bool) -> dict[str, Any]:
    if not MODEL_PATH.is_dir():
        return {"ready": False, "verified": False, "failure": "model_snapshot_missing"}
    if not verify_integrity:
        return {"ready": False, "verified": False, "failure": "model_integrity_not_checked"}
    tree_hash, file_count, size_bytes = _hash_directory(MODEL_PATH)
    ready = tree_hash == MODEL_TREE_SHA256 and size_bytes == MODEL_SIZE_BYTES
    return {
        "ready": ready,
        "verified": True,
        "tree_sha256": tree_hash,
        "file_count": file_count,
        "size_bytes": size_bytes,
        "failure": None if ready else "model_integrity_mismatch",
    }


def _project_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _source_record(path: Path, expected_sha256: str | None) -> dict[str, Any]:
    exists = path.is_file()
    actual = sha256_file(path) if exists else None
    return {
        "path": str(path),
        "exists": exists,
        "expected_sha256": expected_sha256,
        "actual_sha256": actual,
        "hash_matches": bool(exists and expected_sha256 and actual == expected_sha256),
    }


def _phase1_qc_probe() -> dict[str, Any]:
    failures: list[str] = []
    try:
        from hyperreal.identity.qc.registry import verified_model_paths

        models = {name: str(path) for name, path in verified_model_paths().items()}
    except Exception as error:  # readiness must report, never default-pass
        models = {}
        failures.append(f"mediapipe_models:{type(error).__name__}:{error}")
    for module in ("mediapipe", "Foundation", "Quartz", "Vision"):
        try:
            importlib.import_module(module)
        except Exception as error:
            failures.append(f"import:{module}:{type(error).__name__}")
    return {"ready": not failures, "models": models, "failures": failures}


def _mlx_capability_probe() -> dict[str, Any]:
    failures: list[str] = []
    try:
        runtime_version = importlib.metadata.version("mlx-gen")
    except importlib.metadata.PackageNotFoundError:
        runtime_version = None
        failures.append("mlx-gen_not_installed")
    signature = None
    model_name = None
    try:
        from mflux.models.common.config import ModelConfig
        from mflux.models.qwen.variants.edit.qwen_image_edit import QwenImageEdit

        signature = str(inspect.signature(QwenImageEdit.generate_image))
        parameters = inspect.signature(QwenImageEdit.generate_image).parameters
        if "image_paths" not in parameters:
            failures.append("python_api_missing_image_paths")
        if "mask_path" not in parameters:
            failures.append("python_api_missing_mask_path")
        model_name = ModelConfig.from_name(MODEL_REPO).model_name
    except Exception as error:
        failures.append(f"mlx_api_import:{type(error).__name__}:{error}")
    if runtime_version != "0.23.1":
        failures.append(f"mlx-gen_version_mismatch:{runtime_version}")

    # 0.23.1 exposes both arguments, but its public capability table separates
    # qwen.multi-reference (2+, no mask) from qwen.inpaint (1 image, mask).
    # The Python method concatenates every conditioning latent, then supplies
    # that concatenation as the clean image to a one-scene mask blend. This is
    # not authoritative support for the required combined three-image+mask call.
    combined_multi_image_mask_supported = False
    failures.append("mlx_0.23.1_combined_multi_image_plus_mask_not_supported")
    return {
        "ready": not failures,
        "runtime_version": runtime_version,
        "model_name": model_name,
        "generate_image_signature": signature,
        "multi_image_supported_separately": True,
        "mask_supported_separately": True,
        "combined_multi_image_mask_supported": combined_multi_image_mask_supported,
        "input_order": [
            "Picture 1: scene base and output geometry/dimensions",
            "Picture 2: approved master identity",
            "Picture 3: approved matching-angle Qwen identity reference",
        ],
        "prompt_format": "MLX tokenizer prepends Picture N: vision tokens in list order; prompt names Picture 1/2/3.",
        "failures": failures,
    }


def _diffusers_fallback_probe() -> dict[str, Any]:
    model_index = MODEL_PATH / "model_index.json"
    return {
        "ready": False,
        "installed_runtime": _optional_version("diffusers"),
        "pinned_q8_is_diffusers_checkpoint": model_index.is_file(),
        "edit_plus_multi_image": True,
        "edit_plus_region_mask": False,
        "edit_inpaint_region_mask": True,
        "edit_inpaint_multi_reference_contract": False,
        "failures": [
            "pinned_mlx_q8_checkpoint_is_not_diffusers_from_pretrained_layout",
            "diffusers_edit_plus_and_edit_inpaint_are_separate_uncombined_pipelines",
            "no_pinned_upstream_diffusers_2511_checkpoint_available_locally",
        ],
    }


def _optional_version(distribution: str) -> str | None:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def evaluate_readiness(
    plan_path: Path = PLAN_PATH,
    *,
    verify_model_integrity: bool = True,
) -> dict[str, Any]:
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    blockers: list[str] = []
    sources: list[dict[str, Any]] = []
    for pilot in plan["pilots"]:
        for field, hash_field in (
            ("scene_path", "scene_sha256"),
            ("master_path", "master_sha256"),
            ("angle_reference_path", "angle_reference_sha256"),
        ):
            record = _source_record(_project_path(pilot[field]), pilot.get(hash_field))
            record.update({"pilot_id": pilot["pilot_id"], "role": field})
            sources.append(record)
            if not record["hash_matches"]:
                blockers.append(f"{pilot['pilot_id']}:{field}:missing_or_hash_mismatch")
        record_path = _project_path(pilot["angle_reference_record_path"])
        approval_record = _source_record(record_path, pilot["angle_reference_record_sha256"])
        approval_record.update(
            {"pilot_id": pilot["pilot_id"], "role": "angle_reference_approval_record"}
        )
        sources.append(approval_record)
        if not approval_record["hash_matches"]:
            blockers.append(f"{pilot['pilot_id']}:approval_record:missing_or_hash_mismatch")
        else:
            approval = json.loads(record_path.read_text(encoding="utf-8"))
            if (
                approval.get("review_status") != "approved_internal_production_reference"
                or approval.get("human_review", {}).get("approved") is not True
            ):
                blockers.append(f"{pilot['pilot_id']}:angle_reference_not_approved")
        mask_value = pilot.get("mask_path")
        if not mask_value or not pilot.get("mask_sha256"):
            blockers.append(f"{pilot['pilot_id']}:face_hair_mask_not_pinned")
        else:
            mask = _source_record(_project_path(mask_value), pilot["mask_sha256"])
            mask.update({"pilot_id": pilot["pilot_id"], "role": "mask_path"})
            sources.append(mask)
            if not mask["hash_matches"]:
                blockers.append(f"{pilot['pilot_id']}:mask:missing_or_hash_mismatch")

    mlx = _mlx_capability_probe()
    diffusers = _diffusers_fallback_probe()
    qc = _phase1_qc_probe()
    model_integrity = _model_probe(verify_integrity=verify_model_integrity)
    if not model_integrity["ready"]:
        blockers.append(model_integrity["failure"])
    if not mlx["combined_multi_image_mask_supported"]:
        blockers.append("no_verified_local_combined_three_image_plus_region_mask_route")
    if not qc["ready"]:
        blockers.append("phase1_qc_runtime_not_ready")
    if not plan["gates"]["spatial_lpips_provider_ready"]:
        blockers.append("spatial_lpips_provider_not_registered")
    if not plan["gates"]["combined_route_memory_smoke_completed"]:
        blockers.append("combined_three_image_mask_memory_unmeasured")

    return {
        "schema_version": "1.0.0",
        "ready": not blockers,
        "execution_allowed": False if blockers else plan["execution_authorized"],
        "generation_performed": False,
        "training_performed": False,
        "model": {
            "repo": MODEL_REPO,
            "revision": MODEL_REVISION,
            "path": str(MODEL_PATH),
            "expected_tree_sha256": MODEL_TREE_SHA256,
            "expected_size_bytes": MODEL_SIZE_BYTES,
            "checkpoint_layout": "mlx-gen/mflux q8; not Diffusers",
            "integrity": model_integrity,
        },
        "mlx": mlx,
        "diffusers_mps_fallback": diffusers,
        "phase1_qc": qc,
        "memory": {
            "machine_unified_memory_gib": 128,
            "observed_single_reference_peak_mlx_gib": 34.888,
            "observed_single_reference_process_max_rss_gib": 23.033,
            "combined_three_image_mask_peak_gib": None,
            "verdict": "blocked_until_non-generative route validation and authorized memory smoke",
        },
        "sources": sources,
        "blockers": sorted(set(blockers)),
    }


def require_ready(plan_path: Path = PLAN_PATH) -> dict[str, Any]:
    report = evaluate_readiness(plan_path, verify_model_integrity=True)
    if not report["ready"] or not report["execution_allowed"]:
        raise RuntimeError("identity edit lane is fail-closed: " + "; ".join(report["blockers"]))
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, default=PLAN_PATH)
    parser.add_argument("--require-ready", action="store_true")
    args = parser.parse_args()
    report = evaluate_readiness(args.plan)
    print(json.dumps(report, indent=2))
    if args.require_ready and (not report["ready"] or not report["execution_allowed"]):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
