"""Non-generative artifact, runtime, and full-load readiness checks."""

from __future__ import annotations

import gc
import hashlib
import importlib.metadata
import json
import platform
import time
from pathlib import Path
from typing import Any


RINA_ROOT = Path(__file__).resolve().parents[2]
REGISTRY_PATH = RINA_ROOT / "models/registry_seedvr2.yml"
LPIPS_REGISTRY_PATH = RINA_ROOT / "models/lpips/registry_lpips.yml"
CORPUS_MANIFEST_PATH = (
    RINA_ROOT / "hyperreal/finishing/calibration/corpus_manifest.v1.json"
)
CALIBRATION_REPORT_PATH = (
    RINA_ROOT / "hyperreal/finishing/calibration/calibration_report.v1.json"
)
FALLBACK_CALIBRATION_REPORT_PATH = (
    RINA_ROOT
    / "hyperreal/finishing/calibration/fallback_calibration_report.v1.json"
)
MODEL_PATH = (
    RINA_ROOT
    / "models/seedvr2/AbstractFramework-seedvr2-3b-8bit-6e6b162d"
)
PINNED_RUNTIME = {
    "mlx-gen": "0.23.1",
    "mlx": "0.31.2",
    "huggingface-hub": "1.24.0",
    "safetensors": "0.8.0",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _runtime_versions() -> tuple[dict[str, str | None], list[str]]:
    versions: dict[str, str | None] = {}
    failures: list[str] = []
    for distribution, expected in PINNED_RUNTIME.items():
        try:
            actual = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            actual = None
        versions[distribution] = actual
        if actual != expected:
            failures.append(f"runtime_version_mismatch:{distribution}:{actual}!={expected}")
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        failures.append(
            f"unsupported_platform:{platform.system()}:{platform.machine()}"
        )
    return versions, failures


def _registry_probe() -> tuple[dict[str, Any], list[str]]:
    if not REGISTRY_PATH.is_file():
        return {"ready": False}, ["registry_missing"]
    registry = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    failures: list[str] = []
    actual_size = 0
    for relative, record in registry["artifacts"].items():
        path = MODEL_PATH / relative
        if not path.is_file():
            failures.append(f"artifact_missing:{relative}")
            continue
        actual_size += path.stat().st_size
        if path.stat().st_size != record["size_bytes"]:
            failures.append(f"artifact_size_mismatch:{relative}")
        elif _sha256(path) != record["sha256"]:
            failures.append(f"artifact_sha256_mismatch:{relative}")
    if actual_size != registry["size_bytes"]:
        failures.append(
            f"tree_size_mismatch:{actual_size}!={registry['size_bytes']}"
        )
    return {
        "ready": not failures,
        "repo_id": registry["source"]["repo_id"],
        "revision": registry["source"]["revision"],
        "size_bytes": actual_size,
        "file_count": len(registry["artifacts"]),
    }, failures


def _backend_probe() -> tuple[dict[str, Any], list[str]]:
    try:
        import mlx.core as mx

        value = mx.array([1.0, 2.0])
        mx.eval(value)
        return {
            "ready": True,
            "backend": "MLX Metal on Apple Silicon",
            "active_memory_gib": round(mx.get_active_memory() / 1024**3, 3),
        }, []
    except Exception as error:
        return {"ready": False, "error": str(error)}, [
            f"mlx_backend_probe_failed:{type(error).__name__}"
        ]


def _lpips_probe() -> tuple[dict[str, Any], list[str]]:
    if not LPIPS_REGISTRY_PATH.is_file():
        return {"ready": False}, ["lpips_registry_missing"]
    registry = json.loads(LPIPS_REGISTRY_PATH.read_text(encoding="utf-8"))
    failures: list[str] = []
    records = (
        registry["implementation"]["wheel"],
        registry["linear_calibration_weights"],
        registry["feature_trunk"],
    )
    for record in records:
        path = LPIPS_REGISTRY_PATH.parent / record["path"]
        if not path.is_file():
            failures.append(f"lpips_artifact_missing:{record['path']}")
        elif path.stat().st_size != record["size_bytes"]:
            failures.append(f"lpips_artifact_size_mismatch:{record['path']}")
        elif _sha256(path) != record["sha256"]:
            failures.append(f"lpips_artifact_sha256_mismatch:{record['path']}")
    try:
        installed = importlib.metadata.version("lpips")
    except importlib.metadata.PackageNotFoundError:
        installed = None
    if installed != registry["implementation"]["version"]:
        failures.append(f"lpips_version_mismatch:{installed}")
    return {
        "ready": not failures,
        "provider": registry["runtime"]["provider"],
        "version": installed,
        "device": registry["runtime"]["observed_device"],
        "offline": registry["runtime"]["network_access_during_inference"] is False,
        "license_scope": "private_internal_metric_evaluation_only",
    }, failures


def _corpus_probe() -> tuple[dict[str, Any], list[str]]:
    if not CORPUS_MANIFEST_PATH.is_file():
        return {"ready": False}, ["calibration_corpus_manifest_missing"]
    manifest = json.loads(CORPUS_MANIFEST_PATH.read_text(encoding="utf-8"))
    failures: list[str] = []
    assets = manifest.get("assets", [])
    if manifest.get("corpus_size") != len(assets) or not assets:
        failures.append("calibration_corpus_size_mismatch")
    for record in assets:
        path = CORPUS_MANIFEST_PATH.parent / record["path"]
        if "adult" not in str(record.get("source_caption_evidence", "")).lower():
            failures.append(f"adult_evidence_missing:{record.get('id')}")
        if not path.is_file() or _sha256(path) != record.get("sha256"):
            failures.append(f"calibration_corpus_integrity_failure:{record.get('id')}")
    if manifest.get("moodboard_policy", {}).get("used") is not False:
        failures.append("moodboard_must_not_be_used_without_explicit_adult_evidence")
    return {
        "ready": not failures,
        "corpus_size": len(assets),
        "license": manifest.get("license", {}).get("name"),
        "adult_age_inference_performed": manifest.get("adult_policy", {}).get(
            "age_inference_performed"
        ),
        "moodboard_used": manifest.get("moodboard_policy", {}).get("used"),
    }, failures


def _calibration_probe() -> tuple[dict[str, Any], list[str]]:
    if not CALIBRATION_REPORT_PATH.is_file():
        return {"ready": False}, ["calibration_report_missing"]
    report = json.loads(CALIBRATION_REPORT_PATH.read_text(encoding="utf-8"))
    failures: list[str] = []
    if report.get("safety", {}).get("rina_image_processed") is not False:
        failures.append("calibration_touched_rina_image")
    if report.get("configuration_count") != 6:
        failures.append("minimum_calibration_matrix_incomplete")
    enabled_scales = [
        scale
        for scale, record in report.get("scales", {}).items()
        if record.get("enabled") is True
    ]
    return {
        "ready": not failures,
        "run_id": report.get("run_id"),
        "corpus_size": report.get("corpus_size"),
        "configuration_count": report.get("configuration_count"),
        "enabled_scales": enabled_scales,
        "scale_verdicts": {
            scale: record.get("verdict")
            for scale, record in report.get("scales", {}).items()
        },
        "rina_image_processed": report.get("safety", {}).get(
            "rina_image_processed"
        ),
    }, failures


def _fallback_probe() -> tuple[dict[str, Any], list[str]]:
    if not FALLBACK_CALIBRATION_REPORT_PATH.is_file():
        return {"ready": False}, ["fallback_calibration_report_missing"]
    report = json.loads(
        FALLBACK_CALIBRATION_REPORT_PATH.read_text(encoding="utf-8")
    )
    failures: list[str] = []
    if report.get("rina_image_processed") is not False:
        failures.append("fallback_calibration_touched_rina_image")
    if report.get("moodboard_image_processed") is not False:
        failures.append("fallback_calibration_touched_moodboard_image")
    if report.get("generative_model_used") is not False:
        failures.append("fallback_calibration_used_generative_model")
    if report.get("configuration_count") != 15:
        failures.append("fallback_calibration_matrix_incomplete")
    if report.get("thresholds_relaxed") is not False:
        failures.append("fallback_thresholds_were_relaxed")
    enabled = report.get("enabled_presets", [])
    default = report.get("production_default")
    if default not in enabled:
        failures.append("fallback_default_is_not_enabled")
    if default != "original_metadata_only":
        failures.append("preservation_first_default_required")
    return {
        "ready": not failures,
        "run_id": report.get("run_id"),
        "corpus_size": report.get("corpus_size"),
        "configuration_count": report.get("configuration_count"),
        "enabled_presets": enabled,
        "production_default": default,
        "preset_verdicts": {
            preset: record.get("verdict")
            for preset, record in report.get("preset_results", {}).items()
        },
        "generative_model_used": report.get("generative_model_used"),
        "rina_image_processed": report.get("rina_image_processed"),
        "moodboard_image_processed": report.get("moodboard_image_processed"),
    }, failures


def _full_load_probe() -> tuple[dict[str, Any], list[str]]:
    import mlx.core as mx
    from mflux.models.common.config.model_config import ModelConfig
    from mflux.models.seedvr2.variants.upscale.seedvr2 import SeedVR2

    started = time.perf_counter()
    mx.reset_peak_memory()
    model = None
    result: dict[str, Any]
    failures: list[str] = []
    try:
        model = SeedVR2(
            model_path=str(MODEL_PATH),
            model_config=ModelConfig.seedvr2_3b(),
        )
        mx.eval(model.parameters())
        active = mx.get_active_memory()
        result = {
            "ready": True,
            "seconds": round(time.perf_counter() - started, 3),
            "mlx_active_gib": round(active / 1024**3, 3),
            "mlx_peak_gib": round(mx.get_peak_memory() / 1024**3, 3),
            "checkpoint_variant": model.seedvr2_checkpoint_variant,
            "source_layout": model.seedvr2_source_layout,
            "quantization_bits": model.bits,
            "resident_weight_gib": round(
                model.seedvr2_resident_weight_bytes / 1024**3,
                3,
            ),
            "image_inference_performed": False,
        }
    except Exception as error:
        result = {
            "ready": False,
            "error_type": type(error).__name__,
            "error": str(error),
            "image_inference_performed": False,
        }
        failures.append(f"full_model_load_failed:{type(error).__name__}:{error}")
    finally:
        if model is not None:
            del model
        mx.clear_cache()
        gc.collect()
    result["mlx_active_gib_after_cleanup"] = round(
        mx.get_active_memory() / 1024**3,
        3,
    )
    return result, failures


def evaluate_readiness(*, full_load: bool = False) -> dict[str, Any]:
    versions, runtime_failures = _runtime_versions()
    artifact, artifact_failures = _registry_probe()
    backend, backend_failures = _backend_probe()
    lpips, lpips_failures = _lpips_probe()
    corpus, corpus_failures = _corpus_probe()
    calibration, calibration_failures = _calibration_probe()
    fallback, fallback_failures = _fallback_probe()
    load = {
        "ready": False,
        "performed": False,
        "image_inference_performed": False,
    }
    load_failures: list[str] = []
    if full_load and not (runtime_failures or artifact_failures or backend_failures):
        load, load_failures = _full_load_probe()
        load["performed"] = True

    execution_blockers: list[str] = []
    if not lpips["ready"]:
        execution_blockers.append("local_lpips_weights_and_provider_not_registered")
    if not corpus["ready"]:
        execution_blockers.append("non_rina_validation_corpus_not_registered")
    if not calibration["ready"]:
        execution_blockers.append("calibration_report_not_ready")
    if not calibration.get("enabled_scales"):
        execution_blockers.append("no_calibrated_scale_enabled")
    failures = [
        *runtime_failures,
        *artifact_failures,
        *backend_failures,
        *lpips_failures,
        *corpus_failures,
        *calibration_failures,
        *fallback_failures,
        *load_failures,
    ]
    if not full_load:
        failures.append("full_model_load_not_performed")
    return {
        "schema_version": "1.0.0",
        "artifact_and_runtime_ready": not failures,
        "production_execution_ready": not failures and not execution_blockers,
        "fallback_execution_ready": fallback["ready"],
        "image_inference_performed": True,
        "rina_image_inference_performed": False,
        "calibration_image_inference_performed": True,
        "model": artifact,
        "runtime": {
            "versions": versions,
            "platform": f"{platform.system()} {platform.machine()}",
            "backend": backend,
            "mode": "local_offline_mlx_metal_not_pytorch_mps",
        },
        "lpips": lpips,
        "calibration_corpus": corpus,
        "calibration": calibration,
        "fallback": fallback,
        "full_load": load,
        "failures": failures,
        "execution_blockers": execution_blockers,
    }


if __name__ == "__main__":
    print(json.dumps(evaluate_readiness(full_load=True), indent=2, sort_keys=True))
