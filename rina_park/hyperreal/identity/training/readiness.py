#!/usr/bin/env python3
"""Fail-closed, non-training readiness checks for Qwen-Image-2512 identity LoRA."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
from pathlib import Path
from typing import Any

from packaging.requirements import Requirement
from packaging.version import Version

from ..validator import load_spec, training_readiness

CONFIG_PATH = Path(__file__).with_name("qwen_2512_mps.json")
PROJECT_ROOT = Path(__file__).resolve().parents[4]
VENV_ROOT = PROJECT_ROOT / ".venv"
Q8_MODEL_PATH = (
    PROJECT_ROOT
    / "rina_park/models/qwen_image_2512"
    / "AbstractFramework-qwen-image-2512-8bit-f70648e"
)


def load_config(path: Path = CONFIG_PATH) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def physical_memory_gib() -> float:
    pages = os.sysconf("SC_PHYS_PAGES")
    page_size = os.sysconf("SC_PAGE_SIZE")
    return round(pages * page_size / 1024**3, 2)


def package_report(requirements: dict[str, str]) -> tuple[dict[str, Any], list[str]]:
    report: dict[str, Any] = {}
    blockers: list[str] = []
    for distribution, constraint in requirements.items():
        if constraint == "required for per-image captions":
            requirement = Requirement(distribution)
        else:
            requirement = Requirement(f"{distribution}{constraint}")
        try:
            installed = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            report[distribution] = {"installed": None, "satisfies": False}
            blockers.append(f"required package is missing: {distribution}")
            continue
        satisfies = Version(installed) in requirement.specifier
        report[distribution] = {"installed": installed, "satisfies": satisfies}
        if not satisfies:
            blockers.append(
                f"{distribution} {installed} does not satisfy {requirement.specifier}"
            )
    return report, blockers


def estimate_memory(config: dict[str, Any]) -> dict[str, Any]:
    model = config["memory_model"]
    rank = config["adapter"]["rank"]
    dim = 3072
    blocks = 60
    target_count = 4
    trainable = blocks * target_count * rank * (dim + dim)
    # LoRA weights + gradients + two Adam moment tensors, all FP32.
    lora_optimizer_gib = trainable * 4 * 4 / 1024**3
    low = (
        model["fp32_foundation_estimate_gib"]
        + lora_optimizer_gib
        + model["cached_dataset_latents_estimate_gib"]
        + model["activation_and_framework_range_gib"][0]
    )
    high = (
        model["fp32_foundation_estimate_gib"]
        + lora_optimizer_gib
        + model["cached_dataset_latents_estimate_gib"]
        + model["activation_and_framework_range_gib"][1]
    )
    return {
        "trainable_parameters": trainable,
        "lora_optimizer_gib": round(lora_optimizer_gib, 3),
        "estimated_peak_range_gib": [round(low, 1), round(high, 1)],
        "physical_memory_gib": physical_memory_gib(),
        "estimate_only": True,
        "model_loaded": False,
    }


def validate_config(config: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if config.get("training_enabled") is not False:
        errors.append("training_enabled must remain false in Phase-1 scaffolding")
    if config["foundation"]["repo_id"] != "Qwen/Qwen-Image-2512":
        errors.append("foundation must be Qwen/Qwen-Image-2512")
    if config["runtime"]["mixed_precision"] != "no":
        errors.append("MPS Qwen training must use FP32 (--mixed_precision=no)")
    if config["runtime"]["optimizer"] != "torch.optim.AdamW":
        errors.append("only standard torch AdamW is approved on MPS")
    expected_targets = ["to_q", "to_k", "to_v", "to_out.0"]
    if config["adapter"]["target_modules"] != expected_targets:
        errors.append("LoRA targets exceed the MFLUX-compatible attention-only allowlist")
    if config["dataset"]["spatial_loss_masks"].startswith("required") is False:
        errors.append("the required spatial-mask blocker must be preserved")
    if config["training"]["validation_generation"] is not False:
        errors.append("dry-run configuration must disable validation generation")
    if config["training"]["push_to_hub"] is not False:
        errors.append("dry-run configuration must disable hub writes")
    return errors


def _mps_report() -> tuple[dict[str, Any], list[str]]:
    blockers: list[str] = []
    try:
        import torch

        built = torch.backends.mps.is_built()
        available = torch.backends.mps.is_available()
    except Exception as exc:  # pragma: no cover - diagnostic path
        return {"imported": False, "error": f"{type(exc).__name__}: {exc}"}, [
            "PyTorch import or MPS probe failed"
        ]
    if not built or not available:
        blockers.append("PyTorch MPS backend is unavailable")
    return {"imported": True, "built": built, "available": available}, blockers


def _trainer_import_report(expected_commit: str) -> tuple[dict[str, Any], list[str]]:
    blockers: list[str] = []
    imported: list[str] = []
    try:
        from accelerate import Accelerator
        from datasets import load_dataset
        from diffusers import (
            AutoencoderKLQwenImage,
            QwenImagePipeline,
            QwenImageTransformer2DModel,
        )
        from peft import LoraConfig
        from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2Tokenizer

        symbols = (
            Accelerator,
            load_dataset,
            AutoencoderKLQwenImage,
            QwenImagePipeline,
            QwenImageTransformer2DModel,
            LoraConfig,
            Qwen2_5_VLForConditionalGeneration,
            Qwen2Tokenizer,
        )
        imported = [symbol.__name__ for symbol in symbols]
    except Exception as exc:  # pragma: no cover - diagnostic path
        blockers.append(f"Qwen trainer import surface failed: {type(exc).__name__}: {exc}")

    installed_commit: str | None = None
    try:
        direct_url = importlib.metadata.distribution("diffusers").read_text(
            "direct_url.json"
        )
        if direct_url:
            installed_commit = json.loads(direct_url).get("vcs_info", {}).get("commit_id")
    except (importlib.metadata.PackageNotFoundError, json.JSONDecodeError):
        pass
    if installed_commit != expected_commit:
        blockers.append(
            f"diffusers source commit is {installed_commit!r}, expected {expected_commit}"
        )
    return {
        "symbols_imported": imported,
        "diffusers_commit": installed_commit,
        "expected_commit": expected_commit,
        "source_pinned": installed_commit == expected_commit,
    }, blockers


def _mlx_mapping_report(config: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    expected = config["adapter"]["expected_mlx_runner_mapping"]
    blockers: list[str] = []
    try:
        from mflux.models.qwen.weights.qwen_lora_mapping import QwenLoRAMapping

        actual = {target.model_path for target in QwenLoRAMapping.get_mapping()}
    except Exception as exc:  # pragma: no cover - diagnostic path
        return {"imported": False, "error": f"{type(exc).__name__}: {exc}"}, [
            "MFLUX Qwen LoRA mapping import failed"
        ]
    missing = sorted(set(expected) - actual)
    if missing:
        blockers.append(f"MFLUX mapping lacks required targets: {missing}")
    return {
        "imported": True,
        "required_targets_present": not missing,
        "missing": missing,
    }, blockers


def dry_run(config_path: Path = CONFIG_PATH) -> dict[str, Any]:
    """Inspect config/imports/memory only; never load weights, generate, or train."""
    config = load_config(config_path)
    blockers = list(config["blockers"])
    config_errors = validate_config(config)
    blockers.extend(config_errors)

    packages, package_blockers = package_report(config["runtime"]["required_packages"])
    blockers.extend(package_blockers)
    mps, mps_blockers = _mps_report()
    blockers.extend(mps_blockers)
    trainer_imports, trainer_import_blockers = _trainer_import_report(
        config["runtime"]["diffusers_commit"]
    )
    blockers.extend(trainer_import_blockers)
    mapping, mapping_blockers = _mlx_mapping_report(config)
    blockers.extend(mapping_blockers)

    dataset_path = (config_path.parent / config["dataset"]["spec"]).resolve()
    dataset = training_readiness(load_spec(dataset_path))
    blockers.extend(f"dataset: {item}" for item in dataset["blockers"])

    memory = estimate_memory(config)
    if memory["physical_memory_gib"] < memory["estimated_peak_range_gib"][1] + 16:
        blockers.append(
            "estimated high-water mark leaves less than 16 GiB system headroom"
        )

    # Existing q8 presence is evidence for inference compatibility only. It is
    # deliberately not opened, hashed, converted, or used as a training source.
    q8_inference = {
        "path": str(Q8_MODEL_PATH),
        "present": Q8_MODEL_PATH.is_dir(),
        "used_for_training": False,
    }

    unique_blockers = list(dict.fromkeys(blockers))
    return {
        "status": "blocked",
        "ready_to_train": False,
        "verdict": config["verdict"],
        "training_enabled": False,
        "checks": {
            "config_valid": not config_errors,
            "packages": packages,
            "mps": mps,
            "trainer_imports": trainer_imports,
            "mlx_runner_lora_mapping": mapping,
            "dataset": dataset,
            "memory": memory,
            "q8_inference_checkpoint": q8_inference,
            "venv": {
                "expected": str(VENV_ROOT),
                "active_prefix": os.sys.prefix,
                "inside_project_venv": Path(os.sys.prefix).resolve() == VENV_ROOT.resolve(),
            },
        },
        "blockers": unique_blockers,
        "side_effects": {
            "model_loaded": False,
            "images_generated": False,
            "training_started": False,
            "artifacts_modified": False,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--require-ready", action="store_true")
    args = parser.parse_args()
    report = dry_run(args.config)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 2 if args.require_ready and not report["ready_to_train"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
