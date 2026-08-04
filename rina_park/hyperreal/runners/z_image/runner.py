#!/usr/bin/env python3
"""Deterministic offline Z-Image Base runner for Phase-0 camera hypotheses."""

from __future__ import annotations

import argparse
import gc
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .readiness import (
    MODEL_PATH,
    MODEL_REPO,
    MODEL_REVISION,
    config_readiness,
    memory_snapshot,
    no_network,
    sequential_gpu,
)

DEFAULT_STEPS = 50
DEFAULT_GUIDANCE = 4.0
DEFAULT_NEGATIVE_PROMPT = (
    "anatomical errors, extra or missing fingers or limbs, malformed hands, "
    "impossible reflections, plastic skin, oversmoothing, text, logo, watermark"
)
MODEL_ID = "z_image"
ADAPTER_API_VERSION = "phase0-foundation-adapter-v1"


@dataclass(frozen=True)
class BakeoffCase:
    case_id: str
    prompt: str
    negative_prompt: str
    seed: int
    steps: int
    width: int
    height: int
    guidance: float
    camera_hypothesis: dict[str, Any]


class ZImageAdapter:
    """Fail-closed shared adapter using only the pinned local BF16 checkpoint."""

    model_id = MODEL_ID
    adapter_api_version = ADAPTER_API_VERSION
    steps = DEFAULT_STEPS
    guidance = DEFAULT_GUIDANCE
    negative_prompt = DEFAULT_NEGATIVE_PROMPT

    def __init__(self) -> None:
        self._model: Any | None = None
        self._readiness: dict[str, Any] | None = None
        self._smoke_passed = False
        self._load_metadata: dict[str, Any] = {}

    def readiness(self) -> dict[str, Any]:
        error: str | None = None
        if self._readiness is None:
            try:
                self._readiness = config_readiness(verify_hashes=True)
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
                self._readiness = {"status": "failed", "error": error}
        config_ok = self._readiness.get("status") == "config_ready"
        license_data = self._readiness.get("license", {})
        checks = {
            "artifact_integrity": config_ok,
            "runtime_backend": config_ok and "gpu" in str(self._readiness.get("device")).lower(),
            "license": config_ok and license_data.get("weights_spdx") == "Apache-2.0",
            "generation_smoke": self._smoke_passed,
            "deterministic_request": True,
            "no_fallback": True,
        }
        details = dict(self._readiness)
        details["ready_for_generation"] = self._smoke_passed
        details["blocker"] = None if self._smoke_passed else details.get("blocker")
        return {
            "model_id": self.model_id,
            "adapter_api_version": self.adapter_api_version,
            "ready": all(checks.values()),
            "checks": checks,
            "status": "ready" if all(checks.values()) else "smoke_required_or_blocked",
            "details": details,
            "error": error,
        }

    def mark_smoke_passed(self) -> None:
        self._smoke_passed = True

    def _load(self) -> Any:
        if self._model is not None:
            return self._model
        import mlx.core as mx
        from mflux.models.common.config import ModelConfig
        from mflux.models.z_image import ZImage

        started = time.monotonic()
        with no_network():
            self._model = ZImage(
                model_config=ModelConfig.z_image(),
                model_path=str(MODEL_PATH),
            )
            mx.eval(self._model.parameters())
            mx.synchronize()
        self._load_metadata = {
            "load_seconds": round(time.monotonic() - started, 3),
            "load_memory": memory_snapshot(),
        }
        return self._model

    def generate(self, request: Any, output_path: Path) -> dict[str, Any]:
        if request.model_id != self.model_id:
            raise RuntimeError(f"Z adapter refuses model substitution: {request.model_id}")
        if output_path.exists():
            raise FileExistsError(f"refusing to overwrite {output_path}")
        import mlx.core as mx

        model = self._load()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        mx.clear_cache()
        gc.collect()
        mx.reset_peak_memory()
        started = time.monotonic()
        with no_network():
            generated = model.generate_image(
                seed=request.seed,
                prompt=request.prompt,
                negative_prompt=self.negative_prompt,
                num_inference_steps=self.steps,
                width=request.width,
                height=request.height,
                guidance=self.guidance,
                scheduler="flow_match_euler_discrete",
            )
            generated.save(output_path)
            mx.synchronize()
        return {
            "model_repo": MODEL_REPO,
            "model_revision": MODEL_REVISION,
            "model_path": str(MODEL_PATH),
            "precision": "BF16",
            "steps": self.steps,
            "guidance": self.guidance,
            "negative_prompt": self.negative_prompt,
            "scheduler": "flow_match_euler_discrete",
            "network_at_inference": "blocked",
            "fallback_allowed": False,
            "generation_seconds": round(time.monotonic() - started, 3),
            "memory": memory_snapshot(),
            **self._load_metadata,
        }

    def cleanup(self) -> dict[str, Any]:
        if self._model is not None:
            del self._model
            self._model = None
        import mlx.core as mx

        mx.clear_cache()
        gc.collect()
        return memory_snapshot()


ADAPTER = ZImageAdapter()


def load_bakeoff_manifest(path: Path) -> tuple[dict[str, Any], list[BakeoffCase]]:
    """Read the shared manifest once and preserve each camera hypothesis."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    entries = raw.get("cases", raw.get("camera_hypotheses"))
    if not isinstance(entries, list) or not entries:
        raise ValueError(
            "bakeoff manifest requires non-empty cases or camera_hypotheses"
        )
    defaults = raw.get("defaults", {})
    output = raw.get("output", {})
    seed_policy = raw.get("seed_policy", {})
    base_seed = seed_policy.get("base_seed")
    cases: list[BakeoffCase] = []
    seen: set[str] = set()
    for index, entry in enumerate(entries, start=1):
        model_overrides = entry.get("models", {}).get("z_image", {})
        merged = defaults | entry | model_overrides
        case_id = str(merged.get("id", merged.get("case_id", ""))).strip()
        if not case_id or case_id in seen:
            raise ValueError(f"case id must be non-empty and unique: {case_id!r}")
        seen.add(case_id)
        camera = merged.get("camera_hypothesis", entry)
        if not isinstance(camera, dict) or not camera:
            raise ValueError(f"{case_id}: camera_hypothesis is required")
        prompt = merged.get("prompt")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError(f"{case_id}: prompt is required")
        seed = merged.get("seed")
        if seed is None and isinstance(base_seed, int):
            seed = base_seed + index
        steps = merged.get("steps", DEFAULT_STEPS)
        if not isinstance(seed, int) or seed < 0:
            raise ValueError(f"{case_id}: seed must be a non-negative integer")
        if not isinstance(steps, int) or not 28 <= steps <= 50:
            raise ValueError(f"{case_id}: Z-Image Base steps must be 28 through 50")
        cases.append(
            BakeoffCase(
                case_id=case_id,
                prompt=prompt.strip(),
                negative_prompt=str(
                    merged.get("negative_prompt", DEFAULT_NEGATIVE_PROMPT)
                ).strip(),
                seed=seed,
                steps=steps,
                width=int(merged.get("width", output.get("width", 1024))),
                height=int(merged.get("height", output.get("height", 1024))),
                guidance=float(merged.get("guidance", DEFAULT_GUIDANCE)),
                camera_hypothesis=camera,
            )
        )
    return raw, cases


def manifest_plan(path: Path) -> dict[str, Any]:
    raw, cases = load_bakeoff_manifest(path)
    return {
        "status": "manifest_ready_no_generation",
        "manifest_path": str(path.resolve()),
        "manifest_schema_version": raw.get("schema_version"),
        "model_repo": MODEL_REPO,
        "model_revision": MODEL_REVISION,
        "case_count": len(cases),
        "cases": [asdict(case) for case in cases],
    }


def run_bakeoff(path: Path, output_dir: Path) -> dict[str, Any]:
    """Generate sequentially only after the explicit CLI safety switch."""
    import mlx.core as mx
    from mflux.models.common.config import ModelConfig
    from mflux.models.z_image import ZImage

    raw, cases = load_bakeoff_manifest(path)
    readiness = config_readiness(verify_hashes=True)
    output_dir.mkdir(parents=True, exist_ok=False)
    mx.clear_cache()
    mx.reset_peak_memory()
    model = None
    results: list[dict[str, Any]] = []
    load_started = time.monotonic()
    try:
        with sequential_gpu(), no_network():
            model = ZImage(
                model_config=ModelConfig.z_image(),
                model_path=str(MODEL_PATH),
            )
            mx.eval(model.parameters())
            mx.synchronize()
            load_seconds = round(time.monotonic() - load_started, 3)
            load_memory = memory_snapshot()
            for case in cases:
                mx.clear_cache()
                gc.collect()
                mx.reset_peak_memory()
                started = time.monotonic()
                generated = model.generate_image(
                    seed=case.seed,
                    prompt=case.prompt,
                    negative_prompt=case.negative_prompt,
                    num_inference_steps=case.steps,
                    width=case.width,
                    height=case.height,
                    guidance=case.guidance,
                    scheduler="flow_match_euler_discrete",
                )
                output = output_dir / f"{case.case_id}.png"
                generated.save(output)
                mx.synchronize()
                metadata = {
                    **asdict(case),
                    "model_repo": MODEL_REPO,
                    "model_revision": MODEL_REVISION,
                    "precision": "BF16",
                    "scheduler": "flow_match_euler_discrete",
                    "generation_seconds": round(time.monotonic() - started, 3),
                    "memory": memory_snapshot(),
                    "network_at_inference": "blocked",
                    "gpu_execution": "sequential",
                    "output": str(output),
                }
                output.with_suffix(".png.json").write_text(
                    json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
                )
                results.append(metadata)
    finally:
        if model is not None:
            del model
        mx.clear_cache()
        gc.collect()
    summary = {
        "status": "generated_pending_blind_review",
        "manifest_path": str(path.resolve()),
        "manifest_schema_version": raw.get("schema_version"),
        "readiness": readiness,
        "model_load_seconds": load_seconds,
        "model_load_memory": load_memory,
        "cleanup_memory": memory_snapshot(),
        "results": results,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--allow-image-generation",
        action="store_true",
        help="Required safety switch; omit for manifest-only validation.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if not args.allow_image_generation:
        print(json.dumps(manifest_plan(args.manifest), indent=2))
        return
    if args.output_dir is None:
        raise SystemExit("--output-dir is required with --allow-image-generation")
    print(json.dumps(run_bakeoff(args.manifest, args.output_dir), indent=2))


if __name__ == "__main__":
    main()
