"""Smoke-gated execution entry point for the Qwen/Z Phase-0 bake-off."""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .orchestration import (
    DEFAULT_LOCK,
    DEFAULT_MOODBOARD,
    AdapterRequest,
    _memory_snapshot,
    require_all_ready,
    run_bakeoff,
    serialized_gpu,
    validate_sidecar,
)
from .prescreen import run_prescreen
from .runners.qwen_2512.runner import QwenImage2512Adapter
from .runners.z_image.runner import ZImageAdapter
from .spec import DEFAULT_MANIFEST, load_manifest

RINA_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = RINA_ROOT / "out" / "hyperreal_phase0_bakeoff"


def _pre_smoke_ready(report: dict[str, Any]) -> bool:
    checks = report.get("checks", {})
    return all(
        checks.get(name) is True
        for name in (
            "artifact_integrity",
            "runtime_backend",
            "license",
            "deterministic_request",
            "no_fallback",
        )
    ) and checks.get("generation_smoke") is False


def execute(
    *,
    run_directory: Path,
    manifest_path: Path = DEFAULT_MANIFEST,
    moodboard_directory: Path = DEFAULT_MOODBOARD,
) -> Path:
    manifest = load_manifest(manifest_path)
    run_directory.mkdir(parents=True, exist_ok=False)
    smoke_directory = run_directory / "private" / "smoke"
    smoke_directory.mkdir(parents=True)
    moodboards = [
        moodboard_directory / name
        for name in manifest["moodboard_policy"]["references_reviewed"]
    ]
    missing = [str(path) for path in moodboards if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"smoke copy checks require all moodboards: {missing}")

    adapters = {
        "qwen_image_2512": QwenImage2512Adapter(),
        "z_image": ZImageAdapter(),
    }
    first = manifest["camera_hypotheses"][0]
    seed = int(manifest["seed_policy"]["base_seed"]) + 1
    smoke_results: dict[str, Any] = {}
    smoke_passed = False
    full_bakeoff_started = False
    started_all = time.monotonic()
    try:
        initial = {model_id: adapter.readiness() for model_id, adapter in adapters.items()}
        blockers = [model_id for model_id, report in initial.items() if not _pre_smoke_ready(report)]
        if blockers:
            raise RuntimeError(f"pre-smoke readiness failed: {blockers}")

        with serialized_gpu(DEFAULT_LOCK):
            for model_id in manifest["models"]:
                adapter = adapters[model_id]
                image_path = smoke_directory / f"{model_id}_H01_smoke.png"
                request = AdapterRequest(
                    blind_id="H01_A",
                    hypothesis_id="H01",
                    model_id=model_id,
                    prompt=first["prompt"],
                    seed=seed,
                    width=int(manifest["output"]["width"]),
                    height=int(manifest["output"]["height"]),
                    camera_hypothesis=first,
                )
                started_wall = datetime.now(timezone.utc)
                started = time.monotonic()
                memory_before = _memory_snapshot()
                adapter_metadata = adapter.generate(request, image_path)
                elapsed = time.monotonic() - started
                sidecar = {
                    "schema_version": "1.0.0",
                    "experiment_id": manifest["experiment_id"],
                    "blind_id": "H01_A",
                    "hypothesis_id": "H01",
                    "model_id": model_id,
                    "seed": seed,
                    "prompt": first["prompt"],
                    "camera_hypothesis": first,
                    "output": {
                        "path": str(image_path),
                        "dimensions": [
                            manifest["output"]["width"],
                            manifest["output"]["height"],
                        ],
                        "format": manifest["output"]["format"],
                    },
                    "timing": {
                        "started_at_utc": started_wall.isoformat(),
                        "generation_seconds": round(elapsed, 3),
                    },
                    "memory": {
                        "before": memory_before,
                        "after": _memory_snapshot(),
                    },
                    "adapter_metadata": adapter_metadata,
                    "publication_allowed": False,
                    "published": False,
                }
                screen = run_prescreen(
                    image_path,
                    sidecar,
                    expected_dimensions=(
                        int(manifest["output"]["width"]),
                        int(manifest["output"]["height"]),
                    ),
                    moodboard_paths=moodboards,
                )
                sidecar["prescreen"] = screen
                validate_sidecar(sidecar)
                if adapter_metadata.get("network_at_inference") != "blocked":
                    raise RuntimeError(f"{model_id} smoke did not prove offline inference")
                if adapter_metadata.get("fallback_allowed") is not False:
                    raise RuntimeError(f"{model_id} smoke allowed model fallback")
                if not screen["passed"]:
                    raise RuntimeError(
                        f"{model_id} smoke failed pre-screen: {screen['critical_failures']}"
                    )
                sidecar_path = smoke_directory / f"{model_id}_H01_smoke.json"
                sidecar_path.write_text(
                    json.dumps(sidecar, indent=2) + "\n",
                    encoding="utf-8",
                )
                adapter.mark_smoke_passed()
                smoke_results[model_id] = {
                    "passed": True,
                    "image": str(image_path),
                    "sidecar": str(sidecar_path),
                    "seconds": round(elapsed, 3),
                    "memory": adapter_metadata.get("memory"),
                }
                adapter.cleanup()

        readiness = require_all_ready(adapters, manifest["models"])
        smoke_passed = True
        (smoke_directory / "smoke_summary.json").write_text(
            json.dumps(
                {
                    "passed": True,
                    "models": smoke_results,
                    "readiness_after_smoke": readiness,
                    "total_seconds": round(time.monotonic() - started_all, 3),
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        full_bakeoff_started = True
        return run_bakeoff(
            adapters,
            run_directory=run_directory,
            manifest_path=manifest_path,
            moodboard_directory=moodboard_directory,
            create_run_directory=False,
        )
    except Exception as exc:
        (smoke_directory / "smoke_summary.json").write_text(
            json.dumps(
                {
                    "passed": smoke_passed,
                    "models": smoke_results,
                    "error": f"{type(exc).__name__}: {exc}",
                    "full_bakeoff_started": full_bakeoff_started,
                    "total_seconds": round(time.monotonic() - started_all, 3),
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        for adapter in adapters.values():
            adapter.cleanup()
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--moodboard-directory", type=Path, default=DEFAULT_MOODBOARD)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--run-id",
        default=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run = execute(
        run_directory=args.output_root / args.run_id,
        manifest_path=args.manifest,
        moodboard_directory=args.moodboard_directory,
    )
    print(json.dumps({"run_directory": str(run), "published": False}))


if __name__ == "__main__":
    main()
