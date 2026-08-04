"""Calibrate deterministic finishing presets on the isolated adult corpus."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import resource
import time
from datetime import datetime, timezone
from pathlib import Path

import psutil

from hyperreal.finishing.calibration.run_calibration import (
    CALIBRATION_ROOT,
    _load_corpus,
)
from hyperreal.finishing.fallback_validation import validate_fallback
from hyperreal.finishing.non_generative import (
    ALL_PRESETS,
    ORIGINAL_METADATA,
    apply_preset,
)
from hyperreal.finishing.providers import LocalFaceLandmarks, OfflineAlexLpips
from hyperreal.finishing.runner import _exclusive_lock


CANONICAL_REPORT_PATH = CALIBRATION_ROOT / "fallback_calibration_report.v1.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run(run_id: str) -> dict[str, object]:
    output_root = CALIBRATION_ROOT / "fallback_runs" / run_id
    if output_root.exists():
        raise RuntimeError("fallback_calibration_run_directory_already_exists")
    output_root.mkdir(parents=True)
    (output_root / "CALIBRATION_ONLY_NEVER_PROMOTE").write_text(
        "Non-Rina deterministic calibration only. Never publish or promote.\n",
        encoding="utf-8",
    )
    corpus = _load_corpus()
    process = psutil.Process()
    records: list[dict[str, object]] = []

    with _exclusive_lock():
        lpips_provider = OfflineAlexLpips()
        with LocalFaceLandmarks() as landmark_provider:
            for preset in ALL_PRESETS:
                for asset in corpus:
                    source_path = Path(asset["resolved_path"])
                    output_path = (
                        None
                        if preset == ORIGINAL_METADATA
                        else output_root / f"{asset['id']}_{preset}.png"
                    )
                    rss_before = process.memory_info().rss
                    started = time.perf_counter()
                    output = apply_preset(source_path, preset)
                    if output is not None:
                        assert output_path is not None
                        output.save(output_path, format="PNG", compress_level=9)
                        output.close()
                    operation_seconds = time.perf_counter() - started
                    validation = validate_fallback(
                        source_path,
                        output_path,
                        preset=preset,
                        landmark_provider=landmark_provider,
                        lpips_provider=lpips_provider,
                    )
                    wall_seconds = time.perf_counter() - started
                    rss_after = process.memory_info().rss
                    record = {
                        "asset_id": asset["id"],
                        "preset": preset,
                        "source_path": str(source_path.relative_to(CALIBRATION_ROOT)),
                        "source_sha256": asset["sha256"],
                        "output_path": output_path.name if output_path else None,
                        "output_sha256": _sha256(output_path) if output_path else None,
                        "operation_seconds": round(operation_seconds, 6),
                        "wall_seconds_with_metrics": round(wall_seconds, 6),
                        "rss_before_gib": round(rss_before / 1024**3, 6),
                        "rss_after_gib": round(rss_after / 1024**3, 6),
                        "rss_delta_gib": round((rss_after - rss_before) / 1024**3, 6),
                        "process_max_rss_gib": round(
                            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                            / 1024**3,
                            6,
                        ),
                        "validation": validation,
                        "generative_model_used": False,
                        "learned_image_operation_used": False,
                        "publication_allowed": False,
                        "promotion_allowed": False,
                    }
                    records.append(record)
                    (output_root / f"{asset['id']}_{preset}.json").write_text(
                        json.dumps(record, indent=2, sort_keys=True) + "\n",
                        encoding="utf-8",
                    )
                    gc.collect()
        del lpips_provider
        gc.collect()

    preset_results: dict[str, object] = {}
    for preset in ALL_PRESETS:
        selected = [record for record in records if record["preset"] == preset]
        passed = all(record["validation"]["passed"] is True for record in selected)
        metric_names = selected[0]["validation"]["metrics"].keys()
        metric_summary = {}
        for name in metric_names:
            values = [
                float(record["validation"]["metrics"][name]) for record in selected
            ]
            metric_summary[name] = {
                "mean": sum(values) / len(values),
                "min": min(values),
                "max": max(values),
            }
        preset_results[preset] = {
            "sample_count": len(selected),
            "pass_count": sum(
                record["validation"]["passed"] is True for record in selected
            ),
            "enabled": passed,
            "verdict": "enabled" if passed else "blocked",
            "blockers": sorted(
                {
                    blocker
                    for record in selected
                    for blocker in record["validation"]["blockers"]
                }
            ),
            "metrics": metric_summary,
            "runtime": {
                "operation_seconds_mean": sum(
                    float(record["operation_seconds"]) for record in selected
                )
                / len(selected),
                "operation_seconds_max": max(
                    float(record["operation_seconds"]) for record in selected
                ),
                "wall_seconds_with_metrics_mean": sum(
                    float(record["wall_seconds_with_metrics"]) for record in selected
                )
                / len(selected),
            },
            "memory": {
                "rss_after_gib_max": max(
                    float(record["rss_after_gib"]) for record in selected
                ),
                "rss_delta_gib_max": max(
                    float(record["rss_delta_gib"]) for record in selected
                ),
                "process_max_rss_gib": max(
                    float(record["process_max_rss_gib"]) for record in selected
                ),
            },
        }
    enabled = [
        preset
        for preset, result in preset_results.items()
        if result["enabled"] is True
    ]
    default = (
        ORIGINAL_METADATA
        if ORIGINAL_METADATA in enabled
        else None
    )
    summary = {
        "schema_version": "1.0.0",
        "run_id": run_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "corpus_size": len(corpus),
        "configuration_count": len(records),
        "presets": list(ALL_PRESETS),
        "records": records,
        "preset_results": preset_results,
        "enabled_presets": enabled,
        "production_default": default,
        "thresholds_relaxed": False,
        "generative_model_used": False,
        "rina_image_processed": False,
        "moodboard_image_processed": False,
        "publication_allowed": False,
        "promotion_allowed": False,
    }
    rendered = json.dumps(summary, indent=2, sort_keys=True) + "\n"
    (output_root / "summary.json").write_text(rendered, encoding="utf-8")
    temporary = CANONICAL_REPORT_PATH.with_suffix(".json.tmp")
    temporary.write_text(rendered, encoding="utf-8")
    temporary.replace(CANONICAL_REPORT_PATH)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-id",
        default=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
    )
    args = parser.parse_args()
    print(json.dumps(run(args.run_id), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
