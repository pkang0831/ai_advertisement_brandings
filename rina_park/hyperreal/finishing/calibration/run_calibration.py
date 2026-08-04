"""Run the isolated, non-promotable SeedVR2 adult calibration matrix."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import resource
import time
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image

from hyperreal.finishing.providers import LocalFaceLandmarks, OfflineAlexLpips
from hyperreal.finishing.runner import MODEL_PATH, _exclusive_lock
from hyperreal.finishing.validation import ValidationThresholds, validate_restoration


CALIBRATION_ROOT = Path(__file__).resolve().parent
MANIFEST_PATH = CALIBRATION_ROOT / "corpus_manifest.v1.json"
SCALES = (1.5, 2.0)
SEED = 42
STRENGTH = 0.30


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_corpus() -> list[dict[str, object]]:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    if manifest["corpus_size"] != len(manifest["assets"]):
        raise RuntimeError("calibration_corpus_size_mismatch")
    assets: list[dict[str, object]] = []
    for record in manifest["assets"]:
        path = CALIBRATION_ROOT / str(record["path"])
        if not path.is_file() or _sha256(path) != record["sha256"]:
            raise RuntimeError(f"calibration_asset_integrity_failure:{record['id']}")
        if "adult" not in str(record["source_caption_evidence"]).lower():
            raise RuntimeError(f"calibration_adult_evidence_missing:{record['id']}")
        assets.append({**record, "resolved_path": path})
    return assets


def _blend(source_path: Path, restored: Image.Image, strength: float) -> Image.Image:
    with Image.open(source_path) as source:
        baseline = source.convert("RGB").resize(restored.size, Image.Resampling.LANCZOS)
    return Image.blend(baseline, restored.convert("RGB"), strength)


def _calibrate_thresholds(
    records: list[dict[str, object]],
    guardrails: ValidationThresholds,
) -> dict[str, float]:
    metrics = [record["validation"]["metrics"] for record in records]

    def maximum(name: str) -> float:
        return max(float(item[name]) for item in metrics)

    def minimum(name: str) -> float:
        return min(float(item[name]) for item in metrics)

    return {
        "landmark_procrustes_rms_max": min(
            guardrails.landmark_procrustes_rms_max,
            maximum("landmark_procrustes_rms") * 1.20 + 0.0005,
        ),
        "face_bbox_iou_min": max(
            guardrails.face_bbox_iou_min,
            minimum("face_bbox_iou") - 0.01,
        ),
        "global_lpips_max": min(
            guardrails.global_lpips_max,
            maximum("global_lpips") * 1.20 + 0.001,
        ),
        "face_lpips_max": min(
            guardrails.face_lpips_max,
            maximum("face_lpips") * 1.20 + 0.001,
        ),
        "edge_gain_min": max(
            guardrails.edge_gain_min,
            minimum("edge_detail_gain") - 0.02,
        ),
        "edge_gain_max": min(
            guardrails.edge_gain_max,
            maximum("edge_detail_gain") + 0.05,
        ),
        "black_clip_delta_max": min(
            guardrails.black_clip_delta_max,
            max(0.001, maximum("black_clip_delta") * 1.20),
        ),
        "white_clip_delta_max": min(
            guardrails.white_clip_delta_max,
            max(0.001, maximum("white_clip_delta") * 1.20),
        ),
        "low_frequency_change_max": min(
            guardrails.low_frequency_change_max,
            maximum("low_frequency_relative_change") * 1.20 + 0.001,
        ),
        "high_frequency_gain_max": min(
            guardrails.high_frequency_gain_max,
            max(0.05, maximum("high_frequency_relative_gain") * 1.20),
        ),
    }


def run(run_id: str) -> dict[str, object]:
    output_root = CALIBRATION_ROOT / "runs" / run_id
    if output_root.exists():
        raise RuntimeError("calibration_run_directory_already_exists")
    output_root.mkdir(parents=True)
    (output_root / "CALIBRATION_ONLY_NEVER_PROMOTE").write_text(
        "Private non-Rina validation artifacts. Never publish or promote.\n",
        encoding="utf-8",
    )
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    corpus = _load_corpus()
    thresholds = ValidationThresholds()
    records: list[dict[str, object]] = []

    with _exclusive_lock():
        import mlx.core as mx
        from mflux.models.common.config.model_config import ModelConfig
        from mflux.models.common.vae.tiling_config import TilingConfig
        from mflux.models.seedvr2.variants.upscale.seedvr2 import SeedVR2
        from mflux.utils.scale_factor import ScaleFactor

        mx.set_cache_limit(8 * 1024**3)
        model = SeedVR2(
            model_path=str(MODEL_PATH),
            model_config=ModelConfig.seedvr2_3b(),
        )
        model.tiling_config = TilingConfig(
            vae_encode_tile_size=768,
            vae_encode_tile_overlap=128,
            vae_decode_tiles_per_dim=8,
        )
        lpips_provider = OfflineAlexLpips()
        with LocalFaceLandmarks() as landmark_provider:
            for scale in SCALES:
                for asset in corpus:
                    source_path = Path(asset["resolved_path"])
                    output_path = output_root / f"{asset['id']}_{scale:g}x.png"
                    started = time.perf_counter()
                    mx.reset_peak_memory()
                    generated = model.generate_image(
                        seed=SEED,
                        image_path=source_path,
                        resolution=ScaleFactor(scale),
                        softness=0.0,
                        color_correction_mode="wavelet",
                    )
                    candidate = _blend(source_path, generated.image, STRENGTH)
                    candidate.save(output_path, format="PNG", compress_level=9)
                    del candidate
                    del generated
                    mx.clear_cache()
                    validation = validate_restoration(
                        source_path,
                        output_path,
                        landmark_provider=landmark_provider,
                        lpips_provider=lpips_provider,
                        thresholds=thresholds,
                    )
                    with Image.open(output_path) as output_image:
                        output_dimensions = list(output_image.size)
                    record = {
                        "asset_id": asset["id"],
                        "scale": scale,
                        "seed": SEED,
                        "strength": STRENGTH,
                        "source_path": str(source_path.relative_to(CALIBRATION_ROOT)),
                        "source_sha256": asset["sha256"],
                        "output_path": output_path.name,
                        "output_sha256": _sha256(output_path),
                        "output_dimensions": output_dimensions,
                        "wall_seconds": round(time.perf_counter() - started, 3),
                        "mlx_peak_gib": round(mx.get_peak_memory() / 1024**3, 3),
                        "process_max_rss_gib": round(
                            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                            / 1024**3,
                            3,
                        ),
                        "validation": validation,
                        "publication_allowed": False,
                        "promotion_allowed": False,
                    }
                    records.append(record)
                    (output_root / f"{asset['id']}_{scale:g}x.json").write_text(
                        json.dumps(record, indent=2, sort_keys=True) + "\n",
                        encoding="utf-8",
                    )
                    mx.clear_cache()
                    gc.collect()
        del lpips_provider
        del model
        mx.clear_cache()
        gc.collect()
        active_after_cleanup = round(mx.get_active_memory() / 1024**3, 3)

    scale_results: dict[str, object] = {}
    for scale in SCALES:
        selected = [record for record in records if record["scale"] == scale]
        automated_pass = all(
            record["validation"]["passed"] is True for record in selected
        )
        scale_results[f"{scale:g}x"] = {
            "sample_count": len(selected),
            "all_automated_gates_passed": automated_pass,
            "eligible_for_human_scale_review": automated_pass,
            "calibrated_thresholds": (
                _calibrate_thresholds(selected, thresholds)
                if automated_pass
                else None
            ),
            "enabled": False,
            "enablement_reason": "pending_visual_identity_review"
            if automated_pass
            else "automated_validation_failed",
        }
    summary = {
        "schema_version": "1.0.0",
        "run_id": run_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "corpus_size": len(corpus),
        "config_count": len(records),
        "scales": list(SCALES),
        "seed": SEED,
        "strength": STRENGTH,
        "lpips_provider": OfflineAlexLpips.provider_id,
        "landmark_provider": LocalFaceLandmarks.provider_id,
        "records": records,
        "scale_results": scale_results,
        "mlx_active_gib_after_cleanup": active_after_cleanup,
        "rina_image_processed": False,
        "moodboard_image_processed": False,
        "publication_allowed": False,
        "promotion_allowed": False,
    }
    (output_root / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
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
