"""Generate one fail-closed five-view synthetic Rina bootstrap mini-sheet."""

from __future__ import annotations

import argparse
import gc
import json
import math
import resource
import time
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

from hyperreal.identity.audit import (
    _appearance,
    _geometry,
    sha256_file,
)
from hyperreal.identity.qc.integrity import dhash, hamming_distance
from hyperreal.identity.qc.macos_vision import MacOSVisionOCRAdapter
from hyperreal.identity.qc.mediapipe_tasks import MediaPipeTasksAdapter
from hyperreal.identity.qc.registry import verified_model_paths


ROOT = Path(__file__).resolve().parents[3]
RINA = ROOT / "rina_park"
OUTPUT_ROOT = RINA / "out/hyperreal_identity_bootstrap"
MASTER = RINA / "identity/master/rina_master_face.jpg"
MASTER_SHA256 = "70fb81b7af2928518a195210d90b29b5e74e64b305531d1f03fa06e585a7d6dc"
MODEL = (
    RINA
    / "models/qwen_image_edit/AbstractFramework-qwen-image-edit-2511-8bit-5f35885d"
)
MODEL_REPO = "AbstractFramework/qwen-image-edit-2511-8bit"
MODEL_REVISION = "5f35885dfe4061e57c87df2c03123e5e8124edfa"
MODEL_TREE_SHA256 = "6cd4bfa9e260e72b6febb2f94876f402bfa140b9c107fcf3763eba729a3334fd"
MODEL_SIZE_BYTES = 30343385206
WIDTH = HEIGHT = 768
STEPS = 20
GUIDANCE = 4.0
COMMON = (
    "Create a complete new full-frame identity portrait conditioned on Picture 1 from the first denoising step; "
    "this is full-image synthesis, not a localized patch, face swap, or masked edit. Picture 1 is the approved "
    "synthetic identity anchor for the same exact adult fictional Korean-Canadian woman, apparent age 27. Preserve "
    "her identity-defining almond eyes and eyelids, face width, softly tapered jaw, straight nose, natural lips, "
    "hairline, and long natural black hair. Head-and-shoulders framing, plain charcoal crew-neck top, neutral soft "
    "daylight, plain light-gray or off-white studio wall, realistic unretouched skin texture and pores. No glamour, "
    "no sexual styling, no heavy makeup, no beauty filter, no jewelry, no text, no watermark, no second person."
)
VIEWS: tuple[dict[str, Any], ...] = (
    {
        "slug": "01_frontal_neutral",
        "seed": 26073101,
        "view": "frontal",
        "prompt": (
            "True frontal view with a level head, both ears balanced, looking directly at camera, relaxed brows, "
            f"neutral closed mouth. {COMMON}"
        ),
    },
    {
        "slug": "02_left_three_quarter_subtle_smile",
        "seed": 26073102,
        "view": "left_three_quarter",
        "prompt": (
            "Left three-quarter view, nose pointing moderately toward image-left, both eyes visible, gaze slightly "
            f"past camera, subtle natural closed-mouth smile. {COMMON}"
        ),
    },
    {
        "slug": "03_right_three_quarter_neutral",
        "seed": 26073103,
        "view": "right_three_quarter",
        "prompt": (
            "Right three-quarter view, nose pointing moderately toward image-right, both eyes visible, gaze slightly "
            f"past camera, neutral relaxed closed mouth. {COMMON}"
        ),
    },
    {
        "slug": "04_true_left_profile_neutral",
        "seed": 26073104,
        "view": "left_profile",
        "prompt": (
            "True 90-degree left profile, nose pointing image-left, only the near eye visible, profile gaze left, "
            f"level head, neutral relaxed closed mouth. {COMMON}"
        ),
    },
    {
        "slug": "05_true_right_profile_neutral",
        "seed": 26073105,
        "view": "right_profile",
        "prompt": (
            "True 90-degree right profile, nose pointing image-right, only the near eye visible, profile gaze right, "
            f"level head, neutral relaxed closed mouth. {COMMON}"
        ),
    },
)


def _hash_directory(path: Path) -> tuple[str, int, int]:
    import hashlib

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


def validate_readiness() -> dict[str, Any]:
    if len(VIEWS) != 5 or len({item["slug"] for item in VIEWS}) != 5:
        raise RuntimeError("exactly_five_unique_views_required")
    if not MODEL.is_dir():
        raise RuntimeError(f"model_missing:{MODEL}")
    tree_hash, file_count, size_bytes = _hash_directory(MODEL)
    if tree_hash != MODEL_TREE_SHA256 or size_bytes != MODEL_SIZE_BYTES:
        raise RuntimeError(
            f"model_integrity_failed:{tree_hash}:{size_bytes}"
        )
    if not MASTER.is_file() or sha256_file(MASTER) != MASTER_SHA256:
        raise RuntimeError("master_missing_or_changed")
    if mx.default_device() != mx.gpu:
        raise RuntimeError(f"mlx_gpu_required:{mx.default_device()}")
    return {
        "ready": True,
        "route": "qwen-image-edit-2511-single-anchor-full-image-synthesis-v1",
        "route_contract": {
            "image_count": 1,
            "ordered_images": ["approved_synthetic_master_anchor"],
            "mask_argument": False,
            "localized_patch": False,
            "face_swap": False,
            "identity_conditioning": "Picture 1 vision tokens condition full-image denoising",
        },
        "model_repo": MODEL_REPO,
        "model_revision": MODEL_REVISION,
        "model_tree_sha256": tree_hash,
        "model_size_bytes": size_bytes,
        "model_file_count": file_count,
        "weights_license": "Apache-2.0",
        "runtime": f"mlx-gen=={version('mlx-gen')}",
        "runtime_license": "MIT",
        "mlx": version("mlx"),
        "device": str(mx.default_device()),
        "master_path": str(MASTER),
        "master_sha256": MASTER_SHA256,
        "forbidden_routes_used": [],
        "large_model_downloaded": False,
    }


def build_call(spec: dict[str, Any]) -> dict[str, Any]:
    return {
        "seed": int(spec["seed"]),
        "prompt": str(spec["prompt"]),
        "image_path": str(MASTER),
        "image_paths": [str(MASTER)],
        "width": WIDTH,
        "height": HEIGHT,
        "guidance": GUIDANCE,
        "num_inference_steps": STEPS,
        "scheduler": "flow_match_euler_discrete",
        "canvas_policy": "exact_resize",
    }


def _memory() -> dict[str, float]:
    return {
        "mlx_active_gib": round(mx.get_active_memory() / 1024**3, 3),
        "mlx_peak_gib": round(mx.get_peak_memory() / 1024**3, 3),
        "process_max_rss_gib": round(
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**3, 3
        ),
    }


def _sanity(path: Path) -> dict[str, Any]:
    with Image.open(path) as source:
        pixels = np.asarray(source.convert("RGB"), dtype=np.uint8)
        size = list(source.size)
    return {
        "dimensions": size,
        "mean_rgb": round(float(pixels.mean()), 6),
        "std_rgb": round(float(pixels.std()), 6),
        "black_pixel_fraction": round(float((pixels <= 2).all(axis=2).mean()), 8),
        "white_pixel_fraction": round(float((pixels >= 253).all(axis=2).mean()), 8),
        "passed": size == [WIDTH, HEIGHT] and float(pixels.std()) > 8,
    }


def _landmarks(paths: list[Path]) -> dict[Path, list[list[Any]]]:
    import mediapipe as mp
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision

    options = vision.FaceLandmarkerOptions(
        base_options=python.BaseOptions(
            model_asset_path=str(verified_model_paths()["face"])
        ),
        running_mode=vision.RunningMode.IMAGE,
        num_faces=3,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
    )
    results: dict[Path, list[list[Any]]] = {}
    with vision.FaceLandmarker.create_from_options(options) as detector:
        for path in paths:
            with Image.open(path) as source:
                array = np.ascontiguousarray(source.convert("RGB"), dtype=np.uint8)
            image = mp.Image(image_format=mp.ImageFormat.SRGB, data=array)
            results[path] = detector.detect(image).face_landmarks
    return results


def _run_qc(run: Path, outputs: list[Path]) -> list[dict[str, Any]]:
    landmark_sets = _landmarks([MASTER, *outputs])
    master_faces = landmark_sets[MASTER]
    if len(master_faces) != 1:
        raise RuntimeError("master_face_detection_not_exactly_one")
    master_points = master_faces[0]
    master_bbox = (
        min(float(point.x) for point in master_points),
        min(float(point.y) for point in master_points),
        max(float(point.x) for point in master_points),
        max(float(point.y) for point in master_points),
    )
    master_appearance = _appearance(MASTER, master_bbox, None)
    known = [(MASTER, sha256_file(MASTER), dhash(MASTER))]
    qc_results: list[dict[str, Any]] = []
    detector = MediaPipeTasksAdapter()
    ocr = MacOSVisionOCRAdapter()
    for spec, output in zip(VIEWS, outputs, strict=True):
        faces = landmark_sets[output]
        metrics: dict[str, Any] = {
            "scored": len(faces) == 1,
            "identity_score_calibrated": False,
            "age_consistency_score": None,
            "identity_proxy_cannot_auto_approve": True,
        }
        if len(faces) == 1:
            points = faces[0]
            bbox = (
                min(float(point.x) for point in points),
                min(float(point.y) for point in points),
                max(float(point.x) for point in points),
                max(float(point.y) for point in points),
            )
            metrics.update(_geometry(points, master_points))
            metrics.update(_appearance(output, bbox, master_appearance))
            metrics["identity_proxy_score"] = round(
                0.55 * metrics["landmark_geometry_score"]
                + 0.20 * metrics["eye_geometry_score"]
                + 0.15 * metrics["hair_proxy_score"]
                + 0.10 * metrics["artifact_proxy_score"],
                6,
            )
        output_hash = sha256_file(output)
        perceptual_hash = dhash(output)
        matches = [
            {
                "path": str(path),
                "exact": digest == output_hash,
                "dhash_distance": hamming_distance(perceptual_hash, phash),
            }
            for path, digest, phash in known
            if digest == output_hash or hamming_distance(perceptual_hash, phash) <= 5
        ]
        known.append((output, output_hash, perceptual_hash))
        payload = {
            "slug": spec["slug"],
            "requested_view": spec["view"],
            "sha256": output_hash,
            "dhash64": perceptual_hash,
            "sanity": _sanity(output),
            "mediapipe": detector.detect(output),
            "ocr": ocr.detect(output),
            "identity_and_appearance_proxy": metrics,
            "duplicate_check": {
                "algorithm": "sha256+dhash64",
                "threshold": 5,
                "matches": matches,
                "passed": not matches,
            },
            "human_review_required": True,
            "automatic_training_or_production_approval": False,
        }
        qc_path = run / "sidecars" / f"{spec['slug']}.qc.json"
        qc_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        qc_results.append(payload)
    return qc_results


def _contact_sheet(outputs: list[Path], output: Path) -> None:
    thumb, label_height = 300, 56
    sheet = Image.new("RGB", (len(outputs) * thumb, thumb + label_height), "white")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    for index, (spec, path) in enumerate(zip(VIEWS, outputs, strict=True)):
        with Image.open(path) as source:
            image = ImageOps.fit(
                source.convert("RGB"),
                (thumb, thumb),
                method=Image.Resampling.LANCZOS,
            )
        x = index * thumb
        sheet.paste(image, (x, 0))
        draw.text((x + 8, thumb + 8), str(spec["slug"]), fill="black", font=font)
    sheet.save(output, quality=94, subsampling=0)


def execute(run: Path) -> None:
    readiness = validate_readiness()
    finals = run / "finals"
    sidecars = run / "sidecars"
    finals.mkdir(parents=True, exist_ok=False)
    sidecars.mkdir(parents=True, exist_ok=False)
    manifest_path = run / "manifest.json"
    manifest: dict[str, Any] = {
        "schema_version": "1.0.0",
        "run_id": run.name,
        "status": "generation_in_progress",
        "review_status": "pending_user_approval",
        "permanently_non_publishable": True,
        "synthetic_bootstrap_version": "v0",
        "expected_final_image_count": 5,
        "authorized_attempts_per_view": 1,
        "technical_retries": [],
        "readiness": readiness,
        "views": [],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    from mflux.models.common.config import ModelConfig
    from mflux.models.qwen.variants.edit.qwen_image_edit import QwenImageEdit
    from mflux.utils.dimension_resolver import CANVAS_POLICY_EXACT_RESIZE

    model_config = ModelConfig.from_name(MODEL_REPO)
    mx.reset_peak_memory()
    load_started = time.monotonic()
    model = QwenImageEdit(model_config=model_config, model_path=str(MODEL))
    mx.synchronize()
    model_load_seconds = round(time.monotonic() - load_started, 3)
    outputs: list[Path] = []
    try:
        for index, spec in enumerate(VIEWS, start=1):
            mx.clear_cache()
            gc.collect()
            mx.reset_peak_memory()
            output = finals / f"{spec['slug']}.png"
            call = build_call(spec)
            call["canvas_policy"] = CANVAS_POLICY_EXACT_RESIZE
            attempt = {
                "attempt_number": 1,
                "authorized_type": "initial_generation",
                "started_at_utc": datetime.now(timezone.utc).isoformat(),
                "call": {
                    **call,
                    "canvas_policy": "CANVAS_POLICY_EXACT_RESIZE",
                },
            }
            manifest["views"].append(
                {
                    "index": index,
                    "slug": spec["slug"],
                    "requested_view": spec["view"],
                    "output": str(output),
                    "attempts": [attempt],
                }
            )
            manifest_path.write_text(
                json.dumps(manifest, indent=2), encoding="utf-8"
            )
            started = time.monotonic()
            try:
                generated = model.generate_image(**call)
                generated.save(path=output, overwrite=False)
                mx.synchronize()
                seconds = round(time.monotonic() - started, 3)
                sanity = _sanity(output)
                if not sanity["passed"]:
                    raise RuntimeError(f"technical_output_sanity_failed:{sanity}")
                attempt.update(
                    {
                        "status": "completed",
                        "generation_seconds": seconds,
                        "memory": _memory(),
                        "output_sha256": sha256_file(output),
                        "sanity": sanity,
                    }
                )
                outputs.append(output)
                print(
                    json.dumps(
                        {
                            "completed": index,
                            "slug": spec["slug"],
                            "seconds": seconds,
                            "memory": attempt["memory"],
                        }
                    ),
                    flush=True,
                )
            except Exception as error:
                attempt.update(
                    {
                        "status": "failed",
                        "failure": f"{type(error).__name__}:{error}",
                        "elapsed_seconds": round(time.monotonic() - started, 3),
                    }
                )
                manifest["status"] = "failed_closed_generation_interrupted_or_invalid"
                manifest_path.write_text(
                    json.dumps(manifest, indent=2), encoding="utf-8"
                )
                raise
            manifest_path.write_text(
                json.dumps(manifest, indent=2), encoding="utf-8"
            )
    finally:
        del model
        mx.clear_cache()
        gc.collect()

    if len(outputs) != 5:
        raise RuntimeError(f"exactly_five_outputs_required:{len(outputs)}")
    qc_results = _run_qc(run, outputs)
    contact_sheet = run / "contact_sheet_five_view_pending_user_approval.jpg"
    _contact_sheet(outputs, contact_sheet)
    manifest.update(
        {
            "status": "pending_user_approval",
            "generated_final_image_count": len(outputs),
            "model_load_seconds": model_load_seconds,
            "total_generation_seconds": round(
                sum(
                    view["attempts"][0]["generation_seconds"]
                    for view in manifest["views"]
                ),
                3,
            ),
            "peak_memory_gib": max(
                view["attempts"][0]["memory"]["mlx_peak_gib"]
                for view in manifest["views"]
            ),
            "contact_sheet": str(contact_sheet),
            "qc_results": [
                {
                    "slug": result["slug"],
                    "qc_path": str(
                        sidecars / f"{result['slug']}.qc.json"
                    ),
                    "automatic_training_or_production_approval": False,
                }
                for result in qc_results
            ],
            "visual_review": {
                "status": "pending_agent_visual_review",
                "user_approval_required": True,
            },
            "promotion": {
                "training": False,
                "production": False,
                "publishing": False,
                "calendar": False,
                "package": False,
                "mature_paths": False,
            },
        }
    )
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--readiness-only", action="store_true")
    parser.add_argument("--generate", action="store_true")
    args = parser.parse_args()
    if args.readiness_only:
        print(json.dumps(validate_readiness(), indent=2))
    elif args.generate:
        execute(OUTPUT_ROOT / args.run_id)
    else:
        parser.error("choose --readiness-only or --generate")


if __name__ == "__main__":
    main()
