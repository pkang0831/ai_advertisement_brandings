"""Read-only identity asset audit and fail-closed LoRA dataset manifest builder."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from hyperreal.identity.qc.integrity import dhash, hamming_distance
from hyperreal.identity.qc.registry import verified_model_paths


ROOT = Path(__file__).resolve().parents[3]
RINA = ROOT / "rina_park"
AUDIT_DIR = Path(__file__).resolve().parent / "audit"
MASTER = RINA / "identity/master/rina_master_face.jpg"
MASTER_SHA256 = "70fb81b7af2928518a195210d90b29b5e74e64b305531d1f03fa06e585a7d6dc"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
TRAINING_FACE_MIN = 512
MIN_UNIQUE_APPROVED = 32
REQUIRED_YAW_BINS = {
    "frontal": 8,
    "left_three_quarter": 6,
    "right_three_quarter": 6,
    "left_profile": 4,
    "right_profile": 4,
    "rear_three_quarter": 4,
}
FEATURE_INDICES = (
    33,
    133,
    362,
    263,
    1,
    2,
    4,
    61,
    291,
    0,
    17,
    78,
    308,
    152,
    234,
    454,
    10,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sidecar(path: Path) -> tuple[Path | None, dict[str, Any]]:
    candidate = Path(str(path) + ".json")
    if not candidate.is_file():
        return None, {}
    try:
        return candidate, json.loads(candidate.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return candidate, {}


def classify(path: Path) -> str:
    relative = path.relative_to(RINA).as_posix()
    name = path.name.lower()
    if relative.startswith("moodboard/"):
        return "moodboard_reference_only"
    if any(
        token in name
        for token in ("contact_sheet", "mask", "overlay", "heatmap", "before", "diagnostic")
    ):
        return "derived_review_artifact"
    if relative == "identity/master/rina_master_face.jpg":
        return "master_anchor"
    if relative.startswith("identity/qwen_image_edit_reference_pack/"):
        return "qwen_reference_candidate"
    if relative.startswith("identity/qwen_image_edit_smoke/"):
        return "qwen_reference_candidate"
    if relative.startswith("identity/expression_reference_candidates/"):
        return "expression_candidate"
    if relative.startswith("identity/photomaker_v1_smoke/"):
        return "photomaker_candidate"
    if relative.startswith("identity/face_lock/"):
        return "legacy_face_lock_candidate"
    if relative.startswith("identity/beauty_candidates/"):
        return "legacy_beauty_candidate"
    if relative.startswith("out/hyperreal_identity_pilot/"):
        return "identity_pilot_artifact"
    if relative.startswith("out/review_samples/"):
        return "scene_output_candidate"
    if relative.startswith("private/mature_non_explicit/private_media/"):
        return "private_scene_output_candidate"
    return "other_image_asset"


def discover_assets() -> list[Path]:
    roots = (
        RINA / "identity",
        RINA / "moodboard",
        RINA / "out/review_samples",
        RINA / "out/hyperreal_identity_pilot",
        RINA / "private/mature_non_explicit/private_media",
    )
    paths: set[Path] = set()
    for root in roots:
        if root.is_dir():
            paths.update(
                path
                for path in root.rglob("*")
                if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
            )
    return sorted(paths)


def _normalized_points(points: list[Any]) -> np.ndarray:
    array = np.array([[float(point.x), float(point.y)] for point in points], dtype=np.float64)
    left, right = array[33], array[263]
    center = (left + right) / 2
    delta = right - left
    angle = math.atan2(delta[1], delta[0])
    rotation = np.array(
        [[math.cos(-angle), -math.sin(-angle)], [math.sin(-angle), math.cos(-angle)]]
    )
    span = max(float(np.linalg.norm(delta)), 1e-6)
    return ((array - center) @ rotation.T) / span


def _geometry(points: list[Any], master_points: list[Any]) -> dict[str, float]:
    normalized = _normalized_points(points)
    master = _normalized_points(master_points)
    rms = float(
        np.sqrt(
            np.mean(
                (normalized[list(FEATURE_INDICES)] - master[list(FEATURE_INDICES)]) ** 2
            )
        )
    )
    geometry_score = math.exp(-4.0 * rms)
    left_eye = float(np.linalg.norm(normalized[33] - normalized[133]))
    right_eye = float(np.linalg.norm(normalized[362] - normalized[263]))
    master_left = float(np.linalg.norm(master[33] - master[133]))
    master_right = float(np.linalg.norm(master[362] - master[263]))
    eye_delta = (abs(left_eye - master_left) + abs(right_eye - master_right)) / 2
    eye_score = math.exp(-6.0 * eye_delta)
    eye_mid_x = (float(points[33].x) + float(points[263].x)) / 2
    eye_span = max(abs(float(points[263].x) - float(points[33].x)), 1e-6)
    face_height = max(
        abs(float(points[152].y) - (float(points[33].y) + float(points[263].y)) / 2),
        1e-6,
    )
    yaw = max(-90.0, min(90.0, (float(points[1].x) - eye_mid_x) / eye_span * 90.0))
    expected_nose_y = (float(points[33].y) + float(points[263].y)) / 2 + face_height * 0.45
    pitch = max(
        -90.0,
        min(90.0, (float(points[1].y) - expected_nose_y) / face_height * 90.0),
    )
    return {
        "landmark_geometry_rms": round(rms, 6),
        "landmark_geometry_score": round(geometry_score, 6),
        "eye_geometry_score": round(eye_score, 6),
        "yaw_degrees_proxy": round(yaw, 3),
        "pitch_degrees_proxy": round(pitch, 3),
    }


def _appearance(
    image_path: Path,
    bbox: tuple[float, float, float, float],
    master_metrics: dict[str, float] | None,
) -> dict[str, float]:
    with Image.open(image_path) as source:
        pixels = np.asarray(source.convert("RGB"), dtype=np.float32)
    height, width = pixels.shape[:2]
    x0, y0, x1, y1 = bbox
    fw, fh = x1 - x0, y1 - y0
    hx0 = max(0, round((x0 - 0.45 * fw) * width))
    hx1 = min(width, round((x1 + 0.45 * fw) * width))
    hy0 = max(0, round((y0 - 0.65 * fh) * height))
    hy1 = min(height, round((y1 + 0.15 * fh) * height))
    head = pixels[hy0:hy1, hx0:hx1]
    luma = head.mean(axis=2)
    dark_fraction = float((luma < 75).mean()) if head.size else 0.0
    fx0, fx1 = max(0, round(x0 * width)), min(width, round(x1 * width))
    fy0, fy1 = max(0, round(y0 * height)), min(height, round(y1 * height))
    face = pixels[fy0:fy1, fx0:fx1]
    gray = face.mean(axis=2) if face.size else np.zeros((1, 1), dtype=np.float32)
    gradient = (
        float(np.mean(np.abs(np.diff(gray, axis=0))))
        + float(np.mean(np.abs(np.diff(gray, axis=1))))
    ) / 2
    clipping = float(((face <= 2) | (face >= 253)).mean()) if face.size else 1.0
    metrics = {
        "head_dark_fraction": round(dark_fraction, 6),
        "face_gradient_mean": round(gradient, 6),
        "face_clipping_fraction": round(clipping, 6),
    }
    if master_metrics is None:
        metrics.update({"hair_proxy_score": 1.0, "artifact_proxy_score": 1.0})
    else:
        metrics["hair_proxy_score"] = round(
            math.exp(-4.0 * abs(dark_fraction - master_metrics["head_dark_fraction"])),
            6,
        )
        gradient_ratio = abs(
            math.log((gradient + 1e-6) / (master_metrics["face_gradient_mean"] + 1e-6))
        )
        clipping_delta = abs(
            clipping - master_metrics["face_clipping_fraction"]
        )
        metrics["artifact_proxy_score"] = round(
            math.exp(-gradient_ratio - 8.0 * clipping_delta), 6
        )
    return metrics


def _yaw_bin(yaw: float) -> str:
    if yaw <= -65:
        return "left_profile"
    if yaw < -15:
        return "left_three_quarter"
    if yaw <= 15:
        return "frontal"
    if yaw < 65:
        return "right_three_quarter"
    return "right_profile"


def _review_status(sidecar: dict[str, Any]) -> str | None:
    for field in ("review_status", "production_status", "status"):
        value = sidecar.get(field)
        if isinstance(value, str):
            return value
    return None


def _disposition(
    category: str,
    sidecar: dict[str, Any],
    metrics: dict[str, Any] | None,
    duplicate_group_size: int,
    near_duplicate_group_size: int,
) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    status = (_review_status(sidecar) or "").lower()
    if category == "moodboard_reference_only":
        return False, [
            "moodboard_people_are_composition_lighting_only",
            "prohibited_as_rina_identity_or_training_data",
        ]
    if category == "derived_review_artifact":
        return False, ["contact_sheet_mask_overlay_or_diagnostic_not_a_source_asset"]
    if "reject" in status or category in ("photomaker_candidate", "identity_pilot_artifact"):
        reasons.append("explicitly_rejected_or_failed_identity_route")
    if category == "master_anchor":
        reasons.append("master_is_identity_anchor_not_automatically_training_approved")
    if duplicate_group_size > 1:
        reasons.append("exact_duplicate_group_must_not_inflate_dataset")
    if near_duplicate_group_size > 1:
        reasons.append("perceptual_near_duplicate_group_requires_deduplication")
    if metrics is None or metrics.get("face_count") != 1:
        reasons.append("exactly_one_face_not_reliably_detected")
    elif min(metrics["face_resolution"]) < TRAINING_FACE_MIN:
        reasons.append(f"pre_resize_face_short_side_below_{TRAINING_FACE_MIN}")
    reasons.extend(
        [
            "no_calibrated_face_identity_embedding_score",
            "no_calibrated_age_consistency_score",
            "training_specific_human_identity_approval_missing",
            "commercial_training_rights_or_consent_record_missing",
            "aligned_face_hair_loss_mask_missing",
            "complete_training_lineage_and_caption_missing",
        ]
    )
    # Existing internal-reference approval is deliberately not training approval.
    return False, sorted(set(reasons))


def _kmeans(records: list[dict[str, Any]], k: int = 4) -> None:
    scored = [record for record in records if (record.get("metrics") or {}).get("scored")]
    if not scored:
        return
    matrix = np.array(
        [
            [
                record["metrics"]["yaw_degrees_proxy"] / 90.0,
                record["metrics"]["pitch_degrees_proxy"] / 45.0,
                record["metrics"]["landmark_geometry_score"],
                record["metrics"]["eye_geometry_score"],
                record["metrics"]["hair_proxy_score"],
                record["metrics"]["artifact_proxy_score"],
            ]
            for record in scored
        ],
        dtype=np.float64,
    )
    k = min(k, len(scored))
    order = np.argsort(matrix[:, 0])
    centers = matrix[order[np.linspace(0, len(order) - 1, k, dtype=int)]].copy()
    labels = np.zeros(len(scored), dtype=int)
    for _ in range(30):
        distances = ((matrix[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        new_labels = distances.argmin(axis=1)
        new_centers = np.array(
            [
                matrix[new_labels == index].mean(axis=0)
                if np.any(new_labels == index)
                else centers[index]
                for index in range(k)
            ]
        )
        if np.array_equal(labels, new_labels) and np.allclose(centers, new_centers):
            break
        labels, centers = new_labels, new_centers
    for record, label in zip(scored, labels, strict=True):
        record["metrics"]["cluster_id"] = int(label)


def build_audit() -> tuple[dict[str, Any], dict[str, Any]]:
    import mediapipe as mp
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision

    paths = discover_assets()
    model_path = verified_model_paths()["face"]
    options = vision.FaceLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=str(model_path)),
        running_mode=vision.RunningMode.IMAGE,
        num_faces=3,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
    )
    detections: dict[Path, list[list[Any]]] = {}
    with vision.FaceLandmarker.create_from_options(options) as detector:
        for path in paths:
            if classify(path) in ("moodboard_reference_only", "derived_review_artifact"):
                continue
            try:
                with Image.open(path) as source:
                    srgb = np.ascontiguousarray(source.convert("RGB"), dtype=np.uint8)
                image = mp.Image(image_format=mp.ImageFormat.SRGB, data=srgb)
                result = detector.detect(image)
                detections[path] = result.face_landmarks
            except Exception:
                detections[path] = []
    master_faces = detections.get(MASTER, [])
    if len(master_faces) != 1:
        raise RuntimeError("approved master must have exactly one detected face")
    master_points = master_faces[0]
    master_bbox = (
        min(float(point.x) for point in master_points),
        min(float(point.y) for point in master_points),
        max(float(point.x) for point in master_points),
        max(float(point.y) for point in master_points),
    )
    master_appearance = _appearance(MASTER, master_bbox, None)

    records: list[dict[str, Any]] = []
    hash_groups: dict[str, list[int]] = defaultdict(list)
    for path in paths:
        digest = sha256_file(path)
        sidecar_path, sidecar = _sidecar(path)
        category = classify(path)
        faces = detections.get(path, [])
        metrics: dict[str, Any] | None = None
        if category not in ("moodboard_reference_only", "derived_review_artifact"):
            metrics = {"scored": len(faces) == 1, "face_count": len(faces)}
            if len(faces) == 1:
                points = faces[0]
                bbox = (
                    min(float(point.x) for point in points),
                    min(float(point.y) for point in points),
                    max(float(point.x) for point in points),
                    max(float(point.y) for point in points),
                )
                with Image.open(path) as image:
                    width, height = image.size
                resolution = [
                    round((bbox[2] - bbox[0]) * width),
                    round((bbox[3] - bbox[1]) * height),
                ]
                metrics.update(_geometry(points, master_points))
                metrics.update(_appearance(path, bbox, master_appearance))
                metrics["face_resolution"] = resolution
                metrics["yaw_bin"] = _yaw_bin(metrics["yaw_degrees_proxy"])
                metrics["identity_proxy_score"] = round(
                    0.55 * metrics["landmark_geometry_score"]
                    + 0.20 * metrics["eye_geometry_score"]
                    + 0.15 * metrics["hair_proxy_score"]
                    + 0.10 * metrics["artifact_proxy_score"],
                    6,
                )
                metrics["age_consistency_score"] = None
                metrics["identity_score_calibrated"] = False
        record = {
            "asset_id": f"A{len(records) + 1:03d}",
            "path": path.relative_to(ROOT).as_posix(),
            "sha256": digest,
            "dhash64": dhash(path),
            "bytes": path.stat().st_size,
            "category": category,
            "sidecar_path": sidecar_path.relative_to(ROOT).as_posix() if sidecar_path else None,
            "review_status": _review_status(sidecar),
            "metrics": metrics,
        }
        hash_groups[digest].append(len(records))
        records.append(record)
    for digest, indices in hash_groups.items():
        group_id = f"D{min(indices) + 1:03d}" if len(indices) > 1 else None
        for index in indices:
            records[index]["exact_duplicate_group"] = group_id
            records[index]["exact_duplicate_group_size"] = len(indices)
    parent = list(range(len(records)))

    def root(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        left_root, right_root = root(left), root(right)
        if left_root != right_root:
            parent[right_root] = left_root

    for left in range(len(records)):
        if records[left]["category"] == "derived_review_artifact":
            continue
        for right in range(left + 1, len(records)):
            if records[right]["category"] == "derived_review_artifact":
                continue
            if hamming_distance(records[left]["dhash64"], records[right]["dhash64"]) <= 5:
                union(left, right)
    near_groups: dict[int, list[int]] = defaultdict(list)
    for index in range(len(records)):
        near_groups[root(index)].append(index)
    for indices in near_groups.values():
        group_id = f"N{min(indices) + 1:03d}" if len(indices) > 1 else None
        for index in indices:
            records[index]["near_duplicate_group"] = group_id
            records[index]["near_duplicate_group_size"] = len(indices)
    _kmeans(records)
    for record in records:
        # Preserve adjacent sidecar status in the rejection decision.
        sidecar_path = ROOT / record["sidecar_path"] if record["sidecar_path"] else None
        sidecar = (
            json.loads(sidecar_path.read_text(encoding="utf-8"))
            if sidecar_path and sidecar_path.is_file()
            else {}
        )
        include, reasons = _disposition(
            record["category"],
            sidecar,
            record["metrics"],
            record["exact_duplicate_group_size"],
            record["near_duplicate_group_size"],
        )
        record["include_for_training"] = include
        record["include_exclude_reasons"] = reasons

    scored = [record for record in records if (record.get("metrics") or {}).get("scored")]
    face_sizes = [min(record["metrics"]["face_resolution"]) for record in scored]
    yaw_counts = Counter(record["metrics"]["yaw_bin"] for record in scored)
    category_counts = Counter(record["category"] for record in records)
    review_counts = Counter(record["review_status"] or "missing" for record in records)
    duplicate_assets = sum(
        1 for record in records if record["exact_duplicate_group_size"] > 1
    )
    near_duplicate_assets = sum(
        1 for record in records if record["near_duplicate_group_size"] > 1
    )
    internal_approved = sum(
        1
        for record in records
        if record["review_status"] == "approved_internal_production_reference"
    )
    training_resolution_pass = sum(size >= TRAINING_FACE_MIN for size in face_sizes)

    def score_distribution(field: str) -> dict[str, float | None]:
        values = [float(record["metrics"][field]) for record in scored]
        return {
            "min": round(min(values), 6) if values else None,
            "p25": round(float(np.percentile(values, 25)), 6) if values else None,
            "median": round(float(np.median(values)), 6) if values else None,
            "p75": round(float(np.percentile(values, 75)), 6) if values else None,
            "max": round(max(values), 6) if values else None,
        }

    category_score_distribution: dict[str, dict[str, float | int]] = {}
    for category in sorted({record["category"] for record in scored}):
        values = [
            record["metrics"]["identity_proxy_score"]
            for record in scored
            if record["category"] == category
        ]
        category_score_distribution[category] = {
            "count": len(values),
            "median": round(float(np.median(values)), 6),
            "min": round(min(values), 6),
            "max": round(max(values), 6),
        }
    report = {
        "schema_version": "1.0.0",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "verdict": "fail_closed_insufficient_training_data_and_unavailable_identity_validation",
        "counts": {
            "assets_total": len(records),
            "synthetic_or_anchor_scored": len(scored),
            "moodboard_reference_only": category_counts["moodboard_reference_only"],
            "internal_reference_approved_not_training_approved": internal_approved,
            "training_included": 0,
            "training_face_resolution_pass": training_resolution_pass,
            "exact_duplicate_assets": duplicate_assets,
            "perceptual_near_duplicate_assets_dhash5": near_duplicate_assets,
        },
        "distributions": {
            "category_counts": dict(sorted(category_counts.items())),
            "review_status_counts": dict(sorted(review_counts.items())),
            "yaw_bin_counts_scored": dict(sorted(yaw_counts.items())),
            "identity_proxy_score": score_distribution("identity_proxy_score"),
            "landmark_geometry_score": score_distribution("landmark_geometry_score"),
            "eye_geometry_score": score_distribution("eye_geometry_score"),
            "hair_dark_region_proxy_score": score_distribution("hair_proxy_score"),
            "artifact_gradient_clipping_proxy_score": score_distribution(
                "artifact_proxy_score"
            ),
            "age_consistency_score": None,
            "identity_proxy_by_category": category_score_distribution,
            "face_short_side_px": {
                "min": min(face_sizes) if face_sizes else None,
                "median": round(float(np.median(face_sizes)), 1) if face_sizes else None,
                "max": max(face_sizes) if face_sizes else None,
            },
            "cluster_counts": dict(
                sorted(
                    Counter(
                        str(record["metrics"].get("cluster_id"))
                        for record in scored
                        if record["metrics"].get("cluster_id") is not None
                    ).items()
                )
            ),
        },
        "metric_limits": {
            "identity_proxy_is_not_face_recognition": True,
            "age_consistency_unavailable": True,
            "hair_metric_is_dark_hair_region_proxy_only": True,
            "artifact_metric_is_gradient_and_clipping_proxy_only": True,
            "no_proxy_can_approve_identity_or_training_use": True,
        },
        "training_gate": {
            "minimum_unique_approved": MIN_UNIQUE_APPROVED,
            "current_unique_approved": 0,
            "required_yaw_bins": REQUIRED_YAW_BINS,
            "current_training_approved_yaw_bins": {},
            "all_sources_need_512px_face": True,
            "all_sources_need_masks_rights_lineage_caption_hitl": True,
            "ready": False,
        },
        "smallest_data_acquisition_plan": {
            "target": "32 unique owner-controlled or separately licensed source photographs",
            "capture_plan": {
                "frontal": 8,
                "left_three_quarter": 6,
                "right_three_quarter": 6,
                "left_profile": 4,
                "right_profile": 4,
                "rear_three_quarter": 4,
            },
            "requirements": [
                "pre-resize detected face at least 512x512",
                "24 train and 8 held-out validation sources",
                "balanced neutral, soft-smile, serious, teeth-smile/laugh, down/up gaze",
                "unique capture bursts, backgrounds, outfits, and pose combinations",
                "pixel-aligned human-reviewed face/hair masks excluding neck/body",
                "complete rights/consent, lineage, hashes, captions, and QC",
            ],
            "user_approvals_required": [
                "confirm the selected master is the intended identity anchor and may be used as a training source or anchor-only",
                "approve each source identity match, age consistency, hairline, eyes, and artifact quality",
                "approve commercial training/derivative rights for every source",
                "approve every face/hair mask and caption",
                "approve the final 24/8 split after duplicate and leakage review",
                "separately authorize trainer/runtime validation and training start",
            ],
        },
        "architecture": {
            "external_alpha": "closed",
            "native_multi_reference_mask": "closed_for_mlx_gen_0.23.1",
            "direct_qwen_lora": "deferred_trainer_and_dataset_not_ready",
            "next_gate": "owner_controlled_multi_angle_source_acquisition_and_per_asset_approval",
        },
    }
    manifest = {
        "schema_version": "1.0.0",
        "purpose": "candidate inventory only; no files copied or promoted",
        "master_sha256": MASTER_SHA256,
        "moodboard_policy": (
            "Moodboard people are composition/lighting references only and are prohibited "
            "as Rina identity, training data, face references, or promotion candidates."
        ),
        "rejected_asset_policy": (
            "A rejected output remains excluded even if it resembles the master; resemblance "
            "or proxy scores never override rejection or missing provenance."
        ),
        "records": records,
    }
    return manifest, report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=AUDIT_DIR)
    args = parser.parse_args()
    manifest, report = build_audit()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "candidate_dataset_manifest.v1.json"
    report_path = args.output_dir / "audit_report.v1.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "manifest": str(manifest_path),
                "report": str(report_path),
                "verdict": report["verdict"],
                "counts": report["counts"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
