from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
from typing import Any

from .qc import evaluate_qc_report


DEFAULT_SPEC = Path(__file__).with_name("phase1_identity_dataset_spec.v1.json")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def load_spec(path: Path = DEFAULT_SPEC) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _counts(slots: list[dict[str, Any]], field: str) -> Counter[str]:
    return Counter(str(slot[field]) for slot in slots)


def _approved_asset(asset: dict[str, Any]) -> bool:
    return asset.get("human_approval", {}).get("status") == "approved"


def validate_spec(spec: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    slots = spec.get("slots", [])
    planned = spec.get("planned_slots")
    if planned != 32 or len(slots) != 32:
        errors.append("dataset must contain exactly 32 planned slots")

    slot_ids = [slot.get("slot_id") for slot in slots]
    if len(slot_ids) != len(set(slot_ids)):
        errors.append("slot IDs must be unique")
    if set(slot_ids) != {f"ID{index:02d}" for index in range(1, 33)}:
        errors.append("slot IDs must be ID01 through ID32")

    constraints = spec.get("balance_constraints", {})
    for field, key in (
        ("framing", "framing_exact"),
        ("camera_distance", "camera_distance_exact"),
        ("lighting", "lighting_exact"),
    ):
        expected = Counter(constraints.get(key, {}))
        if _counts(slots, field) != expected:
            errors.append(f"{field} balance must equal {dict(expected)}")

    yaw_counts = _counts(slots, "yaw")
    for yaw, minimum in constraints.get("yaw_minimum", {}).items():
        if yaw_counts[yaw] < minimum:
            errors.append(f"yaw {yaw} requires at least {minimum} slots")

    for field in ("expression", "gaze"):
        counts = _counts(slots, field)
        if len(counts) < 5:
            errors.append(f"{field} must cover at least five conditions")
        allowed = constraints.get(f"{field}_allowed_spread")
        if counts and allowed is not None and max(counts.values()) - min(counts.values()) > allowed:
            errors.append(f"{field} counts exceed allowed spread {allowed}")

    for field in ("background_id", "outfit_id", "pose_id"):
        values = [slot.get(field) for slot in slots]
        if len(values) != len(set(values)):
            errors.append(f"{field} must be unique across all slots")

    split_counts = _counts(slots, "split")
    split_policy = spec.get("split_policy", {})
    if split_counts != Counter(
        {
            "train": split_policy.get("train_slots"),
            "validation": split_policy.get("validation_slots"),
        }
    ):
        errors.append("train/validation split counts do not match split policy")
    actual_validation = {slot["slot_id"] for slot in slots if slot["split"] == "validation"}
    if actual_validation != set(split_policy.get("validation_slot_ids", [])):
        errors.append("validation slots do not match held-out slot list")

    assets = spec.get("assets", {})
    if not isinstance(assets, dict):
        return errors + ["assets must be an object keyed by slot ID"]
    unknown_assets = set(assets).difference(slot_ids)
    if unknown_assets:
        errors.append(f"assets reference unknown slots: {sorted(unknown_assets)}")

    required_lineage = set(spec["source_policy"]["required_lineage_fields"])
    for slot_id, asset in assets.items():
        approved = _approved_asset(asset)
        if approved:
            missing_lineage = sorted(field for field in required_lineage if not asset.get(field))
            if missing_lineage:
                errors.append(f"{slot_id} approved asset missing lineage: {missing_lineage}")
            approval = asset.get("human_approval", {})
            if not approval.get("reviewer") or not approval.get("reviewed_at_utc"):
                errors.append(f"{slot_id} approved asset lacks named, dated human approval")
            source_type = str(asset.get("source_type", ""))
            if "qwen" in source_type.lower() and approval.get("automated", False):
                errors.append(f"{slot_id} Qwen-derived candidate cannot be auto-approved")

            resolution = asset.get("face_crop_resolution")
            if (
                not isinstance(resolution, list)
                or len(resolution) != 2
                or min(resolution) < 512
            ):
                errors.append(f"{slot_id} approved face crop must be at least 512x512")
            for field in ("source_sha256", "mask_sha256"):
                if not SHA256_RE.fullmatch(str(asset.get(field, ""))):
                    errors.append(f"{slot_id} requires a valid {field}")
            for field in ("source_path", "caption", "mask_path", "license"):
                if not asset.get(field):
                    errors.append(f"{slot_id} approved asset missing {field}")
            if asset.get("critical_rejections"):
                errors.append(f"{slot_id} has uncleared critical rejections")
            qc_gate = evaluate_qc_report(asset.get("qc_report"))
            for blocker in qc_gate["blockers"]:
                errors.append(f"{slot_id} QC blocker: {blocker}")
            if qc_gate["hitl_required"]:
                errors.append(
                    f"{slot_id} requires mandatory QC HITL: {qc_gate['hitl_reasons']}"
                )
            qc_review = asset.get("qc_report", {}).get("human_qc_review", {})
            if qc_review.get("status") == "cleared" and (
                not qc_review.get("reviewer") or not qc_review.get("reviewed_at_utc")
            ):
                errors.append(f"{slot_id} cleared QC HITL lacks named, dated review")

    for hash_field in ("source_sha256",):
        approved_hashes = [
            (slot_id, asset.get(hash_field))
            for slot_id, asset in assets.items()
            if _approved_asset(asset) and asset.get(hash_field)
        ]
        duplicate_hashes = {
            value
            for _, value in approved_hashes
            if sum(candidate == value for _, candidate in approved_hashes) > 1
        }
        for value in sorted(duplicate_hashes):
            duplicate_slots = sorted(
                slot_id for slot_id, candidate in approved_hashes if candidate == value
            )
            errors.append(f"exact duplicate approved sources: {duplicate_slots}")

    approved_count = sum(_approved_asset(asset) for asset in assets.values())
    if spec.get("approved_slots") != approved_count:
        errors.append("approved_slots must equal the number of approved asset records")
    return errors


def training_readiness(spec: dict[str, Any]) -> dict[str, Any]:
    errors = validate_spec(spec)
    slots_by_id = {slot["slot_id"]: slot for slot in spec.get("slots", [])}
    approved_ids = {
        slot_id for slot_id, asset in spec.get("assets", {}).items() if _approved_asset(asset)
    }
    known_approved_ids = approved_ids.intersection(slots_by_id)
    approved_train = sum(
        slots_by_id[slot_id]["split"] == "train" for slot_id in known_approved_ids
    )
    approved_validation = sum(
        slots_by_id[slot_id]["split"] == "validation" for slot_id in known_approved_ids
    )
    gate = spec["training_gate"]
    foundation = spec["foundation_path"]
    blockers = list(errors)
    if len(approved_ids) < gate["minimum_approved_slots"]:
        blockers.append("approved total is below training minimum")
    if approved_train < gate["minimum_approved_train_slots"]:
        blockers.append("approved train split is below training minimum")
    if approved_validation < gate["minimum_approved_validation_slots"]:
        blockers.append("approved validation split is below training minimum")
    if not foundation["foundation_license_verified"]:
        blockers.append("foundation license review is incomplete")
    if not foundation["trainer_compatibility_verified"]:
        blockers.append("trainer compatibility is unverified")
    return {
        "structurally_valid": not errors,
        "training_ready": not blockers,
        "approved": {
            "total": len(approved_ids),
            "train": approved_train,
            "validation": approved_validation,
        },
        "blockers": blockers,
    }


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description="Dry-run Phase-1 identity dataset gates.")
    parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    parser.add_argument(
        "--require-training-ready",
        action="store_true",
        help="Return non-zero unless every approval and training gate passes.",
    )
    args = parser.parse_args()
    result = training_readiness(load_spec(args.spec))
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["structurally_valid"]:
        return 1
    if args.require_training_ready and not result["training_ready"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
