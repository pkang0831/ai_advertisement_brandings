"""SDXL character LoRA dataset prep from screenshot intake (MediaPipe QC, no InsightFace)."""

from __future__ import annotations

import hashlib
import json
import random
import shutil
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from PIL import Image

from hyperreal.identity.qc.mediapipe_tasks import MediaPipeTasksAdapter

TRIGGER_TOKEN = "rina_park_person"
CHARACTER_ID = "rina_character_v1"
REPO_ROOT = Path(__file__).resolve().parents[4]
TRAINING_ROOT = Path(__file__).resolve().parent
INTAKE_ROOT = TRAINING_ROOT / "intake" / CHARACTER_ID
RAW_DIR = INTAKE_ROOT / "raw"
DATASET_DIR = INTAKE_ROOT / "dataset"
MANIFESTS_DIR = INTAKE_ROOT / "manifests"
TRAIN_DIR = DATASET_DIR / "train"
VAL_DIR = DATASET_DIR / "val"

# QC thresholds tuned for glam/portrait screenshots (~1k–1.8k px)
MIN_FACE_SHORT_SIDE_PX = 96
MIN_FACE_AREA_FRAC = 0.012
TINY_FACE_AREA_FRAC = 0.008
BACK_OF_HEAD_YAW_ABS = 62.0
NEAR_DUP_HAMMING = 6
VAL_FRACTION = 0.15
SPLIT_SEED = 42

APPEARANCE_BASE = (
    "young east asian woman, long brown hair, fair skin, slim athletic physique, "
    "glam makeup, glossy lips, brown eyes"
)


def nfc(name: str) -> str:
    return unicodedata.normalize("NFC", name)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def average_hash(path: Path, hash_size: int = 16) -> int:
    with Image.open(path) as image:
        gray = image.convert("L").resize((hash_size, hash_size), Image.Resampling.LANCZOS)
        if hasattr(gray, "get_flattened_data"):
            pixels = list(gray.get_flattened_data())
        else:
            pixels = list(gray.getdata())
    mean = sum(pixels) / len(pixels)
    bits = 0
    for index, value in enumerate(pixels):
        if value >= mean:
            bits |= 1 << index
    return bits


def _repo_relative(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(REPO_ROOT.resolve()))
    except ValueError:
        return str(resolved)


def hamming(a: int, b: int) -> int:
    return (a ^ b).bit_count()


def framing_tags(landmarks: dict[str, Any], width: int, height: int) -> list[str]:
    tags: list[str] = []
    face_box = landmarks.get("face_bbox_normalized")
    subject = landmarks.get("subject") or {}
    occupancy = subject.get("occupancy_fraction")
    if face_box:
        face_h = (face_box[3] - face_box[1]) * height
        face_frac = (face_box[2] - face_box[0]) * (face_box[3] - face_box[1])
        if face_frac > 0.18 or face_h > height * 0.45:
            tags.append("close-up portrait")
        elif face_frac > 0.06:
            tags.append("upper body")
        else:
            tags.append("full body or medium wide")
    if occupancy is not None:
        if occupancy > 0.45:
            tags.append("subject fills frame")
        elif occupancy < 0.18:
            tags.append("wide environmental framing")
    pose = landmarks.get("pose") or {}
    if pose.get("visible"):
        tags.append("body pose visible")
    else:
        tags.append("face-focused framing")
    head = landmarks.get("head_pose") or {}
    yaw = head.get("yaw_degrees")
    if isinstance(yaw, (int, float)):
        if abs(yaw) < 12:
            tags.append("facing camera")
        elif abs(yaw) < 35:
            tags.append("three-quarter view")
        elif abs(yaw) < BACK_OF_HEAD_YAW_ABS:
            tags.append("profile-ish angle")
        else:
            tags.append("extreme side or rear angle")
    return tags


def caption_for(landmarks: dict[str, Any], width: int, height: int) -> str:
    tags = framing_tags(landmarks, width, height)
    scene = "glam influencer photo, polished skin, soft studio or outdoor lighting"
    parts = [TRIGGER_TOKEN, APPEARANCE_BASE, ", ".join(tags), scene]
    return ", ".join(parts)


def evaluate_keep(landmarks: dict[str, Any], width: int, height: int) -> tuple[bool, list[str], list[str]]:
    """Return (keep, reject_reasons, soft_flags)."""
    reasons: list[str] = []
    flags: list[str] = []
    if landmarks.get("status") != "complete" or not landmarks.get("available"):
        reasons.append("landmark_detector_unavailable_or_failed")
        return False, reasons, flags

    face_count = landmarks.get("face_count") or 0
    if face_count == 0:
        # Body-only may still be useful if pose is strong
        pose = landmarks.get("pose") or {}
        if pose.get("visible") and (landmarks.get("subject") or {}).get("occupancy_fraction", 0) >= 0.12:
            flags.append("no_face_but_body_pose_kept_for_physique")
        else:
            reasons.append("no_face_and_insufficient_body")
            return False, reasons, flags
    if face_count and face_count > 2:
        reasons.append("too_many_faces_ambiguous_identity")
        return False, reasons, flags
    if face_count == 2:
        flags.append("secondary_face_present_review")

    face_res = landmarks.get("face_crop_effective_resolution")
    face_box = landmarks.get("face_bbox_normalized")
    if face_box and face_res:
        short_side = min(face_res)
        area_frac = (face_box[2] - face_box[0]) * (face_box[3] - face_box[1])
        if short_side < MIN_FACE_SHORT_SIDE_PX or area_frac < TINY_FACE_AREA_FRAC:
            reasons.append("tiny_face")
            return False, reasons, flags
        if area_frac < MIN_FACE_AREA_FRAC:
            flags.append("small_face_borderline")

    head = landmarks.get("head_pose") or {}
    yaw = head.get("yaw_degrees")
    if isinstance(yaw, (int, float)) and abs(yaw) >= BACK_OF_HEAD_YAW_ABS:
        # Keep only if body occupancy is strong (physique training)
        occ = (landmarks.get("subject") or {}).get("occupancy_fraction") or 0
        if occ < 0.2:
            reasons.append("extreme_back_of_head_or_profile_without_body")
            return False, reasons, flags
        flags.append("extreme_yaw_kept_for_body")

    # UI chrome heuristic: very thin letterboxing / solid bars via edge variance is hard;
    # flag unusually small subject occupancy with no face as chrome-only.
    occ = (landmarks.get("subject") or {}).get("occupancy_fraction")
    if face_count == 0 and (occ is None or occ < 0.08):
        reasons.append("ui_chrome_or_empty_subject")
        return False, reasons, flags

    if width < 512 or height < 512:
        reasons.append("below_min_resolution_512")
        return False, reasons, flags

    return True, reasons, flags


def discover_raw_images(raw_dir: Path = RAW_DIR) -> list[Path]:
    return sorted(raw_dir.glob("*.png"), key=lambda p: nfc(p.name))


def build_dataset(
    *,
    raw_dir: Path = RAW_DIR,
    dataset_dir: Path = DATASET_DIR,
    manifests_dir: Path = MANIFESTS_DIR,
    adapter: MediaPipeTasksAdapter | None = None,
    val_fraction: float = VAL_FRACTION,
    seed: int = SPLIT_SEED,
) -> dict[str, Any]:
    manifests_dir.mkdir(parents=True, exist_ok=True)
    train_dir = dataset_dir / "train"
    val_dir = dataset_dir / "val"
    if dataset_dir.exists():
        shutil.rmtree(dataset_dir)
    train_dir.mkdir(parents=True)
    val_dir.mkdir(parents=True)

    detector = adapter or MediaPipeTasksAdapter()
    images = discover_raw_images(raw_dir)
    records: list[dict[str, Any]] = []
    hash_to_first: dict[str, str] = {}
    phash_kept: list[tuple[int, str]] = []

    for index, path in enumerate(images, start=1):
        stem = f"{CHARACTER_ID}_{index:03d}"
        file_hash = sha256_file(path)
        with Image.open(path) as im:
            width, height = im.size
        landmarks = detector.detect(path)
        keep, reasons, flags = evaluate_keep(landmarks, width, height)
        decision = "keep" if keep else "exclude"
        exclude_reason = reasons[:] if not keep else []

        if file_hash in hash_to_first:
            decision = "exclude"
            exclude_reason = [f"exact_duplicate_of:{hash_to_first[file_hash]}"]
            keep = False
        else:
            hash_to_first[file_hash] = stem

        phash = average_hash(path)
        if keep:
            for prior_hash, prior_stem in phash_kept:
                if hamming(phash, prior_hash) <= NEAR_DUP_HAMMING:
                    decision = "exclude"
                    exclude_reason = [f"near_duplicate_of:{prior_stem}"]
                    keep = False
                    break
            if keep:
                phash_kept.append((phash, stem))

        caption = caption_for(landmarks, width, height) if keep else ""
        record = {
            "stem": stem,
            "source_name": nfc(path.name),
            "source_path": str(path.resolve()),
            "sha256": file_hash,
            "phash16": format(phash, "064x"),
            "width": width,
            "height": height,
            "decision": decision,
            "exclude_reasons": exclude_reason,
            "flags": flags,
            "framing_tags": framing_tags(landmarks, width, height) if landmarks.get("available") else [],
            "face_count": landmarks.get("face_count"),
            "face_crop_effective_resolution": landmarks.get("face_crop_effective_resolution"),
            "head_pose": landmarks.get("head_pose"),
            "subject_occupancy_fraction": (landmarks.get("subject") or {}).get("occupancy_fraction"),
            "pose_visible": (landmarks.get("pose") or {}).get("visible"),
            "caption": caption,
            "split": None,
            "dataset_relpath": None,
        }
        records.append(record)

    kept = [r for r in records if r["decision"] == "keep"]
    rng = random.Random(seed)
    order = list(range(len(kept)))
    rng.shuffle(order)
    val_count = max(1, round(len(kept) * val_fraction)) if len(kept) >= 7 else max(0, min(1, len(kept) // 7))
    # Ensure train gets majority; if very small, put all in train except 1 val when >=7
    if len(kept) < 7:
        val_indices = set()
    else:
        val_indices = set(order[:val_count])

    for i, record in enumerate(kept):
        split = "val" if i in val_indices else "train"
        dest_dir = val_dir if split == "val" else train_dir
        dest_img = dest_dir / f"{record['stem']}.png"
        dest_txt = dest_dir / f"{record['stem']}.txt"
        shutil.copy2(record["source_path"], dest_img)
        dest_txt.write_text(record["caption"] + "\n", encoding="utf-8")
        record["split"] = split
        record["dataset_relpath"] = str(dest_img.relative_to(dataset_dir))

    shortfall = len(kept) < 25
    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    manifest = {
        "schema_version": "1.0.0",
        "character_id": CHARACTER_ID,
        "trigger_token": TRIGGER_TOKEN,
        "generated_at_utc": generated_at,
        "source_policy": {
            "authoritative_identity": True,
            "source_type": "user_provided_glam_ai_polish_screenshots",
            "user_accepted_glam_over_raw_hyperreal_photos": True,
            "supersedes": "synthetic_master_face_only_bootstrap_training_path",
        },
        "paths": {
            "intake_raw": str(raw_dir.resolve()),
            "dataset": str(dataset_dir.resolve()),
            "train": str(train_dir.resolve()),
            "val": str(val_dir.resolve()),
            "repo_relative_dataset": _repo_relative(dataset_dir),
        },
        "counts": {
            "intake": len(records),
            "keep": len(kept),
            "exclude": sum(1 for r in records if r["decision"] == "exclude"),
            "train": sum(1 for r in kept if r["split"] == "train"),
            "val": sum(1 for r in kept if r["split"] == "val"),
            "exact_duplicates_excluded": sum(
                1 for r in records if any(x.startswith("exact_duplicate") for x in r["exclude_reasons"])
            ),
            "near_duplicates_excluded": sum(
                1 for r in records if any(x.startswith("near_duplicate") for x in r["exclude_reasons"])
            ),
        },
        "qc_thresholds": {
            "min_face_short_side_px": MIN_FACE_SHORT_SIDE_PX,
            "min_face_area_frac": MIN_FACE_AREA_FRAC,
            "tiny_face_area_frac": TINY_FACE_AREA_FRAC,
            "back_of_head_yaw_abs": BACK_OF_HEAD_YAW_ABS,
            "near_dup_hamming": NEAR_DUP_HAMMING,
            "val_fraction": val_fraction,
            "split_seed": seed,
        },
        "shortfall": {
            "below_25_unique_keeps": shortfall,
            "recommended_additions": [
                "more profile / three-quarter face angles",
                "neutral daylight outfit variety",
                "full-body standing front and side",
                "hair up / different hairstyles if available",
            ]
            if shortfall
            else [],
        },
        "records": records,
    }
    out = manifests_dir / "dataset_manifest.v1.json"
    out.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    summary_path = manifests_dir / "dataset_summary.v1.json"
    summary = {k: manifest[k] for k in manifest if k != "records"}
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifest


def main() -> None:
    manifest = build_dataset()
    counts = manifest["counts"]
    print(
        f"intake={counts['intake']} keep={counts['keep']} exclude={counts['exclude']} "
        f"train={counts['train']} val={counts['val']} "
        f"shortfall={manifest['shortfall']['below_25_unique_keeps']}"
    )
    print(f"manifest={MANIFESTS_DIR / 'dataset_manifest.v1.json'}")
    print(f"trigger={TRIGGER_TOKEN}")


if __name__ == "__main__":
    main()
