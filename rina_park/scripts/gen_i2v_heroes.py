#!/usr/bin/env python3
"""Generate I2V hero stills: anatomy-size gen (832x1216) → upscale to 1080x1920.

CLIP budget is tight (~77 tokens). Prompt order:
  identity+beauty → pose/framing (hands/feet hidden) → short SCENE_GLUE

Default poses hide hands/feet (pockets / cropped). FaceDetailer ON by default.

Usage:
  cd /Users/RBIPK031/ai_influencer
  PYTHONPATH=rina_park .venv/bin/python rina_park/scripts/gen_i2v_heroes.py
  PYTHONPATH=rina_park .venv/bin/python rina_park/scripts/gen_i2v_heroes.py --seeds-per-pose 6
  PYTHONPATH=rina_park .venv/bin/python rina_park/scripts/gen_i2v_heroes.py --skip-face-detailer
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
RINA = ROOT / "rina_park"
sys.path.insert(0, str(RINA))

from hyperreal.anatomy.pass2_inpaint import detect_hand_boxes
from hyperreal.anatomy.pose_catalog import PoseEntry, load_pose_catalog
from hyperreal.anatomy.qc_gates import evaluate_anatomy_image
from hyperreal.anatomy.stack import ExtraLora, known_specialty_loras, load_img2img_from, load_txt2img, unload
from hyperreal.finishing.face_detailer import FaceDetailerConfig, apply_face_detailer
from hyperreal.identity.qc.mediapipe_tasks import MediaPipeTasksAdapter
from hyperreal.prompt_combo import (
    ComboJob,
    build_combo_job,
    build_combo_overrides,
    build_interleaved_both,
    build_jobs_cartesian,
    count_axes,
    load_phrase_banks,
)

OUT_ROOT = RINA / "out" / "i2v_heroes"
OPS_I2V = RINA / "ops" / "i2v"

# Match anatomy_lock native gen; I2V pack is 1080x1920 after upscale.
DEFAULT_W = 832
DEFAULT_H = 1216
I2V_W = 1080
I2V_H = 1920

# Face area ratio upper bound (normalized bbox). Medium 3/4 shots sit well below;
# beauty close-ups / double-face crops push past this.
MAX_FACE_AREA_RATIO = 0.18
# Lap-rest poses: hands must not live only in the upper face band.
LAP_HAND_Y_MIN = 0.42  # normalized image y; below upper torso/face

# Default: hands/feet hidden — finger problem avoided; beauty + framing prioritized.
HERO_POSES = [
    "hand_pockets_3q",
    "hand_cropped_out",
]

# hand_visibility values that get detailed_hands LoRA (not used by default hero set)
HAND_VISIBLE = {"soft_resting", "soft_prop_grip"}

# Keep SHORT — CLIP truncates ~77. Beauty is in LOOK_POS; glue stays tiny.
SCENE_GLUE = {
    "hand_pockets_3q": "beige trench, soft morning sidewalk",
    "hand_cropped_out": "sunlit apartment, soft loungewear",
    # Optional overrides if user passes --poses explicitly:
    "hand_resting_lap_seated": "sunlit apartment sofa, soft loungewear",
    "fitness_mat_soft_seated": "home yoga studio, charcoal mat",
}

# Compact pose/framing — always emphasize hands/feet out of play.
HERO_POSE_SHORT = {
    "hand_pockets_3q": (
        "hands fully in pockets hiding fingers, feet cropped out, "
        "standing three-quarter lifestyle"
    ),
    "hand_cropped_out": (
        "hands and feet out of frame, cropped mid-torso, "
        "upper-body three-quarter portrait"
    ),
    "hand_resting_lap_seated": (
        "both hands resting on lap, no hand near face, "
        "seated three-quarter indoor"
    ),
    "fitness_mat_soft_seated": (
        "seated on yoga mat, hands tucked out of view, three-quarter"
    ),
}

# Short beauty + identity (keep ≤ ~25 tokens so pose/glue survive CLIP 77).
LOOK_POS = (
    "rina_park_person, young Korean-Canadian woman mid-20s, long dark hair, "
    "youthful face, dewy skin, soft V-line, soft glam, photoreal"
)
LOOK_NEG_EXTRA = (
    "painterly, oversmooth, text, typing, phone grip, extreme close-up, "
    "visible fingers, splayed fingers, bare feet, deformed feet, "
    "hands near face, hand on cheek, wrinkles, crow's feet, "
    "sagging skin, middle-aged"
)

HERO_FACE_PROMPT = (
    "rina_park_person, young Korean-Canadian mid-20s, youthful dewy skin, "
    "soft V-line, slightly larger symmetrical eyes, soft open eyes, "
    "sharp pupils, looking at camera, flattering face detail, photoreal"
)
HERO_FACE_NEG = (
    "plastic skin, doll face, airbrushed, CGI, 3d render, beauty filter, "
    "acne, blotchy skin, asymmetrical eyes, deformed pupils, "
    "anime eyes, blurry face, watermark, text, logo, "
    "black face, blank face, missing face, wrinkles, crow's feet, "
    "sagging skin, middle-aged"
)


def _run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _mps_cleanup() -> None:
    import gc

    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
        torch.mps.synchronize()


def _needs_hand_lora(pose: PoseEntry) -> bool:
    return pose.hand_visibility in HAND_VISIBLE and pose.expected_visible_hands > 0


def _compress_composition(text: str, max_chars: int = 90) -> str:
    """Flatten multiline composition into a short CLIP-safe phrase."""
    flat = " ".join(text.replace("\n", " ").split())
    if len(flat) <= max_chars:
        return flat
    # Prefer first clause cluster
    cut = flat[:max_chars]
    if "," in cut:
        cut = cut.rsplit(",", 1)[0]
    return cut.strip()


_CLIP_TOK = None


def _count_clip_tokens(prompt: str) -> int | None:
    """Best-effort CLIP token count (77 = hard truncate)."""
    global _CLIP_TOK
    try:
        if _CLIP_TOK is None:
            from transformers import CLIPTokenizer

            _CLIP_TOK = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
        return len(_CLIP_TOK(prompt, truncation=False, add_special_tokens=True)["input_ids"])
    except Exception:  # noqa: BLE001
        # Fallback: rough whitespace split (+2 BOS/EOS)
        return len(prompt.replace(",", " ").split()) + 2


def _load_pose_prompts_json(raw: str) -> dict[str, str]:
    """Parse --pose-prompts-json: file path or inline JSON object {pose_id: short}."""
    text = raw.strip()
    if not text:
        return {}
    path = Path(text)
    if path.is_file():
        text = path.read_text(encoding="utf-8")
    data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError("--pose-prompts-json must be a JSON object {pose_id: short_prompt}")
    out: dict[str, str] = {}
    for k, v in data.items():
        if v is None:
            continue
        s = str(v).strip()
        if s:
            out[str(k)] = s
    return out


def resolve_pose_short(
    pose: PoseEntry,
    *,
    pose_prompt_all: str = "",
    pose_prompts: dict[str, str] | None = None,
) -> str | None:
    """Per-pose JSON wins, then --pose-prompt (all), else None → catalog defaults."""
    overrides = pose_prompts or {}
    if pose.id in overrides:
        return overrides[pose.id]
    if pose_prompt_all.strip():
        return pose_prompt_all.strip()
    return None


def build_hero_prompt(
    pose: PoseEntry,
    glue: str,
    *,
    hand_trigger: bool,
    look_pos: str | None = None,
    pose_short: str | None = None,
) -> str:
    """Identity+beauty → pose/framing (hands hidden) → short glue.

    Injects composition via HERO_POSE_SHORT (or pose_short override).
    Target ≤ ~70–75 CLIP tokens. Hand-visible poses may append hand trigger.
    """
    bits: list[str] = [(look_pos if look_pos is not None else LOOK_POS)]
    # Pose / framing before glue so CLIP keeps three-quarter + hidden hands.
    if pose_short is not None:
        short = pose_short.strip() if pose_short else ""
    else:
        short = HERO_POSE_SHORT.get(pose.id, "")
    if short:
        bits.append(short)
    elif pose_short is None:
        # Only fall back to catalog compression when no explicit override was given.
        if pose.positive_extra.strip():
            bits.append(_compress_composition(pose.positive_extra, max_chars=70))
        comp = _compress_composition(pose.composition, max_chars=70)
        if comp:
            bits.append(comp)
        if pose.camera.strip():
            bits.append(_compress_composition(pose.camera, max_chars=40))
    if hand_trigger:
        bits.append("hand, detailed hands")
    if glue.strip():
        bits.append(glue.strip())
    return ", ".join(b.strip().rstrip(",") for b in bits if b and b.strip())


def upscale_to_i2v(
    image: Image.Image,
    *,
    target_w: int = I2V_W,
    target_h: int = I2V_H,
) -> tuple[Image.Image, dict]:
    """Center-crop to 9:16 then LANCZOS resize to I2V input size.

    Interim non-generative upscale (SeedVR2 / Ultrasharp blocked in finishing).
    832×1216 is wider than 9:16 → left/right crop, then upscale.
    """
    src_w, src_h = image.size
    target_ar = target_w / target_h
    src_ar = src_w / src_h
    crop_box: tuple[int, int, int, int]
    if abs(src_ar - target_ar) < 1e-3:
        cropped = image
        crop_box = (0, 0, src_w, src_h)
    elif src_ar > target_ar:
        new_w = max(1, int(round(src_h * target_ar)))
        left = max(0, (src_w - new_w) // 2)
        crop_box = (left, 0, left + new_w, src_h)
        cropped = image.crop(crop_box)
    else:
        new_h = max(1, int(round(src_w / target_ar)))
        top = max(0, (src_h - new_h) // 2)
        crop_box = (0, top, src_w, top + new_h)
        cropped = image.crop(crop_box)
    out = cropped.resize((target_w, target_h), Image.Resampling.LANCZOS)
    meta = {
        "method": "center_crop_lanczos",
        "note": "interim non-generative upscale; generative upscalers blocked in finishing",
        "gen_size": [src_w, src_h],
        "crop_box": list(crop_box),
        "i2v_size": [target_w, target_h],
    }
    return out, meta


def _face_area_ratio(image_path: Path) -> tuple[float | None, dict]:
    try:
        report = MediaPipeTasksAdapter().detect(image_path)
    except Exception as exc:  # noqa: BLE001
        return None, {"error": f"{type(exc).__name__}: {exc}"}
    box = report.get("face_bbox_normalized")
    if not box or len(box) != 4:
        return None, {"face_bbox_normalized": box, "face_count": report.get("face_count")}
    area = max(0.0, float(box[2] - box[0])) * max(0.0, float(box[3] - box[1]))
    return area, {
        "face_bbox_normalized": box,
        "face_count": report.get("face_count"),
        "face_area_ratio": area,
    }


def _hands_only_in_upper_face_band(image: Image.Image, *, y_min: float = LAP_HAND_Y_MIN) -> bool:
    """True if every detected hand centroid is above y_min (face/chest band)."""
    boxes = detect_hand_boxes(image, pad_ratio=0.05)
    if not boxes:
        return False
    h = image.size[1]
    centroids_y = [((t + b) / 2.0) / h for (_l, t, _r, b) in boxes]
    return all(y < y_min for y in centroids_y)


def hero_auto_pass(
    *,
    qc: dict,
    pose: PoseEntry,
    image: Image.Image,
    image_path: Path,
    res_ok: bool,
) -> tuple[bool, dict]:
    """Stricter gate than anatomy overall_auto_pass for I2V hero promotion candidates."""
    blockers: list[str] = []
    detail: dict = {}

    if not res_ok:
        blockers.append("i2v_resolution_below_1080x1920")
    if not qc.get("identity", {}).get("pass"):
        blockers.append("identity_fail")
        blockers.extend(qc.get("identity", {}).get("blockers") or [])

    hand_count = int(qc.get("hands", {}).get("count") or 0)
    expected = int(pose.expected_visible_hands)
    detail["hand_count"] = hand_count
    detail["expected_hands"] = expected
    if expected > 0:
        # Exact match — undercount (e.g. 1 of 2) was wrongly OK before.
        if hand_count != expected:
            blockers.append(f"hand_count_{hand_count}_!=_expected_{expected}")
        if not qc.get("hands", {}).get("pass"):
            blockers.append("hands_topology_fail")
            blockers.extend(qc.get("hands", {}).get("blockers") or [])
    elif hand_count > 0:
        # Hidden-hand poses: soft warn only (not a blocker), keep detail.
        detail["unexpected_visible_hands"] = hand_count

    face_area, face_meta = _face_area_ratio(image_path)
    detail.update(face_meta)
    if face_area is not None and face_area > MAX_FACE_AREA_RATIO:
        blockers.append(f"face_area_ratio_{face_area:.3f}_>_max_{MAX_FACE_AREA_RATIO}")

    # Lap / soft_resting: reject hands stuck only in upper face region.
    if pose.hand_visibility == "soft_resting" and expected > 0:
        upper_only = _hands_only_in_upper_face_band(image)
        detail["hands_only_upper_face_band"] = upper_only
        if upper_only:
            blockers.append("hands_only_in_upper_face_band")

    detail["blockers"] = blockers
    return (len(blockers) == 0), detail


def _scorecard_stub(
    *,
    path: Path,
    pose_id: str,
    seed: int,
    auto_qc: dict,
    hand_lora: bool,
) -> dict:
    return {
        "image": str(path),
        "pose_id": pose_id,
        "seed": seed,
        "hand_lora": hand_lora,
        "auto_qc": auto_qc,
        "human": {
            "skin": None,  # 0 fail / 1 borderline / 2 pass
            "hands": None,
            "identity": None,
            "i2v_ready": None,
            "notes": "",
        },
        "promoted": False,
    }


def generate_one(
    pipe,
    img2img,
    catalog,
    pose,
    *,
    seed: int,
    steps: int,
    cfg: float,
    lora: float,
    width: int,
    height: int,
    skip_face_detailer: bool,
    look_pos: str | None = None,
    look_neg_extra: str | None = None,
    pose_short: str | None = None,
    scene_glue: str | None = None,
) -> tuple[Image.Image, str, str, int, dict | None, int | None]:
    glue = scene_glue if scene_glue is not None else SCENE_GLUE.get(pose.id, "")
    hand_trigger = _needs_hand_lora(pose)
    prompt = build_hero_prompt(
        pose,
        glue,
        hand_trigger=hand_trigger,
        look_pos=look_pos,
        pose_short=pose_short,
    )
    token_count = _count_clip_tokens(prompt)
    if token_count is not None and token_count > 75:
        print(f"WARN CLIP tokens≈{token_count} (>75); pose may truncate — shorten glue/composition")

    neg_track = "sfw" if pose.track == "sfw" else "nsfw_private"
    neg_extra = look_neg_extra if look_neg_extra is not None else LOOK_NEG_EXTRA
    negative = catalog.build_negative(neg_track) + ", " + neg_extra
    # Ban-check intent includes full composition (not just compressed prompt).
    bans = catalog.reject_if_banned(
        " ".join([prompt, pose.composition, pose.positive_extra, glue])
    )
    if bans:
        raise ValueError(f"banned: {bans}")

    last_mean = 0.0
    image = None
    use_seed = seed
    for attempt in range(3):
        _mps_cleanup()
        use_seed = seed + attempt * 3331
        gen = torch.Generator(device="cpu").manual_seed(use_seed)
        image = pipe(
            prompt=prompt,
            negative_prompt=negative,
            width=width,
            height=height,
            num_inference_steps=steps,
            guidance_scale=cfg,
            generator=gen,
        ).images[0]
        arr = np.asarray(image.convert("RGB"), dtype=np.float32)
        last_mean = float(arr.mean())
        if np.isfinite(arr).all() and last_mean >= 5:
            break
        print(f"WARN black/NaN mean={last_mean} attempt={attempt+1}")
        image = None
    if image is None:
        raise RuntimeError(f"black frame mean={last_mean}")

    face_report = None
    if not skip_face_detailer:
        pre = image.copy()
        _mps_cleanup()
        refined, face_report = apply_face_detailer(
            image,
            img2img,
            FaceDetailerConfig(
                lora_scale=lora,
                eye_refine=False,
                eye_refine_auto=False,
                face_denoise=0.28,
                face_prompt=HERO_FACE_PROMPT,
                negative_prompt=HERO_FACE_NEG,
            ),
            base_seed=use_seed,
        )
        arr = np.asarray(refined.convert("RGB"), dtype=np.float32)
        # Full-frame mean can stay high even when face crop is a black oval.
        upper = arr[: arr.shape[0] // 2]
        black_frac = float((upper.mean(axis=2) < 8.0).mean())
        bad = (
            not np.isfinite(arr).all()
            or float(arr.mean()) < 5
            or float(np.isnan(arr).mean()) > 0
            or black_frac > 0.08
        )
        if bad:
            face_report = {
                **(face_report or {}),
                "reverted": True,
                "reason": "nan_or_black_or_face_blob",
                "upper_black_frac": black_frac,
            }
            image = pre
        else:
            image = refined
    return image, prompt, negative, use_seed, face_report, token_count


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--width", type=int, default=DEFAULT_W, help="Native gen width (default 832)")
    ap.add_argument("--height", type=int, default=DEFAULT_H, help="Native gen height (default 1216)")
    ap.add_argument("--i2v-width", type=int, default=I2V_W)
    ap.add_argument("--i2v-height", type=int, default=I2V_H)
    ap.add_argument("--steps", type=int, default=36)
    ap.add_argument("--cfg", type=float, default=4.2)
    ap.add_argument("--lora", type=float, default=0.80)
    ap.add_argument("--hand-lora", type=float, default=0.45)
    ap.add_argument("--base-seed", type=int, default=30072026)
    ap.add_argument("--seeds-per-pose", type=int, default=4)
    ap.add_argument("--poses", type=str, default="", help="Comma pose ids (default hero set)")
    ap.add_argument(
        "--look-pos",
        type=str,
        default="",
        help="Override LOOK_POS (identity/beauty prefix) when set; omit to keep default",
    )
    ap.add_argument(
        "--look-neg-extra",
        type=str,
        default="",
        help="Replace LOOK_NEG_EXTRA when set; omit to keep default",
    )
    ap.add_argument(
        "--pose-prompt",
        type=str,
        default="",
        help="Short pose/framing prompt applied to ALL poses in this run "
        "(overrides HERO_POSE_SHORT / catalog compression). For single-pose experiments.",
    )
    ap.add_argument(
        "--pose-prompts-json",
        type=str,
        default="",
        help="Path to JSON file or inline JSON mapping {pose_id: short_prompt}. "
        "Per-pose entries override --pose-prompt. Omit to keep HERO_POSE_SHORT/catalog.",
    )
    ap.add_argument(
        "--scene-glue",
        type=str,
        default="",
        help="Optional scene glue applied to ALL poses when set; omit to keep SCENE_GLUE per pose",
    )
    ap.add_argument(
        "--skip-face-detailer",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Skip FaceDetailer (default: OFF = FaceDetailer enabled). Pass --skip-face-detailer to disable.",
    )
    ap.add_argument(
        "--no-upscale",
        action="store_true",
        help="Keep native gen resolution (skip 1080x1920 pack)",
    )
    ap.add_argument("--promote-auto", action="store_true", help="Promote auto_pass stills to current/")
    ap.add_argument(
        "--combo",
        action="store_true",
        help="Enable prompt-bank mixing (pose×expression×outfit_type×color×background×lora). "
        "Off by default — existing HERO_POSE_SHORT / SCENE_GLUE behavior unchanged.",
    )
    ap.add_argument(
        "--combo-banks-dir",
        type=str,
        default="",
        help="Optional public banks dir (default: rina_park/ops/prompt_bank)",
    )
    ap.add_argument(
        "--combo-mode",
        type=str,
        default="random_seeded",
        choices=["random_seeded", "cartesian"],
        help="random_seeded=hash pick per seed; cartesian=1 image per full axis combo",
    )
    ap.add_argument(
        "--combo-track",
        type=str,
        default="",
        help="sfw|nsfw|both bank track for --combo (default: inferred from first pose id). "
        "both = build SFW+NSFW jobs and interleave (requires --combo-interleave or implies it).",
    )
    ap.add_argument(
        "--combo-interleave",
        action="store_true",
        help="With --combo-track both: merge jobs as SFW,NSFW,SFW,NSFW,... (zip-longest). "
        "Implied when --combo-track both.",
    )
    ap.add_argument(
        "--combo-max-jobs",
        type=int,
        default=0,
        help="Cartesian safety cap. 0 = no limit (default; warn if huge). "
        "If total > cap (>0) and --combo-allow-huge not set, refuse. "
        "With --combo-track both, cap applies per track before interleave. "
        "Use 64 for smoke tests.",
    )
    ap.add_argument(
        "--combo-allow-huge",
        action="store_true",
        help="If cartesian total exceeds --combo-max-jobs, take first N with warning instead of error.",
    )
    args = ap.parse_args()

    catalog = load_pose_catalog()
    pose_ids = HERO_POSES
    if args.poses.strip():
        pose_ids = [p.strip() for p in args.poses.split(",") if p.strip()]

    look_pos_override = args.look_pos.strip() or None
    look_neg_override = args.look_neg_extra.strip() or None
    scene_glue_override = args.scene_glue.strip() if args.scene_glue.strip() else None
    pose_prompts_map = _load_pose_prompts_json(args.pose_prompts_json) if args.pose_prompts_json.strip() else {}
    pose_prompt_all = args.pose_prompt.strip()
    if look_pos_override or look_neg_override or pose_prompt_all or pose_prompts_map or scene_glue_override is not None:
        print(
            "prompt overrides:",
            f"look_pos={'set' if look_pos_override else 'default'}",
            f"look_neg_extra={'set' if look_neg_override else 'default'}",
            f"pose_prompt_all={'set' if pose_prompt_all else 'off'}",
            f"pose_prompts_json={len(pose_prompts_map)} ids",
            f"scene_glue={'set' if scene_glue_override is not None else 'default'}",
        )

    combo_banks = None
    combo_track = ""
    combo_jobs: list[ComboJob] | None = None
    combo_axes: dict | None = None
    if args.combo:
        if args.combo_track.strip():
            combo_track = args.combo_track.strip().lower()
        else:
            first = pose_ids[0] if pose_ids else ""
            combo_track = "nsfw" if first.startswith("nsfw_") else "sfw"
        banks_dir = Path(args.combo_banks_dir) if args.combo_banks_dir.strip() else None
        if args.combo_interleave and combo_track != "both":
            print("WARN --combo-interleave ignored unless --combo-track both")
        if combo_track == "both":
            # Single process: SFW+NSFW jobs interleaved s0,n0,s1,n1,...
            sfw_pose_ids = [p for p in pose_ids if not p.startswith("nsfw_")]
            nsfw_pose_ids = [p for p in pose_ids if p.startswith("nsfw_")]
            if not sfw_pose_ids or not nsfw_pose_ids:
                raise SystemExit(
                    "--combo-track both requires both SFW and nsfw_* pose ids in --poses "
                    f"(sfw={len(sfw_pose_ids)} nsfw={len(nsfw_pose_ids)})"
                )
            pose_meta = {
                pid: {
                    "composition": catalog.get(pid).composition,
                    "positive_extra": catalog.get(pid).positive_extra,
                }
                for pid in pose_ids
            }
            sfw_axes = count_axes("sfw", sfw_pose_ids)
            nsfw_axes = count_axes("nsfw", nsfw_pose_ids)
            print(
                "combo ON:",
                "track=both",
                f"mode={args.combo_mode}",
                "interleave=True",
                f"sfw_poses={len(sfw_pose_ids)} nsfw_poses={len(nsfw_pose_ids)}",
                "(LOOK_POS kept; per-track banks; order s0,n0,s1,n1,...)",
            )
            print(
                "combo axes SFW:",
                f"pose={sfw_axes['pose']} total_cartesian={sfw_axes['total_cartesian']}",
            )
            print(
                "combo axes NSFW:",
                f"pose={nsfw_axes['pose']} total_cartesian={nsfw_axes['total_cartesian']}",
            )
            print(
                "combo axes BOTH sum_cartesian=",
                sfw_axes["total_cartesian"] + nsfw_axes["total_cartesian"],
            )
            combo_jobs = build_interleaved_both(
                sfw_pose_ids,
                nsfw_pose_ids,
                args.base_seed,
                mode=args.combo_mode,  # type: ignore[arg-type]
                max_jobs=args.combo_max_jobs,
                allow_huge=args.combo_allow_huge,
                seeds_per_pose=args.seeds_per_pose,
                banks_dir=banks_dir,
                pose_meta=pose_meta,
            )
            # Banks not needed at gen time (ComboJob already has pose_prompt/glue)
            combo_banks = None
            combo_axes = {
                "sfw": sfw_axes,
                "nsfw": nsfw_axes,
                "interleaved": len(combo_jobs),
            }
            _cap = args.combo_max_jobs
            _cap_s = "unlimited" if _cap in (None, 0) else str(_cap)
            print(
                f"interleaved jobs={len(combo_jobs)} "
                f"(max_jobs/track={_cap_s}, allow_huge={args.combo_allow_huge}); "
                "1 image/job; seed=base_seed+index per track"
            )
        else:
            combo_banks = load_phrase_banks(combo_track, banks_dir=banks_dir)
            combo_axes = count_axes(combo_track, pose_ids)
            print(
                "combo ON:",
                f"track={combo_banks.track}",
                f"mode={args.combo_mode}",
                f"counts={combo_banks.counts()}",
                f"private_loaded={combo_banks.private_loaded}",
                "(LOOK_POS kept; outfit/expression→pose_short; background→glue)",
            )
            print(
                "combo axes:",
                f"pose={combo_axes['pose']}",
                f"expression={combo_axes['expression']}",
                f"outfit_type={combo_axes['outfit_type']}",
                f"outfit_color={combo_axes['outfit_color']}",
                f"background={combo_axes['background']}",
                f"lora={combo_axes['lora']}",
                f"total_cartesian={combo_axes['total_cartesian']}",
            )
            if args.combo_mode == "cartesian":
                pose_meta = {
                    pid: {
                        "composition": catalog.get(pid).composition,
                        "positive_extra": catalog.get(pid).positive_extra,
                    }
                    for pid in pose_ids
                }
                combo_jobs = build_jobs_cartesian(
                    combo_track,
                    pose_ids,
                    args.base_seed,
                    max_jobs=args.combo_max_jobs,
                    allow_huge=args.combo_allow_huge,
                    banks_dir=banks_dir,
                    pose_meta=pose_meta,
                )
                _cap = args.combo_max_jobs
                _cap_s = "unlimited" if _cap in (None, 0) else str(_cap)
                print(
                    f"cartesian jobs={len(combo_jobs)} "
                    f"(max_jobs={_cap_s}, allow_huge={args.combo_allow_huge}); "
                    "seeds_per_pose ignored; 1 image/job; seed=base_seed+job_index"
                )
            elif args.combo_mode == "random_seeded":
                pose_meta = {
                    pid: {
                        "composition": catalog.get(pid).composition,
                        "positive_extra": catalog.get(pid).positive_extra,
                    }
                    for pid in pose_ids
                }
                combo_jobs = build_combo_overrides(
                    combo_track,
                    pose_ids,
                    args.base_seed,
                    args.seeds_per_pose,
                    mode="random_seeded",
                    max_jobs=args.combo_max_jobs,
                    allow_huge=args.combo_allow_huge,
                    banks_dir=banks_dir,
                    pose_meta=pose_meta,
                )
                print(f"random_seeded jobs={len(combo_jobs)}")

    run_id = _run_id()
    run_dir = OUT_ROOT / run_id
    reject_dir = run_dir / "_reject"
    run_dir.mkdir(parents=True, exist_ok=True)
    reject_dir.mkdir(parents=True, exist_ok=True)
    (OUT_ROOT / "current").mkdir(parents=True, exist_ok=True)

    known = known_specialty_loras()
    if "detailed_hands" not in known:
        print("WARN detailed_hands LoRA missing — hand-visible poses will use character only")

    t0 = time.time()
    index: list[dict] = []

    # Load pipes per LoRA mode to avoid adapter thrash
    modes = {
        "char_only": [],
        "char_hands": (
            [ExtraLora(path=known["detailed_hands"], name="detailed_hands", weight=args.hand_lora)]
            if "detailed_hands" in known
            else []
        ),
    }

    # Worklist: (pose_id, seed, tag_si, combo_job|None, char_lora)
    work: list[tuple[str, int, int, ComboJob | None, float]] = []
    if combo_jobs is not None:
        # Per-pose seed ordinal for filename sXX (1 image per combo job)
        pose_si: dict[str, int] = {}
        for job in combo_jobs:
            si = pose_si.get(job.pose_id, 0)
            pose_si[job.pose_id] = si + 1
            work.append((job.pose_id, job.seed, si, job, float(job.lora_weight)))
    else:
        for pi, pose_id in enumerate(pose_ids):
            for si in range(args.seeds_per_pose):
                seed = args.base_seed + pi * 1000 + si * 97
                work.append((pose_id, seed, si, None, float(args.lora)))

    pipes: dict[str, tuple] = {}
    active_lora: dict[str, float] = {}
    try:
        needed_poses = {w[0] for w in work} or set(pose_ids)
        for mode, extras in modes.items():
            need = False
            for pid in needed_poses:
                pose = catalog.get(pid)
                if _needs_hand_lora(pose) and mode == "char_hands":
                    need = True
                if not _needs_hand_lora(pose) and mode == "char_only":
                    need = True
            if not need:
                continue
            init_lora = work[0][4] if work else args.lora
            print(f"Loading stack mode={mode} lora={init_lora} extras={[e.name for e in extras]}")
            pipe = load_txt2img(lora_scale=init_lora, extra_loras=extras)
            img2img = load_img2img_from(pipe)
            pipes[mode] = (pipe, img2img)
            active_lora[mode] = float(init_lora)

        prev_pose = ""
        jobs_in_mode = 0
        for wi, (pose_id, seed, si, preset_job, char_lora) in enumerate(work):
            pose = catalog.get(pose_id)
            mode = "char_hands" if _needs_hand_lora(pose) else "char_only"
            if mode not in pipes:
                mode = next(iter(pipes))
            pipe, img2img = pipes[mode]
            pose_dir = run_dir / pose_id
            pose_dir.mkdir(parents=True, exist_ok=True)

            if pose_id != prev_pose:
                print(
                    f"\n=== pose {pose_id} mode={mode} "
                    f"gen={args.width}x{args.height} i2v={args.i2v_width}x{args.i2v_height} "
                    f"fd_skip={args.skip_face_detailer} combo_mode={args.combo_mode if args.combo else 'off'} ==="
                )
                prev_pose = pose_id
                jobs_in_mode = 0

            # Reload when char LoRA weight changes, or every 2 gens (MPS hygiene)
            need_reload = abs(active_lora.get(mode, char_lora) - char_lora) > 1e-6
            if jobs_in_mode > 0 and (jobs_in_mode % 2 == 0 or need_reload):
                reason = "lora change" if need_reload else "MPS hygiene"
                print(f"  reloading stack mode={mode} lora={char_lora} ({reason})")
                try:
                    unload(pipe)
                except Exception:  # noqa: BLE001
                    pass
                _mps_cleanup()
                extras = modes[mode]
                pipe = load_txt2img(lora_scale=char_lora, extra_loras=extras)
                img2img = load_img2img_from(pipe)
                pipes[mode] = (pipe, img2img)
                active_lora[mode] = float(char_lora)

            tag = f"s{si:02d}_seed{seed}"
            native_path = pose_dir / f"{tag}_gen{args.width}x{args.height}.jpg"
            out_path = pose_dir / f"{tag}.jpg"
            pose_short = resolve_pose_short(
                pose,
                pose_prompt_all=pose_prompt_all,
                pose_prompts=pose_prompts_map,
            )
            glue_for_seed = scene_glue_override
            combo_job: ComboJob | None = preset_job
            if combo_banks is not None and pose_short is None and combo_job is None:
                combo_job = build_combo_job(
                    pose_id,
                    seed,
                    combo_banks,
                    composition=pose.composition,
                    positive_extra=pose.positive_extra,
                    mode=args.combo_mode,
                    job_index=wi,
                )
            if combo_job is not None and pose_short is None:
                pose_short = combo_job.pose_prompt
                if glue_for_seed is None:
                    glue_for_seed = combo_job.scene_glue
                char_lora = float(combo_job.lora_weight)

            try:
                image, prompt, negative, use_seed, face_report, token_count = generate_one(
                    pipe,
                    img2img,
                    catalog,
                    pose,
                    seed=seed,
                    steps=args.steps,
                    cfg=args.cfg,
                    lora=char_lora,
                    width=args.width,
                    height=args.height,
                    skip_face_detailer=args.skip_face_detailer,
                    look_pos=look_pos_override,
                    look_neg_extra=look_neg_override,
                    pose_short=pose_short,
                    scene_glue=glue_for_seed,
                )
                gen_w, gen_h = image.size
                image.save(native_path, quality=95)

                upscale_meta: dict | None = None
                if args.no_upscale:
                    i2v_image = image
                else:
                    i2v_image, upscale_meta = upscale_to_i2v(
                        image, target_w=args.i2v_width, target_h=args.i2v_height
                    )
                i2v_image.save(out_path, quality=95)
                w, h = i2v_image.size

                qc = evaluate_anatomy_image(
                    out_path,
                    pose_expected_hands=pose.expected_visible_hands,
                    track="sfw",
                    require_feet=False,
                    require_genitals=False,
                )
                res_ok = w >= 1080 and h >= 1920
                auto_pass, hero_gate = hero_auto_pass(
                    qc=qc,
                    pose=pose,
                    image=i2v_image,
                    image_path=out_path,
                    res_ok=res_ok,
                )

                meta = {
                    "run_id": run_id,
                    "pose_id": pose_id,
                    "seed": use_seed,
                    "gen_res": [gen_w, gen_h],
                    "i2v_res": [w, h],
                    "width": w,
                    "height": h,
                    "steps": args.steps,
                    "cfg": args.cfg,
                    "lora": char_lora,
                    "hand_lora": mode == "char_hands",
                    "hand_lora_weight": args.hand_lora if mode == "char_hands" else 0.0,
                    "prompt": prompt,
                    "negative": negative,
                    "clip_token_count": token_count,
                    "face_detailer": face_report,
                    "face_detailer_skipped": args.skip_face_detailer,
                    "upscale": upscale_meta,
                    "native_path": str(native_path),
                    "res_gate": res_ok,
                    "auto_pass": auto_pass,
                    "hero_gate": hero_gate,
                    "combo": combo_job.as_dict() if combo_job is not None else None,
                    "note": "auto_pass ≠ promote; human QC required before current/",
                }
                out_path.with_suffix(".jpg.json").write_text(
                    json.dumps(meta, indent=2), encoding="utf-8"
                )
                card = _scorecard_stub(
                    path=out_path,
                    pose_id=pose_id,
                    seed=use_seed,
                    auto_qc=qc,
                    hand_lora=mode == "char_hands",
                )
                card["gen_res"] = [gen_w, gen_h]
                card["i2v_res"] = [w, h]
                card["res_gate"] = res_ok
                card["auto_pass"] = auto_pass
                card["hero_gate"] = hero_gate
                card["clip_token_count"] = token_count
                card["native_path"] = str(native_path)
                (pose_dir / f"{tag}_scorecard.json").write_text(
                    json.dumps(card, indent=2), encoding="utf-8"
                )

                if not auto_pass:
                    dest = reject_dir / f"{pose_id}_{tag}.jpg"
                    shutil.copy2(out_path, dest)
                    shutil.copy2(out_path.with_suffix(".jpg.json"), dest.with_suffix(".jpg.json"))

                if args.promote_auto and auto_pass:
                    cur = OUT_ROOT / "current" / f"{pose_id}.jpg"
                    if cur.is_symlink() or cur.exists():
                        cur.unlink()
                    cur.symlink_to(out_path.resolve())
                    card["promoted"] = True
                    (pose_dir / f"{tag}_scorecard.json").write_text(
                        json.dumps(card, indent=2), encoding="utf-8"
                    )

                index.append(
                    {
                        "pose_id": pose_id,
                        "path": str(out_path),
                        "native_path": str(native_path),
                        "seed": use_seed,
                        "auto_pass": auto_pass,
                        "res_ok": res_ok,
                        "gen_res": [gen_w, gen_h],
                        "i2v_res": [w, h],
                        "clip_token_count": token_count,
                        "identity": qc["identity"]["pass"],
                        "hands": qc["hands"]["pass"],
                        "hand_count": qc["hands"]["count"],
                        "hero_gate_blockers": hero_gate.get("blockers", []),
                        "combo_job_index": combo_job.job_index if combo_job else None,
                    }
                )
                print(
                    f"  {tag}: gen={gen_w}x{gen_h} i2v={w}x{h} tokens≈{token_count} "
                    f"auto_pass={auto_pass} id={qc['identity']['pass']} "
                    f"hands={qc['hands']['pass']} count={qc['hands']['count']}/"
                    f"{pose.expected_visible_hands} "
                    f"blockers={hero_gate.get('blockers')}"
                )
            except Exception as exc:  # noqa: BLE001
                print(f"  ERROR {tag}: {type(exc).__name__}: {exc}")
                index.append(
                    {"pose_id": pose_id, "seed": seed, "error": f"{type(exc).__name__}: {exc}"}
                )
            jobs_in_mode += 1
            _mps_cleanup()
    finally:
        for pipe, img2img in pipes.values():
            unload(img2img, pipe)

    summary = {
        "run_id": run_id,
        "elapsed_sec": round(time.time() - t0, 1),
        "gen_res": [args.width, args.height],
        "i2v_res": [args.i2v_width, args.i2v_height],
        "width": args.width,
        "height": args.height,
        "skip_face_detailer": args.skip_face_detailer,
        "upscale": not args.no_upscale,
        "poses": pose_ids,
        "seeds_per_pose": args.seeds_per_pose if combo_jobs is None else 1,
        "combo": bool(args.combo),
        "combo_mode": args.combo_mode if args.combo else None,
        "combo_track": (
            "both"
            if combo_track == "both"
            else (combo_banks.track if combo_banks is not None else (combo_track or None))
        ),
        "combo_interleave": combo_track == "both",
        "combo_counts": combo_banks.counts() if combo_banks is not None else None,
        "combo_axes": combo_axes,
        "combo_jobs": len(combo_jobs) if combo_jobs is not None else None,
        "combo_max_jobs": args.combo_max_jobs if args.combo else None,
        "auto_pass_count": sum(1 for r in index if r.get("auto_pass")),
        "total": len(index),
        "items": index,
        "output_dir": str(run_dir),
        "still_gate": str(OPS_I2V / "STILL_GATE.md"),
        "note": "auto_pass ≠ promote; human QC required before current/",
        "promote_hint": (
            f"PYTHONPATH=rina_park .venv/bin/python rina_park/scripts/promote_i2v_heroes.py "
            f"--run-id {run_id} --pick pose_id:sXX"
        ),
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (OPS_I2V / f"heroes_run_{run_id}.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(
        "\nSUMMARY",
        json.dumps(
            {
                k: summary[k]
                for k in (
                    "run_id",
                    "auto_pass_count",
                    "total",
                    "elapsed_sec",
                    "gen_res",
                    "i2v_res",
                )
            },
            indent=2,
        ),
    )
    print("wrote", run_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
