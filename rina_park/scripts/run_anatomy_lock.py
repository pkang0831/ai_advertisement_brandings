#!/usr/bin/env python3
"""Anatomy lock end-to-end: pose catalog + 2-pass + crop-CN. RUN all three.

Usage:
  cd /Users/RBIPK031/ai_influencer
  PYTHONPATH=rina_park .venv/bin/python rina_park/scripts/run_anatomy_lock.py
  PYTHONPATH=rina_park .venv/bin/python rina_park/scripts/run_anatomy_lock.py --quick
  PYTHONPATH=rina_park .venv/bin/python rina_park/scripts/run_anatomy_lock.py --sfw-only
"""

from __future__ import annotations

import argparse
import json
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

from hyperreal.anatomy.crop_controlnet import ensure_verified_pose_refs, refine_crops_with_controlnet
from hyperreal.anatomy.pass2_inpaint import inpaint_regions
from hyperreal.anatomy.pose_catalog import load_pose_catalog
from hyperreal.anatomy.qc_gates import evaluate_anatomy_image, scorecard_template, write_scorecard
from hyperreal.anatomy.stack import (
    ExtraLora,
    known_specialty_loras,
    load_controlnet_img2img,
    load_img2img_from,
    load_txt2img,
    unload,
)
from hyperreal.finishing.face_detailer import FaceDetailerConfig, apply_face_detailer

OUT_SFW = RINA / "out" / "anatomy_lock"
OUT_NSFW = RINA / "private" / "nsfw_test" / "private_media" / "anatomy_lock"
OPS = RINA / "ops" / "anatomy_lock"

# Default mission set — several hand + foot SFW + private NSFW
DEFAULT_SFW = [
    "hand_holding_cup_soft",
    "hand_resting_lap_seated",
    "hand_holding_tote_side",
    "hand_pockets_3q",
    "foot_standing_flat_sneakers",
    "foot_seated_ankle_soft",
]
DEFAULT_NSFW = [
    "nsfw_reclined_knee_up_partial",
    "nsfw_standing_mirror_hip_angle",
    "nsfw_side_lying_tucked",
]

SCENE_GLUE = {
    "hand_pockets_3q": "beige trench coat, city sidewalk morning, candid",
    "hand_holding_cup_soft": "cafe window light, cream knit sweater, holding paper cup",
    "hand_resting_lap_seated": "sunlit apartment sofa, soft loungewear",
    "hand_holding_tote_side": "linen tote bag, park path, casual jeans",
    "foot_standing_flat_sneakers": "white sneakers, athletic leggings, studio lobby",
    "foot_seated_ankle_soft": "park bench, socks and sneakers, afternoon",
    "foot_mid_stride_side": "city crosswalk, sneakers, soft motion",
}

# NSFW glue: private/anatomy_lock/scene_glue_nsfw.yml (soft-load when present)
_NSFW_GLUE_PATH = RINA / "private" / "anatomy_lock" / "scene_glue_nsfw.yml"
if _NSFW_GLUE_PATH.is_file():
    import yaml

    _nsfw_glue = yaml.safe_load(_NSFW_GLUE_PATH.read_text(encoding="utf-8")) or {}
    if isinstance(_nsfw_glue, dict):
        SCENE_GLUE.update(
            {str(k): str(v) for k, v in (_nsfw_glue.get("scene_glue") or {}).items()}
        )


def _run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _save(img: Image.Image, path: Path, meta: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path, quality=95)
    path.with_suffix(path.suffix + ".json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )


def _mps_cleanup() -> None:
    import gc

    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
        torch.mps.synchronize()


def generate_base(pipe, catalog, pose, seed: int, steps: int, cfg: float, lora: float):
    prompt = catalog.build_prompt(pose, SCENE_GLUE.get(pose.id, ""))
    negative = catalog.build_negative("sfw" if pose.track == "sfw" else "nsfw_private")
    bans = catalog.reject_if_banned(prompt + " " + SCENE_GLUE.get(pose.id, ""))
    if bans:
        raise ValueError(f"banned: {bans}")
    last_mean = 0.0
    for attempt in range(3):
        _mps_cleanup()
        use_seed = seed + attempt * 3331
        gen = torch.Generator(device="cpu").manual_seed(use_seed)
        image = pipe(
            prompt=prompt,
            negative_prompt=negative,
            width=832,
            height=1216,
            num_inference_steps=steps,
            guidance_scale=cfg,
            generator=gen,
        ).images[0]
        arr = np.asarray(image.convert("RGB"), dtype=np.float32)
        last_mean = float(arr.mean())
        if np.isfinite(arr).all() and last_mean >= 5:
            if attempt:
                print(f"recovered black-frame on attempt {attempt+1} seed={use_seed}")
            return image, prompt, negative, use_seed
        print(f"WARN black/NaN frame mean={last_mean} attempt={attempt+1}; retrying")
        _mps_cleanup()
    raise RuntimeError(f"black frame mean={last_mean} after retries")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="2 SFW + 1 NSFW only")
    ap.add_argument("--sfw-only", action="store_true")
    ap.add_argument("--nsfw-only", action="store_true")
    ap.add_argument("--steps", type=int, default=32)
    ap.add_argument("--cfg", type=float, default=4.2)
    ap.add_argument("--lora", type=float, default=0.90)
    ap.add_argument("--seed", type=int, default=29072026)
    ap.add_argument("--skip-face-detailer", action="store_true")
    ap.add_argument("--skip-method2", action="store_true")
    ap.add_argument("--skip-method3", action="store_true")
    ap.add_argument(
        "--poses",
        type=str,
        default="",
        help="Comma-separated pose ids (overrides default/quick sets)",
    )
    ap.add_argument(
        "--extra-lora",
        action="append",
        default=[],
        metavar="NAME:WEIGHT",
        help=(
            "Stack specialty LoRA under character. NAME=detailed_hands|better_hands|real_feet "
            "(repeatable). Example: --extra-lora detailed_hands:0.45"
        ),
    )
    args = ap.parse_args()

    catalog = load_pose_catalog()
    ensure_verified_pose_refs(768)
    run_id = _run_id()
    t0 = time.time()

    sfw_ids = DEFAULT_SFW
    nsfw_ids = DEFAULT_NSFW
    if args.quick:
        sfw_ids = ["hand_holding_cup_soft", "foot_standing_flat_sneakers"]
        nsfw_ids = ["nsfw_side_lying_tucked"]
    if args.sfw_only:
        nsfw_ids = []
    if args.nsfw_only:
        sfw_ids = []

    pose_ids = sfw_ids + nsfw_ids
    if args.poses.strip():
        pose_ids = [p.strip() for p in args.poses.split(",") if p.strip()]
    print(f"run_id={run_id} poses={pose_ids}")

    methods_cfg = catalog.methods
    m2 = methods_cfg.get("2_dedicated_2pass", {})
    m3 = methods_cfg.get("3_crop_controlnet", {})
    seed_offsets = m2.get("seed_offsets", {"hands": 1000, "feet": 2000, "genitals": 3000})
    denoise = m2.get("denoise", {"hands": 0.42, "feet": 0.40, "genitals": 0.38})

    results: list[dict] = []

    known = known_specialty_loras()
    extras: list[ExtraLora] = []
    for spec in args.extra_lora:
        if ":" not in spec:
            raise SystemExit(f"--extra-lora needs NAME:WEIGHT, got {spec!r}")
        name, w_s = spec.split(":", 1)
        name = name.strip()
        if name not in known:
            raise SystemExit(f"unknown specialty LoRA {name!r}; known={list(known)}")
        extras.append(
            ExtraLora(path=known[name], name=name, weight=float(w_s), trigger="")
        )
    print("Loading txt2img stack…", f"extras={[e.name + ':' + str(e.weight) for e in extras]}")
    pipe = load_txt2img(lora_scale=args.lora, extra_loras=extras)
    # img2img used for FaceDetailer + method-2 narrow refine
    img2img = load_img2img_from(pipe)
    cn_pipe = None

    try:
        for i, pose_id in enumerate(pose_ids):
          try:
            pose = catalog.get(pose_id)
            track = "sfw" if pose.track == "sfw" else "nsfw_private"
            out_root = OUT_SFW if track == "sfw" else OUT_NSFW
            out_dir = out_root / run_id / pose_id
            seed = args.seed + i * 97
            print(f"\n=== [{i+1}/{len(pose_ids)}] method1 catalog: {pose_id} seed={seed} ===")
            _mps_cleanup()

            image, prompt, negative, seed = generate_base(
                pipe, catalog, pose, seed, args.steps, args.cfg, args.lora
            )

            face_report = None
            pre_fd = image.copy()
            if not args.skip_face_detailer:
                _mps_cleanup()
                refined, face_report = apply_face_detailer(
                    image,
                    img2img,
                    FaceDetailerConfig(
                        lora_scale=args.lora,
                        eye_refine=False,
                        eye_refine_auto=False,
                        face_denoise=0.28,
                    ),
                    base_seed=seed,
                )
                # Guard: FaceDetailer MPS NaN can black-out the face crop
                arr = np.asarray(refined.convert("RGB"), dtype=np.float32)
                bad = (
                    not np.isfinite(arr).all()
                    or float(arr.mean()) < 5
                    or float(np.isnan(arr).mean()) > 0
                )
                if bad:
                    face_report = {
                        **(face_report or {}),
                        "reverted": True,
                        "reason": "nan_or_black_after_face_detailer",
                    }
                    image = pre_fd
                else:
                    # Also revert if face detector loses the face after refine
                    tmp = out_dir / "_fd_check.jpg"
                    tmp.parent.mkdir(parents=True, exist_ok=True)
                    refined.save(tmp, quality=90)
                    qc_fd = evaluate_anatomy_image(tmp, track=track)
                    if not qc_fd["identity"]["pass"]:
                        face_report = {
                            **(face_report or {}),
                            "reverted": True,
                            "reason": "face_lost_after_face_detailer",
                        }
                        image = pre_fd
                    else:
                        image = refined
                    tmp.unlink(missing_ok=True)
                _mps_cleanup()

            base_path = out_dir / "m1_catalog.jpg"
            meta = {
                "run_id": run_id,
                "method": "1_pose_catalog",
                "pose_id": pose_id,
                "track": track,
                "seed": seed,
                "steps": args.steps,
                "cfg": args.cfg,
                "lora": args.lora,
                "prompt": prompt,
                "negative": negative,
                "face_detailer": face_report,
                "local_only": pose.local_only,
            }
            _save(image, base_path, meta)

            require_feet = "feet" in pose.regions
            require_genitals = track != "sfw" and "genitals" in pose.regions
            qc1 = evaluate_anatomy_image(
                base_path,
                pose_expected_hands=pose.expected_visible_hands,
                track=track,
                require_feet=require_feet,
                require_genitals=require_genitals,
            )
            (out_dir / "m1_qc.json").write_text(json.dumps(qc1, indent=2), encoding="utf-8")
            card = scorecard_template(
                image_path=str(base_path), method="1_pose_catalog", pose_id=pose_id, track=track
            )
            card["auto_qc"] = qc1
            write_scorecard(OPS / "scorecards" / run_id / f"{pose_id}_m1.json", card)
            print(f"m1 auto_pass={qc1['overall_auto_pass']} hands={qc1['hands']} feet_pass={qc1['feet']['pass']}")

            row = {
                "pose_id": pose_id,
                "track": track,
                "m1_path": str(base_path),
                "m1_pass": qc1["overall_auto_pass"],
                "m1_hands_pass": qc1["hands"]["pass"],
                "m1_feet_pass": qc1["feet"]["pass"],
                "m1_genital_pass": qc1["genitals"]["pass"],
                "m2_path": None,
                "m2_pass": None,
                "m3_path": None,
                "m3_pass": None,
            }

            # Identity coarse must pass before 2-pass / crop-CN
            identity_ok = qc1["identity"]["pass"]
            working = image

            # --- Method 2 ---
            if not args.skip_method2 and identity_ok:
                regions = [r for r in pose.regions if r in ("hands", "feet", "genitals")]
                # For hand-hide poses with 0 expected hands, skip hand inpaint
                if pose.expected_visible_hands == 0 and "hands" in regions:
                    regions = [r for r in regions if r != "hands"]
                if regions:
                    print(f"=== method2 2-pass regions={regions} ===")
                    working2, m2_report = inpaint_regions(
                        img2img,
                        working,
                        regions=regions,
                        prompt=prompt,
                        negative=negative,
                        base_seed=seed,
                        seed_offsets=seed_offsets,
                        denoise=denoise,
                        steps=max(24, args.steps - 4),
                        cfg=args.cfg,
                    )
                    if not args.skip_face_detailer:
                        working2, _ = apply_face_detailer(
                            working2,
                            img2img,
                            FaceDetailerConfig(lora_scale=args.lora, face_denoise=0.28),
                            base_seed=seed + 3,
                        )
                    m2_path = out_dir / "m2_2pass.jpg"
                    _save(
                        working2,
                        m2_path,
                        {**meta, "method": "2_dedicated_2pass", "m2_report": m2_report},
                    )
                    qc2 = evaluate_anatomy_image(
                        m2_path,
                        pose_expected_hands=pose.expected_visible_hands,
                        track=track,
                        require_feet=require_feet,
                        require_genitals=require_genitals,
                    )
                    (out_dir / "m2_qc.json").write_text(json.dumps(qc2, indent=2), encoding="utf-8")
                    card2 = scorecard_template(
                        image_path=str(m2_path),
                        method="2_dedicated_2pass",
                        pose_id=pose_id,
                        track=track,
                    )
                    card2["auto_qc"] = qc2
                    write_scorecard(OPS / "scorecards" / run_id / f"{pose_id}_m2.json", card2)
                    row["m2_path"] = str(m2_path)
                    row["m2_pass"] = qc2["overall_auto_pass"]
                    row["m2_hands_pass"] = qc2["hands"]["pass"]
                    row["m2_feet_pass"] = qc2["feet"]["pass"]
                    row["m2_genital_pass"] = qc2["genitals"]["pass"]
                    print(f"m2 auto_pass={qc2['overall_auto_pass']}")
                    working = working2
                else:
                    print("method2 skipped (no applicable regions)")
            elif not identity_ok:
                print("method2 skipped (identity coarse fail)")

            # --- Method 3: crop ControlNet (hands/feet only; not full-frame) ---
            if not args.skip_method3 and identity_ok:
                from hyperreal.anatomy.pass2_inpaint import detect_foot_boxes, detect_hand_boxes

                cn_regions = [r for r in ("hands", "feet") if r in pose.regions]
                if pose.expected_visible_hands == 0:
                    cn_regions = [r for r in cn_regions if r != "hands"]
                # If detector sees hands/feet, attempt crop-CN even when catalog hid them
                if detect_hand_boxes(image) and "hands" not in cn_regions:
                    cn_regions.append("hands")
                if require_feet and detect_foot_boxes(image) and "feet" not in cn_regions:
                    cn_regions.append("feet")
                if cn_regions:
                    _mps_cleanup()
                    if cn_pipe is None:
                        print("Loading crop ControlNet (OpenPose)…")
                        cn_pipe = load_controlnet_img2img(pipe, kind="openpose")
                    print(f"=== method3 crop-CN regions={cn_regions} ===")
                    # Use m1 base for fair comparison of method3 alone; also run on working
                    m3_img, m3_report = refine_crops_with_controlnet(
                        cn_pipe,
                        image,  # from method1 base — crop+verified pose, not full-frame retry
                        pose_id=pose_id,
                        prompt=prompt,
                        negative=negative,
                        base_seed=seed,
                        regions=cn_regions,
                        crop_size=int(m3.get("crop_size", 768)),
                        strength=float(m3.get("denoise", 0.45)),
                        controlnet_weight=float(m3.get("controlnet_weight", 0.55)),
                        steps=max(24, args.steps - 4),
                        cfg=args.cfg,
                    )
                    m3_path = out_dir / "m3_crop_cn.jpg"
                    _save(
                        m3_img,
                        m3_path,
                        {**meta, "method": "3_crop_controlnet", "m3_report": m3_report},
                    )
                    qc3 = evaluate_anatomy_image(
                        m3_path,
                        pose_expected_hands=pose.expected_visible_hands,
                        track=track,
                        require_feet=require_feet,
                        require_genitals=False,  # CN path is hand/foot crop
                    )
                    (out_dir / "m3_qc.json").write_text(json.dumps(qc3, indent=2), encoding="utf-8")
                    card3 = scorecard_template(
                        image_path=str(m3_path),
                        method="3_crop_controlnet",
                        pose_id=pose_id,
                        track=track,
                    )
                    card3["auto_qc"] = qc3
                    write_scorecard(OPS / "scorecards" / run_id / f"{pose_id}_m3.json", card3)
                    row["m3_path"] = str(m3_path)
                    row["m3_pass"] = qc3["overall_auto_pass"]
                    row["m3_hands_pass"] = qc3["hands"]["pass"]
                    row["m3_feet_pass"] = qc3["feet"]["pass"]
                    row["m3_status"] = m3_report.get("status")
                    print(f"m3 auto_pass={qc3['overall_auto_pass']} status={m3_report.get('status')}")
                else:
                    print("method3 skipped (no hand/foot crop regions)")

            results.append(row)
            _mps_cleanup()
          except Exception as exc:  # noqa: BLE001
            print(f"ERROR pose {pose_id}: {type(exc).__name__}: {exc}")
            results.append(
                {
                    "pose_id": pose_id,
                    "error": f"{type(exc).__name__}: {exc}",
                    "m1_pass": False,
                    "m2_pass": None,
                    "m3_pass": None,
                }
            )
            _mps_cleanup()
            continue

    finally:
        unload(cn_pipe, img2img, pipe)

    summary = {
        "run_id": run_id,
        "elapsed_sec": round(time.time() - t0, 1),
        "poses": results,
        "counts": {
            "m1_pass": sum(1 for r in results if r.get("m1_pass")),
            "m2_pass": sum(1 for r in results if r.get("m2_pass")),
            "m3_pass": sum(1 for r in results if r.get("m3_pass")),
            "total": len(results),
        },
        "outputs": {
            "sfw": str(OUT_SFW / run_id),
            "nsfw_private": str(OUT_NSFW / run_id),
            "scorecards": str(OPS / "scorecards" / run_id),
        },
    }
    summary_path = OPS / f"run_summary_{run_id}.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("\nSUMMARY", json.dumps(summary["counts"], indent=2))
    print("wrote", summary_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
