"""Method 3: ControlNet on hand/foot CROP + verified pose reference → composite.

NOT full-frame OpenPose retry (prior pilots failed 200% hand gates).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFilter

from .pass2_inpaint import detect_foot_boxes, detect_hand_boxes

RINA = Path(__file__).resolve().parents[2]
POSE_REF_DIR = RINA / "ops" / "anatomy_lock" / "pose_refs"

# MediaPipe hand connections for stick-figure verified refs
_HAND_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
    (5, 9), (9, 13), (13, 17),
]


# Canonical normalized hand landmarks for high-pass poses (verified templates)
VERIFIED_HAND_TEMPLATES: dict[str, list[tuple[float, float]]] = {
    # Soft resting / slightly curved fingers pointing down-right
    "soft_resting": [
        (0.50, 0.78), (0.40, 0.70), (0.34, 0.60), (0.30, 0.50), (0.28, 0.40),
        (0.48, 0.58), (0.48, 0.44), (0.48, 0.32), (0.48, 0.22),
        (0.56, 0.58), (0.58, 0.42), (0.59, 0.30), (0.60, 0.20),
        (0.64, 0.60), (0.67, 0.46), (0.69, 0.34), (0.70, 0.24),
        (0.72, 0.64), (0.76, 0.54), (0.78, 0.44), (0.80, 0.36),
    ],
    # Simple cup grip — fingers wrapped, thumb opposing
    "cup_grip": [
        (0.55, 0.72), (0.42, 0.62), (0.34, 0.52), (0.30, 0.42), (0.32, 0.32),
        (0.50, 0.50), (0.46, 0.38), (0.44, 0.28), (0.43, 0.20),
        (0.58, 0.50), (0.58, 0.36), (0.58, 0.26), (0.58, 0.18),
        (0.66, 0.52), (0.68, 0.40), (0.70, 0.30), (0.71, 0.22),
        (0.74, 0.56), (0.78, 0.48), (0.80, 0.40), (0.82, 0.34),
    ],
    # Flat hand on surface
    "flat_on_surface": [
        (0.50, 0.80), (0.38, 0.72), (0.30, 0.62), (0.26, 0.52), (0.24, 0.42),
        (0.45, 0.55), (0.42, 0.40), (0.40, 0.28), (0.39, 0.18),
        (0.55, 0.54), (0.55, 0.38), (0.55, 0.26), (0.55, 0.16),
        (0.64, 0.56), (0.66, 0.42), (0.68, 0.30), (0.69, 0.20),
        (0.72, 0.60), (0.76, 0.50), (0.78, 0.40), (0.80, 0.32),
    ],
}


POSE_TEMPLATE_MAP = {
    "hand_holding_cup_soft": "cup_grip",
    "hand_holding_tote_side": "cup_grip",
    "hand_resting_lap_seated": "soft_resting",
    "nsfw_reclined_knee_up_partial": "flat_on_surface",
    "nsfw_standing_mirror_hip_angle": "flat_on_surface",
    "nsfw_side_lying_tucked": "flat_on_surface",
    "foot_standing_flat_sneakers": "soft_resting",
    "foot_seated_ankle_soft": "soft_resting",
    "foot_mid_stride_side": "soft_resting",
}


def draw_openpose_hand(
    size: int,
    landmarks: list[tuple[float, float]],
) -> Image.Image:
    """Draw OpenPose-style hand skeleton on black (verified reference)."""
    img = Image.new("RGB", (size, size), (0, 0, 0))
    draw = ImageDraw.Draw(img)
    pts = [(x * size, y * size) for x, y in landmarks]
    # limbs in OpenPose-ish colors
    colors = [
        (255, 0, 0), (255, 85, 0), (255, 170, 0), (255, 255, 0),
        (170, 255, 0), (85, 255, 0), (0, 255, 0), (0, 255, 85),
        (0, 255, 170), (0, 255, 255), (0, 170, 255), (0, 85, 255),
        (0, 0, 255), (85, 0, 255), (170, 0, 255), (255, 0, 255),
        (255, 0, 170), (255, 0, 85), (255, 128, 128), (128, 255, 128),
        (128, 128, 255), (255, 255, 128), (255, 128, 255),
    ]
    for i, (a, b) in enumerate(_HAND_EDGES):
        if a < len(pts) and b < len(pts):
            draw.line([pts[a], pts[b]], fill=colors[i % len(colors)], width=max(3, size // 128))
    for i, p in enumerate(pts):
        r = max(3, size // 64)
        draw.ellipse((p[0] - r, p[1] - r, p[0] + r, p[1] + r), fill=(255, 255, 255))
    return img


def ensure_verified_pose_refs(size: int = 768) -> dict[str, Path]:
    POSE_REF_DIR.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for name, lms in VERIFIED_HAND_TEMPLATES.items():
        path = POSE_REF_DIR / f"hand_{name}_{size}.png"
        if not path.exists():
            draw_openpose_hand(size, lms).save(path)
            meta = {"template": name, "size": size, "landmarks": lms}
            path.with_suffix(".json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        paths[name] = path
    return paths


def _feather_mask(size: tuple[int, int], feather_frac: float = 0.14) -> Image.Image:
    w, h = size
    mask = Image.new("L", (w, h), 0)
    draw = ImageDraw.Draw(mask)
    inset = max(1, int(min(w, h) * feather_frac * 0.5))
    draw.ellipse((inset, inset, w - inset - 1, h - inset - 1), fill=255)
    blur = max(1, int(min(w, h) * feather_frac * 0.4))
    return mask.filter(ImageFilter.GaussianBlur(radius=blur))


def _round8(v: int) -> int:
    return max(64, int(round(v / 8) * 8))


def refine_crops_with_controlnet(
    cn_pipe: Any,
    image: Image.Image,
    *,
    pose_id: str,
    prompt: str,
    negative: str,
    base_seed: int,
    regions: list[str] | None = None,
    crop_size: int = 768,
    strength: float = 0.45,
    controlnet_weight: float = 0.55,
    steps: int = 28,
    cfg: float = 4.0,
) -> tuple[Image.Image, dict[str, Any]]:
    """Crop hand/foot → CN img2img with verified pose ref → feather composite."""
    refs = ensure_verified_pose_refs(crop_size)
    template_name = POSE_TEMPLATE_MAP.get(pose_id, "soft_resting")
    pose_ref = Image.open(refs[template_name]).convert("RGB")

    regions = regions or ["hands"]
    out = image.convert("RGB")
    report: dict[str, Any] = {
        "method": "3_crop_controlnet",
        "pose_id": pose_id,
        "template": template_name,
        "pose_ref": str(refs[template_name]),
        "crops": [],
    }

    boxes: list[tuple[str, tuple[int, int, int, int]]] = []
    if "hands" in regions:
        for b in detect_hand_boxes(out, pad_ratio=0.45):
            boxes.append(("hands", b))
    if "feet" in regions:
        for b in detect_foot_boxes(out, pad_ratio=0.50):
            boxes.append(("feet", b))

    if not boxes:
        report["status"] = "no_crop_boxes"
        return out, report

    for idx, (region, box) in enumerate(boxes):
        l, t, r, b = box
        crop = out.crop(box)
        cw, ch = crop.size
        scale = crop_size / max(cw, ch)
        tw, th = _round8(int(cw * scale)), _round8(int(ch * scale))
        crop_r = crop.resize((tw, th), Image.Resampling.LANCZOS)
        # Fit pose ref to crop canvas
        control = pose_ref.resize((tw, th), Image.Resampling.LANCZOS)

        region_prompt = (
            f"{prompt}, perfect {region}, five fingers, natural joints, photoreal"
            if region == "hands"
            else f"{prompt}, perfect feet in sneakers, natural ankles, photoreal"
        )
        seed = base_seed + 5000 + idx * 17
        refined = cn_pipe(
            prompt=region_prompt,
            negative_prompt=negative,
            image=crop_r,
            control_image=control,
            strength=strength,
            num_inference_steps=steps,
            guidance_scale=cfg,
            controlnet_conditioning_scale=controlnet_weight,
            generator=torch.Generator(device="cpu").manual_seed(seed),
        ).images[0]

        patch = refined.resize((cw, ch), Image.Resampling.LANCZOS)
        arr = np.asarray(patch.convert("RGB"), dtype=np.float32)
        if (not np.isfinite(arr).all()) or float(arr.mean()) < 8.0:
            report["crops"].append(
                {
                    "region": region,
                    "box": list(box),
                    "seed": seed,
                    "status": "reverted_black_or_nan_crop",
                }
            )
            continue
        mask = _feather_mask((cw, ch))
        out.paste(patch, (l, t), mask)
        report["crops"].append(
            {
                "region": region,
                "box": list(box),
                "seed": seed,
                "strength": strength,
                "controlnet_weight": controlnet_weight,
                "status": "applied",
            }
        )

    report["status"] = "applied"
    return out, report
