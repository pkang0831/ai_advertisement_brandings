"""Dedicated 2-pass: full-body base → narrow crop img2img (ADetailer-style) → composite.

Uses shared SDXL img2img weights (not a separate 9-ch inpaint UNet) so MPS +
RealVis single-file stack stays compatible. Narrow elliptical feather masks.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFilter


def _clamp(v: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, v))


def _round8(v: int) -> int:
    return max(64, int(round(v / 8) * 8))


def _feather_mask(size: tuple[int, int], feather_frac: float = 0.14) -> Image.Image:
    w, h = size
    mask = Image.new("L", (w, h), 0)
    draw = ImageDraw.Draw(mask)
    inset = max(1, int(min(w, h) * feather_frac * 0.5))
    draw.ellipse((inset, inset, w - inset - 1, h - inset - 1), fill=255)
    blur = max(1, int(min(w, h) * feather_frac * 0.4))
    return mask.filter(ImageFilter.GaussianBlur(radius=blur))


def detect_hand_boxes(image: Image.Image, pad_ratio: float = 0.35) -> list[tuple[int, int, int, int]]:
    try:
        import mediapipe as mp
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision
        from hyperreal.identity.qc.registry import verified_model_paths

        models = verified_model_paths()
        rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
        h, w = rgb.shape[:2]
        with vision.HandLandmarker.create_from_options(
            vision.HandLandmarkerOptions(
                base_options=python.BaseOptions(model_asset_path=str(models["hand"])),
                running_mode=vision.RunningMode.IMAGE,
                num_hands=2,
            )
        ) as detector:
            result = detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
        boxes: list[tuple[int, int, int, int]] = []
        for hand in result.hand_landmarks or []:
            xs = [p.x * w for p in hand]
            ys = [p.y * h for p in hand]
            l, t, r, b = min(xs), min(ys), max(xs), max(ys)
            bw, bh = max(1.0, r - l), max(1.0, b - t)
            l = _clamp(int(l - bw * pad_ratio), 0, w - 1)
            t = _clamp(int(t - bh * pad_ratio), 0, h - 1)
            r = _clamp(int(r + bw * pad_ratio), l + 1, w)
            b = _clamp(int(b + bh * pad_ratio), t + 1, h)
            boxes.append((l, t, r, b))
        return boxes
    except Exception:  # noqa: BLE001
        return []


def detect_foot_boxes(image: Image.Image, pad_ratio: float = 0.40) -> list[tuple[int, int, int, int]]:
    try:
        import mediapipe as mp
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision
        from hyperreal.identity.qc.registry import verified_model_paths

        models = verified_model_paths()
        rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
        h, w = rgb.shape[:2]
        with vision.PoseLandmarker.create_from_options(
            vision.PoseLandmarkerOptions(
                base_options=python.BaseOptions(model_asset_path=str(models["pose"])),
                running_mode=vision.RunningMode.IMAGE,
                num_poses=1,
            )
        ) as detector:
            result = detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
        if not result.pose_landmarks:
            return []
        pose = result.pose_landmarks[0]
        boxes: list[tuple[int, int, int, int]] = []
        for idxs in ((27, 29, 31), (28, 30, 32)):
            pts = [pose[i] for i in idxs if i < len(pose)]
            if not pts:
                continue
            vis = [getattr(p, "visibility", 0.5) for p in pts]
            if max(float(v) if v is not None else 0.0 for v in vis) < 0.35:
                continue
            xs = [p.x * w for p in pts]
            ys = [p.y * h for p in pts]
            l, t, r, b = min(xs), min(ys), max(xs), max(ys)
            bw, bh = max(8.0, r - l), max(8.0, b - t)
            side = max(bw, bh) * (1.0 + pad_ratio)
            cx, cy = (l + r) / 2, (t + b) / 2
            l = _clamp(int(cx - side / 2), 0, w - 1)
            t = _clamp(int(cy - side / 2), 0, h - 1)
            r = _clamp(int(cx + side / 2), l + 1, w)
            b = _clamp(int(cy + side / 2), t + 1, h)
            boxes.append((l, t, r, b))
        return boxes
    except Exception:  # noqa: BLE001
        return []


def genital_box(image: Image.Image) -> tuple[int, int, int, int]:
    w, h = image.size
    return int(w * 0.32), int(h * 0.48), int(w * 0.68), int(h * 0.72)


REGION_PROMPTS = {
    "hands": (
        "perfect human hands, five fingers each, natural knuckles, clean fingernails, "
        "correct finger joints, photoreal hands"
    ),
    "feet": (
        "perfect human feet in sneakers, natural ankles, correct shoe shape, "
        "photoreal feet, no extra toes"
    ),
    "genitals": (
        "natural adult female vulva anatomy, coherent labia, realistic skin, "
        "photoreal intimate detail, correct anatomy"
    ),
}


def _refine_box(
    img2img_pipe: Any,
    image: Image.Image,
    box: tuple[int, int, int, int],
    *,
    prompt: str,
    negative: str,
    strength: float,
    seed: int,
    steps: int,
    cfg: float,
    refine_size: int = 768,
) -> Image.Image:
    l, t, r, b = box
    crop = image.crop(box)
    cw, ch = crop.size
    scale = refine_size / max(cw, ch)
    tw, th = _round8(int(cw * max(scale, 1.0))), _round8(int(ch * max(scale, 1.0)))
    crop_r = crop.resize((tw, th), Image.Resampling.LANCZOS)
    refined = img2img_pipe(
        prompt=prompt,
        negative_prompt=negative,
        image=crop_r,
        strength=float(strength),
        num_inference_steps=int(steps),
        guidance_scale=float(cfg),
        generator=torch.Generator(device="cpu").manual_seed(int(seed)),
    ).images[0]
    patch = refined.resize((cw, ch), Image.Resampling.LANCZOS)
    arr = np.asarray(patch.convert("RGB"), dtype=np.float32)
    # MPS occasionally emits NaN/black crops — keep original box instead of pasting void
    if (not np.isfinite(arr).all()) or float(arr.mean()) < 8.0:
        return image.convert("RGB")
    mask = _feather_mask((cw, ch))
    out = image.copy().convert("RGB")
    out.paste(patch, (l, t), mask)
    return out


def inpaint_regions(
    img2img_pipe: Any,
    image: Image.Image,
    *,
    regions: list[str],
    prompt: str,
    negative: str,
    base_seed: int,
    seed_offsets: dict[str, int],
    denoise: dict[str, float],
    steps: int = 28,
    cfg: float = 4.0,
) -> tuple[Image.Image, dict[str, Any]]:
    """Narrow-mask second pass via crop img2img + feather composite. Fixed seeds."""
    out = image.convert("RGB")
    report: dict[str, Any] = {"regions": {}, "applied": [], "mode": "crop_img2img_adetailer"}

    for region in regions:
        if region == "hands":
            boxes = detect_hand_boxes(out)
        elif region == "feet":
            boxes = detect_foot_boxes(out)
        elif region == "genitals":
            boxes = [genital_box(out)]
        else:
            continue

        if not boxes:
            report["regions"][region] = {"status": "no_box", "boxes": []}
            continue

        strength = float(denoise.get(region, 0.4))
        seed_base = int(base_seed) + int(seed_offsets.get(region, 0))
        region_prompt = f"{prompt}, {REGION_PROMPTS.get(region, '')}"
        applied_boxes: list[list[int]] = []
        for bi, box in enumerate(boxes):
            out = _refine_box(
                img2img_pipe,
                out,
                box,
                prompt=region_prompt,
                negative=negative,
                strength=strength,
                seed=seed_base + bi * 13,
                steps=steps,
                cfg=cfg,
            )
            applied_boxes.append(list(box))
        report["applied"].append(region)
        report["regions"][region] = {
            "status": "applied",
            "boxes": applied_boxes,
            "strength": strength,
            "seed_base": seed_base,
        }

    return out, report
