"""FaceDetailer-style post-pass for Diffusers SDXL on MPS.

Detect face (MediaPipe local tasks) → crop → low-denoise img2img refine with the
same RealVis + character LoRA stack → soft alpha composite. Optional eye-region
refine. Not SeedVR2; no InsightFace.

Hands/feet: this module is face-only. Do NOT enable hand ADetailer/inpaint here.
Prefer composition hide (see content/story_3year/COMPOSITION_RULES.md).
Optional hand refine only when a scene sets hands_hero: true (separate path).
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from hyperreal.identity.qc.registry import verified_model_paths

# MediaPipe Face Mesh eye contour indices (subject left = image-right typically).
_RIGHT_EYE_IDX = (33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246)
_LEFT_EYE_IDX = (362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398)

TRIGGER = "rina_park_person"

# Gaze: candid / looking away by default — do not force eye contact to camera.
# Eyes soft preference (future runs): slightly larger / subtle doe — photoreal only.
DEFAULT_FACE_PROMPT = (
    f"{TRIGGER}, soft glam Korean-Canadian 27, clear even skin, "
    "slightly larger symmetrical eyes, subtle doe eyes, soft open eyes, "
    "sharp pupils, balanced eyelids, "
    "looking away from camera, gaze off-camera, "
    "flattering face detail, photoreal portrait face"
)

DEFAULT_EYE_PROMPT = (
    f"{TRIGGER}, soft glam, slightly larger symmetrical eyes, subtle doe eyes, "
    "soft open eyes, sharp iris detail, matched eyelids, even eye size, "
    "looking away, gaze off-camera, photoreal eye detail"
)

DEFAULT_NEGATIVE = (
    "plastic skin, doll face, airbrushed, CGI, 3d render, beauty filter, "
    "acne, rash, heavy pores, blotchy skin, "
    "asymmetrical eyes, uneven eyes, lazy eye, squashed eye, crossed eyes, "
    "misaligned eyes, different sized eyes, deformed pupils, droopy eyelid, "
    "small eyes, squinting, beady eyes, anime eyes, exaggerated eyes, oversized eyes, "
    "looking at camera, eye contact with camera, staring at viewer, "
    "blurry face, watermark, text, logo, Nike logo, swoosh"
)


@dataclass
class FaceDetailerConfig:
    """Configurable FaceDetailer defaults (ON for IG generation).

    Hand detailer is intentionally absent / always off here. Use composition
    (hands_hero: false default) instead of hand ADetailer.
    """

    enabled: bool = True
    face_denoise: float = 0.34
    eye_denoise: float = 0.30
    face_pad_ratio: float = 0.38
    eye_pad_ratio: float = 0.70
    refine_size: int = 1024
    steps: int = 28
    cfg: float = 4.0
    lora_scale: float = 0.90
    feather_frac: float = 0.12
    min_face_short_side: int = 72
    min_eye_short_side: int = 28
    eye_refine: bool = False
    eye_refine_auto: bool = True
    # Below this min/max eye-height ratio → auto eye refine (known left-eye bias).
    eye_height_ratio_trigger: float = 0.90
    # Always false — hand ADetailer not implemented on this path; composition preferred.
    hand_detailer: bool = False
    face_prompt: str = DEFAULT_FACE_PROMPT
    eye_prompt: str = DEFAULT_EYE_PROMPT
    negative_prompt: str = DEFAULT_NEGATIVE
    seed_offset: int = 17

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class _Region:
    box: tuple[int, int, int, int]  # left, top, right, bottom in full image
    kind: str
    meta: dict[str, Any] = field(default_factory=dict)


def _clamp_box(
    left: float, top: float, right: float, bottom: float, width: int, height: int
) -> tuple[int, int, int, int]:
    l = max(0, min(width - 1, int(math.floor(left))))
    t = max(0, min(height - 1, int(math.floor(top))))
    r = max(l + 1, min(width, int(math.ceil(right))))
    b = max(t + 1, min(height, int(math.ceil(bottom))))
    return l, t, r, b


def _pad_box(
    box: tuple[int, int, int, int],
    pad_ratio: float,
    width: int,
    height: int,
) -> tuple[int, int, int, int]:
    l, t, r, b = box
    bw, bh = r - l, b - t
    pad_x = bw * pad_ratio
    pad_y = bh * pad_ratio
    return _clamp_box(l - pad_x, t - pad_y, r + pad_x, b + pad_y, width, height)


def _bbox_from_indices(
    landmarks: list[Any], indices: tuple[int, ...], width: int, height: int
) -> tuple[int, int, int, int] | None:
    xs: list[float] = []
    ys: list[float] = []
    for idx in indices:
        if idx >= len(landmarks):
            return None
        pt = landmarks[idx]
        xs.append(float(pt.x) * width)
        ys.append(float(pt.y) * height)
    if not xs:
        return None
    return _clamp_box(min(xs), min(ys), max(xs), max(ys), width, height)


def detect_face_regions(image: Image.Image) -> dict[str, Any]:
    """Return primary face + eye boxes using local MediaPipe FaceLandmarker."""
    try:
        import mediapipe as mp
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision
    except ImportError as error:
        return {"status": "unavailable", "reason": f"mediapipe_missing:{error}"}

    try:
        models = verified_model_paths()
        face_model = models["face"]
    except Exception as error:  # noqa: BLE001
        return {"status": "unavailable", "reason": f"model_registry:{error}"}

    rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    height, width = rgb.shape[:2]
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    options = vision.FaceLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=str(face_model)),
        running_mode=vision.RunningMode.IMAGE,
        num_faces=1,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
    )
    with vision.FaceLandmarker.create_from_options(options) as detector:
        result = detector.detect(mp_image)

    if not result.face_landmarks:
        return {"status": "no_face", "face_count": 0, "width": width, "height": height}

    face = result.face_landmarks[0]
    xs = [float(p.x) * width for p in face]
    ys = [float(p.y) * height for p in face]
    face_box = _clamp_box(min(xs), min(ys), max(xs), max(ys), width, height)
    left_eye = _bbox_from_indices(face, _LEFT_EYE_IDX, width, height)
    right_eye = _bbox_from_indices(face, _RIGHT_EYE_IDX, width, height)

    eye_meta: dict[str, Any] = {}
    if left_eye and right_eye:
        lh = max(1, left_eye[3] - left_eye[1])
        rh = max(1, right_eye[3] - right_eye[1])
        lw = max(1, left_eye[2] - left_eye[0])
        rw = max(1, right_eye[2] - right_eye[0])
        ratio_h = min(lh, rh) / max(lh, rh)
        ratio_w = min(lw, rw) / max(lw, rw)
        eye_meta = {
            "left_eye_box": list(left_eye),
            "right_eye_box": list(right_eye),
            "height_ratio": round(ratio_h, 4),
            "width_ratio": round(ratio_w, 4),
            # Subject-left tends to be MediaPipe left eye (image-right).
            "smaller_eye": "left" if lh < rh else "right",
        }

    return {
        "status": "ok",
        "face_count": len(result.face_landmarks),
        "face_box": list(face_box),
        "left_eye_box": list(left_eye) if left_eye else None,
        "right_eye_box": list(right_eye) if right_eye else None,
        "eye_meta": eye_meta,
        "width": width,
        "height": height,
    }


def _feather_mask(size: tuple[int, int], feather_frac: float) -> Image.Image:
    w, h = size
    mask = Image.new("L", (w, h), 0)
    draw = ImageDraw.Draw(mask)
    inset_x = max(1, int(w * feather_frac * 0.55))
    inset_y = max(1, int(h * feather_frac * 0.55))
    draw.ellipse((inset_x, inset_y, w - inset_x - 1, h - inset_y - 1), fill=255)
    blur = max(1, int(min(w, h) * feather_frac * 0.45))
    return mask.filter(ImageFilter.GaussianBlur(radius=blur))


def _round_dim(value: int, multiple: int = 8) -> int:
    return max(multiple, int(round(value / multiple) * multiple))


def _prepare_crop(
    image: Image.Image, box: tuple[int, int, int, int], refine_size: int
) -> tuple[Image.Image, tuple[int, int]]:
    crop = image.crop(box)
    cw, ch = crop.size
    # Upscale short side toward refine_size; keep aspect, multiples of 8.
    scale = refine_size / max(cw, ch)
    if scale < 1.0:
        scale = 1.0
    tw = _round_dim(int(cw * scale))
    th = _round_dim(int(ch * scale))
    tw = min(tw, refine_size + 128)
    th = min(th, refine_size + 128)
    resized = crop.resize((tw, th), Image.Resampling.LANCZOS)
    return resized, (cw, ch)


def _img2img_refine(
    img2img_pipe: Any,
    crop: Image.Image,
    *,
    prompt: str,
    negative: str,
    strength: float,
    steps: int,
    cfg: float,
    seed: int,
    lora_scale: float,
) -> Image.Image:
    if hasattr(img2img_pipe, "set_adapters"):
        try:
            img2img_pipe.set_adapters(["rina_person"], adapter_weights=[lora_scale])
        except Exception:  # noqa: BLE001
            pass
    import torch

    result = img2img_pipe(
        prompt=prompt,
        negative_prompt=negative,
        image=crop,
        strength=float(strength),
        num_inference_steps=int(steps),
        guidance_scale=float(cfg),
        generator=torch.Generator(device="cpu").manual_seed(int(seed)),
    ).images[0]
    return result


def _composite_region(
    base: Image.Image,
    refined_full_res: Image.Image,
    box: tuple[int, int, int, int],
    feather_frac: float,
) -> Image.Image:
    l, t, r, b = box
    w, h = r - l, b - t
    patch = refined_full_res.resize((w, h), Image.Resampling.LANCZOS).convert("RGB")
    mask = _feather_mask((w, h), feather_frac)
    out = base.copy().convert("RGB")
    out.paste(patch, (l, t), mask)
    return out


def build_img2img_from_txt2img(txt2img_pipe: Any) -> Any:
    """Share weights: txt2img components → SDXL img2img pipe."""
    from diffusers import StableDiffusionXLImg2ImgPipeline

    return StableDiffusionXLImg2ImgPipeline(**txt2img_pipe.components)


def apply_face_detailer(
    image: Image.Image,
    img2img_pipe: Any,
    config: FaceDetailerConfig | None = None,
    *,
    base_seed: int = 0,
) -> tuple[Image.Image, dict[str, Any]]:
    """Run face (and optional eye) refine. Returns (image, report)."""
    cfg = config or FaceDetailerConfig()
    report: dict[str, Any] = {
        "enabled": cfg.enabled,
        "config": cfg.to_dict(),
        "applied": False,
        "face_pass": None,
        "eye_pass": None,
        "skipped_reason": None,
    }
    if not cfg.enabled:
        report["skipped_reason"] = "disabled"
        return image, report

    detection = detect_face_regions(image)
    report["detection"] = {
        k: detection.get(k)
        for k in (
            "status",
            "face_count",
            "face_box",
            "eye_meta",
            "left_eye_box",
            "right_eye_box",
        )
    }
    if detection.get("status") != "ok":
        report["skipped_reason"] = detection.get("reason") or detection.get("status")
        return image, report

    face_box_raw = tuple(detection["face_box"])
    width, height = image.size
    face_box = _pad_box(face_box_raw, cfg.face_pad_ratio, width, height)
    fw, fh = face_box[2] - face_box[0], face_box[3] - face_box[1]
    if min(fw, fh) < cfg.min_face_short_side:
        report["skipped_reason"] = f"face_too_small:{fw}x{fh}"
        return image, report

    import gc
    import time

    import torch

    started = time.monotonic()
    crop, orig_size = _prepare_crop(image, face_box, cfg.refine_size)
    try:
        refined = _img2img_refine(
            img2img_pipe,
            crop,
            prompt=cfg.face_prompt,
            negative=cfg.negative_prompt,
            strength=cfg.face_denoise,
            steps=cfg.steps,
            cfg=cfg.cfg,
            seed=int(base_seed) + cfg.seed_offset,
            lora_scale=cfg.lora_scale,
        )
        refined = refined.resize(orig_size, Image.Resampling.LANCZOS)
        out = _composite_region(image, refined, face_box, cfg.feather_frac)
        report["face_pass"] = {
            "status": "applied",
            "box": list(face_box),
            "crop_size": list(crop.size),
            "denoise": cfg.face_denoise,
            "seconds": round(time.monotonic() - started, 3),
        }
        report["applied"] = True
    except Exception as error:  # noqa: BLE001
        report["face_pass"] = {
            "status": "failed",
            "error": f"{type(error).__name__}:{error}",
        }
        report["skipped_reason"] = "face_pass_failed"
        return image, report
    finally:
        gc.collect()
        if hasattr(torch, "mps"):
            torch.mps.empty_cache()

    # Decide eye refine.
    do_eye = bool(cfg.eye_refine)
    eye_meta = detection.get("eye_meta") or {}
    if cfg.eye_refine_auto and eye_meta:
        ratio = float(eye_meta.get("height_ratio") or 1.0)
        if ratio < cfg.eye_height_ratio_trigger:
            do_eye = True
            report["eye_auto_trigger"] = {
                "height_ratio": ratio,
                "threshold": cfg.eye_height_ratio_trigger,
                "smaller_eye": eye_meta.get("smaller_eye"),
            }

    if not do_eye:
        report["eye_pass"] = {"status": "skipped", "reason": "not_requested_or_not_needed"}
        return out, report

    # Re-detect on refined image for accurate eye boxes; fall back to pre-face boxes.
    post = detect_face_regions(out)
    report["detection_after_face"] = {
        k: post.get(k) for k in ("status", "eye_meta", "left_eye_box", "right_eye_box")
    }
    if post.get("status") != "ok":
        report["detection_after_face"]["fallback"] = "pre_face_detection_boxes"
        post = detection
        if not (post.get("left_eye_box") or post.get("right_eye_box")):
            report["eye_pass"] = {"status": "skipped", "reason": "redetect_failed_no_fallback"}
            return out, report

    targets: list[tuple[str, tuple[int, int, int, int]]] = []
    # Prefer the smaller / subject-left eye when auto; otherwise both.
    prefer = (post.get("eye_meta") or {}).get("smaller_eye") or "left"
    order = ("left", "right") if prefer == "left" else ("right", "left")
    for name in order:
        key = f"{name}_eye_box"
        raw = post.get(key)
        if not raw:
            continue
        box = _pad_box(tuple(raw), cfg.eye_pad_ratio, out.size[0], out.size[1])
        if min(box[2] - box[0], box[3] - box[1]) < cfg.min_eye_short_side:
            continue
        targets.append((name, box))
        # Auto mode: only the smaller eye; forced eye_refine: both.
        if not cfg.eye_refine and cfg.eye_refine_auto:
            break

    eye_reports: list[dict[str, Any]] = []
    current = out
    for idx, (name, box) in enumerate(targets):
        eye_started = time.monotonic()
        try:
            eye_crop, eye_orig = _prepare_crop(current, box, min(768, cfg.refine_size))
            eye_refined = _img2img_refine(
                img2img_pipe,
                eye_crop,
                prompt=cfg.eye_prompt,
                negative=cfg.negative_prompt,
                strength=cfg.eye_denoise,
                steps=max(20, cfg.steps - 4),
                cfg=cfg.cfg,
                seed=int(base_seed) + cfg.seed_offset + 100 + idx,
                lora_scale=cfg.lora_scale,
            )
            eye_refined = eye_refined.resize(eye_orig, Image.Resampling.LANCZOS)
            current = _composite_region(current, eye_refined, box, min(0.18, cfg.feather_frac + 0.04))
            eye_reports.append(
                {
                    "eye": name,
                    "status": "applied",
                    "box": list(box),
                    "denoise": cfg.eye_denoise,
                    "seconds": round(time.monotonic() - eye_started, 3),
                }
            )
        except Exception as error:  # noqa: BLE001
            eye_reports.append(
                {
                    "eye": name,
                    "status": "failed",
                    "error": f"{type(error).__name__}:{error}",
                }
            )
        finally:
            gc.collect()
            if hasattr(torch, "mps"):
                torch.mps.empty_cache()

    report["eye_pass"] = {"status": "done", "eyes": eye_reports}
    if any(e.get("status") == "applied" for e in eye_reports):
        report["applied"] = True
    return current, report


def detailer_sidecar_snippet(report: dict[str, Any]) -> dict[str, Any]:
    """Compact sidecar fields for manifests."""
    return {
        "face_detailer_applied": bool(report.get("applied")),
        "face_detailer": report,
    }


def save_debug_overlay(
    image: Image.Image, detection: dict[str, Any], path: Path
) -> None:
    """Optional debug: draw face/eye boxes (not used in production path)."""
    vis = image.copy().convert("RGB")
    draw = ImageDraw.Draw(vis)
    if detection.get("face_box"):
        draw.rectangle(tuple(detection["face_box"]), outline=(0, 200, 80), width=3)
    for key, color in (("left_eye_box", (255, 80, 80)), ("right_eye_box", (80, 120, 255))):
        box = detection.get(key)
        if box:
            draw.rectangle(tuple(box), outline=color, width=2)
    path.parent.mkdir(parents=True, exist_ok=True)
    vis.save(path, quality=90)
