"""Identity-sensitive validation for restoration candidates.

The metrics are rejection gates, not proof that an output is correct. A passing
candidate remains private and requires 100% human review.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import numpy as np
from PIL import Image


LpipsProvider = Callable[[np.ndarray, np.ndarray], float]
LandmarkProvider = Callable[[Path], list[np.ndarray]]


@dataclass(frozen=True)
class ValidationThresholds:
    landmark_procrustes_rms_max: float = 0.018
    face_bbox_iou_min: float = 0.94
    global_lpips_max: float = 0.10
    face_lpips_max: float = 0.075
    edge_gain_min: float = 1.0
    edge_gain_max: float = 1.35
    black_clip_delta_max: float = 0.005
    white_clip_delta_max: float = 0.005
    low_frequency_change_max: float = 0.08
    high_frequency_gain_max: float = 0.50


def _rgb(path: Path, size: tuple[int, int] | None = None) -> np.ndarray:
    with Image.open(path) as image:
        image = image.convert("RGB")
        if size is not None and image.size != size:
            image = image.resize(size, Image.Resampling.LANCZOS)
        return np.asarray(image, dtype=np.float32) / 255.0


def _normalized_landmarks(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[0] < 5 or points.shape[1] < 2:
        raise ValueError("face_landmarks_must_be_Nx2_with_at_least_five_points")
    points = points[:, :2]
    centered = points - points.mean(axis=0, keepdims=True)
    norm = float(np.linalg.norm(centered))
    if norm <= 1e-12:
        raise ValueError("face_landmarks_are_collapsed")
    return centered / norm


def _procrustes_rms(before: np.ndarray, after: np.ndarray) -> float:
    first = _normalized_landmarks(before)
    second = _normalized_landmarks(after)
    if first.shape != second.shape:
        raise ValueError("before_after_landmark_counts_differ")
    u, _, vh = np.linalg.svd(second.T @ first)
    rotation = u @ vh
    aligned = second @ rotation
    return float(np.sqrt(np.mean(np.sum((first - aligned) ** 2, axis=1))))


def _bbox(points: np.ndarray) -> tuple[float, float, float, float]:
    xy = np.asarray(points, dtype=np.float64)[:, :2]
    return (
        float(xy[:, 0].min()),
        float(xy[:, 1].min()),
        float(xy[:, 0].max()),
        float(xy[:, 1].max()),
    )


def _bbox_iou(first: tuple[float, ...], second: tuple[float, ...]) -> float:
    left, top = max(first[0], second[0]), max(first[1], second[1])
    right, bottom = min(first[2], second[2]), min(first[3], second[3])
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    area_first = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
    area_second = max(0.0, second[2] - second[0]) * max(0.0, second[3] - second[1])
    union = area_first + area_second - intersection
    return intersection / union if union > 0 else 0.0


def _face_crop(image: np.ndarray, points: np.ndarray, margin: float = 0.15) -> np.ndarray:
    height, width = image.shape[:2]
    x0, y0, x1, y1 = _bbox(points)
    dx, dy = (x1 - x0) * margin, (y1 - y0) * margin
    left = max(0, int((x0 - dx) * width))
    top = max(0, int((y0 - dy) * height))
    right = min(width, int(np.ceil((x1 + dx) * width)))
    bottom = min(height, int(np.ceil((y1 + dy) * height)))
    if right <= left or bottom <= top:
        raise ValueError("face_crop_is_empty")
    return image[top:bottom, left:right]


def _gradient_energy(image: np.ndarray) -> float:
    luminance = image @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
    dx = np.diff(luminance, axis=1)
    dy = np.diff(luminance, axis=0)
    return float((np.mean(np.abs(dx)) + np.mean(np.abs(dy))) / 2)


def _frequency_energy(image: np.ndarray) -> tuple[float, float]:
    luminance = image @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
    spectrum = np.abs(np.fft.fftshift(np.fft.fft2(luminance)))
    height, width = spectrum.shape
    yy, xx = np.ogrid[:height, :width]
    radius = np.sqrt(((yy - height / 2) / max(height, 1)) ** 2 + ((xx - width / 2) / max(width, 1)) ** 2)
    low = float(spectrum[radius <= 0.08].mean())
    high = float(spectrum[radius >= 0.25].mean())
    return low, high


def validate_restoration(
    source_path: Path,
    restored_path: Path,
    *,
    landmark_provider: LandmarkProvider | None,
    lpips_provider: LpipsProvider | None,
    thresholds: ValidationThresholds = ValidationThresholds(),
) -> dict[str, object]:
    """Fail closed on identity drift, hallucinated detail, clipping, or missing metrics."""
    if landmark_provider is None:
        raise RuntimeError("local_before_after_face_landmark_provider_required")
    if lpips_provider is None:
        raise RuntimeError("local_lpips_provider_required")
    with Image.open(restored_path) as restored_image:
        output_size = restored_image.size
    source = _rgb(source_path, output_size)
    restored = _rgb(restored_path)
    if source.shape != restored.shape:
        raise ValueError("restored_dimensions_do_not_match_expected_upscale")

    source_faces = landmark_provider(source_path)
    restored_faces = landmark_provider(restored_path)
    blockers: list[str] = []
    if len(source_faces) != 1 or len(restored_faces) != 1:
        blockers.append(
            f"exactly_one_face_required:before={len(source_faces)}:after={len(restored_faces)}"
        )
        return {
            "passed": False,
            "promotion_allowed": False,
            "human_review_required": True,
            "metrics": {},
            "thresholds": asdict(thresholds),
            "blockers": blockers,
        }

    before_points, after_points = source_faces[0], restored_faces[0]
    landmark_rms = _procrustes_rms(before_points, after_points)
    bbox_iou = _bbox_iou(_bbox(before_points), _bbox(after_points))
    global_lpips = float(lpips_provider(source, restored))
    before_face = _face_crop(source, before_points)
    after_face = _face_crop(restored, after_points)
    after_face = np.asarray(
        Image.fromarray(np.clip(after_face * 255, 0, 255).astype(np.uint8)).resize(
            (before_face.shape[1], before_face.shape[0]),
            Image.Resampling.LANCZOS,
        ),
        dtype=np.float32,
    ) / 255.0
    face_lpips = float(lpips_provider(before_face, after_face))

    source_edge = _gradient_energy(source)
    restored_edge = _gradient_energy(restored)
    edge_gain = restored_edge / max(source_edge, 1e-8)
    black_delta = float(np.mean(restored <= 1 / 255) - np.mean(source <= 1 / 255))
    white_delta = float(np.mean(restored >= 254 / 255) - np.mean(source >= 254 / 255))
    source_low, source_high = _frequency_energy(source)
    restored_low, restored_high = _frequency_energy(restored)
    low_change = abs(restored_low - source_low) / max(source_low, 1e-8)
    high_gain = (restored_high - source_high) / max(source_high, 1e-8)

    metrics = {
        "landmark_procrustes_rms": landmark_rms,
        "face_bbox_iou": bbox_iou,
        "global_lpips": global_lpips,
        "face_lpips": face_lpips,
        "edge_detail_gain": edge_gain,
        "black_clip_delta": black_delta,
        "white_clip_delta": white_delta,
        "low_frequency_relative_change": low_change,
        "high_frequency_relative_gain": high_gain,
    }
    checks = {
        "landmark_identity_shape": landmark_rms <= thresholds.landmark_procrustes_rms_max,
        "face_geometry": bbox_iou >= thresholds.face_bbox_iou_min,
        "global_perceptual_change": global_lpips <= thresholds.global_lpips_max,
        "face_perceptual_change": face_lpips <= thresholds.face_lpips_max,
        "detail_gain": thresholds.edge_gain_min <= edge_gain <= thresholds.edge_gain_max,
        "black_clipping": black_delta <= thresholds.black_clip_delta_max,
        "white_clipping": white_delta <= thresholds.white_clip_delta_max,
        "low_frequency_preservation": low_change <= thresholds.low_frequency_change_max,
        "high_frequency_hallucination": high_gain <= thresholds.high_frequency_gain_max,
    }
    blockers.extend(name for name, passed in checks.items() if not passed)
    return {
        "passed": not blockers,
        "promotion_allowed": False,
        "candidate_status": "pending_100_percent_human_review" if not blockers else "rejected",
        "human_review_required": True,
        "metrics": metrics,
        "checks": checks,
        "thresholds": asdict(thresholds),
        "blockers": blockers,
    }
