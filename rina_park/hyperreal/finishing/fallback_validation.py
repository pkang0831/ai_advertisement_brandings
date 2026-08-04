"""Preservation-first metrics for deterministic non-generative finishing."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from .validation import (
    LandmarkProvider,
    LpipsProvider,
    ValidationThresholds,
    _rgb,
    validate_restoration,
)


@dataclass(frozen=True)
class FallbackThresholds:
    landmark_procrustes_rms_max: float = 0.003
    face_bbox_iou_min: float = 0.98
    global_lpips_max: float = 0.01
    face_lpips_max: float = 0.01
    edge_gain_min: float = 0.98
    edge_gain_max: float = 1.12
    black_clip_delta_max: float = 0.001
    white_clip_delta_max: float = 0.001
    low_frequency_change_max: float = 0.03
    high_frequency_gain_max: float = 0.20
    delta_e76_mean_max: float = 1.0
    delta_e76_p95_max: float = 2.0
    ringing_fraction_max: float = 0.001
    edge_halo_delta_mean_max: float = 0.015

    def core(self) -> ValidationThresholds:
        return ValidationThresholds(
            landmark_procrustes_rms_max=self.landmark_procrustes_rms_max,
            face_bbox_iou_min=self.face_bbox_iou_min,
            global_lpips_max=self.global_lpips_max,
            face_lpips_max=self.face_lpips_max,
            edge_gain_min=self.edge_gain_min,
            edge_gain_max=self.edge_gain_max,
            black_clip_delta_max=self.black_clip_delta_max,
            white_clip_delta_max=self.white_clip_delta_max,
            low_frequency_change_max=self.low_frequency_change_max,
            high_frequency_gain_max=self.high_frequency_gain_max,
        )


def _delta_e76(reference: np.ndarray, candidate: np.ndarray) -> tuple[float, float]:
    reference_lab = cv2.cvtColor(reference.astype(np.float32), cv2.COLOR_RGB2LAB)
    candidate_lab = cv2.cvtColor(candidate.astype(np.float32), cv2.COLOR_RGB2LAB)
    delta = np.linalg.norm(candidate_lab - reference_lab, axis=2)
    return float(delta.mean()), float(np.percentile(delta, 95))


def _ringing_metrics(
    source: np.ndarray,
    candidate: np.ndarray,
) -> tuple[float, float]:
    weights = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
    native = source @ weights
    output = candidate @ weights
    kernel = np.ones((3, 3), dtype=np.uint8)
    target_size = (candidate.shape[1], candidate.shape[0])
    baseline = cv2.resize(native, target_size, interpolation=cv2.INTER_LINEAR)
    local_min = cv2.resize(
        cv2.erode(native, kernel),
        target_size,
        interpolation=cv2.INTER_LINEAR,
    )
    local_max = cv2.resize(
        cv2.dilate(native, kernel),
        target_size,
        interpolation=cv2.INTER_LINEAR,
    )
    excursion = np.maximum(output - local_max, local_min - output)
    gx = cv2.Sobel(baseline, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(baseline, cv2.CV_32F, 0, 1, ksize=3)
    edge = np.hypot(gx, gy) > 0.08
    edge_band = cv2.dilate(edge.astype(np.uint8), kernel).astype(bool)
    if not np.any(edge_band):
        return 0.0, 0.0
    ringing = float(np.mean(excursion[edge_band] > (2.0 / 255.0)))
    halo_delta = float(np.mean(np.abs(output - baseline)[edge_band]))
    return ringing, halo_delta


def validate_fallback(
    source_path: Path,
    output_path: Path | None,
    *,
    preset: str,
    landmark_provider: LandmarkProvider | None,
    lpips_provider: LpipsProvider | None,
    thresholds: FallbackThresholds = FallbackThresholds(),
) -> dict[str, object]:
    """Validate metadata-only or deterministic pixel processing."""
    target_path = source_path if output_path is None else output_path
    core = validate_restoration(
        source_path,
        target_path,
        landmark_provider=landmark_provider,
        lpips_provider=lpips_provider,
        thresholds=thresholds.core(),
    )
    with Image.open(target_path) as target:
        target_size = target.size
    reference = _rgb(source_path, target_size)
    source_native = _rgb(source_path)
    candidate = _rgb(target_path)
    delta_mean, delta_p95 = _delta_e76(reference, candidate)
    ringing_fraction, halo_delta = _ringing_metrics(source_native, candidate)
    extra_metrics = {
        "delta_e76_mean": delta_mean,
        "delta_e76_p95": delta_p95,
        "ringing_fraction": ringing_fraction,
        "edge_halo_delta_mean": halo_delta,
    }
    extra_checks = {
        "color_preservation_mean": delta_mean <= thresholds.delta_e76_mean_max,
        "color_preservation_p95": delta_p95 <= thresholds.delta_e76_p95_max,
        "ringing": ringing_fraction <= thresholds.ringing_fraction_max,
        "edge_halo": halo_delta <= thresholds.edge_halo_delta_mean_max,
    }
    blockers = list(core["blockers"])
    blockers.extend(name for name, passed in extra_checks.items() if not passed)
    metrics = {**core["metrics"], **extra_metrics}
    checks = {**core.get("checks", {}), **extra_checks}
    return {
        "preset": preset,
        "passed": not blockers,
        "promotion_allowed": False,
        "human_review_required": True,
        "candidate_status": (
            "pending_100_percent_human_review" if not blockers else "rejected"
        ),
        "metrics": metrics,
        "checks": checks,
        "thresholds": asdict(thresholds),
        "blockers": blockers,
    }
