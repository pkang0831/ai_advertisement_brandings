"""Non-generative Phase-0 image pre-screening."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Protocol

import numpy as np
from PIL import Image


class TextWatermarkDetector(Protocol):
    detector_id: str

    def detect(self, image_path: Path) -> dict[str, Any]:
        """Return at least detected(bool), regions(list), and confidence(float)."""


class VisionLandmarkDetector(Protocol):
    detector_id: str

    def detect(self, image_path: Path) -> dict[str, Any]:
        """Return pose, face and hands flags or findings."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _gray_thumbnail(path: Path, size: tuple[int, int] = (256, 256)) -> np.ndarray:
    with Image.open(path) as image:
        gray = image.convert("L").resize(size, Image.Resampling.LANCZOS)
    return np.asarray(gray, dtype=np.float64)


def global_ssim(left: np.ndarray, right: np.ndarray) -> float:
    """Dependency-light global SSIM used only as a high-similarity copy safeguard."""
    if left.shape != right.shape:
        raise ValueError("SSIM inputs must have identical shapes")
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    mean_left, mean_right = float(left.mean()), float(right.mean())
    variance_left, variance_right = float(left.var()), float(right.var())
    covariance = float(((left - mean_left) * (right - mean_right)).mean())
    numerator = (2 * mean_left * mean_right + c1) * (2 * covariance + c2)
    denominator = (mean_left**2 + mean_right**2 + c1) * (
        variance_left + variance_right + c2
    )
    return float(numerator / denominator) if denominator else 1.0


def _basic_image_checks(image_path: Path, expected_dimensions: tuple[int, int]) -> dict[str, Any]:
    with Image.open(image_path) as image:
        dimensions = image.size
        pixels = np.asarray(image.convert("RGB"), dtype=np.uint8)
    black_fraction = float(np.all(pixels <= 3, axis=2).mean())
    white_fraction = float(np.all(pixels >= 252, axis=2).mean())
    channel_clipping = {
        "low": [float((pixels[:, :, channel] <= 1).mean()) for channel in range(3)],
        "high": [float((pixels[:, :, channel] >= 254).mean()) for channel in range(3)],
    }
    return {
        "dimensions": list(dimensions),
        "expected_dimensions": list(expected_dimensions),
        "dimensions_ok": dimensions == expected_dimensions,
        "black_pixel_fraction": round(black_fraction, 6),
        "white_pixel_fraction": round(white_fraction, 6),
        "black_frame": black_fraction >= 0.90 or float(pixels.std()) < 3.0,
        "clipping": channel_clipping,
        "severe_clipping": black_fraction >= 0.25 or white_fraction >= 0.25,
    }


def _copy_checks(
    image_path: Path,
    moodboard_paths: list[Path],
    comparison_outputs: list[Path],
    ssim_threshold: float,
) -> dict[str, Any]:
    target_hash = _sha256(image_path)
    target = _gray_thumbnail(image_path)
    comparisons: list[dict[str, Any]] = []
    for kind, paths in (("moodboard", moodboard_paths), ("planned_output", comparison_outputs)):
        for candidate in paths:
            if candidate.resolve() == image_path.resolve() or not candidate.is_file():
                continue
            exact = target_hash == _sha256(candidate)
            score = global_ssim(target, _gray_thumbnail(candidate))
            comparisons.append(
                {
                    "kind": kind,
                    "path": str(candidate),
                    "exact_sha256_match": exact,
                    "ssim": round(score, 6),
                    "suspected_copy": exact or score >= ssim_threshold,
                }
            )
    suspected = [item for item in comparisons if item["suspected_copy"]]
    return {
        "ssim_threshold": ssim_threshold,
        "comparisons": comparisons,
        "suspected_copy": bool(suspected),
        "suspected_matches": suspected,
    }


def _optional_mediapipe(image_path: Path) -> dict[str, Any]:
    try:
        import mediapipe as mp  # type: ignore[import-not-found]
    except ImportError:
        return {
            "available": False,
            "detector_id": "mediapipe",
            "pose": "not_run",
            "face": "not_run",
            "hands": "not_run",
            "flags": [],
        }

    with Image.open(image_path) as image:
        rgb = np.asarray(image.convert("RGB"))
    findings: dict[str, Any] = {
        "available": True,
        "detector_id": f"mediapipe-{getattr(mp, '__version__', 'unknown')}",
        "flags": [],
    }
    solutions = getattr(mp, "solutions", None)
    if solutions is None:
        return {**findings, "pose": "unsupported_api", "face": "unsupported_api", "hands": "unsupported_api"}
    with solutions.pose.Pose(static_image_mode=True) as pose:
        findings["pose"] = bool(pose.process(rgb).pose_landmarks)
    with solutions.face_detection.FaceDetection(model_selection=1) as face:
        findings["face"] = bool(face.process(rgb).detections)
    with solutions.hands.Hands(static_image_mode=True, max_num_hands=2) as hands:
        hand_result = hands.process(rgb).multi_hand_landmarks
        findings["hands"] = len(hand_result or [])
    if not findings["pose"]:
        findings["flags"].append("pose_not_detected")
    if not findings["face"]:
        findings["flags"].append("face_not_detected")
    if findings["hands"] > 2:
        findings["flags"].append("more_than_two_hands")
    return findings


def run_prescreen(
    image_path: Path,
    sidecar: dict[str, Any],
    *,
    expected_dimensions: tuple[int, int],
    moodboard_paths: list[Path],
    comparison_outputs: list[Path] | None = None,
    landmark_detector: VisionLandmarkDetector | None = None,
    text_detector: TextWatermarkDetector | None = None,
    ssim_threshold: float = 0.985,
) -> dict[str, Any]:
    from .spec import REQUIRED_CAMERA_FIELDS

    required_metadata = {
        "blind_id",
        "hypothesis_id",
        "seed",
        "camera_hypothesis",
        "timing",
        "memory",
        "publication_allowed",
    }
    missing_metadata = sorted(required_metadata.difference(sidecar))
    camera = sidecar.get("camera_hypothesis")
    if isinstance(camera, dict):
        missing_camera = sorted(REQUIRED_CAMERA_FIELDS.difference(camera))
    else:
        missing_camera = sorted(REQUIRED_CAMERA_FIELDS)

    basic = _basic_image_checks(image_path, expected_dimensions)
    copies = _copy_checks(
        image_path,
        moodboard_paths,
        comparison_outputs or [],
        ssim_threshold,
    )
    landmarks = (
        landmark_detector.detect(image_path)
        if landmark_detector is not None
        else _optional_mediapipe(image_path)
    )
    text = (
        text_detector.detect(image_path)
        if text_detector is not None
        else {
            "available": False,
            "detector_id": None,
            "detected": None,
            "regions": [],
            "confidence": None,
        }
    )
    critical_failures: list[str] = []
    if not basic["dimensions_ok"]:
        critical_failures.append("wrong_dimensions")
    if basic["black_frame"]:
        critical_failures.append("black_frame")
    if basic["severe_clipping"]:
        critical_failures.append("severe_clipping")
    if copies["suspected_copy"]:
        critical_failures.append("duplicate_or_moodboard_pixel_copy")
    if text.get("detected") is True:
        critical_failures.append("text_or_watermark_detected")
    if missing_metadata or missing_camera:
        critical_failures.append("incomplete_camera_hypothesis_metadata")
    return {
        "schema_version": "1.0.0",
        "image": str(image_path),
        "basic": basic,
        "landmarks": landmarks,
        "copy_safeguards": copies,
        "text_watermark": text,
        "metadata": {
            "complete": not missing_metadata and not missing_camera,
            "missing_sidecar_fields": missing_metadata,
            "missing_camera_fields": missing_camera,
        },
        "critical_failures": critical_failures,
        "passed": not critical_failures,
    }
