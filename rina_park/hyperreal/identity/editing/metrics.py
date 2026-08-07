"""Mask-exterior preservation metrics with an injected, local LPIPS provider."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
from PIL import Image

from .contract import PRESERVATION_THRESHOLD


SpatialLpips = Callable[[np.ndarray, np.ndarray], np.ndarray | float]


@dataclass(frozen=True)
class PreservationResult:
    exterior_mean_absolute_diff: float
    exterior_lpips: float
    threshold: float
    passed: bool
    exterior_pixel_count: int


def _rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0


def _editable_mask(path: Path, size: tuple[int, int]) -> np.ndarray:
    with Image.open(path) as image:
        mask = image.convert("L")
        if mask.size != size:
            raise ValueError("mask_must_align_pixel_for_pixel_with_scene")
        return np.asarray(mask, dtype=np.float32) / 255.0 >= 0.5


def measure_mask_exterior(
    source_path: Path,
    edited_path: Path,
    mask_path: Path,
    *,
    spatial_lpips: SpatialLpips | None,
    threshold: float = PRESERVATION_THRESHOLD,
) -> PreservationResult:
    """Reject if either normalized RGB diff or spatial LPIPS exceeds 0.02."""
    if spatial_lpips is None:
        raise RuntimeError("local_spatial_lpips_provider_required")
    source = _rgb(source_path)
    edited = _rgb(edited_path)
    if source.shape != edited.shape:
        raise ValueError("edited_dimensions_must_equal_source")
    height, width = source.shape[:2]
    editable = _editable_mask(mask_path, (width, height))
    exterior = ~editable
    exterior_count = int(exterior.sum())
    if exterior_count == 0:
        raise ValueError("mask_exterior_must_not_be_empty")

    mean_diff = float(np.abs(source - edited)[exterior].mean())
    lpips_map = np.asarray(spatial_lpips(source, edited), dtype=np.float32)
    if lpips_map.ndim == 0:
        raise ValueError("lpips_provider_must_return_a_spatial_map")
    lpips_map = np.squeeze(lpips_map)
    if lpips_map.shape != exterior.shape:
        lpips_image = Image.fromarray(lpips_map.astype(np.float32), mode="F")
        lpips_image = lpips_image.resize((width, height), Image.Resampling.BILINEAR)
        lpips_map = np.asarray(lpips_image, dtype=np.float32)
    exterior_lpips = float(lpips_map[exterior].mean())
    passed = mean_diff <= threshold and exterior_lpips <= threshold
    return PreservationResult(
        exterior_mean_absolute_diff=mean_diff,
        exterior_lpips=exterior_lpips,
        threshold=threshold,
        passed=passed,
        exterior_pixel_count=exterior_count,
    )
