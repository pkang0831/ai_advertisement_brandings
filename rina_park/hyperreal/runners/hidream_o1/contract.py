"""Deterministic manifest types shared with the future foundation bake-off."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


def _finite_positive(value: float | None, name: str) -> None:
    if value is not None and value <= 0:
        raise ValueError(f"{name} must be positive")


@dataclass(frozen=True)
class CameraMetadata:
    """Physical camera intent; unknown values remain explicit nulls."""

    camera_make: str | None = None
    camera_model: str | None = None
    lens_model: str | None = None
    focal_length_mm: float | None = None
    aperture_f: float | None = None
    shutter_seconds: float | None = None
    iso: int | None = None
    camera_height_m: float | None = None
    subject_distance_m: float | None = None
    yaw_degrees: float | None = None
    pitch_degrees: float | None = None
    roll_degrees: float | None = None

    def __post_init__(self) -> None:
        for field_name in (
            "focal_length_mm",
            "aperture_f",
            "shutter_seconds",
            "camera_height_m",
            "subject_distance_m",
        ):
            _finite_positive(getattr(self, field_name), field_name)
        if self.iso is not None and self.iso <= 0:
            raise ValueError("iso must be positive")


@dataclass(frozen=True)
class BakeoffRequest:
    """Versioned, deterministic input contract; does not imply generation approval."""

    case_id: str
    prompt: str
    seed: int
    width: int
    height: int
    camera: CameraMetadata
    steps: int = 28
    model_id: str = "hidream-o1-image-dev-2604-official"
    manifest_version: int = 1

    def __post_init__(self) -> None:
        if not self.case_id.strip():
            raise ValueError("case_id is required")
        if not self.prompt.strip():
            raise ValueError("prompt is required")
        if self.seed < 0 or self.seed > 2**63 - 1:
            raise ValueError("seed must be between 0 and 2^63-1")
        if self.width <= 0 or self.height <= 0:
            raise ValueError("dimensions must be positive")
        if self.width % 32 or self.height % 32:
            raise ValueError("dimensions must be divisible by 32")
        if self.steps != 28:
            raise ValueError("the pinned Dev recipe requires exactly 28 steps")
        if self.model_id != "hidream-o1-image-dev-2604-official":
            raise ValueError("model substitution is forbidden")

    def canonical_dict(self) -> dict[str, Any]:
        return asdict(self)

    def canonical_json(self) -> str:
        return json.dumps(
            self.canonical_dict(),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )

    @property
    def request_sha256(self) -> str:
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class RunMetrics:
    wall_seconds: float
    process_max_rss_bytes: int
    accelerator_peak_bytes: int | None


@dataclass(frozen=True)
class RunRecord:
    request_sha256: str
    model_revision: str
    code_revision: str
    runtime: str
    offline: bool
    metrics: RunMetrics
    output_path: str | None
    status: str
    blocker: str | None = None

    def write(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(
            asdict(self), ensure_ascii=False, indent=2, sort_keys=True
        )
        path.write_text(payload + "\n", encoding="utf-8")
