"""Fail-closed HiDream-O1-Image-Dev-2604 readiness and runner contract."""

from .contract import BakeoffRequest, CameraMetadata, RunMetrics, RunRecord
from .readiness import ReadinessReport, inspect_readiness
from .runner import HiDreamO1Runner, ModelNotReadyError

__all__ = [
    "BakeoffRequest",
    "CameraMetadata",
    "HiDreamO1Runner",
    "ModelNotReadyError",
    "ReadinessReport",
    "RunMetrics",
    "RunRecord",
    "inspect_readiness",
]
