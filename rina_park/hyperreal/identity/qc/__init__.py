"""Non-generative, local QC adapters for Phase-1 identity assets."""

from .gate import evaluate_qc_report
from .integrity import build_integrity_report, dhash, inspect_provenance
from .macos_vision import MacOSVisionOCRAdapter
from .mediapipe_tasks import MediaPipeTasksAdapter
from .registry import load_model_registry, verified_model_paths

__all__ = [
    "MacOSVisionOCRAdapter",
    "MediaPipeTasksAdapter",
    "build_integrity_report",
    "dhash",
    "evaluate_qc_report",
    "inspect_provenance",
    "load_model_registry",
    "verified_model_paths",
]
