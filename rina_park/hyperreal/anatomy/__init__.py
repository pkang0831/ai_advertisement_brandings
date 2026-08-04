"""Anatomy lock: pose catalog, QC gates, 2-pass inpaint, crop ControlNet."""

from .pose_catalog import PoseCatalog, load_pose_catalog
from .qc_gates import evaluate_anatomy_image, scorecard_template

__all__ = [
    "PoseCatalog",
    "load_pose_catalog",
    "evaluate_anatomy_image",
    "scorecard_template",
]
