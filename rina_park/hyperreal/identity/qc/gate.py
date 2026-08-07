"""Interpret detector reports without overstating detector certainty."""

from __future__ import annotations

from typing import Any


REQUIRED_COMPONENTS = ("landmarks", "text_watermark", "integrity", "provenance")
FINISHED_STATUSES = {"complete", "completed"}


def _result(report: dict[str, Any], name: str) -> dict[str, Any] | None:
    value = report.get(name)
    return value if isinstance(value, dict) else None


def evaluate_qc_report(report: Any) -> dict[str, Any]:
    """Return hard blockers and uncertainty that requires explicit human review."""
    blockers: list[str] = []
    hitl_reasons: list[str] = []
    if not isinstance(report, dict):
        return {
            "detectors_complete": False,
            "approval_eligible": False,
            "blockers": ["missing_qc_report"],
            "hitl_required": False,
            "hitl_reasons": [],
        }

    for component in REQUIRED_COMPONENTS:
        result = _result(report, component)
        if result is None:
            blockers.append(f"missing_detector_result:{component}")
        elif result.get("status") not in FINISHED_STATUSES:
            blockers.append(
                f"detector_not_complete:{component}:{result.get('status', 'missing')}"
            )

    landmarks = _result(report, "landmarks") or {}
    face_count = landmarks.get("face_count")
    if isinstance(face_count, int):
        if face_count == 0:
            blockers.append("no_face_detected")
        elif face_count > 1:
            blockers.append("multiple_faces_detected")
    resolution = landmarks.get("face_crop_effective_resolution")
    if (
        isinstance(resolution, list)
        and len(resolution) == 2
        and all(isinstance(value, (int, float)) for value in resolution)
        and min(resolution) < 512
    ):
        blockers.append("low_face_effective_resolution")

    for flag in landmarks.get("topology_flags", []):
        blockers.append(f"landmark_topology:{flag}")
    if landmarks.get("occlusion", {}).get("blocks_required_face_review") is True:
        blockers.append("face_review_blocked_by_occlusion")

    uncertainty = landmarks.get("uncertainty", [])
    if isinstance(uncertainty, list):
        hitl_reasons.extend(f"landmarks:{item}" for item in uncertainty)
    for section in ("eyes", "gaze", "head_pose", "pose", "hands", "subject", "occlusion"):
        value = landmarks.get(section)
        if isinstance(value, dict) and value.get("status") == "uncertain":
            hitl_reasons.append(f"{section}_uncertain")

    text = _result(report, "text_watermark") or {}
    if text.get("detected") is True:
        blockers.append("text_or_watermark_detected")
    if text.get("detected") is None and text.get("status") in FINISHED_STATUSES:
        hitl_reasons.append("text_detection_inconclusive")

    integrity = _result(report, "integrity") or {}
    if integrity.get("duplicate") is True:
        blockers.append("duplicate_or_near_duplicate")
    if integrity.get("source_hash_matches") is False:
        blockers.append("source_hash_mismatch")

    provenance = _result(report, "provenance") or {}
    if provenance.get("complete") is False:
        missing = ",".join(str(item) for item in provenance.get("missing", []))
        blockers.append(f"incomplete_provenance:{missing or 'unspecified'}")

    hitl_reasons.extend(str(item) for item in report.get("uncertainty", []))
    hitl_reasons = sorted(set(hitl_reasons))
    blockers = sorted(set(blockers))
    hitl_cleared = report.get("human_qc_review", {}).get("status") == "cleared"
    return {
        "detectors_complete": not any(
            item.startswith(("missing_detector_result:", "detector_not_complete:"))
            for item in blockers
        ),
        "approval_eligible": not blockers and (not hitl_reasons or hitl_cleared),
        "blockers": blockers,
        "hitl_required": bool(hitl_reasons and not hitl_cleared),
        "hitl_reasons": hitl_reasons,
    }
