"""Measurable anatomy QC gates for hands / feet / genitals + scorecard."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from hyperreal.identity.qc.mediapipe_tasks import MediaPipeTasksAdapter


def scorecard_template(
    *,
    image_path: str,
    method: str,
    pose_id: str,
    track: str,
) -> dict[str, Any]:
    return {
        "image": image_path,
        "method": method,
        "pose_id": pose_id,
        "track": track,
        "human_review": {
            "hands": {"score_0_to_2": None, "notes": ""},
            "feet": {"score_0_to_2": None, "notes": ""},
            "genitals": {"score_0_to_2": None, "notes": "", "na_if_sfw": track == "sfw"},
            "identity": {"score_0_to_2": None, "notes": ""},
            "overall_pass": None,
        },
        "rubric": {
            "0": "fail — mutated / unusable",
            "1": "borderline — maybe salvage with refine",
            "2": "pass — production-usable for region",
        },
    }


def _finger_spread_flags(hand: list[Any]) -> list[str]:
    """Heuristic extra/fused finger proxies from 21 landmarks."""
    flags: list[str] = []
    if len(hand) != 21:
        return [f"unexpected_count_{len(hand)}"]
    tips = [4, 8, 12, 16, 20]
    tip_pts = [(hand[i].x, hand[i].y) for i in tips]
    # Pairwise tip distances; extremely small → fused; wildly uneven cluster → melt
    dists: list[float] = []
    for i in range(len(tip_pts)):
        for j in range(i + 1, len(tip_pts)):
            dists.append(
                math.hypot(tip_pts[i][0] - tip_pts[j][0], tip_pts[i][1] - tip_pts[j][1])
            )
    if dists and min(dists) < 0.008:
        flags.append("tips_too_close_possible_fusion")
    # Finger chain length should increase wrist→tip roughly
    chains = {
        "thumb": (1, 2, 3, 4),
        "index": (5, 6, 7, 8),
        "middle": (9, 10, 11, 12),
        "ring": (13, 14, 15, 16),
        "pinky": (17, 18, 19, 20),
    }
    wrist = hand[0]
    for name, idxs in chains.items():
        lengths = [
            math.hypot(hand[i].x - wrist.x, hand[i].y - wrist.y) for i in idxs
        ]
        if lengths[-1] + 1e-6 < lengths[0] * 0.55:
            flags.append(f"{name}_implausible_short")
        # Non-monotonic collapse
        if max(lengths) < 1e-4:
            flags.append(f"{name}_collapsed")
    return flags


def _detect_raw(image_path: Path) -> dict[str, Any]:
    adapter = MediaPipeTasksAdapter()
    return adapter.detect(image_path)


def _hand_landmarks_from_path(image_path: Path) -> tuple[dict[str, Any], list[list[Any]]]:
    """Re-run hand landmarker to get landmark objects for heuristics."""
    report = _detect_raw(image_path)
    hands_meta = report.get("hands") or {}
    # MediaPipeTasksAdapter doesn't return raw landmarks; re-detect for flags.
    try:
        import mediapipe as mp
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision
        from hyperreal.identity.qc.registry import verified_model_paths

        models = verified_model_paths()
        with vision.HandLandmarker.create_from_options(
            vision.HandLandmarkerOptions(
                base_options=python.BaseOptions(model_asset_path=str(models["hand"])),
                running_mode=vision.RunningMode.IMAGE,
                num_hands=2,
            )
        ) as detector:
            mp_image = mp.Image.create_from_file(str(image_path))
            result = detector.detect(mp_image)
            raw_hands = list(result.hand_landmarks or [])
    except Exception as error:  # noqa: BLE001
        return report, []
        # unreachable — keep type checkers calm
        raise error
    return report, raw_hands


def _pose_foot_visibility(image_path: Path) -> dict[str, Any]:
    try:
        import mediapipe as mp
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision
        from hyperreal.identity.qc.registry import verified_model_paths

        models = verified_model_paths()
        with vision.PoseLandmarker.create_from_options(
            vision.PoseLandmarkerOptions(
                base_options=python.BaseOptions(model_asset_path=str(models["pose"])),
                running_mode=vision.RunningMode.IMAGE,
                num_poses=1,
            )
        ) as detector:
            mp_image = mp.Image.create_from_file(str(image_path))
            result = detector.detect(mp_image)
        if not result.pose_landmarks:
            return {"status": "not_visible", "ankles_visible": 0, "feet_score_proxy": 0.0}
        pose = result.pose_landmarks[0]
        # BlazePose: left_ankle=27, right_ankle=28, left_heel=29, right_heel=30,
        # left_foot_index=31, right_foot_index=32
        idxs = [27, 28, 29, 30, 31, 32]
        vis = []
        for i in idxs:
            if i < len(pose):
                # tasks API: visibility may be on landmark
                v = getattr(pose[i], "visibility", None)
                vis.append(float(v) if v is not None else 0.5)
        ankles = sum(1 for v in vis[:2] if v >= 0.45)
        proxy = float(np.mean(vis)) if vis else 0.0
        return {
            "status": "estimated",
            "ankles_visible": ankles,
            "mean_foot_visibility": round(proxy, 4),
            "feet_score_proxy": round(min(2.0, proxy * 2.5), 3),
        }
    except Exception as error:  # noqa: BLE001
        return {"status": "unavailable", "reason": str(error), "feet_score_proxy": None}


def _genital_region_proxy(image: Image.Image, track: str) -> dict[str, Any]:
    """Cheap automated proxy — not a medical detector. Human review required."""
    if track == "sfw":
        return {"status": "na_sfw", "pass": True, "score_proxy": None}
    arr = np.asarray(image.convert("RGB"), dtype=np.float32)
    h, w, _ = arr.shape
    # Lower-mid torso band (approximate pelvic region)
    y0, y1 = int(h * 0.45), int(h * 0.82)
    x0, x1 = int(w * 0.25), int(w * 0.75)
    crop = arr[y0:y1, x0:x1]
    if crop.size == 0:
        return {"status": "empty", "pass": False, "score_proxy": 0.0}
    mean = float(crop.mean())
    std = float(crop.std())
    # Skin-ish chroma: R>B modestly, not pure black/white void
    r, g, b = crop[:, :, 0].mean(), crop[:, :, 1].mean(), crop[:, :, 2].mean()
    skinish = (r > b * 0.9) and (40 < mean < 230) and std > 8
    # Fail hard voids / flat blobs
    fail = mean < 12 or std < 4
    score = 0.0 if fail else (1.5 if skinish else 0.5)
    return {
        "status": "proxy",
        "mean": round(mean, 2),
        "std": round(std, 2),
        "skinish": bool(skinish),
        "score_proxy": score,
        "pass": (not fail) and skinish,
        "note": "human genital anatomy review still required",
    }


def evaluate_anatomy_image(
    image_path: Path,
    *,
    pose_expected_hands: int | None = None,
    track: str = "sfw",
    require_feet: bool = False,
    require_genitals: bool = False,
) -> dict[str, Any]:
    """Return structured pass/fail for hands/feet/genitals with blockers."""
    image_path = Path(image_path)
    image = Image.open(image_path).convert("RGB")
    report, raw_hands = _hand_landmarks_from_path(image_path)
    landmarks = report if report.get("available") else report

    hand_flags: list[str] = []
    for hand in raw_hands:
        hand_flags.extend(_finger_spread_flags(hand))
    topology = list((landmarks.get("hands") or {}).get("finger_topology_flags") or [])
    topology.extend(hand_flags)
    topology.extend(landmarks.get("topology_flags") or [])
    topology = sorted(set(str(t) for t in topology))

    hand_count = int((landmarks.get("hands") or {}).get("count") or 0)
    face_count = landmarks.get("face_count")
    hands_blockers: list[str] = []
    if pose_expected_hands is not None:
        # Allow fewer detections when pose expects hidden hands (0)
        if pose_expected_hands == 0 and hand_count > 0:
            # Soft: detected hands when expecting hidden — not hard fail
            pass
        if pose_expected_hands > 0 and hand_count == 0:
            hands_blockers.append("expected_hands_not_detected")
        if pose_expected_hands > 0 and hand_count > pose_expected_hands + 1:
            hands_blockers.append("too_many_hands_detected")
    if any("collapsed" in f for f in topology):
        hands_blockers.append("collapsed_finger_topology")
    if any("fusion" in f or "fused" in f for f in topology):
        hands_blockers.append("possible_fused_digits")
    if any("implausible" in f for f in topology):
        hands_blockers.append("implausible_finger_lengths")

    # Identity coarse: need exactly one face
    identity_blockers: list[str] = []
    if face_count == 0:
        identity_blockers.append("no_face_detected")
    elif isinstance(face_count, int) and face_count > 1:
        identity_blockers.append("multiple_faces_detected")

    feet = _pose_foot_visibility(image_path)
    feet_blockers: list[str] = []
    if require_feet:
        if feet.get("status") == "not_visible":
            feet_blockers.append("feet_not_visible")
        elif (feet.get("ankles_visible") or 0) < 1 and (feet.get("mean_foot_visibility") or 0) < 0.35:
            feet_blockers.append("feet_visibility_low")

    genitals = _genital_region_proxy(image, track)
    genital_blockers: list[str] = []
    if require_genitals and not genitals.get("pass"):
        genital_blockers.append("genital_region_proxy_fail")

    # Hide poses (expected 0): do not hard-fail on accidental hand detections/topology.
    if pose_expected_hands == 0:
        hands_blockers = [
            b
            for b in hands_blockers
            if b
            not in {
                "expected_hands_not_detected",
                "collapsed_finger_topology",
                "possible_fused_digits",
                "implausible_finger_lengths",
                "too_many_hands_detected",
            }
        ]
    hands_pass = not hands_blockers

    feet_pass = not feet_blockers if require_feet else True
    genital_pass = not genital_blockers if require_genitals else True
    identity_pass = not identity_blockers

    overall = identity_pass and hands_pass and feet_pass and genital_pass

    return {
        "status": "complete" if landmarks.get("available") else landmarks.get("status", "incomplete"),
        "image": str(image_path),
        "track": track,
        "identity": {
            "pass": identity_pass,
            "face_count": face_count,
            "blockers": identity_blockers,
        },
        "hands": {
            "pass": hands_pass,
            "count": hand_count,
            "expected": pose_expected_hands,
            "topology_flags": topology,
            "blockers": hands_blockers,
            "auto_score_proxy": 2 if hands_pass and not topology else (1 if hands_pass else 0),
        },
        "feet": {
            "pass": feet_pass,
            "require": require_feet,
            "blockers": feet_blockers,
            **feet,
        },
        "genitals": {
            "pass": genital_pass,
            "require": require_genitals,
            "blockers": genital_blockers,
            **genitals,
        },
        "overall_auto_pass": overall,
        "detector": landmarks.get("detector_id"),
    }


def write_scorecard(path: Path, card: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(card, indent=2), encoding="utf-8")
