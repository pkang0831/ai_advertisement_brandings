"""MediaPipe Tasks adapter using only local model assets.

MediaPipe is Apache-2.0. This module never downloads model files and performs
landmark-based QC, not face recognition or identity matching.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any, Iterable

from PIL import Image

from .registry import verified_model_paths


def _bbox(points: Iterable[Any]) -> list[float] | None:
    coordinates = [(float(point.x), float(point.y)) for point in points]
    if not coordinates:
        return None
    xs, ys = zip(*coordinates)
    return [min(xs), min(ys), max(xs), max(ys)]


def _clipped_bbox(box: list[float] | None) -> list[float] | None:
    if box is None:
        return None
    return [
        max(0.0, min(1.0, box[0])),
        max(0.0, min(1.0, box[1])),
        max(0.0, min(1.0, box[2])),
        max(0.0, min(1.0, box[3])),
    ]


def _finite_topology_flags(groups: Iterable[Iterable[Any]]) -> list[str]:
    flags: list[str] = []
    for group_index, points in enumerate(groups):
        coordinates = [(float(point.x), float(point.y), float(point.z)) for point in points]
        if any(not all(math.isfinite(value) for value in point) for point in coordinates):
            flags.append(f"group_{group_index}_non_finite_landmark")
        if len(coordinates) > 1 and len(set(coordinates)) < len(coordinates) * 0.5:
            flags.append(f"group_{group_index}_collapsed_landmark_topology")
    return flags


def _hand_topology_flags(hands: list[list[Any]]) -> list[str]:
    flags: list[str] = []
    for hand_index, hand in enumerate(hands):
        if len(hand) != 21:
            flags.append(f"hand_{hand_index}_unexpected_landmark_count_{len(hand)}")
            continue
        wrist = hand[0]
        for finger, joints in {
            "thumb": (1, 2, 3, 4),
            "index": (5, 6, 7, 8),
            "middle": (9, 10, 11, 12),
            "ring": (13, 14, 15, 16),
            "pinky": (17, 18, 19, 20),
        }.items():
            distances = [
                math.hypot(hand[index].x - wrist.x, hand[index].y - wrist.y)
                for index in joints
            ]
            if max(distances) < 1e-4:
                flags.append(f"hand_{hand_index}_{finger}_collapsed")
    return flags


def _head_pose(face: list[Any]) -> dict[str, Any]:
    if len(face) <= 263:
        return {"status": "uncertain", "yaw_degrees": None, "pitch_degrees": None}
    left_eye, right_eye, nose, chin = face[33], face[263], face[1], face[152]
    eye_mid_x = (left_eye.x + right_eye.x) / 2
    eye_span = max(abs(right_eye.x - left_eye.x), 1e-6)
    face_height = max(abs(chin.y - (left_eye.y + right_eye.y) / 2), 1e-6)
    yaw = max(-90.0, min(90.0, (nose.x - eye_mid_x) / eye_span * 90.0))
    expected_nose_y = (left_eye.y + right_eye.y) / 2 + face_height * 0.45
    pitch = max(-90.0, min(90.0, (nose.y - expected_nose_y) / face_height * 90.0))
    return {
        "status": "estimated",
        "yaw_degrees": round(yaw, 2),
        "pitch_degrees": round(pitch, 2),
        "method": "2d_landmark_ratio_not_calibrated_pose",
    }


def _eye_gaze(face: list[Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    if len(face) <= 477:
        unavailable = {
            "status": "uncertain",
            "available": False,
            "reason": "iris_landmarks_not_returned",
        }
        return unavailable, unavailable.copy()
    left_span = max(abs(face[33].x - face[133].x), 1e-6)
    right_span = max(abs(face[362].x - face[263].x), 1e-6)
    left_ratio = (face[468].x - min(face[33].x, face[133].x)) / left_span
    right_ratio = (face[473].x - min(face[362].x, face[263].x)) / right_span
    vertical_delta = abs(face[468].y - face[473].y)
    return (
        {
            "status": "estimated",
            "available": True,
            "iris_vertical_delta_normalized": round(vertical_delta, 5),
            "alignment_flag": vertical_delta > 0.035,
        },
        {
            "status": "estimated",
            "available": True,
            "left_horizontal_ratio": round(left_ratio, 4),
            "right_horizontal_ratio": round(right_ratio, 4),
            "asymmetry_flag": abs(left_ratio - right_ratio) > 0.25,
            "method": "iris_within_eye_ratio_not_semantic_gaze_classification",
        },
    )


@dataclass(frozen=True)
class MediaPipeModelPaths:
    face: Path
    pose: Path
    hand: Path


class MediaPipeTasksAdapter:
    """Run local Face/Pose/Hand Landmarker Tasks when model paths are supplied."""

    detector_id = "mediapipe-tasks-local"

    def __init__(self, model_paths: MediaPipeModelPaths | None = None) -> None:
        self.configuration_error: str | None = None
        if model_paths is not None:
            self.model_paths = model_paths
            return
        try:
            verified = verified_model_paths()
            self.model_paths = MediaPipeModelPaths(
                face=verified["face"],
                pose=verified["pose"],
                hand=verified["hand"],
            )
        except (KeyError, OSError, RuntimeError, ValueError) as error:
            self.model_paths = None
            self.configuration_error = str(error)

    def _unavailable(self, reason: str) -> dict[str, Any]:
        return {
            "status": "unavailable",
            "available": False,
            "detector_id": self.detector_id,
            "reason": reason,
            "face_count": None,
            "uncertainty": ["landmark_detector_unavailable"],
        }

    def detect(self, image_path: Path) -> dict[str, Any]:
        if self.model_paths is None:
            return self._unavailable(
                self.configuration_error or "local_model_paths_not_configured"
            )
        missing = [
            str(path)
            for path in (
                self.model_paths.face,
                self.model_paths.pose,
                self.model_paths.hand,
            )
            if not path.is_file()
        ]
        if missing:
            return self._unavailable(f"local_model_files_missing:{','.join(missing)}")
        try:
            import mediapipe as mp  # type: ignore[import-not-found]
            from mediapipe.tasks import python  # type: ignore[import-not-found]
            from mediapipe.tasks.python import vision  # type: ignore[import-not-found]
        except ImportError:
            return self._unavailable("mediapipe_package_not_installed_in_project_venv")

        base = python.BaseOptions
        options = vision.RunningMode.IMAGE
        with (
            vision.FaceLandmarker.create_from_options(
                vision.FaceLandmarkerOptions(
                    base_options=base(model_asset_path=str(self.model_paths.face)),
                    running_mode=options,
                    num_faces=3,
                    output_face_blendshapes=False,
                    output_facial_transformation_matrixes=False,
                )
            ) as face_detector,
            vision.PoseLandmarker.create_from_options(
                vision.PoseLandmarkerOptions(
                    base_options=base(model_asset_path=str(self.model_paths.pose)),
                    running_mode=options,
                    num_poses=2,
                )
            ) as pose_detector,
            vision.HandLandmarker.create_from_options(
                vision.HandLandmarkerOptions(
                    base_options=base(model_asset_path=str(self.model_paths.hand)),
                    running_mode=options,
                    num_hands=2,
                )
            ) as hand_detector,
        ):
            image = mp.Image.create_from_file(str(image_path))
            faces = face_detector.detect(image).face_landmarks
            poses = pose_detector.detect(image).pose_landmarks
            hands = hand_detector.detect(image).hand_landmarks

        with Image.open(image_path) as source:
            width, height = source.size
        primary_face = faces[0] if faces else []
        face_box = _bbox(primary_face)
        face_resolution = (
            [
                round((face_box[2] - face_box[0]) * width),
                round((face_box[3] - face_box[1]) * height),
            ]
            if face_box
            else None
        )
        subject_box_raw = _bbox(poses[0]) if poses else face_box
        subject_box = _clipped_bbox(subject_box_raw)
        occupancy = (
            round((subject_box[2] - subject_box[0]) * (subject_box[3] - subject_box[1]), 5)
            if subject_box
            else None
        )
        eyes, gaze = _eye_gaze(primary_face)
        uncertainty: list[str] = []
        if gaze.get("available") is False:
            uncertainty.append("iris_or_gaze_unavailable")
        if subject_box_raw != subject_box:
            uncertainty.append("subject_bbox_extends_outside_frame")
        return {
            "status": "complete",
            "available": True,
            "detector_id": f"{self.detector_id}-{getattr(mp, '__version__', 'unknown')}",
            "face_count": len(faces),
            "face_bbox_normalized": face_box,
            "face_crop_effective_resolution": face_resolution,
            "eyes": eyes,
            "gaze": gaze,
            "head_pose": _head_pose(primary_face),
            "pose": {
                "status": "estimated" if poses else "not_visible",
                "person_count": len(poses),
                "visible": bool(poses),
            },
            "hands": {
                "status": "estimated" if hands else "not_visible",
                "count": len(hands),
                "visible": bool(hands),
                "landmarks_per_hand": [len(hand) for hand in hands],
                "finger_topology_flags": _hand_topology_flags(hands),
            },
            "subject": {
                "status": "estimated" if subject_box else "uncertain",
                "bbox_normalized": subject_box,
                "unclipped_bbox_normalized": subject_box_raw,
                "occupancy_fraction": occupancy,
            },
            "occlusion": {
                "status": "uncertain",
                "blocks_required_face_review": None,
                "reason": "landmarks_cannot_reliably_classify_occluders",
            },
            "topology_flags": [
                *_finite_topology_flags([*faces, *poses, *hands]),
                *_hand_topology_flags(hands),
            ],
            "uncertainty": uncertainty,
        }
