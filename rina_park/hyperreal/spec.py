"""Phase-0 manifest validation and blinded bake-off planning."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PACKAGE_ROOT = Path(__file__).resolve().parent
DEFAULT_MANIFEST = PACKAGE_ROOT / "phase0_manifest.v1.json"
DEFAULT_RUBRIC = PACKAGE_ROOT / "rubric.v1.json"
MANIFEST_SCHEMA_VERSION = "1.1.0"
PHASE0_MODEL_IDS = ("qwen_image_2512", "z_image")
LEGACY_MODEL_IDS = {
    "hidream_o1",
    "hidream-o1-image-dev-2604",
    "hidream-o1-image-dev-2604-official",
}
REQUIRED_CAMERA_FIELDS = {
    "camera_type",
    "equivalent_focal_length_mm",
    "camera_height_m",
    "camera_distance_m",
    "orientation",
    "subject_bbox_xywh_normalized",
    "subject_occupancy_percent",
    "action",
    "gaze_expression",
    "lighting",
    "material_state",
    "background_graph",
    "wardrobe_sfw",
    "deliberate_imperfections",
    "prompt",
}
IDENTITY_MARKERS = ("rina", "rina park", "identity token", "face reference", "lora")


@dataclass(frozen=True)
class PlannedSlot:
    blind_id: str
    hypothesis_id: str
    model_id: str
    seed: int
    filename: str
    prompt: str

    def private_record(self) -> dict[str, Any]:
        return {
            "blind_id": self.blind_id,
            "hypothesis_id": self.hypothesis_id,
            "model_id": self.model_id,
            "seed": self.seed,
            "filename": self.filename,
        }

    def review_record(self) -> dict[str, Any]:
        return {
            "blind_id": self.blind_id,
            "hypothesis_id": self.hypothesis_id,
            "seed_pair": self.seed,
            "filename": self.filename,
            "scores": {
                "anatomy": None,
                "lighting_reflection": None,
                "lens_dof_motion": None,
                "skin_fabric_water": None,
                "sensor_color_compression": None,
                "background": None,
                "candid_composition": None,
            },
            "critical_defect": None,
            "notes": "",
        }


def load_manifest(path: Path = DEFAULT_MANIFEST) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    validate_manifest(data)
    return data


def validate_manifest(data: dict[str, Any]) -> None:
    models = data.get("models")
    hypotheses = data.get("camera_hypotheses")
    if data.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        raise ValueError(
            f"Phase 0 manifest schema must be {MANIFEST_SCHEMA_VERSION}; "
            f"got {data.get('schema_version')!r}"
        )
    if isinstance(models, list) and LEGACY_MODEL_IDS.intersection(models):
        legacy = sorted(LEGACY_MODEL_IDS.intersection(models))
        raise ValueError(
            f"legacy HiDream model identifiers are incompatible with this bake-off: {legacy}; "
            f"required models are {list(PHASE0_MODEL_IDS)}"
        )
    if models != list(PHASE0_MODEL_IDS):
        raise ValueError(
            "Phase 0 requires the exact ordered model matrix "
            f"{list(PHASE0_MODEL_IDS)}; got {models!r}"
        )
    if not isinstance(hypotheses, list) or len(hypotheses) != 6:
        raise ValueError("Phase 0 requires exactly six camera hypotheses")
    if data.get("output", {}).get("planned_slots") != 12:
        raise ValueError("manifest must declare exactly 12 planned slots")
    identity = data.get("identity_policy", {})
    if identity.get("identity_conditioning_allowed") is not False:
        raise ValueError("Phase 0 identity conditioning must be disabled")

    generic_subject = str(data.get("generic_subject", "")).strip()
    if not generic_subject:
        raise ValueError("generic_subject is mandatory")
    ids: set[str] = set()
    for item in hypotheses:
        missing = REQUIRED_CAMERA_FIELDS.difference(item)
        if missing:
            raise ValueError(f"{item.get('id', '<unknown>')} missing camera fields: {sorted(missing)}")
        hypothesis_id = str(item.get("id", ""))
        if not hypothesis_id or hypothesis_id in ids:
            raise ValueError("camera hypothesis IDs must be unique and non-empty")
        ids.add(hypothesis_id)
        prompt = str(item["prompt"])
        if generic_subject not in prompt:
            raise ValueError(f"{hypothesis_id} does not use the shared generic subject verbatim")
        lowered = prompt.lower()
        if any(marker in lowered for marker in IDENTITY_MARKERS):
            raise ValueError(f"{hypothesis_id} contains an identity marker")
        bbox = item["subject_bbox_xywh_normalized"]
        if not isinstance(bbox, list) or len(bbox) != 4 or any(not 0 <= float(v) <= 1 for v in bbox):
            raise ValueError(f"{hypothesis_id} has invalid normalized bbox")
        if not isinstance(item["abstracted_from"], list) or not item["abstracted_from"]:
            raise ValueError(f"{hypothesis_id} must document abstract reference provenance")


def build_plan(data: dict[str, Any]) -> list[PlannedSlot]:
    validate_manifest(data)
    models = list(data["models"])
    base_seed = int(data["seed_policy"]["base_seed"])
    rng = random.Random(int(data["seed_policy"]["mapping_randomization_seed"]))
    slots: list[PlannedSlot] = []
    for index, hypothesis in enumerate(data["camera_hypotheses"], start=1):
        assignment = models[:]
        rng.shuffle(assignment)
        seed = base_seed + index
        for arm, model_id in zip(("A", "B"), assignment, strict=True):
            blind_id = f"{hypothesis['id']}_{arm}"
            slots.append(
                PlannedSlot(
                    blind_id=blind_id,
                    hypothesis_id=hypothesis["id"],
                    model_id=model_id,
                    seed=seed,
                    filename=f"{blind_id}.png",
                    prompt=hypothesis["prompt"],
                )
            )
    if len(slots) != 12:
        raise AssertionError("planner invariant violated: expected exactly 12 slots")
    return slots


def write_blind_packet(
    run_directory: Path,
    data: dict[str, Any],
    *,
    create_run_directory: bool = True,
) -> list[PlannedSlot]:
    """Write reviewer-visible data separately from the private randomized key."""
    slots = build_plan(data)
    if create_run_directory:
        run_directory.mkdir(parents=True, exist_ok=False)
    elif not run_directory.is_dir():
        raise FileNotFoundError(f"pre-created run directory is missing: {run_directory}")
    private_directory = run_directory / "private"
    private_directory.mkdir(exist_ok=True)
    review_sheet = {
        "schema_version": "1.0.0",
        "experiment_id": data["experiment_id"],
        "model_labels_hidden": True,
        "rubric": "rubric.json",
        "items": [slot.review_record() for slot in slots],
    }
    mapping_key = {
        "schema_version": "1.0.0",
        "experiment_id": data["experiment_id"],
        "private": True,
        "mapping_randomization_seed": data["seed_policy"]["mapping_randomization_seed"],
        "items": [slot.private_record() for slot in slots],
    }
    (run_directory / "review_sheet.json").write_text(
        json.dumps(review_sheet, indent=2) + "\n", encoding="utf-8"
    )
    (run_directory / "manifest.json").write_text(
        json.dumps(data, indent=2) + "\n", encoding="utf-8"
    )
    (run_directory / "rubric.json").write_text(
        DEFAULT_RUBRIC.read_text(encoding="utf-8"), encoding="utf-8"
    )
    (private_directory / "mapping_key.json").write_text(
        json.dumps(mapping_key, indent=2) + "\n", encoding="utf-8"
    )
    return slots
