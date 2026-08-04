#!/usr/bin/env python3
"""Apply strict visual review decisions to the eight Qwen platform v4 candidates."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

REVIEWS: dict[str, dict[str, object]] = {
    "01_ig_candid_laugh": {
        "selected": True,
        "identity_consistent": True,
        "reference_expression_gaze_preserved": True,
        "eyes_natural": True,
        "hands_anatomy_natural": True,
        "body_direction_matches": True,
        "subject_fraction_estimate": 0.46,
        "scene_track_matches": True,
        "catalog_pose": False,
        "text_present": False,
        "ai_gloss": False,
        "caveats": "Lower legs meet the frame edge, but framing remains environmental medium-wide and candid.",
        "verdict": "selected: strongest genuine off-camera moment with stable Rina identity and public-pool context",
    },
    "02_ig_over_shoulder": {
        "selected": False,
        "identity_consistent": True,
        "reference_expression_gaze_preserved": True,
        "eyes_natural": False,
        "hands_anatomy_natural": True,
        "body_direction_matches": True,
        "subject_fraction_estimate": 0.49,
        "scene_track_matches": True,
        "catalog_pose": True,
        "text_present": False,
        "ai_gloss": False,
        "caveats": "Eyes read overly wide/direct and the stopped over-shoulder stance feels arranged rather than candid.",
        "verdict": "rejected: uncanny gaze and catalog-like pose",
    },
    "03_patreon_a_notes_down": {
        "selected": False,
        "identity_consistent": True,
        "reference_expression_gaze_preserved": True,
        "eyes_natural": True,
        "hands_anatomy_natural": True,
        "body_direction_matches": True,
        "subject_fraction_estimate": 0.53,
        "scene_track_matches": False,
        "catalog_pose": False,
        "text_present": False,
        "ai_gloss": False,
        "caveats": "Notebook is correctly blank, but the robe is sheer rather than the required elegant opaque cover-up.",
        "verdict": "rejected: wardrobe violates opaque robe requirement",
    },
    "04_patreon_a_towel_soft_smile": {
        "selected": False,
        "identity_consistent": False,
        "reference_expression_gaze_preserved": False,
        "eyes_natural": True,
        "hands_anatomy_natural": True,
        "body_direction_matches": True,
        "subject_fraction_estimate": 0.59,
        "scene_track_matches": True,
        "catalog_pose": True,
        "text_present": False,
        "ai_gloss": True,
        "caveats": "Direct beauty-camera smile, softened facial structure, close subject scale, and polished robe-catalog styling.",
        "verdict": "rejected: identity drift, catalog pose, and subject over 55%",
    },
    "05_patreon_b_wind_soft_smile": {
        "selected": True,
        "identity_consistent": True,
        "reference_expression_gaze_preserved": True,
        "eyes_natural": True,
        "hands_anatomy_natural": True,
        "body_direction_matches": True,
        "subject_fraction_estimate": 0.44,
        "scene_track_matches": True,
        "catalog_pose": False,
        "text_present": False,
        "ai_gloss": False,
        "caveats": "Editorial polish is present, but wind, stride, full-body proportions, and environmental space remain credible.",
        "verdict": "selected: strongest distinct B concept with natural motion and stable profile cues",
    },
    "06_patreon_b_profile_stride": {
        "selected": False,
        "identity_consistent": True,
        "reference_expression_gaze_preserved": True,
        "eyes_natural": True,
        "hands_anatomy_natural": False,
        "body_direction_matches": True,
        "subject_fraction_estimate": 0.58,
        "scene_track_matches": True,
        "catalog_pose": True,
        "text_present": False,
        "ai_gloss": True,
        "caveats": "Not a clean true profile, lower body is cropped, forward hand is stiff, and framing is too tight.",
        "verdict": "rejected: tight catalog framing and hand/profile weaknesses",
    },
    "07_patreon_c_night_serious": {
        "selected": False,
        "identity_consistent": True,
        "reference_expression_gaze_preserved": True,
        "eyes_natural": True,
        "hands_anatomy_natural": False,
        "body_direction_matches": True,
        "subject_fraction_estimate": 0.59,
        "scene_track_matches": True,
        "catalog_pose": False,
        "text_present": False,
        "ai_gloss": False,
        "caveats": "Night concept is strong, but the lowered hand has corrupted finger/skin detail and subject exceeds 55%.",
        "verdict": "rejected: bad hand anatomy and overly close framing",
    },
    "08_patreon_c_night_over_shoulder": {
        "selected": False,
        "identity_consistent": True,
        "reference_expression_gaze_preserved": False,
        "eyes_natural": True,
        "hands_anatomy_natural": True,
        "body_direction_matches": True,
        "subject_fraction_estimate": 0.60,
        "scene_track_matches": True,
        "catalog_pose": True,
        "text_present": False,
        "ai_gloss": False,
        "caveats": "Night lighting succeeds, but pose is a static three-quarter stance rather than over-shoulder and is too close.",
        "verdict": "rejected: wrong pose and subject over 55%",
    },
}

TRACK_VERDICTS = {
    "instagram": "selected 01; 02 rejected for uncanny wide gaze and arranged catalog stance",
    "patreon_a": "none selected; 03 violates opaque robe requirement and 04 drifts/catalog-poses",
    "patreon_b": "selected 05; 06 rejected for tight framing, weak profile, and stiff hand",
    "patreon_c": "none selected; 07 has bad hand anatomy and 08 misses over-shoulder framing",
}


def finalize(run: Path) -> None:
    sidecars = sorted((run / "candidates").glob("*.png.json"))
    slugs = {path.name.removesuffix(".png.json") for path in sidecars}
    if slugs != set(REVIEWS):
        raise RuntimeError(f"unexpected v4 candidates: {sorted(slugs)}")
    selected_dir = run / "selected"
    selected_dir.mkdir(exist_ok=False)
    payloads: dict[str, dict[str, object]] = {}
    selected_paths: list[str] = []

    for sidecar in sidecars:
        payload = json.loads(sidecar.read_text(encoding="utf-8"))
        slug = str(payload["slug"])
        review = dict(REVIEWS[slug])
        selected = bool(review.pop("selected"))
        payload["manual_qc"] = review
        payload["review_status"] = "selected_for_v4_review" if selected else "rejected_after_visual_review"
        payload["production_status"] = "selected_not_published" if selected else "rejected_not_promoted"
        sidecar.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        payloads[slug] = payload
        if selected:
            image = run / "candidates" / f"{slug}.png"
            selected_image = selected_dir / image.name
            selected_sidecar = selected_dir / sidecar.name
            shutil.copy2(image, selected_image)
            shutil.copy2(sidecar, selected_sidecar)
            selected_paths.append(str(selected_image))

    summary_path = run / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["review_status"] = "strict_visual_review_complete"
    summary["selected_count"] = len(selected_paths)
    summary["selected_paths"] = selected_paths
    summary["track_verdicts"] = TRACK_VERDICTS
    summary["results"] = [payloads[str(item["slug"])] for item in summary["results"]]
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    finalize(args.run.resolve())


if __name__ == "__main__":
    main()
