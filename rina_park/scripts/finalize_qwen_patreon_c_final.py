#!/usr/bin/env python3
"""Record strict C-final review and build the four-track provisional sheet."""

from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps

RINA = Path(__file__).resolve().parents[1]
IG = RINA / "out/review_samples/20260726T073925Z/selected/01_ig_candid_laugh.png"
PATREON_B = RINA / "out/review_samples/20260726T073925Z/selected/05_patreon_b_wind_soft_smile.png"
PATREON_A = RINA / "out/review_samples/repair_20260726T083329Z/selected/01_patreon_a_opaque_cream_robe.png"
REVIEWS: dict[str, dict[str, object]] = {
    "01_c_final_serious_rail_occlusion": {
        "selected": True,
        "identity_consistent": True,
        "source_expression_gaze_preserved": True,
        "eyes_natural": True,
        "night_scene_pass": True,
        "coverup_fully_opaque": True,
        "both_hands_fully_hidden": True,
        "subject_fraction_estimate": 0.43,
        "foreground_occlusion_present": True,
        "narrative_pass": True,
        "text_present": False,
        "caveats": (
            "The centered rail creates a deliberately obstructed composition and the opaque wrap reads more like a "
            "heavy towel than a tailored cover-up. The guarded expression, identity, night story, hand concealment, "
            "and environmental pool context remain credible."
        ),
        "verdict": "selected provisionally: all critical C-final gates pass with both hands fully concealed",
    },
    "02_c_final_over_shoulder_doorframe": {
        "selected": False,
        "identity_consistent": True,
        "source_expression_gaze_preserved": False,
        "eyes_natural": True,
        "night_scene_pass": True,
        "coverup_fully_opaque": False,
        "both_hands_fully_hidden": True,
        "subject_fraction_estimate": 0.52,
        "foreground_occlusion_present": True,
        "narrative_pass": True,
        "text_present": False,
        "caveats": (
            "Hands are safely hidden and the doorway/night narrative works, but the wrap is visibly transparent over "
            "the hip, subject scale exceeds 45%, and the source's alert over-shoulder gaze softened toward camera."
        ),
        "verdict": "rejected: opacity, subject fraction, and source-gaze preservation failed",
    },
}


def make_contact_sheet(items: list[tuple[str, Path]], output: Path) -> None:
    thumb_w, thumb_h, label_h, columns = 264, 336, 62, 4
    rows = math.ceil(len(items) / columns)
    sheet = Image.new("RGB", (columns * thumb_w, rows * (thumb_h + label_h)), "white")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    for index, (label, path) in enumerate(items):
        with Image.open(path) as source:
            image = ImageOps.fit(source.convert("RGB"), (thumb_w, thumb_h), method=Image.Resampling.LANCZOS)
        x = index % columns * thumb_w
        y = index // columns * (thumb_h + label_h)
        sheet.paste(image, (x, y))
        draw.multiline_text((x + 6, y + thumb_h + 5), label, fill="black", font=font, spacing=2)
    sheet.save(output, quality=92, subsampling=0)


def finalize(run: Path) -> None:
    sidecars = sorted((run / "candidates").glob("*.png.json"))
    slugs = {path.name.removesuffix(".png.json") for path in sidecars}
    if slugs != set(REVIEWS):
        raise RuntimeError(f"unexpected C-final candidates: {sorted(slugs)}")
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
        payload["review_status"] = (
            "provisional_candidate_pending_user_approval" if selected else "rejected_after_strict_visual_review"
        )
        payload["production_status"] = "not_approved_not_published" if selected else "rejected_not_promoted"
        sidecar.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        payloads[slug] = payload
        if selected:
            image = run / "candidates" / f"{slug}.png"
            selected_image = selected_dir / image.name
            shutil.copy2(image, selected_image)
            shutil.copy2(sidecar, selected_dir / sidecar.name)
            selected_paths.append(str(selected_image))

    if len(selected_paths) != 1:
        raise RuntimeError(f"expected one provisional C candidate, got {len(selected_paths)}")
    final_sheet = run / "contact_sheet_final_provisional_4_track_pending_user_approval.jpg"
    make_contact_sheet(
        [
            ("Instagram / unchanged\nPENDING USER APPROVAL", IG),
            ("Patreon A / repaired\nPENDING USER APPROVAL", PATREON_A),
            ("Patreon B / unchanged\nPENDING USER APPROVAL", PATREON_B),
            ("Patreon C / final clean-ref\nPENDING USER APPROVAL", Path(selected_paths[0])),
        ],
        final_sheet,
    )
    summary_path = run / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["review_status"] = "strict_visual_review_complete_pending_user_approval"
    summary["selected_count"] = len(selected_paths)
    summary["selected_paths"] = selected_paths
    summary["track_verdict"] = (
        "selected 01 provisionally; 02 rejected for transparent wrap, over-45% subject scale, and gaze drift"
    )
    summary["final_provisional_candidates"] = [str(IG), str(PATREON_A), str(PATREON_B), *selected_paths]
    summary["final_provisional_contact_sheet"] = str(final_sheet)
    summary["results"] = [payloads[str(item["slug"])] for item in summary["results"]]
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    finalize(args.run.resolve())


if __name__ == "__main__":
    main()
