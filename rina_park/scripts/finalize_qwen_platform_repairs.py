#!/usr/bin/env python3
"""Record strict visual review and build the provisional platform contact sheet."""

from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps

SOURCE_RUN = Path(__file__).resolve().parents[1] / "out/review_samples/20260726T073925Z"
IG = SOURCE_RUN / "selected/01_ig_candid_laugh.png"
PATREON_B = SOURCE_RUN / "selected/05_patreon_b_wind_soft_smile.png"
REVIEWS: dict[str, dict[str, object]] = {
    "01_patreon_a_opaque_cream_robe": {
        "selected": True,
        "identity_consistent": True,
        "eyes_gaze_preserved": True,
        "hands_natural": True,
        "robe_fully_opaque": True,
        "subject_fraction_estimate": 0.53,
        "text_present": False,
        "scene_preserved": True,
        "unaffected_content_preserved": True,
        "caveats": (
            "The edit lightly re-rendered facial and fabric microtexture, but identity, downward gaze, hands, blank "
            "notebook, props, background, and framing remain consistent. Cream robe reads solid and opaque."
        ),
        "verdict": "selected provisionally: only repair satisfying opacity while preserving the passed A composition",
    },
    "02_patreon_a_opaque_charcoal_robe": {
        "selected": False,
        "identity_consistent": True,
        "eyes_gaze_preserved": True,
        "hands_natural": True,
        "robe_fully_opaque": False,
        "subject_fraction_estimate": 0.55,
        "text_present": False,
        "scene_preserved": True,
        "unaffected_content_preserved": False,
        "caveats": (
            "Charcoal fabric remains visibly translucent at the sleeves and upper torso, and the robe coverage changed "
            "more than the cream edit."
        ),
        "verdict": "rejected: failed the fully opaque wardrobe requirement",
    },
    "03_patreon_c_zoomout_hidden_hand": {
        "selected": False,
        "identity_consistent": True,
        "eyes_gaze_preserved": True,
        "hands_natural": False,
        "robe_fully_opaque": False,
        "subject_fraction_estimate": 0.57,
        "text_present": False,
        "scene_preserved": True,
        "unaffected_content_preserved": False,
        "caveats": (
            "The corrupted lowered hand remains exposed with smeared finger/skin texture. The subject is still too large "
            "and the cover-up reads sheer instead of opaque."
        ),
        "verdict": "rejected: hand repair, opacity, and 35-45% framing all failed",
    },
    "04_patreon_c_zoomout_towel_hand": {
        "selected": False,
        "identity_consistent": True,
        "eyes_gaze_preserved": True,
        "hands_natural": False,
        "robe_fully_opaque": False,
        "subject_fraction_estimate": 0.59,
        "text_present": False,
        "scene_preserved": True,
        "unaffected_content_preserved": False,
        "caveats": (
            "A plausible towel was added to the opposite hand, but the original corrupted lowered hand remains. The "
            "subject was not zoomed out to 35-45%, and the shoulder drape remains sheer."
        ),
        "verdict": "rejected: towel did not repair the bad hand and framing remained too close",
    },
}


def make_contact_sheet(items: list[tuple[str, Path]], output: Path) -> None:
    thumb_w, thumb_h, label_h, columns = 264, 336, 62, 3
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
        raise RuntimeError(f"unexpected repair candidates: {sorted(slugs)}")
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
        payload["repair_status"] = (
            "provisional_candidate_pending_user_approval" if selected else "rejected_after_strict_visual_review"
        )
        payload["production_status"] = "not_approved_not_published" if selected else "rejected_not_promoted"
        sidecar.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        payloads[slug] = payload
        if selected:
            image = run / "candidates" / f"{slug}.png"
            selected_image = selected_dir / image.name
            selected_sidecar = selected_dir / sidecar.name
            shutil.copy2(image, selected_image)
            shutil.copy2(sidecar, selected_sidecar)
            selected_paths.append(str(selected_image))

    items = [
        ("IG v4 / unchanged\nPENDING USER APPROVAL", IG),
        ("Patreon B v4 / unchanged\nPENDING USER APPROVAL", PATREON_B),
    ]
    items.extend(
        (
            f"Patreon A repair / {Path(path).stem}\nPENDING USER APPROVAL",
            Path(path),
        )
        for path in selected_paths
    )
    final_sheet = run / "contact_sheet_final_provisional_pending_user_approval.jpg"
    make_contact_sheet(items, final_sheet)
    summary_path = run / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["review_status"] = "strict_visual_review_complete_pending_user_approval"
    summary["selected_count"] = len(selected_paths)
    summary["selected_paths"] = selected_paths
    summary["track_verdicts"] = {
        "patreon_a": "selected cream opaque robe repair provisionally; charcoal repair rejected for transparency",
        "patreon_c": "none selected; both retained corrupted hand and overly close framing",
    }
    summary["final_provisional_candidates"] = [str(IG), str(PATREON_B), *selected_paths]
    summary["contact_sheets"]["final_provisional"] = str(final_sheet)
    summary["results"] = [payloads[str(item["slug"])] for item in summary["results"]]
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    finalize(args.run.resolve())


if __name__ == "__main__":
    main()
