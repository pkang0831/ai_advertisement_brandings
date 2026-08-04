#!/usr/bin/env python3
"""Generate eight sequential, review-only platform v4 candidates with Qwen Edit."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path

import mlx.core as mx
import numpy as np
from generate_qwen_image_edit_smoke import (
    MODEL,
    MODEL_REPO,
    MODEL_REVISION,
    MODEL_TREE_SHA256,
    memory_snapshot,
    validate_readiness,
)
from mflux.models.common.config import ModelConfig
from mflux.models.qwen.variants.edit.qwen_image_edit import QwenImageEdit
from mflux.utils.dimension_resolver import CANVAS_POLICY_EXACT_RESIZE
from PIL import Image, ImageDraw, ImageFont, ImageOps

ROOT = Path(__file__).resolve().parents[2]
RINA = ROOT / "rina_park"
REFERENCE_PACK = RINA / "identity/qwen_image_edit_reference_pack/20260726T065746Z"
OUTPUT_ROOT = RINA / "out/review_samples"
WIDTH, HEIGHT = 704, 896
STEPS, GUIDANCE = 20, 4.0
BODY = (
    "She has a slender athletic swimmer build, narrow defined waist, and long balanced natural proportions; "
    "no stocky or heavy build and no exaggerated anatomy. Preserve realistic hands, skin texture, and fabric."
)
IDENTITY = (
    "Preserve the exact adult Rina facial identity, eyes, nose, lips, jaw, hairline, expression, and gaze from "
    "the reference image. Natural documentary photography, no AI gloss, no text, logo, watermark, or writing."
)
SPECS = (
    {
        "slug": "01_ig_candid_laugh",
        "track": "instagram",
        "reference": "10_candid_mid_laugh",
        "seed": 26072801,
        "prompt": (
            f"{IDENTITY} Expand to an environmental medium-wide full-body candid at a natural public indoor pool. "
            "She is walking beside the pool after a swim, genuinely laughing off-camera, holding swim goggles and "
            "a folded towel loosely in one hand. Casual dark teal athletic one-piece with an opaque open linen cover-up. "
            "Subject occupies about 40% of the portrait frame; visible pool, deck, windows, and everyday swimmers softly "
            f"in the distance. Unposed off-camera moment. {BODY}"
        ),
    },
    {
        "slug": "02_ig_over_shoulder",
        "track": "instagram",
        "reference": "09_over_shoulder_alert",
        "seed": 26072802,
        "prompt": (
            f"{IDENTITY} Expand to an environmental medium-wide three-quarter-body candid at a natural public indoor "
            "pool. She glances back over her right shoulder while walking toward a bench, goggles dangling from one "
            "hand and towel tucked under the other arm. Dark navy athletic one-piece under an opaque lightweight "
            "cover-up. Subject occupies 45% of frame; pool lanes, tiled deck, and public architecture visible. "
            f"Authentic unstaged moment, not a catalog pose. {BODY}"
        ),
    },
    {
        "slug": "03_patreon_a_notes_down",
        "track": "patreon_a",
        "reference": "07_looking_down_thoughtful",
        "seed": 26072803,
        "prompt": (
            f"{IDENTITY} Private quiet poolside notes story, environmental medium-wide seated portrait. She wears an "
            "elegant opaque charcoal robe over swimwear and looks thoughtfully down while opening a completely blank "
            "notebook on her lap, pen resting unused beside it; absolutely no writing or marks. Folded towel and calm "
            "water nearby. Subject occupies 45% of frame, candid documentary composition rather than a catalog pose. "
            f"{BODY}"
        ),
    },
    {
        "slug": "04_patreon_a_towel_soft_smile",
        "track": "patreon_a",
        "reference": "03_three_quarter_soft_smile",
        "seed": 26072804,
        "prompt": (
            f"{IDENTITY} Private poolside notes story, environmental medium-wide three-quarter-body candid. Wearing "
            "an elegant opaque deep-green robe, she softly smiles past camera while folding a plain towel beside a "
            "closed blank notebook with no writing or text. Quiet daylight pool terrace, intimate reflective mood, "
            f"subject occupies 40% of frame, hands naturally engaged, not posing for a catalog. {BODY}"
        ),
    },
    {
        "slug": "05_patreon_b_wind_soft_smile",
        "track": "patreon_b",
        "reference": "03_three_quarter_soft_smile",
        "seed": 26072805,
        "prompt": (
            f"{IDENTITY} Confident medium-wide pool editorial clearly distinct from a quiet notes scene. She stands "
            "mid-stride along an outdoor pool edge in a tasteful dark teal athletic one-piece as an opaque white linen "
            "cover-up moves in the breeze. Natural wind lifts strands of her long hair; soft smile and gaze remain from "
            "reference. Subject occupies 45% of frame with long full-body proportions and strong environmental space, "
            f"dynamic candid movement, not a catalog pose. {BODY}"
        ),
    },
    {
        "slug": "06_patreon_b_profile_stride",
        "track": "patreon_b",
        "reference": "06_true_right_profile_neutral",
        "seed": 26072806,
        "prompt": (
            f"{IDENTITY} Confident medium-wide profile pool editorial. Full-body right-profile stride beside a modern "
            "pool, wearing a tasteful dark navy athletic one-piece and an opaque teal cover-up flowing behind her in "
            "the wind. Long hair moving naturally, profile gaze forward. Subject occupies 40% of frame with wide "
            f"architectural context, authentic motion and natural hands, not a posed catalog image. {BODY}"
        ),
    },
    {
        "slug": "07_patreon_c_night_serious",
        "track": "patreon_c",
        "reference": "04_three_quarter_serious",
        "seed": 26072807,
        "prompt": (
            f"{IDENTITY} Cinematic NIGHT pool scene, unmistakably after dark. Environmental medium-wide three-quarter "
            "body portrait with wet hair, serious off-camera gaze, dark athletic one-piece, and an opaque draped black "
            "cover-up around her shoulders. Blue water reflections, deep shadows, practical amber lights, damp stone. "
            "Subject occupies 45% of frame; sensual SFW unresolved story, no lingerie, nudity, or explicit pose. "
            f"Natural film grain, not glossy. {BODY}"
        ),
    },
    {
        "slug": "08_patreon_c_night_over_shoulder",
        "track": "patreon_c",
        "reference": "09_over_shoulder_alert",
        "seed": 26072808,
        "prompt": (
            f"{IDENTITY} Cinematic NIGHT pool scene, unmistakably after dark. Environmental medium-wide over-shoulder "
            "candid as she pauses near reflected blue water with damp hair, wearing a dark teal athletic one-piece and "
            "an opaque charcoal cover-up draped securely around her body. Moody reflections, shadow, rim light, distant "
            "practical lamps. Subject occupies 40% of frame; sensual SFW unresolved story, no lingerie, nudity, or "
            f"explicit pose. Natural film texture, not glossy. {BODY}"
        ),
    },
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def image_qc(image: Image.Image) -> dict[str, object]:
    pixels = np.asarray(image.convert("RGB"))
    black = float((pixels < 3).all(axis=2).mean())
    white = float((pixels > 252).all(axis=2).mean())
    return {
        "dimensions": list(image.size),
        "mean_rgb": round(float(pixels.mean()), 3),
        "std_rgb": round(float(pixels.std()), 3),
        "black_pixel_fraction": round(black, 6),
        "white_pixel_fraction": round(white, 6),
        "passed": image.size == (WIDTH, HEIGHT) and float(pixels.std()) > 8 and black < 0.9 and white < 0.9,
    }


def make_contact_sheet(items: list[tuple[str, Path]], output: Path) -> None:
    thumb_w, thumb_h, label_h, columns = 264, 336, 54, 4
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


def generate(run_id: str) -> Path:
    readiness = validate_readiness()
    reference_summary = json.loads((REFERENCE_PACK / "summary.json").read_text(encoding="utf-8"))
    approved = set(reference_summary["approved_internal_reference_labels"])
    required = {str(spec["reference"]) for spec in SPECS}
    if not required <= approved:
        raise RuntimeError(f"unapproved reference requested: {sorted(required - approved)}")
    if reference_summary.get("master_replaced") is not False:
        raise RuntimeError("reference pack must not replace the master")

    run = OUTPUT_ROOT / run_id
    candidates = run / "candidates"
    candidates.mkdir(parents=True, exist_ok=False)
    model_config = ModelConfig.from_name(MODEL_REPO)
    model = QwenImageEdit(model_config=model_config, model_path=str(MODEL))
    mx.synchronize()
    results: list[dict[str, object]] = []
    contact_items: list[tuple[str, Path]] = []
    run_started = time.monotonic()
    try:
        for spec in SPECS:
            reference = REFERENCE_PACK / "candidates" / f"{spec['reference']}.png"
            mx.clear_cache()
            gc.collect()
            mx.reset_peak_memory()
            output = candidates / f"{spec['slug']}.png"
            started = time.monotonic()
            generated = model.generate_image(
                seed=int(spec["seed"]),
                prompt=str(spec["prompt"]),
                image_path=str(reference),
                image_paths=[str(reference)],
                width=WIDTH,
                height=HEIGHT,
                guidance=GUIDANCE,
                num_inference_steps=STEPS,
                scheduler="flow_match_euler_discrete",
                canvas_policy=CANVAS_POLICY_EXACT_RESIZE,
            )
            generated.save(path=output, overwrite=True)
            mx.synchronize()
            seconds = round(time.monotonic() - started, 3)
            with Image.open(output) as source:
                qc = image_qc(source)
            if not qc["passed"]:
                raise RuntimeError(f"image sanity failed for {spec['slug']}: {qc}")
            payload = {
                "schema_version": 1,
                "review_status": "candidate_not_approved",
                "production_status": "not_promoted",
                "track": spec["track"],
                "slug": spec["slug"],
                "source_reference_label": spec["reference"],
                "source_reference_path": str(reference),
                "source_reference_sha256": sha256(reference),
                "model_repo": MODEL_REPO,
                "model_revision": MODEL_REVISION,
                "model_tree_sha256": MODEL_TREE_SHA256,
                "runtime": readiness["runtime"],
                "mlx": readiness["mlx"],
                "seed": spec["seed"],
                "steps": STEPS,
                "guidance": GUIDANCE,
                "prompt": spec["prompt"],
                "dimensions": [WIDTH, HEIGHT],
                "generation_seconds": seconds,
                "memory": memory_snapshot(),
                "qc": qc,
                "manual_qc": {
                    "identity_consistent": None,
                    "reference_expression_gaze_preserved": None,
                    "eyes_natural": None,
                    "hands_anatomy_natural": None,
                    "body_direction_matches": None,
                    "subject_fraction_estimate": None,
                    "scene_track_matches": None,
                    "catalog_pose": None,
                    "text_present": None,
                    "ai_gloss": None,
                    "caveats": None,
                    "verdict": None,
                },
            }
            output.with_suffix(".png.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
            results.append(payload)
            contact_items.append((f"{spec['track']} / {spec['reference']}\n{spec['slug']}", output))
            print(json.dumps({"saved": str(output), "seconds": seconds, "memory": payload["memory"]}), flush=True)
            mx.clear_cache()
            gc.collect()
    finally:
        del model
        mx.clear_cache()
        gc.collect()

    contact_sheet = run / "contact_sheet_platform_candidates_only.jpg"
    make_contact_sheet(contact_items, contact_sheet)
    summary = {
        "run_id": run_id,
        "purpose": "Qwen Image Edit platform review samples v4",
        "candidate_count": len(results),
        "selected_count": 0,
        "selected_paths": [],
        "platform_only_contact_sheet": str(contact_sheet),
        "reference_pack": str(REFERENCE_PACK),
        "master_replaced": False,
        "prior_ip_adapter_used": False,
        "mature_content_generated": False,
        "published": False,
        "packaged": False,
        "review_status": "pending_manual_visual_review",
        "total_generation_seconds": round(time.monotonic() - run_started, 3),
        "peak_mlx_memory_gib": max(float(item["memory"]["mlx_peak_gib"]) for item in results),
        "readiness": readiness,
        "results": results,
    }
    (run / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return run


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
    args = parser.parse_args()
    print(json.dumps({"run": str(generate(args.run_id))}))


if __name__ == "__main__":
    main()
