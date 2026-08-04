#!/usr/bin/env python3
"""Generate exactly two final Patreon C candidates from clean identity references."""

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
REFERENCE_PACK = RINA / "identity/qwen_image_edit_reference_pack/20260726T065746Z/candidates"
OUTPUT_ROOT = RINA / "out/review_samples"
PROTECTED = (
    RINA / "out/review_samples/20260726T073925Z/selected/01_ig_candid_laugh.png",
    RINA / "out/review_samples/20260726T073925Z/selected/05_patreon_b_wind_soft_smile.png",
    RINA / "out/review_samples/repair_20260726T083329Z/selected/01_patreon_a_opaque_cream_robe.png",
)
WIDTH, HEIGHT = 704, 896
STEPS, GUIDANCE = 20, 4.0
SPECS = (
    {
        "slug": "01_c_final_serious_rail_occlusion",
        "reference_label": "04_three_quarter_serious",
        "seed": 26073001,
        "prompt": (
            "Environmental medium-wide NIGHT pool portrait, subject 35% of frame. Put a broad dark foreground pool rail "
            "across her waist and lower torso; both hands and forearms are completely hidden behind the rail and an opaque "
            "wrapped charcoal towel, with no hands visible. Preserve the exact adult Rina face and serious off-camera gaze "
            "from the reference. Wet long black hair, dark athletic one-piece, fully opaque navy draped cover-up. Blue water "
            "reflections, deep darkness, distant amber practical lights, damp stone. Guarded unresolved sensual SFW moment, "
            "natural documentary texture, slender athletic proportions. No daylight, lingerie, nudity, text, catalog pose, "
            "visible hands, transparent fabric, or glossy AI finish."
        ),
    },
    {
        "slug": "02_c_final_over_shoulder_doorframe",
        "reference_label": "09_over_shoulder_alert",
        "seed": 26073002,
        "prompt": (
            "Environmental medium-wide NIGHT pool candid, subject 32% of frame, looking back over her shoulder with the "
            "exact guarded sideways gaze and adult Rina identity from the reference. Frame her through a partly open dark "
            "pool-house doorway: the foreground door frame partially occludes her lower body, while both arms and hands are "
            "fully hidden behind her body and an opaque wrapped charcoal cover-up; no hands visible. Wet long black hair, "
            "dark athletic one-piece, blue water reflections, distant amber practical lights, deep shadows and damp stone. "
            "Unresolved sensual SFW story, natural film texture, slender athletic proportions. No daylight, lingerie, "
            "nudity, text, catalog pose, visible hands, transparent fabric, or glossy AI finish."
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
    thumb_w, thumb_h, label_h, columns = 352, 448, 58, 2
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
        draw.multiline_text((x + 8, y + thumb_h + 7), label, fill="black", font=font, spacing=2)
    sheet.save(output, quality=92, subsampling=0)


def generate(run_id: str) -> Path:
    readiness = validate_readiness()
    protected_hashes = {str(path): sha256(path) for path in PROTECTED}
    run = OUTPUT_ROOT / f"c_final_{run_id}"
    candidates = run / "candidates"
    candidates.mkdir(parents=True, exist_ok=False)
    model = QwenImageEdit(model_config=ModelConfig.from_name(MODEL_REPO), model_path=str(MODEL))
    mx.synchronize()
    results: list[dict[str, object]] = []
    items: list[tuple[str, Path]] = []
    started_run = time.monotonic()
    try:
        for spec in SPECS:
            reference = REFERENCE_PACK / f"{spec['reference_label']}.png"
            if not reference.is_file():
                raise RuntimeError(f"clean approved reference missing: {reference}")
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
            with Image.open(output) as image:
                qc = image_qc(image)
            if not qc["passed"]:
                raise RuntimeError(f"image sanity failed for {spec['slug']}: {qc}")
            payload = {
                "schema_version": 1,
                "review_status": "candidate_pending_strict_visual_review",
                "production_status": "not_promoted",
                "track": "patreon_c",
                "slug": spec["slug"],
                "source_reference_label": spec["reference_label"],
                "source_reference_path": str(reference),
                "source_reference_sha256": sha256(reference),
                "corrupted_prior_c_used_as_source": False,
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
                    "source_expression_gaze_preserved": None,
                    "eyes_natural": None,
                    "night_scene_pass": None,
                    "coverup_fully_opaque": None,
                    "both_hands_fully_hidden": None,
                    "subject_fraction_estimate": None,
                    "foreground_occlusion_present": None,
                    "narrative_pass": None,
                    "text_present": None,
                    "caveats": None,
                    "verdict": None,
                },
            }
            output.with_suffix(".png.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
            results.append(payload)
            items.append((f"{spec['reference_label']}\n{spec['slug']}", output))
            print(json.dumps({"saved": str(output), "seconds": seconds, "memory": payload["memory"]}), flush=True)
    finally:
        del model
        mx.clear_cache()
        gc.collect()

    for path, expected in protected_hashes.items():
        if sha256(Path(path)) != expected:
            raise RuntimeError(f"protected provisional candidate changed: {path}")
    contact_sheet = run / "contact_sheet_c_final_candidates.jpg"
    make_contact_sheet(items, contact_sheet)
    summary = {
        "run_id": run_id,
        "purpose": "final clean-reference Patreon C attempt",
        "candidate_count": len(results),
        "expected_candidate_count": 2,
        "selected_count": 0,
        "selected_paths": [],
        "protected_provisional_candidates": protected_hashes,
        "protected_candidates_unchanged": True,
        "corrupted_prior_c_used_as_source": False,
        "model_repo": MODEL_REPO,
        "model_revision": MODEL_REVISION,
        "steps": STEPS,
        "review_status": "pending_strict_visual_review",
        "total_generation_seconds": round(time.monotonic() - started_run, 3),
        "peak_mlx_memory_gib": max(float(item["memory"]["mlx_peak_gib"]) for item in results),
        "contact_sheet": str(contact_sheet),
        "final_provisional_contact_sheet": None,
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
