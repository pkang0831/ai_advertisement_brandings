#!/usr/bin/env python3
"""Generate exactly two targeted Patreon A and two Patreon C Qwen repairs."""

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
SOURCE_RUN = RINA / "out/review_samples/20260726T073925Z"
SOURCE_A = SOURCE_RUN / "candidates/03_patreon_a_notes_down.png"
SOURCE_C = SOURCE_RUN / "candidates/07_patreon_c_night_serious.png"
PROTECTED = (
    SOURCE_RUN / "selected/01_ig_candid_laugh.png",
    SOURCE_RUN / "selected/05_patreon_b_wind_soft_smile.png",
)
OUTPUT_ROOT = RINA / "out/review_samples"
WIDTH, HEIGHT = 704, 896
STEPS, GUIDANCE = 20, 4.0
SPECS = (
    {
        "slug": "01_patreon_a_opaque_cream_robe",
        "track": "patreon_a",
        "source": SOURCE_A,
        "seed": 26072901,
        "prompt": (
            "Make one localized wardrobe correction only. Replace the sheer charcoal robe with an elegant fully opaque "
            "cream linen robe/cover-up that covers the same areas and shows no skin through the fabric. Preserve every "
            "other pixel-level aspect as closely as possible: exact adult Rina face, downward gaze, hair, body position, "
            "blank notebook with absolutely no writing, both natural hands, towel, pen, pool background, lighting, and "
            "framing. No catalog polish, text, logo, extra exposure, lingerie, or anatomy changes."
        ),
    },
    {
        "slug": "02_patreon_a_opaque_charcoal_robe",
        "track": "patreon_a",
        "source": SOURCE_A,
        "seed": 26072902,
        "prompt": (
            "Make one localized wardrobe correction only. Replace the sheer robe with an elegant fully opaque matte "
            "charcoal linen robe/cover-up, solid fabric with zero transparency, covering the same areas. Preserve every "
            "other element exactly: adult Rina identity, downward eyes and expression, hair, body, completely blank "
            "notebook, both hands, towel, pen, pool, light, composition, and framing. No writing, text, catalog polish, "
            "new exposure, lingerie, or anatomy changes."
        ),
    },
    {
        "slug": "03_patreon_c_zoomout_hidden_hand",
        "track": "patreon_c",
        "source": SOURCE_C,
        "seed": 26072903,
        "prompt": (
            "Targeted repair only: zoom out to an environmental medium-wide portrait where the subject occupies 40% "
            "of frame, revealing more wet pool deck, dark negative space, blue night water and amber reflections. "
            "Preserve the exact adult Rina face, serious off-camera gaze, wet hair, dark athletic one-piece, body, night "
            "lighting and unresolved SFW mood. Replace the sheer drape with an opaque charcoal cover-up arranged naturally "
            "so the previously corrupted lowered hand is fully and plausibly hidden beneath it. Keep the other hand "
            "natural. Absolutely night, no daylight, lingerie, nudity, text, glossy skin, or other scene changes."
        ),
    },
    {
        "slug": "04_patreon_c_zoomout_towel_hand",
        "track": "patreon_c",
        "source": SOURCE_C,
        "seed": 26072904,
        "prompt": (
            "Targeted repair only: reframe wider so the subject occupies 35-45% of the portrait frame with more wet pool "
            "deck, dark negative space, blue night reflections and distant amber lights. Preserve exact adult Rina face, "
            "serious off-camera gaze, wet hair, dark athletic one-piece, body proportions, night lighting and SFW unresolved "
            "story. Fully naturalize the lowered hand by having it loosely hold a small plain charcoal towel with correct "
            "fingers; use an opaque draped charcoal cover-up. Keep the other hand natural. No daylight, lingerie, nudity, "
            "text, catalog pose, glossy skin, or unrelated changes."
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


def make_contact_sheet(items: list[tuple[str, Path]], output: Path, columns: int) -> None:
    thumb_w, thumb_h, label_h = 264, 336, 52
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
    for path in (SOURCE_A, SOURCE_C, *PROTECTED):
        if not path.is_file():
            raise RuntimeError(f"required source missing: {path}")
    protected_hashes = {str(path): sha256(path) for path in PROTECTED}
    run = OUTPUT_ROOT / f"repair_{run_id}"
    candidates = run / "candidates"
    candidates.mkdir(parents=True, exist_ok=False)
    model = QwenImageEdit(model_config=ModelConfig.from_name(MODEL_REPO), model_path=str(MODEL))
    mx.synchronize()
    results: list[dict[str, object]] = []
    items: list[tuple[str, Path]] = []
    started_run = time.monotonic()
    try:
        for spec in SPECS:
            mx.clear_cache()
            gc.collect()
            mx.reset_peak_memory()
            output = candidates / f"{spec['slug']}.png"
            started = time.monotonic()
            generated = model.generate_image(
                seed=int(spec["seed"]),
                prompt=str(spec["prompt"]),
                image_path=str(spec["source"]),
                image_paths=[str(spec["source"])],
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
            with Image.open(output) as source_image:
                qc = image_qc(source_image)
            if not qc["passed"]:
                raise RuntimeError(f"image sanity failed for {spec['slug']}: {qc}")
            payload = {
                "schema_version": 1,
                "repair_status": "candidate_pending_visual_review",
                "production_status": "not_promoted",
                "track": spec["track"],
                "slug": spec["slug"],
                "source_path": str(spec["source"]),
                "source_sha256": sha256(Path(spec["source"])),
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
                    "eyes_gaze_preserved": None,
                    "hands_natural": None,
                    "robe_fully_opaque": None,
                    "subject_fraction_estimate": None,
                    "text_present": None,
                    "scene_preserved": None,
                    "unaffected_content_preserved": None,
                    "caveats": None,
                    "verdict": None,
                },
            }
            output.with_suffix(".png.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
            results.append(payload)
            items.append((f"{spec['track']}\n{spec['slug']}", output))
            print(json.dumps({"saved": str(output), "seconds": seconds, "memory": payload["memory"]}), flush=True)
    finally:
        del model
        mx.clear_cache()
        gc.collect()

    for path, expected in protected_hashes.items():
        if sha256(Path(path)) != expected:
            raise RuntimeError(f"protected provisional candidate changed: {path}")
    make_contact_sheet(items, run / "contact_sheet_all_repairs.jpg", columns=4)
    make_contact_sheet(items[:2], run / "contact_sheet_patreon_a_repairs.jpg", columns=2)
    make_contact_sheet(items[2:], run / "contact_sheet_patreon_c_repairs.jpg", columns=2)
    summary = {
        "run_id": run_id,
        "purpose": "targeted Qwen v4 Patreon A/C repair",
        "candidate_count": len(results),
        "expected_candidate_count": 4,
        "selected_count": 0,
        "selected_paths": [],
        "protected_provisional_candidates": protected_hashes,
        "protected_candidates_unchanged": True,
        "source_run": str(SOURCE_RUN),
        "model_repo": MODEL_REPO,
        "model_revision": MODEL_REVISION,
        "steps": STEPS,
        "review_status": "pending_strict_visual_review",
        "total_generation_seconds": round(time.monotonic() - started_run, 3),
        "peak_mlx_memory_gib": max(float(item["memory"]["mlx_peak_gib"]) for item in results),
        "contact_sheets": {
            "all_repairs": str(run / "contact_sheet_all_repairs.jpg"),
            "patreon_a": str(run / "contact_sheet_patreon_a_repairs.jpg"),
            "patreon_c": str(run / "contact_sheet_patreon_c_repairs.jpg"),
            "final_provisional": None,
        },
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
