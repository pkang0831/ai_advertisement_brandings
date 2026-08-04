#!/usr/bin/env python3
"""Build a review-only ten-label Rina reference pack after the Qwen smoke passes."""

from __future__ import annotations

import argparse
import gc
import json
import math
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

import mlx.core as mx
from generate_qwen_image_edit_smoke import (
    GUIDANCE,
    HEIGHT,
    MASTER,
    MASTER_SHA256,
    MODEL,
    MODEL_REPO,
    MODEL_REVISION,
    MODEL_TREE_SHA256,
    STEPS,
    WIDTH,
    image_qc,
    memory_snapshot,
    validate_readiness,
)
from mflux.models.common.config import ModelConfig
from mflux.models.qwen.variants.edit.qwen_image_edit import QwenImageEdit
from mflux.utils.dimension_resolver import CANVAS_POLICY_EXACT_RESIZE
from PIL import Image, ImageDraw, ImageFont, ImageOps

ROOT = Path(__file__).resolve().parents[2]
RINA = ROOT / "rina_park"
OUTPUT_ROOT = RINA / "identity/qwen_image_edit_reference_pack"
COMMON = (
    "Preserve exactly the same adult woman's eyes, nose, lips, jaw, hairline, and long black hair. "
    "Plain light-gray wall, simple black crew-neck top, natural unretouched head-and-shoulders photo."
)
LABELS = (
    {"slug": "01_frontal_neutral", "reuse": "01_true_frontal_neutral.png"},
    {
        "slug": "02_frontal_genuine_teeth_smile",
        "seed": 26072702,
        "prompt": f"True frontal view, looking at camera, genuine warm smile with visible teeth and smiling eyes. {COMMON}",
    },
    {
        "slug": "03_three_quarter_soft_smile",
        "seed": 26072703,
        "prompt": f"Right three-quarter view, gaze slightly past camera, small closed soft smile, calm eyes. {COMMON}",
    },
    {
        "slug": "04_three_quarter_serious",
        "seed": 26072704,
        "prompt": f"Left three-quarter view, serious closed mouth, focused gaze off-camera left. {COMMON}",
    },
    {"slug": "05_true_left_profile_neutral", "reuse": "02_true_left_profile_neutral.png"},
    {
        "slug": "06_true_right_profile_neutral",
        "seed": 26072706,
        "prompt": f"True 90-degree right profile, nose pointing right, neutral closed mouth, profile gaze right. {COMMON}",
    },
    {
        "slug": "07_looking_down_thoughtful",
        "seed": 26072707,
        "prompt": f"Head slightly turned left, chin lowered, eyes looking down, thoughtful neutral closed mouth. {COMMON}",
    },
    {
        "slug": "08_looking_up_away_curious",
        "seed": 26072708,
        "prompt": f"Head slightly turned right, chin raised, eyes looking up and away right, curious expression, lips slightly parted. {COMMON}",
    },
    {
        "slug": "09_over_shoulder_alert",
        "seed": 26072709,
        "prompt": f"Looking back over her right shoulder, head turned left, alert sideways off-camera gaze, lips slightly parted. {COMMON}",
    },
    {"slug": "10_candid_mid_laugh", "reuse": "03_candid_off_camera_genuine_laugh.png"},
)


def make_contact_sheet(items: list[tuple[str, Path]], output: Path) -> None:
    thumb, label_height, columns = 320, 48, 4
    rows = math.ceil(len(items) / columns)
    sheet = Image.new("RGB", (columns * thumb, rows * (thumb + label_height)), "white")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    for index, (label, path) in enumerate(items):
        with Image.open(path) as source:
            image = ImageOps.fit(source.convert("RGB"), (thumb, thumb), method=Image.Resampling.LANCZOS)
        x = index % columns * thumb
        y = index // columns * (thumb + label_height)
        sheet.paste(image, (x, y))
        draw.text((x + 8, y + thumb + 8), label, fill="black", font=font)
    sheet.save(output, quality=92, subsampling=0)


def build_pack(run_id: str, smoke_run: Path) -> Path:
    readiness = validate_readiness()
    smoke_summary = json.loads((smoke_run / "summary.json").read_text(encoding="utf-8"))
    if smoke_summary.get("gate_status") != "passed_3_of_3_identity_geometry_smoke":
        raise RuntimeError("the three-image Qwen smoke has not passed")

    run = OUTPUT_ROOT / run_id
    candidates = run / "candidates"
    candidates.mkdir(parents=True, exist_ok=False)
    results: list[dict[str, object]] = []
    contact_items: list[tuple[str, Path]] = []
    generated_specs = [spec for spec in LABELS if "prompt" in spec]

    model_config = ModelConfig.from_name(MODEL_REPO)
    load_started = time.monotonic()
    model = QwenImageEdit(model_config=model_config, model_path=str(MODEL))
    mx.synchronize()
    load_seconds = round(time.monotonic() - load_started, 3)
    try:
        for spec in LABELS:
            output = candidates / f"{spec['slug']}.png"
            if "reuse" in spec:
                source = smoke_run / "candidates" / str(spec["reuse"])
                shutil.copy2(source, output)
                generation_seconds = 0.0
                memory = None
                provenance = {"reused_passed_smoke": str(source)}
            else:
                mx.clear_cache()
                gc.collect()
                mx.reset_peak_memory()
                started = time.monotonic()
                generated = model.generate_image(
                    seed=int(spec["seed"]),
                    prompt=str(spec["prompt"]),
                    image_path=str(MASTER),
                    image_paths=[str(MASTER)],
                    width=WIDTH,
                    height=HEIGHT,
                    guidance=GUIDANCE,
                    num_inference_steps=STEPS,
                    scheduler="flow_match_euler_discrete",
                    canvas_policy=CANVAS_POLICY_EXACT_RESIZE,
                )
                generated.save(path=output, overwrite=True)
                mx.synchronize()
                generation_seconds = round(time.monotonic() - started, 3)
                memory = memory_snapshot()
                provenance = {"generated_for_pack": True}
                print(
                    json.dumps({"saved": str(output), "seconds": generation_seconds, "memory": memory}),
                    flush=True,
                )
            with Image.open(output) as source_image:
                qc = image_qc(source_image)
            if not qc["passed"]:
                raise RuntimeError(f"image sanity failed for {spec['slug']}: {qc}")
            payload = {
                "review_status": "reference_candidate_not_approved",
                "production_status": "not_promoted",
                "slug": spec["slug"],
                "prompt": spec.get("prompt"),
                "seed": spec.get("seed"),
                "identity_source": str(MASTER),
                "identity_source_sha256": MASTER_SHA256,
                "model_repo": MODEL_REPO,
                "model_revision": MODEL_REVISION,
                "model_tree_sha256": MODEL_TREE_SHA256,
                "dimensions": [WIDTH, HEIGHT],
                "steps": STEPS if "prompt" in spec else None,
                "guidance": GUIDANCE if "prompt" in spec else None,
                "generation_seconds": generation_seconds,
                "memory": memory,
                "qc": qc,
                "manual_qc": {
                    "identity_consistent": None,
                    "angle_matches": None,
                    "gaze_matches": None,
                    "expression_matches": None,
                    "hairline_matches": None,
                    "verdict": None,
                },
                **provenance,
            }
            output.with_suffix(".png.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
            results.append(payload)
            contact_items.append((spec["slug"], output))
            mx.clear_cache()
            gc.collect()
    finally:
        del model
        mx.clear_cache()
        gc.collect()

    contact_sheet = run / "contact_sheet_10_reference_candidates.jpg"
    make_contact_sheet(contact_items, contact_sheet)
    summary = {
        "run_id": run_id,
        "purpose": "review-only Qwen Image Edit Rina ten-label reference pack",
        "candidate_count": len(results),
        "generated_for_pack_count": len(generated_specs),
        "reused_passed_smoke_count": len(LABELS) - len(generated_specs),
        "production_references_promoted": False,
        "platform_samples_generated": False,
        "mature_content_generated": False,
        "review_status": "pending_manual_visual_review",
        "model_load_seconds": load_seconds,
        "readiness": readiness,
        "source_smoke_run": str(smoke_run),
        "contact_sheet": str(contact_sheet),
        "results": results,
    }
    (run / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return run


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
    parser.add_argument("--smoke-run", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps({"run": str(build_pack(args.run_id, args.smoke_run.resolve()))}))


if __name__ == "__main__":
    main()
