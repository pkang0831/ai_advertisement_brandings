#!/usr/bin/env python3
"""Generate the five serialized, local-only Rina review stills on MPS."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from diffusers import DPMSolverMultistepScheduler, StableDiffusionXLPipeline
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[2]
RINA = ROOT / "rina_park"
MODEL = RINA / "models/checkpoints/RealVisXL_V5.0_fp16.safetensors"
MODEL_SHA256 = "6a35a7855770ae9820a3c931d4964c3817b6d9e3c6f9c4dabb5b3a94e5643b80"
WIDTH, HEIGHT, STEPS, CFG = 832, 1216, 32, 4.2
IDENTITY_METHOD = (
    "not_applied: no commercially approved identity adapter; "
    "descriptive base workflow only"
)
BASE = (
    "RAW phone photo, clearly adult Korean-Canadian woman age 27, long dark hair, "
    "environmental medium-wide, full body, subject 40 percent of frame, 28mm lens, "
)
PLATFORM_NEGATIVE = (
    "minor, underage, teen, child, youthful ambiguity, nude, nudity, explicit, nipples, "
    "genitals, lingerie, sexualized wet-look, seductive pose, body-part framing, tight face crop, "
    "tight chest crop, centered ID photo, plastic skin, beauty filter, studio softbox, CGI, anime, "
    "deformed anatomy, bad hands, watermark, text"
)
SCENES = [
    {
        "track": "instagram",
        "slug": "01_instagram_environmental_swim",
        "seed": 26072601,
        "prompt": BASE + (
            "busy public indoor pool, navy athletic one-piece swimsuit, walking with goggles, "
            "pool rail foreground, candid flat daylight, SFW"
        ),
        "negative": PLATFORM_NEGATIVE,
    },
    {
        "track": "patreon_a_poolside_notes",
        "slug": "02_patreon_a_poolside_notes",
        "seed": 26072601,
        "prompt": BASE + (
            "poolside transition, oversized zipped hoodie over navy athletic swimsuit, "
            "towel and swim bag, doorway framing, available daylight, SFW"
        ),
        "negative": PLATFORM_NEGATIVE,
    },
    {
        "track": "patreon_b_extended_cut",
        "slug": "03_patreon_b_extended_cut",
        "seed": 26072601,
        "prompt": BASE + (
            "evening public pool editorial, long-sleeve swim top, high-waisted swim shorts, "
            "towel on shoulders, lane-light reflections, tasteful athletic styling, SFW"
        ),
        "negative": PLATFORM_NEGATIVE,
    },
    {
        "track": "patreon_c_season_archive",
        "slug": "04_patreon_c_season_archive",
        "seed": 26072601,
        "prompt": BASE + (
            "home studio archive desk, printed contact sheets and production notes, oversized "
            "hoodie and lounge pants, view through doorway, warm cinematic lamp light, SFW"
        ),
        "negative": PLATFORM_NEGATIVE,
    },
]

# Optional local-only mature track: private/scripts/review_set_mature_tracks.json
_MATURE_TRACKS = RINA / "private" / "scripts" / "review_set_mature_tracks.json"
if _MATURE_TRACKS.is_file():
    _mature_blob = json.loads(_MATURE_TRACKS.read_text(encoding="utf-8")).get("v1") or {}
    _scene = dict(_mature_blob.get("scene") or {})
    if _scene.get("prompt_suffix"):
        SCENES.append(
            {
                "track": _scene.get("track", "mature_non_explicit_local_only"),
                "slug": _scene.get("slug", "05_local_mature_non_explicit"),
                "seed": int(_scene.get("seed", 26072601)),
                "prompt": BASE + str(_scene["prompt_suffix"]),
                "negative": str(
                    _mature_blob.get("mature_negative") or PLATFORM_NEGATIVE
                ),
                "mature": True,
            }
        )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def qc(image: Image.Image) -> dict[str, object]:
    pixels = np.asarray(image.convert("RGB"))
    mean = float(pixels.mean())
    std = float(pixels.std())
    black = float((pixels < 3).all(axis=2).mean())
    white = float((pixels > 252).all(axis=2).mean())
    return {
        "dimensions": list(image.size),
        "mean_rgb": round(mean, 3),
        "std_rgb": round(std, 3),
        "black_pixel_fraction": round(black, 6),
        "white_pixel_fraction": round(white, 6),
        "passed": image.size == (WIDTH, HEIGHT) and 5.0 < mean < 250.0 and std > 8.0
        and black < 0.95 and white < 0.95,
    }


def contact_sheet(items: list[tuple[str, Path]], output: Path) -> None:
    thumb_w, thumb_h, label_h = 312, 456, 44
    sheet = Image.new("RGB", (thumb_w * 2, (thumb_h + label_h) * 2), "white")
    draw = ImageDraw.Draw(sheet)
    for index, (label, path) in enumerate(items):
        image = Image.open(path).convert("RGB")
        image.thumbnail((thumb_w, thumb_h))
        x = (index % 2) * thumb_w + (thumb_w - image.width) // 2
        y = (index // 2) * (thumb_h + label_h)
        sheet.paste(image, (x, y))
        draw.text((8 + (index % 2) * thumb_w, y + thumb_h + 8), label, fill="black")
    sheet.save(output, quality=92)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-id",
        default=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
    )
    parser.add_argument("--scene-index", type=int, choices=range(len(SCENES)))
    parser.add_argument("--finalize", action="store_true")
    args = parser.parse_args()
    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS is required")
    if sha256(MODEL) != MODEL_SHA256:
        raise RuntimeError("approved checkpoint hash mismatch")

    review_dir = RINA / "out/review_samples" / args.run_id
    mature_root = RINA / "private/mature_non_explicit/private_media"
    mature_dir = mature_root / args.run_id
    review_dir.mkdir(parents=True, exist_ok=True)
    mature_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(mature_root, 0o700)
    os.chmod(mature_dir, 0o700)

    if args.finalize:
        platform_items = [
            (str(scene["track"]), review_dir / f"{scene['slug']}.jpg")
            for scene in SCENES if not scene.get("mature")
        ]
        mature_items = [
            (str(scene["track"]), mature_dir / f"{scene['slug']}.jpg")
            for scene in SCENES if scene.get("mature")
        ]
        expected = [path for _, path in platform_items + mature_items]
        if not all(path.is_file() for path in expected):
            raise RuntimeError("cannot finalize: one or more review stills are missing")
        contact_sheet(platform_items, review_dir / "contact_sheet_platform_sfw.jpg")
        contact_sheet(mature_items, mature_dir / "contact_sheet_mature_local.jpg")
        results = [
            json.loads(path.with_suffix(".jpg.json").read_text())
            for path in expected
        ]
        summary = {
            "run_id": args.run_id,
            "generated_count": len(results),
            "platform_contact_sheet": str(review_dir / "contact_sheet_platform_sfw.jpg"),
            "mature_contact_sheet": str(mature_dir / "contact_sheet_mature_local.jpg"),
            "identity_lock_applied": False,
            "results": results,
        }
        (review_dir / "summary.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )
        return

    scenes = SCENES if args.scene_index is None else [SCENES[args.scene_index]]
    pipe = StableDiffusionXLPipeline.from_single_file(
        str(MODEL), torch_dtype=torch.float32, use_safetensors=True,
        local_files_only=True,
    )
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(
        pipe.scheduler.config,
        use_karras_sigmas=True,
        algorithm_type="sde-dpmsolver++",
    )
    pipe.enable_attention_slicing()
    pipe.to("mps")
    pipe.vae.to(dtype=torch.float32)

    platform_items: list[tuple[str, Path]] = []
    mature_items: list[tuple[str, Path]] = []
    results: list[dict[str, object]] = []
    for scene in scenes:
        destination = mature_dir if scene.get("mature") else review_dir
        output = destination / f"{scene['slug']}.jpg"
        started = time.monotonic()
        image = pipe(
            prompt=scene["prompt"],
            negative_prompt=scene["negative"],
            width=WIDTH,
            height=HEIGHT,
            num_inference_steps=STEPS,
            guidance_scale=CFG,
            generator=torch.Generator(device="cpu").manual_seed(scene["seed"]),
        ).images[0]
        result_qc = qc(image)
        retry = False
        effective_seed = scene["seed"]
        if not result_qc["passed"]:
            retry = True
            effective_seed = 20_300_101
            torch.mps.empty_cache()
            gc.collect()
            image = pipe(
                prompt=scene["prompt"],
                negative_prompt=scene["negative"],
                width=WIDTH,
                height=HEIGHT,
                num_inference_steps=STEPS,
                guidance_scale=CFG,
                generator=torch.Generator(device="cpu").manual_seed(effective_seed),
            ).images[0]
            result_qc = qc(image)
        if not result_qc["passed"]:
            raise RuntimeError(f"image sanity failed for {scene['track']}: {result_qc}")
        image.save(output, quality=95, subsampling=0)
        metadata = {
            "intended_track": scene["track"],
            "output": str(output),
            "prompt": scene["prompt"],
            "negative_prompt": scene["negative"],
            "seed": scene["seed"],
            "effective_seed": effective_seed,
            "model": MODEL.name,
            "model_sha256": MODEL_SHA256,
            "loras": {},
            "identity_reference": str(RINA / "identity/master/rina_master_face.jpg"),
            "identity_method": IDENTITY_METHOD,
            "dimensions": [WIDTH, HEIGHT],
            "steps": STEPS,
            "cfg": CFG,
            "device": "mps",
            "pipeline_dtype": "float32",
            "vae_dtype": "float32",
            "generation_seconds": round(time.monotonic() - started, 3),
            "black_frame_retry": retry,
            "qc": result_qc,
        }
        output.with_suffix(".jpg.json").write_text(
            json.dumps(metadata, indent=2), encoding="utf-8"
        )
        results.append(metadata)
        pair = (str(scene["track"]), output)
        (mature_items if scene.get("mature") else platform_items).append(pair)
        print(json.dumps({"saved": str(output), "qc": result_qc}), flush=True)
        gc.collect()
        torch.mps.empty_cache()

    if args.scene_index is None:
        contact_sheet(platform_items, review_dir / "contact_sheet_platform_sfw.jpg")
        contact_sheet(mature_items, mature_dir / "contact_sheet_mature_local.jpg")
        summary = {
            "run_id": args.run_id,
            "generated_count": len(results),
            "platform_contact_sheet": str(review_dir / "contact_sheet_platform_sfw.jpg"),
            "mature_contact_sheet": str(mature_dir / "contact_sheet_mature_local.jpg"),
            "identity_lock_applied": False,
            "results": results,
        }
        (review_dir / "summary.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )
    del pipe
    gc.collect()
    torch.mps.empty_cache()


if __name__ == "__main__":
    main()
