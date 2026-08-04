#!/usr/bin/env python3
"""Generate identity-locked Rina review candidates serially on Apple MPS."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import shutil
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
ADAPTER_ROOT = RINA / "models/ipadapter/h94-ip-adapter-018e402"
ADAPTER = ADAPTER_ROOT / "sdxl_models/ip-adapter-plus-face_sdxl_vit-h.safetensors"
ADAPTER_SHA256 = "677ad8860204f7d0bfba12d29e6c31ded9beefdf3e4bbd102518357d31a292c1"
ENCODER = ADAPTER_ROOT / "models/image_encoder/model.safetensors"
ENCODER_SHA256 = "6ca9667da1ca9e0b0f75e46bb030f7e011f44f86cbfb8d5a36590fcd7507b030"
ADAPTER_REVISION = "018e402774aeeddd60609b4ecdb7e298259dc729"
REFERENCE = RINA / "identity/master/rina_master_face.jpg"
WIDTH, HEIGHT, STEPS, CFG = 832, 1216, 32, 4.0
ADAPTER_SCALE = 0.68
REFERENCE_CROP = (0.14, 0.04, 0.80, 0.56)
IDENTITY_METHOD = (
    "CLIP-only h94 IP-Adapter Plus Face SDXL ViT-H; reference image conditioning; "
    "no InsightFace, AntelopeV2, buffalo, or face-recognition embeddings"
)

BASE = (
    " documentary photo, adult Korean-Canadian woman age 27, exact reference face, striking beauty, "
    "long black hair, slender athletic swimmer, narrow waist, toned long limbs, realistic skin, "
    "medium-wide full-body candid, subject 40 percent, asymmetric framing, "
)
PLATFORM_NEGATIVE = (
    "crossed eyes, misaligned eyes, asymmetrical pupils, facial drift, different person, heavy body, "
    "stocky body, broad waist, bodybuilder, short limbs, close-up, headshot, centered lineup, ID photo, "
    "catalog pose, static pose, plastic skin, AI gloss, CGI, deformed anatomy, bad hands, extra fingers, "
    "minor, child, teen, nude, nipples, genitals, lingerie, sexual act, watermark, text"
)
TRACKS = [
    {
        "track": "instagram",
        "slug": "01_instagram_environmental_swim",
        "seeds": [26072711, 26072712],
        "prompt": (
            "public indoor lap pool in soft morning window light, navy athletic one-piece swimsuit, "
            "walking along the wet deck while glancing over her shoulder and lifting goggles toward "
            "her hair, pool rail and another swimmer softly crossing foreground, candid movement, SFW"
        ) + BASE,
        "negative": PLATFORM_NEGATIVE,
    },
    {
        "track": "patreon_a_poolside_notes",
        "slug": "02_patreon_a_poolside_notes",
        "seeds": [26072721, 26072722],
        "prompt": (
            "quiet private pool lounge after swimming, elegant cream linen robe securely wrapped over "
            "a covered navy one-piece, seated sideways on a wooden bench writing a handwritten note, "
            "damp towel, goggles and rippling pool reflection establish the story, candid downward glance, SFW"
        ) + BASE,
        "negative": PLATFORM_NEGATIVE,
    },
    {
        "track": "patreon_b_extended_cut",
        "slug": "03_patreon_b_extended_cut",
        "seeds": [26072731, 26072732],
        "prompt": (
            "natural late-afternoon pool editorial, dark teal athletic one-piece with an open linen "
            "cover-up, walking beside turquoise water while gathering wet hair with one hand, foliage "
            "and lane rope layer the foreground, subtle breeze and available sunlight, quiet candid mood, SFW"
        ) + BASE,
        "negative": PLATFORM_NEGATIVE,
    },
    {
        "track": "patreon_c_season_archive",
        "slug": "04_patreon_c_season_archive",
        "seeds": [26072741, 26072742],
        "prompt": (
            "cinematic night pool archive scene, elegant dark one-piece under a loose black silk "
            "cover-up, wet hair, turning away from the pool after noticing something beyond frame, "
            "blue reflections across glass, deep shadows, foreground curtain edge, unresolved nocturnal "
            "story, sensual but fully covered platform-safe styling, SFW"
        ) + BASE,
        "negative": PLATFORM_NEGATIVE,
    },
]

VISUAL_REVIEWS = {
    "01_instagram_environmental_swim": {
        1: ("selected", "Best candid over-shoulder movement, aligned eyes, slender athletic proportions, and closest facial structure to the master."),
        2: ("rejected", "More frontal and centered with a comparatively static catalog-like stance."),
    },
    "02_patreon_a_poolside_notes": {
        1: ("rejected", "Tighter upper-body emphasis and a less securely covered robe neckline reduce fit for the covered post-swim note brief."),
        2: ("selected", "Clear note-writing story, secure elegant robe coverage, useful pool and swim-bag context, and coherent hands."),
    },
    "03_patreon_b_extended_cut": {
        1: ("selected", "Natural hair movement, layered pool environment, medium-wide framing, aligned eyes, and strong identity continuity."),
        2: ("rejected", "More centered and frontal, with a comparatively posed editorial stance and weaker swimwear read."),
    },
    "04_patreon_c_season_archive": {
        1: ("rejected", "Tighter framing, static direct presentation, and less cover-up coverage weaken the cinematic archive narrative."),
        2: ("selected", "Better environmental framing, dark draped cover-up, coherent anatomy, and a more restrained sensual mood."),
    },
}

# Optional local-only mature track: private/scripts/review_set_mature_tracks.json
_MATURE_TRACKS = RINA / "private" / "scripts" / "review_set_mature_tracks.json"
if _MATURE_TRACKS.is_file():
    _mature_blob = json.loads(_MATURE_TRACKS.read_text(encoding="utf-8")).get("v2") or {}
    _track = dict(_mature_blob.get("track") or {})
    if _track.get("prompt_prefix"):
        TRACKS.append(
            {
                "track": _track.get("track", "mature_non_explicit_local_only"),
                "slug": _track.get("slug", "05_local_mature_non_explicit"),
                "seeds": list(_track.get("seeds") or [26072751, 26072752]),
                "prompt": str(_track["prompt_prefix"]) + BASE,
                "negative": str(
                    _mature_blob.get("mature_negative") or PLATFORM_NEGATIVE
                ),
                "mature": True,
            }
        )
    for slug, reviews in (_mature_blob.get("visual_reviews") or {}).items():
        VISUAL_REVIEWS[str(slug)] = {
            int(k): (str(v[0]), str(v[1])) for k, v in reviews.items()
        }


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def validate_artifacts() -> None:
    expected = {
        MODEL: MODEL_SHA256,
        ADAPTER: ADAPTER_SHA256,
        ENCODER: ENCODER_SHA256,
        REFERENCE: None,
    }
    for path, digest in expected.items():
        if not path.is_file():
            raise RuntimeError(f"required artifact missing: {path}")
        if digest and sha256(path) != digest:
            raise RuntimeError(f"approved artifact hash mismatch: {path}")


def identity_reference() -> Image.Image:
    image = Image.open(REFERENCE).convert("RGB")
    width, height = image.size
    left, top, right, bottom = REFERENCE_CROP
    return image.crop((
        round(width * left), round(height * top),
        round(width * right), round(height * bottom),
    ))


def qc(image: Image.Image, size: tuple[int, int] = (WIDTH, HEIGHT)) -> dict[str, object]:
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
        "passed": (
            image.size == size and 5.0 < mean < 250.0 and std > 8.0
            and black < 0.95 and white < 0.95
        ),
    }


def build_pipeline() -> StableDiffusionXLPipeline:
    pipe = StableDiffusionXLPipeline.from_single_file(
        str(MODEL), torch_dtype=torch.float32, use_safetensors=True,
        local_files_only=True,
    )
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(
        pipe.scheduler.config,
        use_karras_sigmas=True,
        algorithm_type="sde-dpmsolver++",
    )
    pipe.load_ip_adapter(
        str(ADAPTER_ROOT),
        subfolder="sdxl_models",
        weight_name=ADAPTER.name,
        image_encoder_folder="models/image_encoder",
        local_files_only=True,
        low_cpu_mem_usage=True,
    )
    pipe.set_ip_adapter_scale(ADAPTER_SCALE)
    pipe.to("mps")
    pipe.vae.to(dtype=torch.float32)
    return pipe


def generate(
    pipe: StableDiffusionXLPipeline,
    reference: Image.Image,
    prompt: str,
    negative: str,
    seed: int,
    width: int = WIDTH,
    height: int = HEIGHT,
    steps: int = STEPS,
) -> tuple[Image.Image, dict[str, object], int, bool]:
    effective_seed = seed
    retried = False
    for attempt in range(2):
        image = pipe(
            prompt=prompt,
            negative_prompt=negative,
            ip_adapter_image=reference,
            width=width,
            height=height,
            num_inference_steps=steps,
            guidance_scale=CFG,
            generator=torch.Generator(device="cpu").manual_seed(effective_seed),
        ).images[0]
        result = qc(image, (width, height))
        if result["passed"]:
            return image, result, effective_seed, retried
        retried = True
        effective_seed = seed + 90_000
        gc.collect()
        torch.mps.empty_cache()
    raise RuntimeError(f"image sanity failed after black-frame retry: {result}")


def metadata(
    track: dict[str, object],
    output: Path,
    seed: int,
    effective_seed: int,
    elapsed: float,
    retry: bool,
    result_qc: dict[str, object],
    candidate: int,
) -> dict[str, object]:
    return {
        "schema_version": 2,
        "review_status": "candidate",
        "intended_track": track["track"],
        "candidate_index": candidate,
        "output": str(output),
        "prompt": track["prompt"],
        "negative_prompt": track["negative"],
        "seed": seed,
        "effective_seed": effective_seed,
        "model": MODEL.name,
        "model_sha256": MODEL_SHA256,
        "adapter": ADAPTER.name,
        "adapter_source": "https://huggingface.co/h94/IP-Adapter",
        "adapter_revision": ADAPTER_REVISION,
        "adapter_sha256": ADAPTER_SHA256,
        "adapter_scale": ADAPTER_SCALE,
        "clip_vision": "OpenCLIP ViT-H-14",
        "clip_vision_sha256": ENCODER_SHA256,
        "identity_reference": str(REFERENCE),
        "identity_reference_sha256": sha256(REFERENCE),
        "identity_reference_crop_normalized": list(REFERENCE_CROP),
        "identity_method": IDENTITY_METHOD,
        "insightface_weights_used": False,
        "dimensions": [WIDTH, HEIGHT],
        "steps": STEPS,
        "cfg": CFG,
        "device": "mps",
        "pipeline_dtype": "float32",
        "vae_dtype": "float32",
        "generation_seconds": round(elapsed, 3),
        "black_frame_retry": retry,
        "qc": result_qc,
    }


def contact_sheet(items: list[tuple[str, Path]], output: Path, columns: int = 2) -> None:
    thumb_w, thumb_h, label_h = 312, 456, 44
    rows = (len(items) + columns - 1) // columns
    sheet = Image.new("RGB", (thumb_w * columns, (thumb_h + label_h) * rows), "white")
    draw = ImageDraw.Draw(sheet)
    for index, (label, path) in enumerate(items):
        with Image.open(path) as source:
            image = source.convert("RGB")
            image.thumbnail((thumb_w, thumb_h))
        x0 = (index % columns) * thumb_w
        y0 = (index // columns) * (thumb_h + label_h)
        sheet.paste(image, (x0 + (thumb_w - image.width) // 2, y0))
        draw.text((x0 + 8, y0 + thumb_h + 8), label, fill="black")
    sheet.save(output, quality=92, subsampling=0)


def smoke_test(pipe: StableDiffusionXLPipeline, run_id: str) -> None:
    output_dir = RINA / "out/identity_smoke" / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    reference = identity_reference()
    started = time.monotonic()
    image, result_qc, effective_seed, retry = generate(
        pipe,
        reference,
        BASE + (
            "bright pool lounge, elegant navy one-piece, waist-up environmental portrait, "
            "relaxed three-quarter angle, natural expression, SFW"
        ),
        PLATFORM_NEGATIVE,
        26072700,
        width=512,
        height=768,
        steps=16,
    )
    output = output_dir / "rina_identity_smoke.jpg"
    image.save(output, quality=95, subsampling=0)
    payload = {
        "output": str(output),
        "identity_method": IDENTITY_METHOD,
        "adapter": ADAPTER.name,
        "adapter_revision": ADAPTER_REVISION,
        "adapter_sha256": ADAPTER_SHA256,
        "clip_vision_sha256": ENCODER_SHA256,
        "adapter_scale": ADAPTER_SCALE,
        "seed": 26072700,
        "effective_seed": effective_seed,
        "generation_seconds": round(time.monotonic() - started, 3),
        "black_frame_retry": retry,
        "qc": result_qc,
    }
    output.with_suffix(".jpg.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload), flush=True)


def generate_candidates(pipe: StableDiffusionXLPipeline, run_id: str) -> None:
    review = RINA / "out/review_samples" / run_id
    mature = RINA / "private/mature_non_explicit/private_media" / run_id
    platform_candidates = review / "candidates"
    mature_candidates = mature / "candidates"
    platform_candidates.mkdir(parents=True, exist_ok=True)
    mature_candidates.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(mature.parent, 0o700)
    os.chmod(mature, 0o700)
    os.chmod(mature_candidates, 0o700)
    reference = identity_reference()
    all_results: list[dict[str, object]] = []
    for track in TRACKS:
        destination = mature_candidates if track.get("mature") else platform_candidates
        for candidate, seed in enumerate(track["seeds"], start=1):
            output = destination / f"{track['slug']}_c{candidate}.jpg"
            started = time.monotonic()
            image, result_qc, effective_seed, retry = generate(
                pipe, reference, str(track["prompt"]), str(track["negative"]), int(seed)
            )
            image.save(output, quality=95, subsampling=0)
            payload = metadata(
                track, output, int(seed), effective_seed, time.monotonic() - started,
                retry, result_qc, candidate,
            )
            output.with_suffix(".jpg.json").write_text(
                json.dumps(payload, indent=2), encoding="utf-8"
            )
            all_results.append(payload)
            print(json.dumps({"saved": str(output), "qc": result_qc}), flush=True)
            gc.collect()
            torch.mps.empty_cache()
    (review / "candidates_summary.json").write_text(
        json.dumps({"run_id": run_id, "candidate_count": len(all_results), "results": all_results}, indent=2),
        encoding="utf-8",
    )


def finalize(run_id: str, selections: list[int]) -> None:
    if len(selections) != len(TRACKS) or any(item not in {1, 2} for item in selections):
        raise ValueError("--selections requires exactly five values, each 1 or 2")
    review = RINA / "out/review_samples" / run_id
    mature = RINA / "private/mature_non_explicit/private_media" / run_id
    platform_items: list[tuple[str, Path]] = []
    mature_items: list[tuple[str, Path]] = []
    results: list[dict[str, object]] = []
    for track, selected in zip(TRACKS, selections):
        root = mature if track.get("mature") else review
        for candidate in (1, 2):
            candidate_sidecar = (
                root / "candidates" / f"{track['slug']}_c{candidate}.jpg.json"
            )
            candidate_payload = json.loads(candidate_sidecar.read_text())
            decision, reason = VISUAL_REVIEWS[str(track["slug"])][candidate]
            candidate_payload["review_status"] = decision
            candidate_payload["visual_review"] = {
                "decision": decision,
                "reason": reason,
                "reviewed_against_master": True,
                "checks": [
                    "identity", "eye_alignment", "body_proportions", "hands",
                    "anatomy", "composition", "pose", "surface_realism",
                ],
            }
            candidate_sidecar.write_text(
                json.dumps(candidate_payload, indent=2), encoding="utf-8"
            )
        source = root / "candidates" / f"{track['slug']}_c{selected}.jpg"
        destination = root / f"{track['slug']}.jpg"
        if not source.is_file():
            raise RuntimeError(f"selected candidate is missing: {source}")
        shutil.copy2(source, destination)
        payload = json.loads(source.with_suffix(".jpg.json").read_text())
        payload.update({
            "review_status": "selected_for_v2_review",
            "selected_candidate_index": selected,
            "output": str(destination),
            "selection_rationale": VISUAL_REVIEWS[str(track["slug"])][selected][1],
            "qc": qc(Image.open(destination).convert("RGB")),
        })
        destination.with_suffix(".jpg.json").write_text(
            json.dumps(payload, indent=2), encoding="utf-8"
        )
        results.append(payload)
        pair = (str(track["track"]), destination)
        (mature_items if track.get("mature") else platform_items).append(pair)
    contact_sheet(platform_items, review / "contact_sheet_platform_sfw.jpg")
    contact_sheet(mature_items, mature / "contact_sheet_mature_local.jpg", columns=1)
    summary = {
        "run_id": run_id,
        "schema_version": 2,
        "generated_candidate_count": 10,
        "selected_count": 5,
        "identity_lock_applied": True,
        "identity_method": IDENTITY_METHOD,
        "adapter_revision": ADAPTER_REVISION,
        "selections": selections,
        "platform_contact_sheet": str(review / "contact_sheet_platform_sfw.jpg"),
        "mature_contact_sheet": str(mature / "contact_sheet_mature_local.jpg"),
        "results": results,
    }
    (review / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--generate-candidates", action="store_true")
    parser.add_argument("--finalize", action="store_true")
    parser.add_argument("--selections", nargs=5, type=int, default=[])
    args = parser.parse_args()
    validate_artifacts()
    if args.finalize:
        finalize(args.run_id, args.selections)
        return
    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS is required")
    pipe = build_pipeline()
    try:
        if args.smoke_test:
            smoke_test(pipe, args.run_id)
        if args.generate_candidates:
            generate_candidates(pipe, args.run_id)
        if not args.smoke_test and not args.generate_candidates:
            parser.error("choose --smoke-test, --generate-candidates, or --finalize")
    finally:
        del pipe
        gc.collect()
        torch.mps.empty_cache()


if __name__ == "__main__":
    main()
