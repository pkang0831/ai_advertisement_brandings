#!/usr/bin/env python3
"""Generate platform-only Rina v3 review candidates serially on Apple MPS."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
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
V2_RUN = RINA / "out/review_samples/20260726T014600Z"
WIDTH, HEIGHT, STEPS, CFG = 832, 1216, 32, 4.0
IDENTITY_METHOD = (
    "CLIP-only h94 IP-Adapter Plus Face SDXL ViT-H; no InsightFace, "
    "AntelopeV2, buffalo, or face-recognition embeddings"
)
BASE = (
    "adult Korean-Canadian woman age 27, Rina identity geometry, long black hair, "
    "slender athletic swimmer, narrow defined waist, natural toned limbs, balanced proportions, "
    "documentary realism, natural skin texture, SFW"
)
NEGATIVE = (
    "same three-quarter face, repeated reference pose, mirrored pose, crossed eyes, misaligned eyes, "
    "uncanny pupils, facial drift, different person, heavy body, stocky body, broad waist, bodybuilder, "
    "short limbs, close crop, headshot, subject over half frame, catalog stance, static pose, fake text, "
    "writing, letters, watermark, plastic skin, AI gloss, CGI, bad hands, extra fingers, minor, child, "
    "teen, nude, nipples, genitals, lingerie, transparent fabric, sexual act"
)
CROPS = {
    "wide_face_context": (0.06, 0.01, 0.92, 0.72),
    "v2_tight_face": (0.14, 0.04, 0.80, 0.56),
}

TRACKS = [
    {
        "track": "instagram",
        "slug": "01_instagram_environmental_swim",
        "candidates": [
            {
                "seed": 26072811, "scale": 0.50, "crop": "wide_face_context",
                "angle": "left side profile; slight downward pitch; low oblique camera",
                "gaze": "looking away toward lane",
                "expression": "focused neutral; relaxed brows; closed mouth",
                "action": "walking while loosely carrying goggles",
                "prompt": (
                    "environmental wide public lap pool, subject 35 percent of frame, left side profile, "
                    "chin slightly down, eyes looking away toward a lane, focused relaxed face, walking "
                    "mid-stride with swim goggles loose in one hand, navy athletic one-piece, 28mm candid"
                ),
            },
            {
                "seed": 26072812, "scale": 0.56, "crop": "wide_face_context",
                "angle": "three-quarter back; head turned back; high wide camera",
                "gaze": "looking back toward camera",
                "expression": "genuine open smile/laugh; lifted cheeks",
                "action": "turning back with towel in hand",
                "prompt": (
                    "environmental wide indoor pool, subject 40 percent of frame, body moving away, head "
                    "turning back toward camera, genuine open smile and spontaneous laugh, towel swinging "
                    "in one hand, goggles in the other, navy athletic one-piece, high 28mm candid angle"
                ),
            },
            {
                "seed": 26072813, "scale": 0.62, "crop": "v2_tight_face",
                "angle": "right three-quarter side; eye-level distant camera",
                "gaze": "off-camera toward teammate",
                "expression": "broad toothy smile; animated eyes",
                "action": "pausing while drying hair with towel",
                "prompt": (
                    "super-wide busy lap pool, subject 45 percent of frame, right three-quarter side angle, "
                    "looking off-camera at a teammate with broad natural toothy smile, drying wet hair with "
                    "a towel while holding goggles, navy athletic one-piece, candid eye-level 28mm photo"
                ),
            },
        ],
    },
    {
        "track": "patreon_a_poolside_notes",
        "slug": "02_patreon_a_poolside_notes",
        "candidates": [
            {
                "seed": 26072821, "scale": 0.50, "crop": "wide_face_context",
                "angle": "left side profile; downward pitch; elevated camera",
                "gaze": "down at blank notebook",
                "expression": "thoughtful neutral; soft brow; closed mouth",
                "action": "placing closed blank notebook beside towel",
                "prompt": (
                    "quiet pool lounge, left side profile from elevated medium-wide camera, thoughtful "
                    "downward gaze, placing a completely blank closed notebook beside a folded towel, "
                    "no pen and no marks, cream opaque robe over navy one-piece, candid post-swim moment"
                ),
            },
            {
                "seed": 26072822, "scale": 0.56, "crop": "wide_face_context",
                "angle": "near profile; slight downward pitch; doorway camera",
                "gaze": "down toward hands",
                "expression": "soft candid closed-mouth smile; relaxed eyes",
                "action": "folding towel over a blank notebook",
                "prompt": (
                    "private pool deck viewed through doorway, near-profile seated pose, eyes down with "
                    "soft candid closed-mouth smile, folding a towel over a plain blank notebook with no "
                    "visible pages or text, cream linen robe securely wrapped, medium-wide natural photo"
                ),
            },
            {
                "seed": 26072823, "scale": 0.62, "crop": "v2_tight_face",
                "angle": "right three-quarter; overhead diagonal camera",
                "gaze": "away toward water",
                "expression": "quiet reflective half-smile; lowered brows",
                "action": "wiping goggles beside closed blank notebook",
                "prompt": (
                    "poolside bench from overhead diagonal medium-wide angle, right three-quarter face "
                    "turned toward water, quiet reflective half-smile, wiping goggles with towel beside "
                    "a closed blank unmarked notebook, cream opaque cover-up, candid post-swim story"
                ),
            },
        ],
    },
    {
        "track": "patreon_b_extended_cut",
        "slug": "03_patreon_b_extended_cut",
        "candidates": [
            {
                "seed": 26072831, "scale": 0.50, "crop": "wide_face_context",
                "angle": "left mid-turn profile; low medium-wide camera",
                "gaze": "direct glance during turn",
                "expression": "confident serious; one brow slightly raised",
                "action": "mid-turn as wind lifts hair and cover-up",
                "prompt": (
                    "late-afternoon swim editorial, low medium-wide camera, caught mid-turn in left profile, "
                    "brief direct confident serious glance with one brow subtly raised, wind lifting wet hair "
                    "and opaque teal cover-up over dark one-piece, walking beside pool, dynamic candid"
                ),
            },
            {
                "seed": 26072832, "scale": 0.56, "crop": "wide_face_context",
                "angle": "right three-quarter; tilted camera from pool level",
                "gaze": "past camera",
                "expression": "confident small asymmetric smile; narrowed eyes",
                "action": "sweeping windblown hair from face",
                "prompt": (
                    "distinct medium-wide pool fashion story from water level, right three-quarter moving "
                    "angle, eyes looking past camera, confident asymmetric small smile, windblown wet hair "
                    "swept from face with one hand, dark teal athletic one-piece and opaque linen cover-up"
                ),
            },
            {
                "seed": 26072833, "scale": 0.62, "crop": "v2_tight_face",
                "angle": "frontal body with face in profile; long-lens medium-wide",
                "gaze": "off-frame left",
                "expression": "energized open smile; raised cheeks",
                "action": "stepping from water as hair swings",
                "prompt": (
                    "medium-wide editorial stepping from pool, frontal moving body but face turned into "
                    "clean profile, looking off-frame left with energized open smile, wet hair swinging in "
                    "breeze, dark athletic one-piece beneath opaque teal cover-up, layered reflections"
                ),
            },
        ],
    },
    {
        "track": "patreon_c_season_archive",
        "slug": "04_patreon_c_season_archive",
        "candidates": [
            {
                "seed": 26072841, "scale": 0.50, "crop": "wide_face_context",
                "angle": "profile silhouette; low distant camera",
                "gaze": "toward dark doorway",
                "expression": "guarded and intent; lips gently parted",
                "action": "pausing after hearing something off-frame",
                "prompt": (
                    "cinematic nighttime pool, low distant medium-wide camera, profile silhouette pausing "
                    "after hearing something near a dark doorway, guarded intent gaze, lips gently parted, "
                    "wet hair, opaque dark one-piece and black cover-up, blue reflections, deep shadow, SFW"
                ),
            },
            {
                "seed": 26072842, "scale": 0.56, "crop": "wide_face_context",
                "angle": "three-quarter downward; camera across reflective water",
                "gaze": "up from water reflection toward camera",
                "expression": "sultry restrained gaze; neutral mouth",
                "action": "pulling opaque cover-up around shoulders",
                "prompt": (
                    "night pool scene photographed across reflective water, subject under 45 percent, head "
                    "angled down then eyes lifted toward camera, restrained sultry gaze with neutral mouth, "
                    "wet hair, pulling opaque black cover-up around dark one-piece, moonlit shadow, unresolved"
                ),
            },
            {
                "seed": 26072843, "scale": 0.62, "crop": "v2_tight_face",
                "angle": "back view with face turned left; high distant camera",
                "gaze": "sideways beyond frame",
                "expression": "wary half-smile; asymmetrical brows",
                "action": "walking away along wet deck",
                "prompt": (
                    "cinematic blue-black nighttime pool from high distant camera, walking away along wet "
                    "deck, back view with face turned left, wary sideways gaze beyond frame and faint uneasy "
                    "half-smile, wet hair, opaque dark one-piece under long black cover-up, reflection, mystery"
                ),
            },
        ],
    },
]

V2_FEEDBACK = {
    "instagram": {
        "issue_codes": ["facial_angle_mode_collapse", "gaze_mode_collapse", "expression_mode_collapse"],
        "observations": [
            "Selected and candidate faces repeatedly resolve to the master's mild three-quarter gaze.",
            "Differences are effectively mirrored variants rather than distinct yaw, pitch, gaze, or mouth states.",
            "The set lacks a true profile/looking-away frame and a genuine open smile or laugh.",
        ],
        "required_changes": "Vary yaw, pitch, gaze and mouth; add candid action with goggles/towel and subject at or below 50 percent.",
    },
    "patreon_a_poolside_notes": {
        "issue_codes": ["illegible_generated_writing", "weak_natural_action"],
        "observations": [
            "Notebook contains synthetic illegible marks.",
            "Pen-to-page pose reads as staged and directs attention to the artifact.",
        ],
        "required_changes": "Use a blank/closed notebook or towel with no visible writing and a natural post-swim handling action.",
    },
    "patreon_b_extended_cut": {
        "issue_codes": ["insufficient_track_differentiation", "weak_editorial_appeal"],
        "observations": [
            "Expression and three-quarter face remain too similar to Instagram and Patreon A.",
            "Front-facing robe walk reads as a catalog stance despite hair movement.",
        ],
        "required_changes": "Use a confident mid-turn, stronger wind/hair movement, distinct expression, and medium-wide swim editorial framing.",
    },
    "patreon_c_season_archive": {
        "issue_codes": ["daylight_story_mismatch", "insufficient_sensual_atmosphere"],
        "observations": [
            "Scene is visibly daylight rather than nighttime.",
            "Bright pool setting and seated catalog pose remove cinematic tension and unresolved narrative.",
        ],
        "required_changes": "Use a nighttime pool, wet hair, reflection and shadow, opaque dark styling, restrained sensual gaze, and unresolved story.",
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def validate_artifacts() -> None:
    for path, digest in {
        MODEL: MODEL_SHA256, ADAPTER: ADAPTER_SHA256, ENCODER: ENCODER_SHA256,
    }.items():
        if not path.is_file() or sha256(path) != digest:
            raise RuntimeError(f"approved artifact missing or hash mismatch: {path}")
    if not REFERENCE.is_file():
        raise RuntimeError(f"identity reference missing: {REFERENCE}")


def reference_crop(name: str) -> Image.Image:
    image = Image.open(REFERENCE).convert("RGB")
    width, height = image.size
    left, top, right, bottom = CROPS[name]
    return image.crop((round(width * left), round(height * top), round(width * right), round(height * bottom)))


def qc(image: Image.Image) -> dict[str, object]:
    pixels = np.asarray(image.convert("RGB"))
    mean, std = float(pixels.mean()), float(pixels.std())
    black = float((pixels < 3).all(axis=2).mean())
    white = float((pixels > 252).all(axis=2).mean())
    return {
        "dimensions": list(image.size), "mean_rgb": round(mean, 3), "std_rgb": round(std, 3),
        "black_pixel_fraction": round(black, 6), "white_pixel_fraction": round(white, 6),
        "passed": image.size == (WIDTH, HEIGHT) and 5 < mean < 250 and std > 8 and black < 0.95 and white < 0.95,
    }


def build_pipeline() -> StableDiffusionXLPipeline:
    pipe = StableDiffusionXLPipeline.from_single_file(
        str(MODEL), torch_dtype=torch.float32, use_safetensors=True, local_files_only=True,
    )
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(
        pipe.scheduler.config, use_karras_sigmas=True, algorithm_type="sde-dpmsolver++",
    )
    pipe.load_ip_adapter(
        str(ADAPTER_ROOT), subfolder="sdxl_models", weight_name=ADAPTER.name,
        image_encoder_folder="models/image_encoder", local_files_only=True, low_cpu_mem_usage=True,
    )
    pipe.to("mps")
    pipe.vae.to(dtype=torch.float32)
    return pipe


def generate(
    pipe: StableDiffusionXLPipeline, candidate: dict[str, object],
) -> tuple[Image.Image, dict[str, object], int, bool]:
    seed = int(candidate["seed"])
    effective_seed, retried = seed, False
    pipe.set_ip_adapter_scale(float(candidate["scale"]))
    reference = reference_crop(str(candidate["crop"]))
    for _ in range(2):
        image = pipe(
            prompt=f"{candidate['prompt']}, {BASE}", negative_prompt=NEGATIVE,
            ip_adapter_image=reference, width=WIDTH, height=HEIGHT,
            num_inference_steps=STEPS, guidance_scale=CFG,
            generator=torch.Generator(device="cpu").manual_seed(effective_seed),
        ).images[0]
        result = qc(image)
        if result["passed"]:
            return image, result, effective_seed, retried
        retried = True
        effective_seed = seed + 90_000
        gc.collect()
        torch.mps.empty_cache()
    raise RuntimeError(f"image sanity failed after black-frame retry: {result}")


def mark_v2_rejected() -> None:
    if not V2_RUN.is_dir():
        raise RuntimeError(f"v2 run missing: {V2_RUN}")
    updated = 0
    for sidecar in sorted(V2_RUN.glob("*.jpg.json")) + sorted((V2_RUN / "candidates").glob("*.jpg.json")):
        payload = json.loads(sidecar.read_text(encoding="utf-8"))
        track = str(payload.get("intended_track"))
        feedback = V2_FEEDBACK.get(track)
        if not feedback:
            continue
        payload["review_status"] = "rejected_needs_regeneration"
        payload["v3_regeneration_feedback"] = {"decision": "rejected", **feedback}
        sidecar.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        updated += 1
    summary_path = V2_RUN / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    for result in summary.get("results", []):
        track = str(result.get("intended_track"))
        if track in V2_FEEDBACK:
            result["review_status"] = "rejected_needs_regeneration"
            result["v3_regeneration_feedback"] = {"decision": "rejected", **V2_FEEDBACK[track]}
    summary["platform_v2_review_status"] = "rejected_needs_regeneration"
    summary["platform_v2_feedback"] = V2_FEEDBACK
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"v2_platform_sidecars_updated": updated}), flush=True)


def metadata(
    track: dict[str, object], candidate: dict[str, object], output: Path, index: int,
    result_qc: dict[str, object], effective_seed: int, retried: bool, elapsed: float,
) -> dict[str, object]:
    return {
        "schema_version": 3, "review_status": "candidate", "intended_track": track["track"],
        "candidate_index": index, "output": str(output), "prompt": f"{candidate['prompt']}, {BASE}",
        "negative_prompt": NEGATIVE, "seed": candidate["seed"], "effective_seed": effective_seed,
        "angle_label": candidate["angle"], "gaze_label": candidate["gaze"],
        "expression_label": candidate["expression"], "action_label": candidate["action"],
        "model": MODEL.name, "model_sha256": MODEL_SHA256, "adapter": ADAPTER.name,
        "adapter_source": "https://huggingface.co/h94/IP-Adapter",
        "adapter_revision": ADAPTER_REVISION, "adapter_sha256": ADAPTER_SHA256,
        "adapter_scale": candidate["scale"], "clip_vision": "OpenCLIP ViT-H-14",
        "clip_vision_sha256": ENCODER_SHA256, "identity_reference": str(REFERENCE),
        "identity_reference_sha256": sha256(REFERENCE), "identity_reference_crop_name": candidate["crop"],
        "identity_reference_crop_normalized": list(CROPS[str(candidate["crop"])]),
        "identity_method": IDENTITY_METHOD, "insightface_weights_used": False,
        "expression_constraint_diagnostic": (
            "Candidate is part of a 0.50/0.56/0.62 scale and wide/tight crop comparison; "
            "lower scale and wider context are expected to reduce replication of the master's three-quarter expression."
        ),
        "dimensions": [WIDTH, HEIGHT], "steps": STEPS, "cfg": CFG, "device": "mps",
        "pipeline_dtype": "float32", "vae_dtype": "float32",
        "generation_seconds": round(elapsed, 3), "black_frame_retry": retried, "qc": result_qc,
    }


def generate_candidates(pipe: StableDiffusionXLPipeline, run_id: str) -> None:
    review = RINA / "out/review_samples" / run_id
    destination = review / "candidates"
    destination.mkdir(parents=True, exist_ok=False)
    results = []
    for track in TRACKS:
        for index, candidate in enumerate(track["candidates"], start=1):
            output = destination / f"{track['slug']}_c{index}.jpg"
            started = time.monotonic()
            image, result_qc, effective_seed, retried = generate(pipe, candidate)
            image.save(output, quality=95, subsampling=0)
            payload = metadata(
                track, candidate, output, index, result_qc, effective_seed, retried,
                time.monotonic() - started,
            )
            output.with_suffix(".jpg.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
            results.append(payload)
            print(json.dumps({"saved": str(output), "qc": result_qc}), flush=True)
            gc.collect()
            torch.mps.empty_cache()
    (review / "candidates_summary.json").write_text(
        json.dumps({"run_id": run_id, "schema_version": 3, "candidate_count": 12, "results": results}, indent=2),
        encoding="utf-8",
    )


def contact_sheet(items: list[tuple[str, Path]], output: Path, columns: int = 4) -> None:
    thumb_w, thumb_h, label_h = 280, 410, 54
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
        draw.text((x0 + 7, y0 + thumb_h + 7), label, fill="black")
    sheet.save(output, quality=92, subsampling=0)


def finalize(run_id: str, selections: list[int], reviews_path: Path) -> None:
    if len(selections) != 4 or any(item not in {1, 2, 3} for item in selections):
        raise ValueError("--selections requires exactly four values, each 1, 2, or 3")
    reviews = json.loads(reviews_path.read_text(encoding="utf-8"))
    review = RINA / "out/review_samples" / run_id
    selected_items, candidate_items, results = [], [], []
    for track, selected in zip(TRACKS, selections):
        slug = str(track["slug"])
        track_reviews = reviews[slug]
        for index in (1, 2, 3):
            source = review / "candidates" / f"{slug}_c{index}.jpg"
            sidecar = source.with_suffix(".jpg.json")
            payload = json.loads(sidecar.read_text(encoding="utf-8"))
            decision = track_reviews[str(index)]
            payload["review_status"] = "selected" if index == selected else "rejected"
            payload["visual_review"] = decision
            sidecar.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            candidate_items.append((f"{track['track']} c{index}", source))
        source = review / "candidates" / f"{slug}_c{selected}.jpg"
        destination = review / f"{slug}.jpg"
        shutil.copy2(source, destination)
        payload = json.loads(source.with_suffix(".jpg.json").read_text(encoding="utf-8"))
        payload.update({
            "review_status": "selected_for_v3_review", "selected_candidate_index": selected,
            "output": str(destination), "selection_rationale": track_reviews[str(selected)]["reason"],
            "remaining_caveats": track_reviews[str(selected)].get("remaining_caveats", []),
            "qc": qc(Image.open(destination).convert("RGB")),
        })
        destination.with_suffix(".jpg.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
        selected_items.append((str(track["track"]), destination))
        results.append(payload)
    contact_sheet(candidate_items, review / "contact_sheet_all_candidates.jpg")
    contact_sheet(selected_items, review / "contact_sheet_platform_sfw.jpg", columns=4)
    summary = {
        "run_id": run_id, "schema_version": 3, "generated_candidate_count": 12,
        "selected_count": 4, "platform_only": True, "mature_content_generated": False,
        "identity_lock_applied": True, "identity_method": IDENTITY_METHOD,
        "adapter_revision": ADAPTER_REVISION, "selections": selections,
        "scale_crop_diagnosis": reviews["scale_crop_diagnosis"],
        "candidate_contact_sheet": str(review / "contact_sheet_all_candidates.jpg"),
        "platform_contact_sheet": str(review / "contact_sheet_platform_sfw.jpg"), "results": results,
    }
    (review / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
    parser.add_argument("--mark-v2-rejected", action="store_true")
    parser.add_argument("--generate-candidates", action="store_true")
    parser.add_argument("--finalize", action="store_true")
    parser.add_argument("--selections", nargs=4, type=int, default=[])
    parser.add_argument("--reviews", type=Path)
    args = parser.parse_args()
    validate_artifacts()
    if args.mark_v2_rejected:
        mark_v2_rejected()
    if args.finalize:
        if not args.reviews:
            parser.error("--finalize requires --reviews")
        finalize(args.run_id, args.selections, args.reviews)
        return
    if args.generate_candidates:
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS is required")
        pipe = build_pipeline()
        try:
            generate_candidates(pipe, args.run_id)
        finally:
            del pipe
            gc.collect()
            torch.mps.empty_cache()
    elif not args.mark_v2_rejected:
        parser.error("choose --mark-v2-rejected, --generate-candidates, or --finalize")


if __name__ == "__main__":
    main()
