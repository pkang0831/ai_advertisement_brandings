#!/usr/bin/env python3
"""Build a review-only Rina expression/angle reference candidate pack on MPS."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from diffusers import DPMSolverMultistepScheduler, StableDiffusionXLImg2ImgPipeline
from PIL import Image, ImageDraw, ImageFont, ImageOps

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
MASTER = RINA / "identity/master/rina_master_face.jpg"
V3_RUN = RINA / "out/review_samples/20260726T055318Z"
OUTPUT_ROOT = RINA / "identity/expression_reference_candidates"
WIDTH, HEIGHT, STEPS, CFG = 768, 1024, 30, 4.0
IDENTITY_METHOD = (
    "RealVisXL V5 + h94 IP-Adapter Plus Face SDXL ViT-H, CLIP-only; "
    "master-seeded img2img; no face-recognition embeddings"
)
STYLE = (
    "same adult Korean-Canadian woman, long natural black hair, chest-up reference photo, "
    "simple dark crew-neck top, realistic skin, plain subtle background, everyday unretouched light"
)
NEGATIVE = (
    "different person, identity drift, direct gaze unless requested, repeated mild closed smile, "
    "crossed eyes, misaligned pupils, uncanny eyes, exaggerated expression, glam studio, beauty campaign, "
    "heavy makeup, plastic skin, mirror duplicate, swimsuit, cleavage, text, watermark, CGI, child"
)

LABELS = [
    {
        "slug": "01_frontal_neutral", "label": "frontal neutral",
        "yaw": "0 degrees", "pitch": "level", "gaze": "straight ahead",
        "expression": "neutral mouth, relaxed distinct brows",
        "critical": "frontal face, level head, straight gaze, neutral closed mouth, relaxed brows",
        "light": "soft flat window light", "seed": 26072901, "strengths": [0.44, 0.52], "scales": [0.52, 0.46],
    },
    {
        "slug": "02_frontal_teeth_smile", "label": "frontal genuine teeth smile",
        "yaw": "0 degrees", "pitch": "level", "gaze": "straight ahead",
        "expression": "genuine warm teeth smile, lifted cheeks, smiling eyes",
        "critical": "frontal face, genuine warm teeth smile, lifted cheeks, smiling eyes",
        "light": "soft golden window light", "seed": 26072911, "strengths": [0.50, 0.58], "scales": [0.48, 0.42],
    },
    {
        "slug": "03_three_quarter_soft_smile", "label": "three-quarter closed soft smile",
        "yaw": "30 degrees right", "pitch": "level", "gaze": "slightly past camera",
        "expression": "small closed soft smile, calm eyes, gently lifted brows",
        "critical": "right three-quarter face, gaze slightly past camera, small closed soft smile",
        "light": "neutral indoor window light", "seed": 26072921, "strengths": [0.44, 0.52], "scales": [0.50, 0.44],
    },
    {
        "slug": "04_three_quarter_serious", "label": "three-quarter serious",
        "yaw": "35 degrees left", "pitch": "level", "gaze": "off-camera left",
        "expression": "serious closed mouth, focused eyes, one brow subtly tense",
        "critical": "left three-quarter face, off-camera gaze, serious closed mouth, focused eyes",
        "light": "soft overcast flat light", "seed": 26072931, "strengths": [0.50, 0.58], "scales": [0.46, 0.40],
    },
    {
        "slug": "05_left_profile_neutral", "label": "left profile neutral",
        "yaw": "90 degrees left", "pitch": "level", "gaze": "forward in profile",
        "expression": "neutral relaxed mouth and brow",
        "critical": "clean full left profile, nose pointing left, profile gaze, neutral mouth",
        "light": "even natural daylight", "seed": 26072941, "strengths": [0.56, 0.62], "scales": [0.42, 0.38],
    },
    {
        "slug": "06_right_profile_neutral", "label": "right profile neutral",
        "yaw": "90 degrees right", "pitch": "level", "gaze": "forward in profile",
        "expression": "neutral relaxed mouth, distinct natural brow",
        "critical": "clean full right profile, nose pointing right, profile gaze, neutral mouth",
        "light": "warm indoor daylight", "seed": 26072951, "strengths": [0.56, 0.62], "scales": [0.42, 0.38],
    },
    {
        "slug": "07_looking_down_thoughtful", "label": "looking down thoughtful",
        "yaw": "10 degrees left", "pitch": "20 degrees down", "gaze": "eyes down away from camera",
        "expression": "thoughtful closed mouth, softened brow",
        "critical": "chin lowered, eyes looking down, no camera gaze, thoughtful neutral mouth",
        "light": "quiet soft indoor light", "seed": 26072961, "strengths": [0.48, 0.56], "scales": [0.48, 0.42],
    },
    {
        "slug": "08_looking_up_away_curious", "label": "looking up/away curious",
        "yaw": "20 degrees right", "pitch": "15 degrees up", "gaze": "up and away right",
        "expression": "subtle curiosity, gently raised asymmetric brows, parted lips",
        "critical": "chin slightly raised, eyes looking up and away right, curious raised brow, parted lips",
        "light": "late-afternoon golden ambient light", "seed": 26072971, "strengths": [0.52, 0.60], "scales": [0.44, 0.40],
    },
    {
        "slug": "09_over_shoulder_alert", "label": "over-shoulder alert",
        "yaw": "head turned 60 degrees left over right shoulder", "pitch": "level",
        "gaze": "sideways past camera", "expression": "alert eyes, slightly parted lips, raised inner brow",
        "critical": "over right shoulder turn, alert sideways gaze, slightly parted lips, raised inner brow",
        "light": "natural doorway light", "seed": 26072981, "strengths": [0.54, 0.62], "scales": [0.42, 0.38],
    },
    {
        "slug": "10_candid_mid_laugh", "label": "candid mid-laugh",
        "yaw": "20 degrees left", "pitch": "slightly down", "gaze": "off-camera left",
        "expression": "natural mid-laugh, teeth visible, open mouth, uneven joyful cheeks",
        "critical": "candid mid-laugh, teeth visible, mouth naturally open, off-camera gaze, joyful uneven cheeks",
        "light": "warm casual indoor light", "seed": 26072991, "strengths": [0.54, 0.62], "scales": [0.44, 0.38],
    },
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def validate_artifacts() -> None:
    for path, digest in {MODEL: MODEL_SHA256, ADAPTER: ADAPTER_SHA256, ENCODER: ENCODER_SHA256}.items():
        if not path.is_file() or sha256(path) != digest:
            raise RuntimeError(f"approved artifact missing or hash mismatch: {path}")
    if not MASTER.is_file():
        raise RuntimeError(f"master missing: {MASTER}")


def master_image() -> Image.Image:
    with Image.open(MASTER) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
    return ImageOps.fit(image, (WIDTH, HEIGHT), method=Image.Resampling.LANCZOS, centering=(0.5, 0.42))


def image_qc(image: Image.Image) -> dict[str, object]:
    pixels = np.asarray(image.convert("RGB"))
    mean, std = float(pixels.mean()), float(pixels.std())
    black = float((pixels < 3).all(axis=2).mean())
    white = float((pixels > 252).all(axis=2).mean())
    passed = image.size == (WIDTH, HEIGHT) and 5 < mean < 250 and std > 8 and black < 0.95 and white < 0.95
    return {
        "dimensions": list(image.size), "mean_rgb": round(mean, 3), "std_rgb": round(std, 3),
        "black_pixel_fraction": round(black, 6), "white_pixel_fraction": round(white, 6),
        "passed": passed,
    }


def build_pipeline() -> StableDiffusionXLImg2ImgPipeline:
    pipe = StableDiffusionXLImg2ImgPipeline.from_single_file(
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


def token_report(pipe: StableDiffusionXLImg2ImgPipeline, prompt: str) -> dict[str, object]:
    report: dict[str, object] = {}
    for name in ("tokenizer", "tokenizer_2"):
        tokenizer = getattr(pipe, name)
        ids = tokenizer(prompt, truncation=False, add_special_tokens=True).input_ids
        report[name] = {"tokens_with_special": len(ids), "model_max_length": tokenizer.model_max_length}
    report["within_sdxl_clip_limit"] = all(int(value["tokens_with_special"]) <= 77 for value in report.values())
    return report


def clip_embedding(pipe: StableDiffusionXLImg2ImgPipeline, image: Image.Image) -> torch.Tensor:
    pixels = pipe.feature_extractor(images=image, return_tensors="pt").pixel_values
    device = next(pipe.image_encoder.parameters()).device
    dtype = next(pipe.image_encoder.parameters()).dtype
    with torch.no_grad():
        embeds = pipe.image_encoder(pixels.to(device=device, dtype=dtype)).image_embeds
    return F.normalize(embeds.float(), dim=-1).cpu()


def make_contact_sheet(items: list[tuple[str, Path]], output: Path, columns: int = 4) -> None:
    thumb_w, thumb_h, label_h = 300, 400, 72
    rows = math.ceil(len(items) / columns)
    sheet = Image.new("RGB", (columns * thumb_w, rows * (thumb_h + label_h)), "white")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    for index, (label, path) in enumerate(items):
        with Image.open(path) as source:
            image = source.convert("RGB")
            image.thumbnail((thumb_w - 8, thumb_h - 8))
        x = (index % columns) * thumb_w
        y = (index // columns) * (thumb_h + label_h)
        sheet.paste(image, (x + (thumb_w - image.width) // 2, y))
        draw.multiline_text((x + 7, y + thumb_h + 4), label, fill="black", font=font, spacing=2)
    sheet.save(output, quality=92, subsampling=0)


def mark_v3_rejected() -> None:
    reason = (
        "Selected v3 platform samples repeat direct gaze, left three-quarter angle, and a closed mild smile "
        "despite mirrored/composition changes; build an approved identity expression/angle reference pack first."
    )
    updated = 0
    for sidecar in sorted(V3_RUN.glob("*.jpg.json")):
        payload = json.loads(sidecar.read_text(encoding="utf-8"))
        if payload.get("review_status") != "selected_for_v3_review":
            continue
        payload["review_status"] = "rejected_needs_identity_reference_diversity"
        payload["rejection_reason"] = reason
        sidecar.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        updated += 1
    summary_path = V3_RUN / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    for result in summary.get("results", []):
        if result.get("review_status") == "selected_for_v3_review":
            result["review_status"] = "rejected_needs_identity_reference_diversity"
            result["rejection_reason"] = reason
    summary["platform_v3_review_status"] = "rejected_needs_identity_reference_diversity"
    summary["platform_v3_rejection_reason"] = reason
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"v3_selected_sidecars_updated": updated}), flush=True)


def generate(run_id: str) -> Path:
    run = OUTPUT_ROOT / run_id
    candidates = run / "candidates"
    candidates.mkdir(parents=True, exist_ok=False)
    pipe = build_pipeline()
    init = master_image()
    master_embed = clip_embedding(pipe, init)
    all_items: list[tuple[str, Path]] = []
    results: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []
    try:
        for label_index, spec in enumerate(LABELS, start=1):
            for candidate_index in (1, 2):
                strength = float(spec["strengths"][candidate_index - 1])
                scale = float(spec["scales"][candidate_index - 1])
                seed = int(spec["seed"]) + candidate_index - 1
                prompt = f"{spec['critical']}, {STYLE}, {spec['light']}"
                tokens = token_report(pipe, prompt)
                if not tokens["within_sdxl_clip_limit"]:
                    raise RuntimeError(f"prompt exceeds CLIP limit for {spec['slug']}: {tokens}")
                pipe.set_ip_adapter_scale(scale)
                output = candidates / f"{spec['slug']}_c{candidate_index}.jpg"
                started = time.monotonic()
                try:
                    image = pipe(
                        prompt=prompt, negative_prompt=NEGATIVE, image=init,
                        ip_adapter_image=init, strength=strength,
                        num_inference_steps=STEPS, guidance_scale=CFG,
                        generator=torch.Generator(device="cpu").manual_seed(seed),
                    ).images[0]
                    qc = image_qc(image)
                    if not qc["passed"]:
                        raise RuntimeError(f"image sanity failed: {qc}")
                    image.save(output, quality=95, subsampling=0)
                    similarity = float(F.cosine_similarity(master_embed, clip_embedding(pipe, image)).item())
                    payload = {
                        "schema_version": 1, "review_status": "candidate_not_approved",
                        "label": spec["label"], "label_slug": spec["slug"], "candidate_index": candidate_index,
                        "yaw": spec["yaw"], "pitch": spec["pitch"], "gaze": spec["gaze"],
                        "expression": spec["expression"], "prompt": prompt, "prompt_tokens": tokens,
                        "negative_prompt": NEGATIVE, "seed": seed, "denoise_strength": strength,
                        "ip_adapter_scale": scale, "identity_method": IDENTITY_METHOD,
                        "identity_reference": str(MASTER), "identity_reference_sha256": sha256(MASTER),
                        "model": MODEL.name, "model_sha256": MODEL_SHA256,
                        "adapter": ADAPTER.name, "adapter_sha256": ADAPTER_SHA256,
                        "adapter_revision": ADAPTER_REVISION, "clip_vision_sha256": ENCODER_SHA256,
                        "clip_image_cosine_to_master_non_face_metric": round(similarity, 6),
                        "device": "mps", "pipeline_dtype": "float32", "steps": STEPS, "cfg": CFG,
                        "generation_seconds": round(time.monotonic() - started, 3), "qc": qc,
                        "manual_qc": {
                            "identity_consistent": None, "angle_matches": None, "gaze_matches": None,
                            "expression_matches": None, "eyes_natural": None, "selection_reason": None,
                        },
                        "production_status": "not_promoted_pending_user_approval",
                    }
                    output.with_suffix(".jpg.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
                    results.append(payload)
                    all_items.append((f"{label_index:02d} c{candidate_index}: {spec['label']}\nstr {strength:.2f} / ip {scale:.2f}", output))
                    print(json.dumps({"saved": str(output), "clip_cosine": similarity, "qc": qc}), flush=True)
                except Exception as exc:
                    failure = {
                        "label": spec["label"], "label_slug": spec["slug"], "candidate_index": candidate_index,
                        "seed": seed, "denoise_strength": strength, "ip_adapter_scale": scale,
                        "error_type": type(exc).__name__, "error": str(exc),
                    }
                    failures.append(failure)
                    (run / "failures.json").write_text(json.dumps(failures, indent=2), encoding="utf-8")
                    print(json.dumps({"failure": failure}), flush=True)
                gc.collect()
                torch.mps.empty_cache()
    finally:
        del pipe
        gc.collect()
        torch.mps.empty_cache()
    make_contact_sheet(all_items, run / "contact_sheet_all_candidates.jpg")
    summary = {
        "run_id": run_id, "schema_version": 1, "purpose": "identity expression/angle reference candidates",
        "review_only": True, "platform_samples_generated": False, "mature_content_generated": False,
        "generated_candidate_count": len(results), "failed_generation_count": len(failures),
        "selected_count": 0, "selected_labels": [], "master_replaced": False,
        "production_references_promoted": False, "identity_method": IDENTITY_METHOD,
        "candidate_contact_sheet": str(run / "contact_sheet_all_candidates.jpg"),
        "selected_contact_sheet": None, "results": results, "generation_failures": failures,
    }
    (run / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return run


def finalize(run_id: str, reviews_path: Path) -> None:
    run = OUTPUT_ROOT / run_id
    reviews = json.loads(reviews_path.read_text(encoding="utf-8"))
    selections = reviews["selections"]
    selected_items: list[tuple[str, Path]] = []
    reviewed_results: list[dict[str, object]] = []
    failures: list[dict[str, object]] = reviews.get("label_failures", [])
    selected_labels: list[str] = []
    for spec in LABELS:
        slug = str(spec["slug"])
        selected = selections.get(slug)
        for candidate_index in (1, 2):
            path = run / "candidates" / f"{slug}_c{candidate_index}.jpg"
            if not path.is_file():
                continue
            sidecar = path.with_suffix(".jpg.json")
            payload = json.loads(sidecar.read_text(encoding="utf-8"))
            review = reviews["candidate_reviews"][slug][str(candidate_index)]
            payload["manual_qc"] = review
            payload["review_status"] = "selected_pending_user_approval" if selected == candidate_index else "rejected"
            sidecar.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            reviewed_results.append(payload)
        if selected is None:
            continue
        source = run / "candidates" / f"{slug}_c{selected}.jpg"
        destination = run / f"{slug}.jpg"
        shutil.copy2(source, destination)
        selected_payload = json.loads(source.with_suffix(".jpg.json").read_text(encoding="utf-8"))
        selected_payload["output"] = str(destination)
        selected_payload["review_status"] = "selected_pending_user_approval"
        destination.with_suffix(".jpg.json").write_text(json.dumps(selected_payload, indent=2), encoding="utf-8")
        selected_labels.append(str(spec["label"]))
        selected_items.append((str(spec["label"]), destination))
    selected_sheet = run / "contact_sheet_selected_pending_approval.jpg"
    if selected_items:
        make_contact_sheet(selected_items, selected_sheet, columns=4)
    summary_path = run / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary.update({
        "selected_count": len(selected_items), "selected_labels": selected_labels,
        "selected_contact_sheet": str(selected_sheet) if selected_items else None,
        "manual_review_method": reviews.get("manual_review_method"),
        "label_failures": failures, "review_assessment": reviews.get("assessment"),
        "results": reviewed_results,
        "production_references_promoted": False, "pending_user_approval": True,
    })
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
    parser.add_argument("--mark-v3-rejected", action="store_true")
    parser.add_argument("--generate", action="store_true")
    parser.add_argument("--finalize", action="store_true")
    parser.add_argument("--reviews", type=Path)
    args = parser.parse_args()
    validate_artifacts()
    if args.mark_v3_rejected:
        mark_v3_rejected()
    if args.generate:
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS is required")
        run = generate(args.run_id)
        print(json.dumps({"run": str(run)}), flush=True)
    if args.finalize:
        if not args.reviews:
            parser.error("--finalize requires --reviews")
        finalize(args.run_id, args.reviews)
    if not (args.mark_v3_rejected or args.generate or args.finalize):
        parser.error("choose an action")


if __name__ == "__main__":
    main()
