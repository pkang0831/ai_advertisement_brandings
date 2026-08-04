#!/usr/bin/env python3
"""Local MPS smoke test: one Rina Park image from image_prompt_queue.csv."""
from __future__ import annotations

import gc
import json
import time
from pathlib import Path

import pandas as pd
import torch
from diffusers import DPMSolverMultistepScheduler, StableDiffusionXLPipeline
from PIL import Image

ROOT = Path(__file__).resolve().parent
MODEL_FILE = ROOT / "realvisxlV50_v50LightningBakedvae.safetensors"
QUEUE_PATH = ROOT / "image_prompt_queue.csv"
OUT_DIR = ROOT / "local_gen"
OUT_PATH = OUT_DIR / "rina_park_RIN_0001_cover_hook.jpg"

QUALITY_PROMPT_ADDON = (
    "natural facial detail, relaxed lips, closed mouth, clean skin texture"
)
QUALITY_NEGATIVE_ADDON = (
    "bad teeth, distorted teeth, extra teeth, open mouth, warped mouth, bad lips, "
    "asymmetrical mouth, melted skin, lumpy skin, sagging skin, loose skin folds, "
    "deformed mouth, dark skin, tan skin, brown skin, deep brown skin, warm brown skin, "
    "olive skin, Black woman, Indian woman, South Asian woman"
)
CLIP_TOKEN_LIMIT = 77


def inject_prompt_addon(prompt: str, addon: str) -> str:
    addon = addon.strip()
    if not addon:
        return prompt
    anchor = "beautiful adult face,"
    if anchor in prompt:
        return prompt.replace(anchor, f"{anchor} {addon},", 1)
    return f"{addon}, {prompt}"


def truncate_for_clip(text: str, tokenizers) -> str:
    value = str(text)
    for _ in range(3):
        changed = False
        for tokenizer in tokenizers:
            ids = tokenizer(value, truncation=False).input_ids
            if len(ids) > CLIP_TOKEN_LIMIT:
                value = tokenizer.decode(ids[:CLIP_TOKEN_LIMIT], skip_special_tokens=True).strip()
                changed = True
        if not changed:
            break
    return value


def main() -> None:
    if not MODEL_FILE.exists():
        raise FileNotFoundError(MODEL_FILE)
    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS not available")

    df = pd.read_csv(QUEUE_PATH)
    row = df[(df["account_slug"] == "rina_park") & (df["slide_role"] == "cover_hook")].iloc[0]

    prompt = inject_prompt_addon(str(row["prompt"]), QUALITY_PROMPT_ADDON)
    negative = f"{QUALITY_NEGATIVE_ADDON}, {row['negative_prompt']}"
    seed = int(row["seed"])
    width = int(row["width"])
    height = int(row["height"])
    steps = int(row["steps"])
    cfg = float(row["cfg"])

    print("character:", row["assigned_character"])
    print("production_id:", row["production_id"], "role:", row["slide_role"])
    print("seed/size/steps/cfg:", seed, f"{width}x{height}", steps, cfg)
    print("model:", MODEL_FILE.name)

    device = "mps"
    dtype = torch.float16

    t0 = time.time()
    print("Loading SDXL pipeline...")
    pipe = StableDiffusionXLPipeline.from_single_file(
        str(MODEL_FILE),
        torch_dtype=dtype,
        use_safetensors=True,
    )
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(
        pipe.scheduler.config,
        use_karras_sigmas=True,
        algorithm_type="sde-dpmsolver++",
    )
    pipe.enable_attention_slicing()
    pipe.to(device)

    tokenizers = [pipe.tokenizer]
    if getattr(pipe, "tokenizer_2", None) is not None:
        tokenizers.append(pipe.tokenizer_2)
    prompt = truncate_for_clip(prompt, tokenizers)
    negative = truncate_for_clip(negative, tokenizers)
    print("prompt:", prompt)
    print("negative (head):", negative[:160], "...")

    generator = torch.Generator(device="cpu").manual_seed(seed)
    print("Generating...")
    image: Image.Image = pipe(
        prompt=prompt,
        negative_prompt=negative,
        width=width,
        height=height,
        num_inference_steps=steps,
        guidance_scale=cfg,
        generator=generator,
    ).images[0]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    image.save(OUT_PATH, quality=95)
    meta = {
        "production_id": row["production_id"],
        "assigned_character": row["assigned_character"],
        "slide_role": row["slide_role"],
        "seed": seed,
        "width": width,
        "height": height,
        "steps": steps,
        "cfg": cfg,
        "prompt": prompt,
        "negative_prompt": negative,
        "model_file": str(MODEL_FILE),
        "device": device,
        "elapsed_sec": round(time.time() - t0, 1),
        "output_path": str(OUT_PATH),
        "note": "local MPS smoke test; face detailer + upscale skipped",
    }
    OUT_PATH.with_suffix(".jpg.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print("saved:", OUT_PATH)
    print("elapsed_sec:", meta["elapsed_sec"])

    del pipe
    gc.collect()


if __name__ == "__main__":
    main()
