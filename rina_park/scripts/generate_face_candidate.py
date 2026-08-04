#!/usr/bin/env python3
"""Generate one photorealistic Rina face candidate in a fresh MPS process."""
from __future__ import annotations

import gc
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from diffusers import DPMSolverMultistepScheduler, StableDiffusionXLPipeline

ROOT = Path(__file__).resolve().parents[2]
RINA = ROOT / "rina_park"
MODEL = RINA / "models/checkpoints/juggernautXL_ragnarok.safetensors"
LORAS = RINA / "models/loras"
OUT_DIR = RINA / "identity/beauty_candidates"

PROMPT = (
    "RAW candid smartphone photo, strikingly beautiful adult Korean fitness influencer, age 27, "
    "delicate heart-shaped oval face, high cheekbones, softly tapered jaw, large elegant almond eyes, "
    "refined straight nose, softly full lips, fair skin with natural pores, subtle Korean makeup, "
    "long glossy black hair, navy one-piece swimsuit, public indoor pool, soft daylight, three-quarter view"
)
NEGATIVE = (
    "underage, teen, child, average plain face, masculine face, broad jaw, long face, tired face, old-looking, "
    "doll face, anime, plastic surgery, overfilled lips, heavy makeup, beauty filter, waxy skin, airbrushed, "
    "studio portrait, CGI, painting, oversaturated, harsh HDR, deformed face, bad anatomy, text, watermark"
)


def main(index: int) -> None:
    seed = 20520000 + index * 193
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"rina_glam_candidate_{index:02d}.jpg"

    print(f"candidate={index} seed={seed}")
    started = time.time()
    pipe = StableDiffusionXLPipeline.from_single_file(
        str(MODEL), torch_dtype=torch.float16, use_safetensors=True
    )
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(
        pipe.scheduler.config,
        use_karras_sigmas=True,
        algorithm_type="sde-dpmsolver++",
    )
    pipe.enable_attention_slicing()
    pipe.to("mps")
    pipe.vae.to(dtype=torch.float32)

    loaded_names: list[str] = []
    loaded_weights: list[float] = []
    specs = [
        ("skin_realism_sdxl.safetensors", "skin", 0.32),
        ("RealSkin_xxXL_v1.safetensors", "realskin", 0.38),
        ("iphone_mirror_selfie_v01b.safetensors", "iphone", 0.18),
        ("add_detail_xl.safetensors", "detail", 0.12),
    ]
    for filename, adapter, weight in specs:
        path = LORAS / filename
        if not path.exists():
            continue
        try:
            pipe.load_lora_weights(str(path), adapter_name=adapter)
            loaded_names.append(adapter)
            loaded_weights.append(weight)
            print(f"LoRA {adapter}={weight}")
        except Exception as exc:
            print(f"skip {filename}: {type(exc).__name__}")
    if loaded_names:
        pipe.set_adapters(loaded_names, adapter_weights=loaded_weights)

    image = pipe(
        prompt=PROMPT,
        negative_prompt=NEGATIVE,
        width=832,
        height=1216,
        num_inference_steps=32,
        guidance_scale=3.7,
        generator=torch.Generator(device="cpu").manual_seed(seed),
    ).images[0]
    mean = float(np.asarray(image).mean())
    if mean < 5:
        raise RuntimeError(f"black frame: mean={mean}")
    image.save(out, quality=96)
    out.with_suffix(".jpg.json").write_text(
        json.dumps(
            {
                "candidate": index,
                "seed": seed,
                "model": MODEL.name,
                "loras": dict(zip(loaded_names, loaded_weights)),
                "steps": 32,
                "cfg": 3.7,
                "prompt": PROMPT,
                "negative_prompt": NEGATIVE,
                "elapsed_sec": round(time.time() - started, 1),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"saved={out} mean={mean:.1f}")
    del pipe
    gc.collect()
    torch.mps.empty_cache()


if __name__ == "__main__":
    main(int(sys.argv[1]))
