#!/usr/bin/env python3
"""Quality IG pool plog: RealVisXL V5 full + realism/phone LoRAs (CUDA/MPS)."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from diffusers import DPMSolverMultistepScheduler, StableDiffusionXLPipeline
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from rina_park.runtime_device import empty_cache, require_accelerator  # noqa: E402

RINA = ROOT / "rina_park"
MODEL = RINA / "models" / "checkpoints" / "RealVisXL_V5.0_fp16.safetensors"
LORAS = RINA / "models" / "loras"
OUT = RINA / "out" / "ig" / "rina_ig_0001_quality.jpg"

PROMPT = (
    "devenira_rina_park, beautiful adult Korean-American woman, fair skin, "
    "long straight black hair, natural facial detail, relaxed closed mouth, "
    "candid smartphone photo, busy public indoor pool, navy athletic one-piece swimsuit, "
    "friend-taken standing portrait, slight imperfect crop, other swimmers blurred in background, "
    "flat facility lighting, natural skin texture, pores, realistic photo"
)
NEGATIVE = (
    "underage, teen, babyface, nude, explicit, plastic skin, beauty filter, "
    "studio lighting, softbox, model catalog pose, oversmooth skin, CGI, painting, "
    "illustration, anime, watermark, text, bad hands, deformed face, oversaturated"
)


def main() -> None:
    assert MODEL.exists(), MODEL
    device = require_accelerator()
    print("device:", device)

    t0 = time.time()
    print("Loading", MODEL.name)
    pipe = StableDiffusionXLPipeline.from_single_file(
        str(MODEL), torch_dtype=torch.float16, use_safetensors=True
    )
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(
        pipe.scheduler.config,
        use_karras_sigmas=True,
        algorithm_type="sde-dpmsolver++",
    )
    pipe.enable_attention_slicing()
    pipe.to(device)
    pipe.vae.to(dtype=torch.float32)

    # Adapter stack — keep weights modest to avoid AI gloss
    adapters = []
    weights = []
    # Prefer SDXL-native LoRAs; ZIT/Instax variants often fail to inject
    lora_specs = [
        ("skin_realism_sdxl.safetensors", "skin", 0.50),
        ("RealSkin_xxXL_v1.safetensors", "realskin", 0.60),
        ("SDXL_FILM_PHOTOGRAPHY_STYLE_V1.safetensors", "film", 0.35),
        ("iphone_mirror_selfie_v01b.safetensors", "iphone", 0.35),
        ("add_detail_xl.safetensors", "detail", 0.22),
    ]
    for fname, name, w in lora_specs:
        path = LORAS / fname
        if not path.exists():
            print("skip missing", fname)
            continue
        try:
            print(f"load LoRA {fname} @ {w}")
            pipe.load_lora_weights(str(path), adapter_name=name)
            adapters.append(name)
            weights.append(w)
        except Exception as e:
            print(f"skip incompatible {fname}: {type(e).__name__}: {e}")
    if adapters:
        pipe.set_adapters(adapters, adapter_weights=weights)

    seed = 20300101
    gen = torch.Generator(device="cpu").manual_seed(seed)
    print("Generating…")
    image: Image.Image = pipe(
        prompt=PROMPT,
        negative_prompt=NEGATIVE,
        width=832,
        height=1216,
        num_inference_steps=32,
        guidance_scale=4.2,
        generator=gen,
    ).images[0]

    mean = float(np.array(image).mean())
    if mean < 5:
        raise RuntimeError(f"black frame mean={mean}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    image.save(OUT, quality=95)
    meta = {
        "output": str(OUT),
        "model": MODEL.name,
        "seed": seed,
        "steps": 32,
        "cfg": 4.2,
        "loras": dict(zip(adapters, weights)),
        "elapsed_sec": round(time.time() - t0, 1),
        "prompt": PROMPT,
        "negative": NEGATIVE,
        "note": "quality pass; FaceID skipped (insightface not installed)",
    }
    OUT.with_suffix(".jpg.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print("saved", OUT, f"mean={mean:.1f}", f"elapsed={meta['elapsed_sec']}s")
    del pipe
    empty_cache()


if __name__ == "__main__":
    main()
