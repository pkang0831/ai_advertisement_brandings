#!/usr/bin/env python3
"""Regenerate one face_lock index (1-based) in a fresh process."""
import gc
import sys
from pathlib import Path

import numpy as np
import torch
from diffusers import DPMSolverMultistepScheduler, StableDiffusionXLPipeline

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from generate_track_smoke import MODEL, RINA, face_lock_rows  # noqa: E402


def main(idx: int) -> None:
    row = face_lock_rows().iloc[idx - 1]
    print("gen", row.image_filename)
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
    seed = int(row.seed) + 17
    img = pipe(
        prompt=str(row.prompt),
        negative_prompt=str(row.negative_prompt),
        width=832,
        height=1216,
        num_inference_steps=12,
        guidance_scale=4.5,
        generator=torch.Generator("cpu").manual_seed(seed),
    ).images[0]
    mean = float(np.array(img).mean())
    print("mean", mean)
    if mean < 5:
        raise SystemExit("still black")
    out = RINA / row.output_subdir / row.image_filename
    out.parent.mkdir(parents=True, exist_ok=True)
    img.save(out, quality=95)
    print("saved", out)
    del pipe
    gc.collect()
    torch.mps.empty_cache()


if __name__ == "__main__":
    main(int(sys.argv[1]))
