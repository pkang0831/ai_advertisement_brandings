"""Shared RealVisXL + Rina person LoRA stack for anatomy-lock runs (MPS)."""

from __future__ import annotations

import gc
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import torch
from diffusers import (
    ControlNetModel,
    DPMSolverMultistepScheduler,
    StableDiffusionXLControlNetImg2ImgPipeline,
    StableDiffusionXLImg2ImgPipeline,
    StableDiffusionXLPipeline,
)

RINA = Path(__file__).resolve().parents[2]
MODEL = RINA / "models" / "checkpoints" / "RealVisXL_V5.0_fp16.safetensors"
LORA_DIR = RINA / "models" / "loras"
PERSON_LORA = LORA_DIR / "rina_park_person_sdxl_lora.safetensors"
OPENPOSE_CN = RINA / "models" / "controlnet" / "OpenPoseXL2.safetensors"
DEPTH_CN_DIR = RINA / "models" / "controlnet" / "depth"

# Specialty anatomy LoRAs (prefer snake_case symlinks; fall back to on-disk names)
HAND_DETAIL_LORA = LORA_DIR / "hand_detail_xl_v2.safetensors"
BETTER_HANDS_LORA = LORA_DIR / "better_hands_sdxl_v1.safetensors"
REAL_FEET_LORA = LORA_DIR / "real_feet_xl_v1.safetensors"
# Upstream filenames if symlinks missing
_HAND_DETAIL_UPSTREAM = LORA_DIR / "detailed_hands-000002.safetensors"
_BETTER_HANDS_UPSTREAM = LORA_DIR / "Better Hands SDXL v1.0.safetensors"
_REAL_FEET_UPSTREAM = LORA_DIR / "RealFeet_xl_v1.safetensors"


@dataclass(frozen=True)
class ExtraLora:
    """Additional LoRA stacked under the character adapter."""

    path: Path
    name: str
    weight: float
    trigger: str = ""


def resolve_lora_path(path: Path | str) -> Path:
    p = Path(path)
    if not p.is_absolute():
        p = LORA_DIR / p
    if p.exists():
        return p.resolve()
    raise FileNotFoundError(p)


def known_specialty_loras() -> dict[str, Path]:
    """Map short ids → resolved paths (symlink preferred)."""
    pairs = {
        "detailed_hands": (HAND_DETAIL_LORA, _HAND_DETAIL_UPSTREAM),
        "better_hands": (BETTER_HANDS_LORA, _BETTER_HANDS_UPSTREAM),
        "real_feet": (REAL_FEET_LORA, _REAL_FEET_UPSTREAM),
    }
    out: dict[str, Path] = {}
    for key, (preferred, upstream) in pairs.items():
        if preferred.exists():
            out[key] = preferred.resolve()
        elif upstream.exists():
            out[key] = upstream.resolve()
    return out


def _configure_scheduler(pipe: Any) -> None:
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(
        pipe.scheduler.config,
        use_karras_sigmas=True,
        algorithm_type="sde-dpmsolver++",
    )


def _unet_only_lora_state(path: Path) -> dict[str, Any]:
    """Kohya TE+UNet LoRAs break multi-adapter TE load under character LoRA.

    Keep UNet keys only so specialty adapters stack cleanly via PEFT.
    """
    from safetensors.torch import load_file

    state = load_file(str(path))
    unet = {
        k: v
        for k, v in state.items()
        if ("lora_unet" in k) or k.startswith("unet.") or ".unet." in k
    }
    if not unet:
        # Already UNet-only / diffusers format — pass through
        return state
    return unet


def apply_lora_stack(
    pipe: Any,
    *,
    lora_scale: float = 0.90,
    extra_loras: Sequence[ExtraLora] = (),
) -> list[str]:
    """Load/set character + optional specialty adapters. Returns active names."""
    names: list[str] = []
    weights: list[float] = []
    if PERSON_LORA.exists():
        pipe.load_lora_weights(str(PERSON_LORA), adapter_name="rina_person")
        names.append("rina_person")
        weights.append(float(lora_scale))
    for extra in extra_loras:
        path = resolve_lora_path(extra.path)
        # Multi-adapter: load specialty as UNet-only to avoid TE rank IndexError
        state = _unet_only_lora_state(path)
        pipe.load_lora_weights(state, adapter_name=extra.name)
        names.append(extra.name)
        weights.append(float(extra.weight))
    if names:
        pipe.set_adapters(names, adapter_weights=weights)
    return names


def load_txt2img(
    *,
    lora_scale: float = 0.90,
    device: str = "mps",
    extra_loras: Sequence[ExtraLora] = (),
) -> StableDiffusionXLPipeline:
    assert MODEL.exists(), MODEL
    assert torch.backends.mps.is_available(), "MPS required"
    pipe = StableDiffusionXLPipeline.from_single_file(
        str(MODEL), torch_dtype=torch.float16, use_safetensors=True
    )
    _configure_scheduler(pipe)
    pipe.enable_attention_slicing()
    pipe.to(device)
    pipe.vae.to(dtype=torch.float32)
    apply_lora_stack(pipe, lora_scale=lora_scale, extra_loras=extra_loras)
    return pipe


def load_img2img_from(txt2img: StableDiffusionXLPipeline) -> StableDiffusionXLImg2ImgPipeline:
    pipe = StableDiffusionXLImg2ImgPipeline(**txt2img.components)
    _configure_scheduler(pipe)
    return pipe


def load_controlnet_img2img(
    txt2img: StableDiffusionXLPipeline,
    *,
    kind: str = "openpose",
    device: str = "mps",
) -> StableDiffusionXLControlNetImg2ImgPipeline:
    if kind == "openpose":
        controlnet = ControlNetModel.from_single_file(
            str(OPENPOSE_CN), torch_dtype=torch.float16
        )
    elif kind == "depth":
        controlnet = ControlNetModel.from_pretrained(
            str(DEPTH_CN_DIR), torch_dtype=torch.float16
        )
    else:
        raise ValueError(kind)
    controlnet.to(device)
    pipe = StableDiffusionXLControlNetImg2ImgPipeline(
        vae=txt2img.vae,
        text_encoder=txt2img.text_encoder,
        text_encoder_2=txt2img.text_encoder_2,
        tokenizer=txt2img.tokenizer,
        tokenizer_2=txt2img.tokenizer_2,
        unet=txt2img.unet,
        controlnet=controlnet,
        scheduler=txt2img.scheduler,
    )
    pipe.enable_attention_slicing()
    pipe.to(device)
    return pipe


def unload(*pipes: Any) -> None:
    for pipe in pipes:
        if pipe is None:
            continue
        try:
            del pipe
        except Exception:  # noqa: BLE001
            pass
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
