#@title 5) Sample — PRIMARY next step (RealVis + soft glam; no retrain)
# 학습 이미 끝난 뒤: Config(0) → Mount(1) → 이 셀만.
# torchao uninstall guard → RealVisXL V5→V4→base → LORA_SCALE default 0.80
# Target: soft glam photoreal (clear skin + beauty light; not plastic, not raw pores)
# Scale sweep: 0.65 / 0.75 / 0.85 / 0.95  (more photo ← → more glam)
# Optional: SAMPLE_MID_CKPTS samples default scale on step 800/1200 if present.

from __future__ import annotations

import os
import re
import subprocess
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", message=".*FrozenDict.*")
warnings.filterwarnings("ignore", message=".*flax.*", category=DeprecationWarning)
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
os.environ.setdefault("DIFFUSERS_FORCE_TORCH", "1")


def _torchao_version() -> str | None:
    try:
        import importlib.metadata as md
        return md.version("torchao")
    except Exception:
        pass
    try:
        import torchao  # noqa: F401
        return getattr(sys.modules["torchao"], "__version__", "unknown")
    except Exception:
        return None


def _parse_ver(v: str) -> tuple:
    parts = []
    for p in v.replace("a", ".").replace("b", ".").replace("rc", ".").split("."):
        try:
            parts.append(int(p))
        except ValueError:
            break
    return tuple(parts) if parts else (0,)


def fix_torchao(*, prefer_uninstall: bool = True) -> bool:
    """Return True if caller should Restart session before continuing."""
    ver = _torchao_version()
    if ver is None:
        print("torchao: not installed — OK (peft will skip torchao dispatcher)")
        return False

    ok = _parse_ver(ver) >= (0, 16, 0)
    print(f"torchao found: {ver}  (need >0.16.0 / >=0.16.0)")
    if ok:
        print("torchao version OK")
        return False

    if prefer_uninstall:
        print(">>> safest Colab fix: uninstall torchao (old package breaks peft.load_lora)")
        print("    alternative: pip install -U 'torchao>=0.16.0'")
        subprocess.check_call([sys.executable, "-m", "pip", "uninstall", "-y", "torchao"])
    else:
        print(">>> upgrading torchao>=0.16.0")
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-q", "-U", "torchao>=0.16.0"]
        )
        ver2 = _torchao_version()
        print("torchao now:", ver2)
        if ver2 and _parse_ver(ver2) >= (0, 16, 0):
            return "torchao" in sys.modules
        return False

    ver_after = _torchao_version()
    still_old = ver_after is not None and _parse_ver(ver_after) < (0, 16, 0)
    if "torchao" in sys.modules or still_old:
        print(
            "\n*** RESTART REQUIRED ***\n"
            "Runtime → Restart session\n"
            "Then re-run: Config (0) → Mount (1) → THIS sample cell only.\n"
            "Do NOT retrain.\n"
        )
        return True

    print("torchao removed — continuing in this session")
    return False


need_restart = fix_torchao(prefer_uninstall=True)
if need_restart:
    raise SystemExit("Stop here → Restart session → Config + Mount + this cell (no retrain)")

WORK = Path(globals().get("WORK", "/content/rina_lora_work"))
OUT_DIR = Path(
    globals().get(
        "OUT_DIR",
        "/content/drive/MyDrive/rina_lora/outputs/rina_park_person_sdxl_lora",
    )
)
TRAIN_BASE_MODEL = globals().get(
    "BASE_MODEL", "stabilityai/stable-diffusion-xl-base-1.0"
)
OUTPUT_NAME = globals().get("OUTPUT_NAME", "rina_park_person_sdxl_lora")
TRIGGER = globals().get("TRIGGER", "rina_park_person")
MIXED_PRECISION = globals().get("MIXED_PRECISION", "auto")
SEED = int(globals().get("SEED", 42))

#@markdown ### Soft glam sample knobs (lower scale = more RealVis raw; higher = more trained glam)
LORA_SCALE = 0.80  #@param {type:"slider", min:0.35, max:1.0, step:0.05}
NUM_INFERENCE_STEPS = 36  #@param {type:"slider", min:20, max:50, step:1}
GUIDANCE_SCALE = 7.0  #@param {type:"slider", min:4.0, max:9.0, step:0.5}
INFER_MODEL = ""  #@param {type:"string"}
RUN_SCALE_SWEEP = True  #@param {type:"boolean"}
SAMPLE_MID_CKPTS = True  #@param {type:"boolean"}
LORA_SCALE = float(LORA_SCALE)
NUM_INFERENCE_STEPS = int(NUM_INFERENCE_STEPS)
GUIDANCE_SCALE = float(GUIDANCE_SCALE)
INFER_MODEL = str(INFER_MODEL or "").strip()
RUN_SCALE_SWEEP = bool(RUN_SCALE_SWEEP)
SAMPLE_MID_CKPTS = bool(SAMPLE_MID_CKPTS)

PREFERRED_INFER_MODELS = list(
    globals().get(
        "SAMPLE_BASE_MODEL_PREFERENCE",
        [
            "SG161222/RealVisXL_V5.0",
            "SG161222/RealVisXL_V4.0",
        ],
    )
)
# Drop base from preference list used for "photoreal" probes; keep as final fallback only
PREFERRED_INFER_MODELS = [
    m for m in PREFERRED_INFER_MODELS if "stable-diffusion-xl-base" not in m
] or [
    "SG161222/RealVisXL_V5.0",
    "SG161222/RealVisXL_V4.0",
]
# more photo (RealVis raw) ← → more glam (trained LoRA beauty)
SCALE_SWEEP = (0.65, 0.75, 0.85, 0.95)
SCALE_SWEEP_LABELS = (
    "0.65 more photo",
    "0.75",
    "0.85 soft glam",
    "0.95 more glam",
)
MID_CKPT_STEPS = (800, 1200)  # also looks for 400; documents 400/800/1200

import torch
from diffusers import StableDiffusionXLPipeline
from PIL import Image, ImageDraw, ImageFont

if MIXED_PRECISION == "auto":
    MIXED_PRECISION = (
        "bf16" if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else "fp16"
    )

lora_path = OUT_DIR / f"{OUTPUT_NAME}.safetensors"
assert lora_path.is_file(), f"Missing LoRA (train already done?): {lora_path}"

SAMPLE_PROMPTS = [
    (
        f"{TRIGGER}, beautiful Korean-Canadian influencer, soft glam photoreal portrait "
        f"on soft beach daylight, clear even skin, soft beauty lighting, flattering light, "
        f"attractive features, subtle soft glam makeup, long brown hair, looking at viewer, "
        f"photographic, natural photo quality"
    ),
    (
        f"{TRIGGER}, medium shot beach walk, soft glam beauty photography, "
        f"clear even skin, soft diffused beauty light, flattering, attractive, "
        f"subtle makeup, photorealistic influencer look"
    ),
    (
        f"{TRIGGER}, close-up soft glam portrait, window beauty light, "
        f"clear even skin, flattering soft focus glow, attractive features, "
        f"subtle makeup, photographic (not plastic doll)"
    ),
]
NEG = (
    "plastic skin, doll face, airbrushed, overly smooth skin, CGI, 3d render, "
    "wax figure, porcelain skin, beauty filter, instagram face, "
    "acne, rash, heavy pores, blotchy skin, weathered skin, extreme skin detail, "
    "ugly, unattractive, "
    "deformed hands, extra limbs, lowres, blurry face, different person, watermark"
)


def _resolve_infer_model() -> tuple[str, bool]:
    if INFER_MODEL:
        return INFER_MODEL, "realvis" in INFER_MODEL.lower() or "photoreal" in INFER_MODEL.lower()
    try:
        from huggingface_hub import model_info
        for mid in PREFERRED_INFER_MODELS:
            try:
                model_info(mid, token=os.environ.get("HF_TOKEN") or True)
                return mid, True
            except Exception as e:
                print(f"skip {mid}: {type(e).__name__}: {e}")
    except Exception as e:
        print(f"huggingface_hub probe skipped ({e}); will try from_pretrained order")
    if PREFERRED_INFER_MODELS:
        return PREFERRED_INFER_MODELS[0], True
    return TRAIN_BASE_MODEL, False


def _load_pipe(model_id: str, dtype: torch.dtype):
    errors: list[str] = []
    try:
        return StableDiffusionXLPipeline.from_pretrained(
            model_id, torch_dtype=dtype, use_safetensors=True
        )
    except Exception as e:
        errors.append(f"from_pretrained: {e}")
    try:
        from huggingface_hub import hf_hub_download
        for fname in (
            "RealVisXL_V5.0_fp16.safetensors",
            "RealVisXL_V5.0.safetensors",
            "realvisxlV50_v5.safetensors",
            "RealVisXL_V4.0.safetensors",
        ):
            try:
                path = hf_hub_download(
                    model_id, fname, token=os.environ.get("HF_TOKEN") or True
                )
                print(f"from_single_file: {model_id}/{fname}")
                return StableDiffusionXLPipeline.from_single_file(path, torch_dtype=dtype)
            except Exception as e2:
                errors.append(f"{fname}: {e2}")
    except Exception as e:
        errors.append(f"hf_hub_download: {e}")
    raise RuntimeError(" ; ".join(errors[:4]))


def _set_lora_scale(pipe, scale: float) -> dict:
    try:
        names = pipe.get_active_adapters()
        if names:
            pipe.set_adapters(names, adapter_weights=[float(scale)] * len(names))
            return {}
    except Exception:
        pass
    return {"scale": float(scale)}


def _contact_sheet(images: list[Image.Image], labels: list[str], pad: int = 8) -> Image.Image:
    w, h = images[0].size
    cols = 2
    rows = (len(images) + cols - 1) // cols
    label_h = 36
    sheet = Image.new(
        "RGB",
        (cols * w + pad * (cols + 1), rows * (h + label_h) + pad * (rows + 1)),
        (24, 24, 24),
    )
    draw = ImageDraw.Draw(sheet)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None
    for i, (im, lab) in enumerate(zip(images, labels)):
        r, c = divmod(i, cols)
        x = pad + c * (w + pad)
        y = pad + r * (h + label_h + pad)
        draw.text((x, y), lab, fill=(240, 240, 240), font=font)
        sheet.paste(im, (x, y + label_h))
    return sheet


def _find_step_ckpt(step: int) -> Path | None:
    """Kohya names vary: name-000800.safetensors / name-step000800.safetensors etc."""
    patterns = [
        f"{OUTPUT_NAME}-{step:06d}.safetensors",
        f"{OUTPUT_NAME}-step{step:06d}.safetensors",
        f"{OUTPUT_NAME}-{step}.safetensors",
        f"{OUTPUT_NAME}-step{step}.safetensors",
    ]
    for name in patterns:
        p = OUT_DIR / name
        if p.is_file():
            return p
    # fuzzy: any safetensors containing the step digits
    rx = re.compile(rf"{re.escape(OUTPUT_NAME)}.*0*{step}\.safetensors$")
    for p in sorted(OUT_DIR.glob("*.safetensors")):
        if p.name == lora_path.name:
            continue
        if rx.search(p.name) or f"{step:06d}" in p.name or f"-{step}." in p.name:
            return p
    return None


dtype = torch.float16 if MIXED_PRECISION == "fp16" else torch.bfloat16
infer_id, photoreal = _resolve_infer_model()
pipe = None
tried = []
for mid, is_photo in (
    [(infer_id, photoreal)]
    + [(m, True) for m in PREFERRED_INFER_MODELS if m != infer_id]
    + [(TRAIN_BASE_MODEL, False)]
):
    if mid in tried:
        continue
    tried.append(mid)
    try:
        print("Loading SDXL …", mid, dtype)
        pipe = _load_pipe(mid, dtype).to("cuda")
        infer_id, photoreal = mid, is_photo
        break
    except Exception as e:
        print(f"FAILED {mid}: {e}")

if pipe is None:
    raise RuntimeError(f"Could not load any SDXL checkpoint. Tried: {tried}")

if not photoreal or "stable-diffusion-xl-base" in infer_id:
    print(
        "\n*** WARNING ***\n"
        "Using SDXL base for samples — doll/porcelain face is exaggerated.\n"
        "Prefer HF: SG161222/RealVisXL_V5.0 (or V4.0). Set INFER_MODEL if needed.\n"
    )
else:
    print("OK — photoreal checkpoint:", infer_id)

# List intermediate ckpts for docs
print("\n===== LoRA files in OUT_DIR =====")
for p in sorted(OUT_DIR.glob("*.safetensors")):
    print(" ", p.name)
for step in (400, 800, 1200):
    found = _find_step_ckpt(step)
    print(f"  step {step}:", found.name if found else "(not found)")

pipe.load_lora_weights(str(OUT_DIR), weight_name=lora_path.name)

sample_dir = OUT_DIR / "samples"
sample_dir.mkdir(exist_ok=True)
g = torch.Generator("cuda").manual_seed(SEED)
sweep_prompt = SAMPLE_PROMPTS[0]
best_scale = LORA_SCALE

if RUN_SCALE_SWEEP:
    sweep_imgs = []
    for scale in SCALE_SWEEP:
        kwargs = _set_lora_scale(pipe, scale)
        img = pipe(
            sweep_prompt,
            negative_prompt=NEG,
            num_inference_steps=NUM_INFERENCE_STEPS,
            guidance_scale=GUIDANCE_SCALE,
            generator=torch.Generator("cuda").manual_seed(SEED),
            cross_attention_kwargs=kwargs or None,
        ).images[0]
        out = sample_dir / f"sweep_scale_{scale:.2f}.png"
        img.save(out)
        print(out)
        sweep_imgs.append(img)
    labels = list(SCALE_SWEEP_LABELS) if len(SCALE_SWEEP_LABELS) == len(SCALE_SWEEP) else [
        f"lora_scale={s:.2f}" for s in SCALE_SWEEP
    ]
    sheet = _contact_sheet(sweep_imgs, labels)
    sheet_path = sample_dir / "contact_sheet_lora_scale_sweep.png"
    sheet.save(sheet_path)
    print("Contact sheet:", sheet_path)
    print("Scale axis: more photo (RealVis raw) ← → more glam (trained LoRA beauty)")
    # pick mid scale as default "best" hint (user picks visually); use LORA_SCALE for mid-ckpt
    best_scale = LORA_SCALE
    print(f"Mid-ckpt sampling will use LORA_SCALE={best_scale:.2f} (edit slider after viewing sheet)")

_set_lora_scale(pipe, LORA_SCALE)
for i, prompt in enumerate(SAMPLE_PROMPTS, 1):
    kwargs = _set_lora_scale(pipe, LORA_SCALE)
    img = pipe(
        prompt,
        negative_prompt=NEG,
        num_inference_steps=NUM_INFERENCE_STEPS,
        guidance_scale=GUIDANCE_SCALE,
        generator=g,
        cross_attention_kwargs=kwargs or None,
    ).images[0]
    out = sample_dir / f"sample_{i:02d}_s{LORA_SCALE:.2f}.png"
    img.save(out)
    print(out)

# Optional: sample intermediate checkpoints at best/default scale
if SAMPLE_MID_CKPTS:
    print("\n===== Mid-checkpoint samples (scale={:.2f}) =====".format(best_scale))
    print("Tip: after picking best scale from contact sheet, re-run with that LORA_SCALE.")
    for step in MID_CKPT_STEPS:
        ckpt = _find_step_ckpt(step)
        if ckpt is None:
            print(f"skip step {step}: no checkpoint file")
            continue
        try:
            pipe.unload_lora_weights()
        except Exception:
            pass
        pipe.load_lora_weights(str(OUT_DIR), weight_name=ckpt.name)
        kwargs = _set_lora_scale(pipe, best_scale)
        img = pipe(
            sweep_prompt,
            negative_prompt=NEG,
            num_inference_steps=NUM_INFERENCE_STEPS,
            guidance_scale=GUIDANCE_SCALE,
            generator=torch.Generator("cuda").manual_seed(SEED),
            cross_attention_kwargs=kwargs or None,
        ).images[0]
        out = sample_dir / f"mid_step{step:04d}_s{best_scale:.2f}.png"
        img.save(out)
        print(out)
    # reload final
    try:
        pipe.unload_lora_weights()
    except Exception:
        pass
    pipe.load_lora_weights(str(OUT_DIR), weight_name=lora_path.name)
else:
    print(
        "\nSAMPLE_MID_CKPTS=False — to compare step 800/1200 manually:\n"
        "  1) Set SAMPLE_MID_CKPTS=True and re-run, OR\n"
        f"  2) Temporarily point weight_name to e.g. {OUTPUT_NAME}-000800.safetensors\n"
        "  Intermediate files live in OUT_DIR next to the final .safetensors."
    )

print("OK — soft glam samples under", sample_dir)
print(f"infer_model={infer_id}  lora_scale={LORA_SCALE}  steps={NUM_INFERENCE_STEPS}  cfg={GUIDANCE_SCALE}")
print("Tip: lower LORA_SCALE → more RealVis raw face; higher → more trained glam beauty (try 0.75–0.85)")
print("Download LoRA:", lora_path)
print("Drive samples:", sample_dir)
print("Next: pick best scale from contact sheet; see FOLLOWUP_FACE_NATURALNESS.md")
