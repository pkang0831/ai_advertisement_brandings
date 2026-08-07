#@title 4) Train — SKIP if already trained (diagnose + live logs / sdxl_train_network)
# ★ SKIP THIS CELL if final LoRA already exists on Drive (you already trained).
# Face naturalness → cell 5 Sample, NOT retrain.
# SDXL MUST use sdxl_train_network.py — never plain train_network.py.
# Live tee logs; on failure prints last 100 lines (not silent CalledProcessError).

import os, re, subprocess, sys
from datetime import datetime, timezone
from pathlib import Path

WORK = Path(globals().get("WORK", "/content/rina_lora_work"))
OUT_DIR = Path(globals().get("OUT_DIR", "/content/drive/MyDrive/rina_lora/outputs/rina_park_person_sdxl_lora"))
BASE_MODEL = globals().get("BASE_MODEL", "stabilityai/stable-diffusion-xl-base-1.0")
OUTPUT_NAME = globals().get("OUTPUT_NAME", "rina_park_person_sdxl_lora")
TRIGGER = globals().get("TRIGGER", "rina_park_person")
RESOLUTION = int(globals().get("RESOLUTION", 1024))
NETWORK_DIM = int(globals().get("NETWORK_DIM", 16))
NETWORK_ALPHA = int(globals().get("NETWORK_ALPHA", 16))
LEARNING_RATE = float(globals().get("LEARNING_RATE", 1e-4))
MAX_TRAIN_STEPS = int(globals().get("MAX_TRAIN_STEPS", 1600))
TRAIN_BATCH_SIZE = int(globals().get("TRAIN_BATCH_SIZE", 1))
SEED = int(globals().get("SEED", 42))
MIXED_PRECISION = globals().get("MIXED_PRECISION", "auto")

_final_lora = OUT_DIR / f"{OUTPUT_NAME}.safetensors"
if _final_lora.is_file():
    print(f"*** SKIP HINT *** Final LoRA already exists:\n  {_final_lora}")
    print("If you only need samples / face naturalness, STOP and run cell 5 instead.")
    print("Re-running this cell WILL overwrite / continue training — only proceed if intentional.\n")

print("===== 1) nvidia-smi / GPU =====")
try:
    print(subprocess.check_output(["nvidia-smi"], text=True, stderr=subprocess.STDOUT))
except Exception as e:
    print("nvidia-smi failed:", e)

import torch
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    print("capability:", torch.cuda.get_device_capability(0))
    print("bf16_supported:", torch.cuda.is_bf16_supported())
    print("VRAM_GB:", round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1))
if MIXED_PRECISION == "auto":
    MIXED_PRECISION = "bf16" if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else "fp16"
print("MIXED_PRECISION=", MIXED_PRECISION)

print("\n===== 2) bitsandbytes / xformers / HF =====")
try:
    import bitsandbytes as bnb
    print("bitsandbytes", getattr(bnb, "__version__", "ok"))
except Exception as e:
    print("bitsandbytes FAIL:", e)
try:
    import xformers
    print("xformers", xformers.__version__)
    XFORMERS_OK = True
except Exception as e:
    print("xformers FAIL:", e)
    XFORMERS_OK = False
hf = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
if not hf:
    try:
        from google.colab import userdata
        hf = userdata.get("HF_TOKEN")
    except Exception:
        hf = None
if hf:
    os.environ["HF_TOKEN"] = hf
    os.environ["HUGGING_FACE_HUB_TOKEN"] = hf
    print("HF_TOKEN: set")
else:
    print("HF_TOKEN: MISSING — SDXL gated download may 401. Set Colab Secret HF_TOKEN.")

print("\n===== 3) train_data tree (must have N_name subfolder) =====")
train_root = WORK / "train_data"
if not train_root.is_dir():
    print("MISSING", train_root, "→ re-run unpack cell first")
else:
    for dirpath, dirnames, filenames in os.walk(train_root):
        rel = os.path.relpath(dirpath, train_root)
        depth = 0 if rel == "." else rel.count(os.sep) + 1
        if depth > 2:
            dirnames.clear()
            continue
        imgs = [f for f in filenames if f.lower().endswith((".png", ".jpg", ".jpeg"))]
        txts = [f for f in filenames if f.lower().endswith(".txt")]
        print(f"{rel}/  subdirs={dirnames[:8]} imgs={len(imgs)} txts={len(txts)} eg={filenames[:6]}")

buckets = []
if train_root.is_dir():
    buckets = sorted(
        p for p in train_root.iterdir()
        if p.is_dir() and re.match(r"^\d+_.+", p.name)
    )
if not buckets:
    raise RuntimeError(
        "Kohya layout invalid / 잘못된 데이터 구조.\n"
        "Need: /content/rina_lora_work/train_data/<repeats>_<name>/*.png + .txt\n"
        "예: train_data/10_rina_park_person/rina_character_v1_001.png\n"
        "→ unpack 셀(제목 3)을 다시 실행하세요."
    )
print("OK buckets:", [b.name for b in buckets])

print("\n===== 4) accelerate config + cwd =====")
sd = Path("/content/sd-scripts")
# SDXL MUST use sdxl_train_network.py — train_network.py is SD1/2 only
train_py = sd / "sdxl_train_network.py"
alt_py = sd / "train_network.py"
print("cwd will be:", sd)
print("sdxl_train_network.py:", train_py.exists())
print("train_network.py (do NOT use for SDXL):", alt_py.exists())
if not train_py.exists():
    raise FileNotFoundError("Re-run install cell — sdxl_train_network.py missing")

acc_dir = Path.home() / ".cache" / "huggingface" / "accelerate"
acc_dir2 = Path.home() / ".accelerate"
for d in (acc_dir, acc_dir2):
    d.mkdir(parents=True, exist_ok=True)
acc_yaml = """compute_environment: LOCAL_MACHINE
debug: false
distributed_type: 'NO'
downcast_bf16: 'no'
gpu_ids: all
machine_rank: 0
main_training_function: main
mixed_precision: {mp}
num_machines: 1
num_processes: 1
rdzv_backend: static
same_network: true
tpu_env: []
tpu_use_cluster: false
tpu_use_sudo: false
use_cpu: false
""".format(mp=MIXED_PRECISION)
(acc_dir / "default_config.yaml").write_text(acc_yaml)
(acc_dir2 / "default_config.yaml").write_text(acc_yaml)
print("Wrote accelerate config mixed_precision=", MIXED_PRECISION)

OUT_DIR.mkdir(parents=True, exist_ok=True)
(LOG := OUT_DIR / "logs").mkdir(parents=True, exist_ok=True)

cmd = [
    sys.executable, "sdxl_train_network.py",
    "--pretrained_model_name_or_path", BASE_MODEL,
    "--train_data_dir", str(train_root),
    "--output_dir", str(OUT_DIR),
    "--output_name", OUTPUT_NAME,
    "--resolution", str(RESOLUTION),
    "--network_module", "networks.lora",
    "--network_dim", str(NETWORK_DIM),
    "--network_alpha", str(NETWORK_ALPHA),
    "--learning_rate", str(LEARNING_RATE),
    "--unet_lr", str(LEARNING_RATE),
    "--network_train_unet_only",
    "--lr_scheduler", "cosine_with_restarts",
    "--lr_scheduler_num_cycles", "1",
    "--max_train_steps", str(MAX_TRAIN_STEPS),
    "--train_batch_size", str(TRAIN_BATCH_SIZE),
    "--save_every_n_steps", "400",
    "--save_model_as", "safetensors",
    "--mixed_precision", MIXED_PRECISION,
    "--optimizer_type", "AdamW8bit",
    "--max_data_loader_n_workers", "2",
    "--cache_latents",
    "--cache_latents_to_disk",
    "--gradient_checkpointing",
    "--caption_extension", ".txt",
    "--shuffle_caption",
    "--keep_tokens", "1",
    "--seed", str(SEED),
    "--logging_dir", str(LOG),
    "--log_prefix", OUTPUT_NAME,
]
if XFORMERS_OK:
    cmd.append("--xformers")
if MIXED_PRECISION == "fp16":
    cmd.append("--no_half_vae")

stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
log_path = LOG / f"debug_train_{stamp}.log"
print("\n===== 5) run sdxl_train_network.py (tee) =====")
print("CMD:", " ".join(cmd))
print("LOG:", log_path)

env = os.environ.copy()
env["PYTHONUNBUFFERED"] = "1"
env["ACCELERATE_MIXED_PRECISION"] = MIXED_PRECISION

with open(log_path, "w", encoding="utf-8") as logf:
    logf.write("CMD: " + " ".join(cmd) + "\n\n")
    logf.flush()
    proc = subprocess.Popen(
        cmd,
        cwd=str(sd),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=env,
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        print(line, end="")
        logf.write(line)
    rc = proc.wait()

print(f"\nexit_code={rc} log={log_path}")
if rc != 0:
    lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
    print("\n===== LAST 100 LOG LINES (root cause) =====")
    print("\n".join(lines[-100:]))
    raise RuntimeError(
        f"Training failed exit={rc}. Scroll UP for live errors, or see last 100 lines above.\n"
        f"전체 로그: {log_path}"
    )
print("Training finished OK:", sorted(OUT_DIR.glob("*.safetensors")))
print("Next: run cell 5 (Sample). Intermediate ckpts at steps 400/800/1200 if save_every_n_steps=400.")
