
# # Rina Park — Still 대량생산 노트북
#
# 사람+스크립트 워크플로: **Config → Prompt → Preflight → Run → Pack 복사 → Finder → QC**.
# 확산 재구현 없이 `gen_i2v_heroes.py`를 subprocess로 호출합니다.
#
# ### 여기만 고치면 됨
# 1. **§2 Config** — `poses` / `seeds_per_pose` / `base_seed` / `lora` / `skip_face_detailer` / `track` / `dry_run`
# 2. **§3 Prompt** — `look_pos` / `look_neg_extra` / `scene_glue` / `pose_prompts`
# 3. Preflight에서 샘플 프롬프트 확인 → `dry_run=False` 후 Run
#
# ### dual50와 함께 쓸 때
# - `screen still_dual50` / `_run_dual50_from_s00.sh`가 **이미 돌고 있으면** 이 노트북에서 배치를 또 돌리지 마세요 (Metal/MPS 충돌).
# - 기다리거나, 돌릴 때는 **새 pack 이름/타임스탬프**를 쓰세요. dual50를 **끊지 마세요**.
# - `track=both`는 **한 gen 프로세스**에서 SFW↔NSFW **인터리브** (`s0,n0,s1,n1,...`). Qwen FaceDetailer 켜면 SDXL과 Metal 경합 주의.
#
# ### 트랙
# - **SFW pack**: `rina_park/out/packs/`
# - **NSFW pack**: `rina_park/private/nsfw_test/private_media/` (로컬 전용, promote 금지)
# - NSFW 포즈는 **id만** 설정. 프롬프트 본문은 노트북에 넣지 않음 (`private/pose_catalog_nsfw.yml`).
# - NSFW일 때 `pose_prompts`는 **비워 두세요** → 카탈로그/private overlay 사용.
# - 풀 카테시안 both: `track=both`, `max_combos=0`, `seeds_per_pose=1`, `combo_mode=cartesian`.
#
# ### HF SSL (필요 시)
# `SSL_CERT_FILE=$HOME/combined-cert.pem` (Setup 셀에서 기본 설정)


# ## 1. Setup
# 경로 · venv 파이썬 · env · import. **생성은 안 함.**

# %% cell 2
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

REPO = Path("/Users/RBIPK031/ai_influencer").resolve()
RINA = REPO / "rina_park"
VENV_PY = REPO / ".venv" / "bin" / "python"
GEN_SCRIPT = RINA / "scripts" / "gen_i2v_heroes.py"
PROMOTE_SCRIPT = RINA / "scripts" / "promote_i2v_heroes.py"
OUT_HEROES = RINA / "out" / "i2v_heroes"
SCRATCH_DIR = OUT_HEROES / "_notebook_scratch"
SFW_PACK_ROOT = RINA / "out" / "packs"
NSFW_PACK_ROOT = RINA / "private" / "nsfw_test" / "private_media"
NSFW_CATALOG = RINA / "private" / "pose_catalog_nsfw.yml"
CERT = Path.home() / "combined-cert.pem"

os.environ.setdefault("PYTHONPATH", str(RINA))
os.environ.setdefault("PYTHONUNBUFFERED", "1")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
if CERT.is_file():
    os.environ.setdefault("SSL_CERT_FILE", str(CERT))
    os.environ.setdefault("REQUESTS_CA_BUNDLE", str(CERT))

if str(RINA) not in sys.path:
    sys.path.insert(0, str(RINA))

assert REPO.is_dir(), REPO
assert VENV_PY.is_file(), f"missing venv python: {VENV_PY}"
assert GEN_SCRIPT.is_file(), GEN_SCRIPT

ver = subprocess.check_output([str(VENV_PY), "-c", "import sys; print(sys.executable)"], text=True).strip()
print("REPO", REPO)
print("venv python", ver)
print("SSL_CERT_FILE", os.environ.get("SSL_CERT_FILE", "(unset)"))
print("PYTORCH_ENABLE_MPS_FALLBACK", os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK"))
print("Setup OK — no generation yet")


# ## 2. Config panel — 여기만 고치면 됨 (생성 파라미터)
# 트랙·**combo 토글**·포즈·시드·lora. NSFW는 **pose id 리스트만**.
# - `use_combo=True` → gen에 `--combo` 전달 (시드마다 outfit/concept/expression 변동)
# - 가변성↑: `seeds_per_pose`를 6–12 권장 (combo ON일 때)

# %% cell 4
@dataclass
class BatchConfig:
    # "sfw" | "nsfw" | "both" — both = interleaved SFW↔NSFW in one gen process
    track: str = "both"

    # dual50-style SFW hero set (hidden-hand / lifestyle)
    sfw_poses: list[str] = field(
        default_factory=lambda: [
            "hand_pockets_3q",
            "hand_cropped_out",
            "hand_resting_lap_seated",
            "fitness_mat_soft_seated",
            "foot_standing_flat_sneakers",
        ]
    )
    # NSFW: ids only — prompts live in private overlay, never paste bodies here
    nsfw_poses: list[str] = field(
        default_factory=lambda: [
            "nsfw_reclined_knee_up_partial",
            "nsfw_standing_mirror_hip_angle",
            "nsfw_side_lying_tucked",
        ]
    )

    seeds_per_pose: int = 4
    # favorite s00 family
    base_seed: int = 30072026
    lora: float = 0.90
    skip_face_detailer: bool = True

    # pack names; empty pack_ts → UTC stamp at preflight
    pack_ts: str = ""
    sfw_pack_prefix: str = "sfw_mass"
    nsfw_pack_prefix: str = "batch_nsfw"

    # safety: keep True until you intentionally run
    dry_run: bool = True


CFG = BatchConfig()
# --- edit here ---
CFG.track = "both"  # "sfw" | "nsfw" | "both"
CFG.sfw_poses = [
    "hand_pockets_3q",
    "hand_cropped_out",
]
CFG.seeds_per_pose = 4
CFG.base_seed = 30072026
CFG.lora = 0.90
CFG.skip_face_detailer = True
CFG.dry_run = True  # flip to False only when ready (and dual50 idle)
# -----------------

if not CFG.pack_ts:
    CFG.pack_ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

SFW_PACK = SFW_PACK_ROOT / f"{CFG.sfw_pack_prefix}_{CFG.pack_ts}"
NSFW_PACK = NSFW_PACK_ROOT / f"{CFG.nsfw_pack_prefix}_{CFG.pack_ts}"

print(json.dumps(asdict(CFG), indent=2))
print("SFW_PACK", SFW_PACK)
print("NSFW_PACK", NSFW_PACK)
print("NOTE: dual50 running? wait or use a new pack_ts — do not kill still_dual50")


# ## 3. Prompt panel — 여기만 고치면 됨 (프롬프트)
# - 빈 문자열/`{}`이면 스크립트 기본값(LOOK_POS / LOOK_NEG_EXTRA / HERO_POSE_SHORT·카탈로그) 유지.
# - `pose_prompts`: `{pose_id: short_text}` — 해당 포즈만 짧은 framing 오버라이드.
# - NSFW 실험: `pose_prompts` **비움** (본문 넣지 말 것).

# %% cell 6
# --- edit prompts here (empty = script defaults) ---
LOOK_POS = "rina_park_person, young Korean-Canadian woman mid-20s, long dark hair, soft glam, detailed skin texture, natural skin pores, realistic lighting, photorealistic"  #
LOOK_NEG_EXTRA = "cartoon, illustration, painting, blurry, deformed, low quality, bad anatomy"  #
SCENE_GLUE = "shot on 85mm lens, soft volumetric lighting, hyper-detailed, sharp focus"  #[cite: 1]

# Single-pose experiment: put short framing here for that id.
# Multi-pose: map each id. Leave {} for catalog / HERO_POSE_SHORT defaults.
POSE_PROMPTS: dict[str, str] = {
    "hand_pockets_3q": "sitting by a cafe window, natural side lighting, holding a ceramic coffee cup with both hands, soft casual expression, medium shot"
}
# -----------------------------------------

PROMPT_CFG = {
    "look_pos": LOOK_POS.strip(),
    "look_neg_extra": LOOK_NEG_EXTRA.strip(),
    "scene_glue": SCENE_GLUE.strip(),
    "pose_prompts": {k: v.strip() for k, v in POSE_PROMPTS.items() if str(v).strip()},
}
print("PROMPT_CFG")
print(json.dumps(PROMPT_CFG, indent=2, ensure_ascii=False))
if CFG.track in ("nsfw", "both") and PROMPT_CFG["pose_prompts"]:
    print("WARN: NSFW track + pose_prompts set — prefer empty dict so private overlay is used")


# ## 4. Preflight
# 카탈로그 id 확인 + **첫 포즈 effective prompt 샘플**. NSFW 본문은 출력하지 않음.
# dual50 / gen 충돌 경고.

# %% cell 8
from hyperreal.anatomy.pose_catalog import load_pose_catalog

# Import hero prompt helpers without loading diffusion stacks
sys.path.insert(0, str(RINA / "scripts"))
import gen_i2v_heroes as heroes  # noqa: E402

catalog = load_pose_catalog()
all_ids = sorted(catalog.by_id.keys())
sfw_ids = [p for p in all_ids if not p.startswith("nsfw_")]
nsfw_ids = [p for p in all_ids if p.startswith("nsfw_")]

print(f"catalog poses: total={len(all_ids)} sfw≈{len(sfw_ids)} nsfw={len(nsfw_ids)}")
print(f"nsfw_overlay_loaded={catalog.nsfw_overlay_loaded} path={catalog.nsfw_overlay_path}")
if not NSFW_CATALOG.is_file():
    print("WARN: private NSFW overlay missing — nsfw track will fail")
elif not catalog.nsfw_overlay_loaded:
    print("WARN: overlay file exists but not loaded")
else:
    print("NSFW overlay OK (ids only):", nsfw_ids)

missing_sfw = [p for p in CFG.sfw_poses if p not in catalog.by_id]
missing_nsfw = [p for p in CFG.nsfw_poses if p not in catalog.by_id]
if missing_sfw:
    print("WARN missing SFW pose ids:", missing_sfw)
if CFG.track in ("nsfw", "both") and missing_nsfw:
    print("WARN missing NSFW pose ids:", missing_nsfw)

n_sfw = len(CFG.sfw_poses) * CFG.seeds_per_pose if CFG.track in ("sfw", "both") else 0
n_nsfw = len(CFG.nsfw_poses) * CFG.seeds_per_pose if CFG.track in ("nsfw", "both") else 0
print(f"planned stills: sfw={n_sfw} nsfw={n_nsfw} total={n_sfw + n_nsfw}")
print(f"CLI: --lora {CFG.lora} --base-seed {CFG.base_seed} skip_fd={CFG.skip_face_detailer}")

# Effective prompt sample (first pose of active track) — SFW only body text
sample_poses = CFG.sfw_poses if CFG.track in ("sfw", "both") else CFG.nsfw_poses
if sample_poses:
    pid0 = sample_poses[0]
    pose0 = catalog.get(pid0)
    look = PROMPT_CFG["look_pos"] or None
    pose_short = heroes.resolve_pose_short(
        pose0,
        pose_prompt_all="",
        pose_prompts=PROMPT_CFG["pose_prompts"],
    )
    glue = PROMPT_CFG["scene_glue"] if PROMPT_CFG["scene_glue"] else heroes.SCENE_GLUE.get(pid0, "")
    if pose0.track != "sfw" or pid0.startswith("nsfw_"):
        print(f"sample pose={pid0} track=nsfw — prompt body omitted (private overlay)")
        print(
            "  overrides:",
            f"look_pos={'set' if look else 'default'}",
            f"pose_short={'set' if pose_short else 'catalog'}",
            f"glue={'set' if PROMPT_CFG['scene_glue'] else 'default'}",
        )
    else:
        sample = heroes.build_hero_prompt(
            pose0,
            glue,
            hand_trigger=heroes._needs_hand_lora(pose0),
            look_pos=look,
            pose_short=pose_short,
        )
        tok = heroes._count_clip_tokens(sample)
        print(f"\n=== effective prompt sample pose={pid0} clip_tokens≈{tok} ===")
        print(sample)
        neg_extra = PROMPT_CFG["look_neg_extra"] or heroes.LOOK_NEG_EXTRA
        print("look_neg_extra:", (neg_extra[:120] + "…") if len(neg_extra) > 120 else neg_extra)

# dual50 / gen collision hint
busy = subprocess.run(["pgrep", "-fl", "gen_i2v_heroes"], capture_output=True, text=True)
dual = subprocess.run(["pgrep", "-fl", "still_dual50|_run_dual50"], capture_output=True, text=True)
if busy.stdout.strip() or dual.stdout.strip():
    print("\nWARN: still_dual50 / gen_i2v_heroes already running — Metal parallel is a bad idea.")
    print("Wait until dual50 finishes before dry_run=False. Do NOT kill still_dual50.")
    if busy.stdout.strip():
        print(busy.stdout.strip()[:500])
    if dual.stdout.strip():
        print(dual.stdout.strip()[:300])
else:
    print("\nNo gen_i2v_heroes / still_dual50 process detected")


# ## 5. Promote favorite (optional)
# `out/i2v_heroes/current/`로 심볼릭 링크. NSFW에는 쓰지 마세요.

# %% cell 10
# Example: promote hand_pockets_3q s00 from a known run
PROMOTE_RUN_ID = "20260802T145425Z"  # edit
PROMOTE_PICK = "hand_pockets_3q:s00"  # pose_id:tag  OR leave "" and use --auto-best
DO_PROMOTE = False  # flip True to run

cmd = [
    str(VENV_PY), "-u", str(PROMOTE_SCRIPT),
    "--run-id", PROMOTE_RUN_ID,
]
if PROMOTE_PICK.strip():
    cmd += ["--pick", PROMOTE_PICK.strip()]
else:
    cmd += ["--auto-best"]

print("promote cmd:", " ".join(cmd))
if DO_PROMOTE:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(RINA)
    subprocess.run(cmd, cwd=str(REPO), env=env, check=False)
    print("current/", list((OUT_HEROES / "current").glob("*")))
else:
    print("skipped (DO_PROMOTE=False). List picks:")
    list_cmd = [str(VENV_PY), "-u", str(PROMOTE_SCRIPT), "--run-id", PROMOTE_RUN_ID, "--list"]
    subprocess.run(list_cmd, cwd=str(REPO), env={**os.environ, "PYTHONPATH": str(RINA)}, check=False)


# ## 6. Run batch
# `CFG.dry_run=False`일 때만 실제 생성. Prompt 오버라이드는 CLI 플래그 + scratch JSON으로 전달.
# `both`면 SFW 끝난 뒤 NSFW 순차. 스크립트에 `--out-dir` 없음 → 실행 전후 `out/i2v_heroes/<run_id>/`를 스냅샷으로 잡습니다.

# %% cell 12
def latest_run_dir(heroes_root: Path) -> Path | None:
    runs = sorted(
        [p for p in heroes_root.iterdir() if p.is_dir() and p.name[:4].isdigit()],
        key=lambda p: p.name,
        reverse=True,
    )
    return runs[0] if runs else None


def write_pose_prompts_json(poses: list[str], label: str) -> Path | None:
    """Write scratch JSON for poses that have overrides; None if empty."""
    mapping = {p: PROMPT_CFG["pose_prompts"][p] for p in poses if p in PROMPT_CFG["pose_prompts"]}
    if not mapping:
        return None
    SCRATCH_DIR.mkdir(parents=True, exist_ok=True)
    path = SCRATCH_DIR / f"pose_prompts_{label.lower()}_{CFG.pack_ts}.json"
    path.write_text(json.dumps(mapping, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def build_gen_cmd(poses: list[str], label: str) -> list[str]:
    cmd = [
        str(VENV_PY), "-u", str(GEN_SCRIPT),
        "--poses", ",".join(poses),
        "--seeds-per-pose", str(CFG.seeds_per_pose),
        "--base-seed", str(CFG.base_seed),
        "--lora", str(CFG.lora),
    ]
    if CFG.skip_face_detailer:
        cmd.append("--skip-face-detailer")
    else:
        cmd.append("--no-skip-face-detailer")
    if PROMPT_CFG["look_pos"]:
        cmd += ["--look-pos", PROMPT_CFG["look_pos"]]
    if PROMPT_CFG["look_neg_extra"]:
        cmd += ["--look-neg-extra", PROMPT_CFG["look_neg_extra"]]
    if PROMPT_CFG["scene_glue"]:
        cmd += ["--scene-glue", PROMPT_CFG["scene_glue"]]
    jp = write_pose_prompts_json(poses, label)
    if jp is not None:
        cmd += ["--pose-prompts-json", str(jp)]
    return cmd


def stream_run(cmd: list[str], label: str) -> tuple[int, Path | None]:
    """Run gen script; stream logs; return (exit_code, new_run_dir)."""
    before = latest_run_dir(OUT_HEROES)
    before_name = before.name if before else None
    print(f"\n===== {label} START =====")
    print("cmd:", " ".join(cmd))
    print("before_run", before_name)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(RINA)
    env["PYTHONUNBUFFERED"] = "1"
    proc = subprocess.Popen(
        cmd,
        cwd=str(REPO),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        print(line, end="")
    ec = proc.wait()
    after = latest_run_dir(OUT_HEROES)
    after_name = after.name if after else None
    new_dir = after if after and after_name != before_name else None
    print(f"===== {label} EXIT={ec} run_dir={new_dir} =====")
    return ec, new_dir


# Re-check Metal conflict before real gen
busy = subprocess.run(["pgrep", "-fl", "gen_i2v_heroes|still_dual50|_run_dual50"], capture_output=True, text=True)
if busy.stdout.strip() and not CFG.dry_run:
    raise RuntimeError(
        "still_dual50 / gen_i2v_heroes still running — wait (Metal conflict). Do not kill dual50.\n"
        + busy.stdout.strip()[:500]
    )
elif busy.stdout.strip():
    print("WARN: dual50/gen still running — dry_run only is OK; do not flip dry_run=False yet")
    print(busy.stdout.strip()[:400])

RUN_RESULTS: dict[str, dict] = {}

jobs: list[tuple[str, list[str], Path]] = []
if CFG.track in ("sfw", "both"):
    jobs.append(("SFW", CFG.sfw_poses, SFW_PACK))
if CFG.track in ("nsfw", "both"):
    if not catalog.nsfw_overlay_loaded:
        raise RuntimeError("NSFW overlay missing — cannot run nsfw track")
    jobs.append(("NSFW", CFG.nsfw_poses, NSFW_PACK))

for label, poses, pack in jobs:
    cmd = build_gen_cmd(poses, label)
    print(f"\n[{label}] poses={poses} pack={pack}")
    if CFG.dry_run:
        print("[dry_run] would run:", " ".join(cmd))
        RUN_RESULTS[label] = {"exit": None, "run_dir": None, "pack": str(pack), "dry_run": True, "cmd": cmd}
        continue
    ec, run_dir = stream_run(cmd, label)
    RUN_RESULTS[label] = {
        "exit": ec,
        "run_dir": str(run_dir) if run_dir else None,
        "pack": str(pack),
        "dry_run": False,
        "cmd": cmd,
    }

print("\nRUN_RESULTS")
print(json.dumps({k: {**v, "cmd": " ".join(v["cmd"]) if isinstance(v.get("cmd"), list) else v.get("cmd")} for k, v in RUN_RESULTS.items()}, indent=2, ensure_ascii=False))
if CFG.dry_run:
    print("dry_run=True — set CFG.dry_run=False and re-run Config + Prompt + this cell to generate")


# ## 7. Collect pack
# 최종 **1080×1920** `s*_seed*.jpg`만 복사 (`*_gen832*` 제외). run_dir이 비면 최신 런을 수동으로 지정.

# %% cell 14
def copy_finals(src: Path, dest: Path) -> int:
    """Copy 1080x1920 finals (not *_gen*) + sidecars/index."""
    dest.mkdir(parents=True, exist_ok=True)
    n = 0
    for f in sorted(src.rglob("s*_seed*.jpg")):
        if "_gen" in f.name:
            continue
        if f.suffix.lower() != ".jpg":
            continue
        rel = f.relative_to(src)
        out = dest / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(f, out)
        side = Path(str(f) + ".json")
        if side.is_file():
            shutil.copy2(side, Path(str(out) + ".json"))
        sc = f.with_name(f.stem + "_scorecard.json")
        if sc.is_file():
            shutil.copy2(sc, out.with_name(out.stem + "_scorecard.json"))
        n += 1
    for extra in ("index.json", "summary.json", "heroes_index.json"):
        p = src / extra
        if p.is_file():
            shutil.copy2(p, dest / extra)
    for p in src.glob("*.json"):
        if p.name not in {"index.json", "summary.json", "heroes_index.json"}:
            shutil.copy2(p, dest / p.name)
    (dest / "SOURCE_RUN_DIR.txt").write_text(str(src.resolve()) + "\n", encoding="utf-8")
    return n


# Optional manual overrides if auto discovery failed:
MANUAL_SFW_RUN: str | None = None  # e.g. "20260802T145425Z"
MANUAL_NSFW_RUN: str | None = None

PACK_STATS: dict[str, dict] = {}

for label, pack_path in (("SFW", SFW_PACK), ("NSFW", NSFW_PACK)):
    info = RUN_RESULTS.get(label)
    if info is None:
        continue
    if info.get("dry_run"):
        print(f"[{label}] dry_run — skip collect")
        continue
    run_s = info.get("run_dir")
    if not run_s:
        manual = MANUAL_SFW_RUN if label == "SFW" else MANUAL_NSFW_RUN
        if manual:
            run_s = str(OUT_HEROES / manual)
        else:
            print(f"[{label}] WARN: no run_dir — set MANUAL_{label}_RUN")
            continue
    src = Path(run_s)
    if not src.is_dir():
        print(f"[{label}] missing run dir", src)
        continue
    n = copy_finals(src, Path(pack_path))
    if n == 0:
        print(f"[{label}] WARN: 0 finals; rsync tree excluding _gen / _reject")
        Path(pack_path).mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [
                "rsync", "-a",
                "--exclude=_reject", "--exclude=*_gen*.jpg",
                f"{src}/", f"{pack_path}/",
            ],
            check=False,
        )
        n = len(list(Path(pack_path).rglob("s*_seed*.jpg")))
    reject = src / "_reject"
    if label == "NSFW" and reject.is_dir():
        dest_r = Path(pack_path) / "_reject"
        dest_r.mkdir(parents=True, exist_ok=True)
        subprocess.run(["rsync", "-a", f"{reject}/", f"{dest_r}/"], check=False)
    PACK_STATS[label] = {"copied": n, "pack": str(pack_path), "src": str(src)}
    print(f"[{label}] copied_finals={n} -> {pack_path}")

print(json.dumps(PACK_STATS, indent=2))


# ## 8. Open Finder
# 완료된 pack 폴더를 macOS Finder로 엽니다.

# %% cell 16
to_open: list[str] = []
if CFG.track in ("sfw", "both") and SFW_PACK.is_dir():
    to_open.append(str(SFW_PACK))
if CFG.track in ("nsfw", "both") and NSFW_PACK.is_dir():
    to_open.append(str(NSFW_PACK))

# Also useful: raw heroes + current
# to_open.append(str(OUT_HEROES / "current"))

if to_open:
    print("Opening Finder:", to_open)
    subprocess.run(["open", *to_open], check=False)
else:
    print("Nothing to open yet — run Collect first (or packs missing)")


# ## 9. Quick QC helpers
# scorecard `auto_pass` 집계 + 경로 요약. 최종 promote는 사람 QC 후.

# %% cell 18
def qc_summary(root: Path) -> dict:
    cards = list(root.rglob("*_scorecard.json"))
    passed, failed = [], []
    for c in cards:
        try:
            data = json.loads(c.read_text(encoding="utf-8"))
        except Exception as e:  # noqa: BLE001
            failed.append({"path": str(c), "error": str(e)})
            continue
        row = {
            "pose_id": data.get("pose_id"),
            "seed": data.get("seed"),
            "auto_pass": data.get("auto_pass"),
            "path": str(c),
        }
        if data.get("auto_pass"):
            passed.append(row)
        else:
            failed.append(row)
    return {
        "root": str(root),
        "scorecards": len(cards),
        "auto_pass": len(passed),
        "not_pass": len(failed),
        "pass_examples": passed[:8],
        "fail_examples": failed[:8],
    }


qc_targets: list[Path] = []
for label in ("SFW", "NSFW"):
    info = RUN_RESULTS.get(label) or {}
    rd = info.get("run_dir")
    if rd:
        qc_targets.append(Path(rd))
    pk = info.get("pack")
    if pk and Path(pk).is_dir():
        qc_targets.append(Path(pk))

if not qc_targets:
    # fallback: latest heroes run
    latest = latest_run_dir(OUT_HEROES)
    if latest:
        qc_targets.append(latest)
        print("fallback QC root", latest)

for root in qc_targets:
    if not root.is_dir():
        continue
    s = qc_summary(root)
    print("\n==", root)
    print(f"scorecards={s['scorecards']} auto_pass={s['auto_pass']} not_pass={s['not_pass']}")
    if s["pass_examples"]:
        print("pass sample:", [(x["pose_id"], x["seed"]) for x in s["pass_examples"]])

print("\nPaths")
print("  heroes", OUT_HEROES)
print("  current", OUT_HEROES / "current")
print("  SFW_PACK", SFW_PACK)
print("  NSFW_PACK", NSFW_PACK)
print("  promote hint: PYTHONPATH=rina_park .venv/bin/python rina_park/scripts/promote_i2v_heroes.py --run-id <RUN> --pick pose_id:sXX")


# ## (Optional) Wan I2V — commented / off by default
# 이 노트북 기본 범위 밖. 필요하면 아래 코드 셀 주석을 해제하고, **스틸 QC+promote 후**에만 실행.

# %% cell 20
# Optional Wan I2V — DO NOT enable during still mass-produce.
# After promote to out/i2v_heroes/current/, typically:
#
#   # !screen -dmS i2v bash -lc 'cd /Users/RBIPK031/ai_influencer && bash rina_park/scripts/run_i2v_from_hero.sh'
#
# Or inspect MOTION_PROMPTS / ops/i2v/STILL_GATE.md first.
print("Wan I2V cell is intentionally inert (still-only notebook)")
