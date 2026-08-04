#!/usr/bin/env python3
"""Hand/foot specialty LoRA A/B on anatomy-lock whitelist poses (base gen only).

Arms:
  baseline — character LoRA only
  arm_a    — character + detailed_hands @ 0.45
  arm_b    — character + Better Hands @ 0.4
  arm_c    — character + RealFeet @ 0.4 (foot poses only)
  arm_d    — character + winning hand + RealFeet (optional combo on foot pose)

Process-isolates each arm to limit MPS NaN cascades. Skips FaceDetailer / 2-pass / CN.

Usage:
  cd /Users/RBIPK031/ai_influencer
  PYTHONPATH=rina_park .venv/bin/python -u rina_park/scripts/run_hand_foot_lora_ab.py
  PYTHONPATH=rina_park .venv/bin/python -u rina_park/scripts/run_hand_foot_lora_ab.py --worker --arm arm_a
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
RINA = ROOT / "rina_park"
sys.path.insert(0, str(RINA))

from hyperreal.anatomy.pose_catalog import load_pose_catalog
from hyperreal.anatomy.qc_gates import evaluate_anatomy_image, scorecard_template, write_scorecard
from hyperreal.anatomy.stack import ExtraLora, known_specialty_loras, load_txt2img, unload

OUT_ROOT = RINA / "out" / "anatomy_lock"
OPS = RINA / "ops" / "anatomy_lock"

# Whitelist first-pass: resting + pockets (no hard grip); sneakers feet
HAND_POSES = ["hand_resting_lap_seated", "hand_pockets_3q"]
FOOT_POSES = ["foot_standing_flat_sneakers", "foot_seated_ankle_soft"]
ALL_POSES = HAND_POSES + FOOT_POSES

SCENE_GLUE = {
    "hand_pockets_3q": "beige trench coat, city sidewalk morning, candid",
    "hand_resting_lap_seated": "sunlit apartment sofa, soft loungewear",
    "foot_standing_flat_sneakers": "white sneakers, athletic leggings, studio lobby",
    "foot_seated_ankle_soft": "park bench, socks and sneakers, afternoon",
}

SEED = 29072026
STEPS = 32
CFG = 4.2
CHAR_LORA = 0.90


def _run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_hand_lora_ab"


def _mps_cleanup() -> None:
    import gc

    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
        try:
            torch.mps.synchronize()
        except Exception:  # noqa: BLE001
            pass


def _arm_specs(known: dict[str, Path]) -> dict[str, dict]:
    specs: dict[str, dict] = {
        "baseline": {
            "label": "character only",
            "extras": [],
            "poses": ALL_POSES,
            "prompt_trigger": "",
        },
    }
    if "detailed_hands" in known:
        specs["arm_a"] = {
            "label": "detailed_hands @ 0.45",
            "extras": [
                ExtraLora(
                    path=known["detailed_hands"],
                    name="detailed_hands",
                    weight=0.45,
                    trigger="hand, detailed hands",
                )
            ],
            "poses": ALL_POSES,
            "prompt_trigger": "hand, detailed hands",
        }
    if "better_hands" in known:
        specs["arm_b"] = {
            "label": "better_hands @ 0.40",
            "extras": [
                ExtraLora(
                    path=known["better_hands"],
                    name="better_hands",
                    weight=0.40,
                    trigger="Perfect hand, Detailed hand",
                )
            ],
            "poses": ALL_POSES,
            "prompt_trigger": "Perfect hand, Detailed hand",
        }
    if "real_feet" in known:
        specs["arm_c"] = {
            "label": "real_feet @ 0.40",
            "extras": [
                ExtraLora(
                    path=known["real_feet"],
                    name="real_feet",
                    weight=0.40,
                    trigger="feet",
                )
            ],
            "poses": FOOT_POSES,
            "prompt_trigger": "feet",
        }
    return specs


def _save(img: Image.Image, path: Path, meta: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path, quality=95)
    path.with_suffix(path.suffix + ".json").write_text(
        json.dumps(meta, indent=2, default=str), encoding="utf-8"
    )


def generate_base(pipe, catalog, pose, seed: int, steps: int, cfg: float, trigger: str):
    glue = SCENE_GLUE.get(pose.id, "")
    prompt = catalog.build_prompt(pose, glue)
    if trigger:
        # Keep trigger early (after identity) for CLIP
        parts = prompt.split(", ", 2)
        if len(parts) >= 2:
            prompt = f"{parts[0]}, {parts[1]}, {trigger}, " + (
                parts[2] if len(parts) > 2 else ""
            )
        else:
            prompt = f"{prompt}, {trigger}"
    negative = catalog.build_negative("sfw" if pose.track == "sfw" else "nsfw_private")
    last_mean = 0.0
    for attempt in range(3):
        _mps_cleanup()
        use_seed = seed + attempt * 3331
        gen = torch.Generator(device="cpu").manual_seed(use_seed)
        image = pipe(
            prompt=prompt,
            negative_prompt=negative,
            width=832,
            height=1216,
            num_inference_steps=steps,
            guidance_scale=cfg,
            generator=gen,
        ).images[0]
        arr = np.asarray(image.convert("RGB"), dtype=np.float32)
        last_mean = float(arr.mean())
        if np.isfinite(arr).all() and last_mean >= 5:
            if attempt:
                print(f"recovered black-frame on attempt {attempt+1} seed={use_seed}")
            return image, prompt, negative, use_seed
        print(f"WARN black/NaN frame mean={last_mean} attempt={attempt+1}; retrying")
        _mps_cleanup()
    raise RuntimeError(f"black frame mean={last_mean} after retries")


def run_worker(
    arm: str,
    run_id: str,
    seed: int,
    steps: int,
    cfg: float,
    char_lora: float,
    pose_filter: str = "",
) -> int:
    known = known_specialty_loras()
    specs = _arm_specs(known)
    if arm not in specs:
        print(f"unknown arm {arm}; available={list(specs)}")
        return 2
    spec = specs[arm]
    catalog = load_pose_catalog()
    out_arm = OUT_ROOT / run_id / arm
    out_arm.mkdir(parents=True, exist_ok=True)

    poses = list(spec["poses"])
    if pose_filter.strip():
        poses = [p.strip() for p in pose_filter.split(",") if p.strip()]
        for p in poses:
            if p not in spec["poses"] and p not in ALL_POSES:
                print(f"WARN pose {p} not in arm default set; still trying")

    print(f"WORKER arm={arm} label={spec['label']} poses={poses}")
    print(f"known specialty: { {k: str(v) for k, v in known.items()} }")
    extras = list(spec["extras"])
    # One pipe per worker process — orchestrator isolates per pose to avoid MPS NaN cascade
    pipe = load_txt2img(lora_scale=char_lora, extra_loras=extras)
    results: list[dict] = []
    t0 = time.time()
    try:
        for pose_id in poses:
            pose = catalog.get(pose_id)
            out_dir = out_arm / pose_id
            pose_seed = seed + ALL_POSES.index(pose_id) * 97
            print(f"\n=== [{arm}] {pose_id} seed={pose_seed} ===")
            try:
                image, prompt, negative, used_seed = generate_base(
                    pipe,
                    catalog,
                    pose,
                    pose_seed,
                    steps,
                    cfg,
                    spec.get("prompt_trigger", ""),
                )
                path = out_dir / "m1_catalog.jpg"
                meta = {
                    "run_id": run_id,
                    "arm": arm,
                    "arm_label": spec["label"],
                    "method": "1_pose_catalog_lora_ab",
                    "pose_id": pose_id,
                    "seed": used_seed,
                    "steps": steps,
                    "cfg": cfg,
                    "character_lora": char_lora,
                    "extra_loras": [
                        {
                            "name": e.name,
                            "weight": e.weight,
                            "path": str(e.path),
                            "trigger": e.trigger,
                        }
                        for e in extras
                    ],
                    "prompt": prompt,
                    "negative": negative,
                }
                _save(image, path, meta)
                require_feet = "feet" in pose.regions
                qc = evaluate_anatomy_image(
                    path,
                    pose_expected_hands=pose.expected_visible_hands,
                    track="sfw",
                    require_feet=require_feet,
                    require_genitals=False,
                )
                (out_dir / "m1_qc.json").write_text(json.dumps(qc, indent=2), encoding="utf-8")
                card = scorecard_template(
                    image_path=str(path),
                    method=f"lora_ab_{arm}",
                    pose_id=pose_id,
                    track="sfw",
                )
                card["auto_qc"] = qc
                card["arm"] = arm
                write_scorecard(OPS / "scorecards" / run_id / f"{arm}_{pose_id}_m1.json", card)
                row = {
                    "arm": arm,
                    "pose_id": pose_id,
                    "path": str(path),
                    "seed": used_seed,
                    "overall_pass": qc["overall_auto_pass"],
                    "hands_pass": qc["hands"]["pass"],
                    "hands": qc["hands"],
                    "feet_pass": qc["feet"]["pass"],
                    "identity_pass": qc["identity"]["pass"],
                }
                results.append(row)
                print(
                    f"OK pass={qc['overall_auto_pass']} hands={qc['hands']['pass']} "
                    f"feet={qc['feet']['pass']} id={qc['identity']['pass']}"
                )
                # Persist per-pose result for orchestrator merge (process-isolated)
                (out_dir / "pose_result.json").write_text(
                    json.dumps(row, indent=2), encoding="utf-8"
                )
            except Exception as exc:  # noqa: BLE001
                print(f"ERROR {pose_id}: {type(exc).__name__}: {exc}")
                row = {
                    "arm": arm,
                    "pose_id": pose_id,
                    "error": f"{type(exc).__name__}: {exc}",
                    "overall_pass": False,
                    "hands_pass": False,
                    "feet_pass": False,
                }
                results.append(row)
                (out_dir / "pose_result.json").write_text(
                    json.dumps(row, indent=2), encoding="utf-8"
                )
            _mps_cleanup()
    finally:
        unload(pipe)

    summary = {
        "arm": arm,
        "label": spec["label"],
        "elapsed_sec": round(time.time() - t0, 1),
        "results": results,
        "counts": {
            "overall_pass": sum(1 for r in results if r.get("overall_pass")),
            "hands_pass": sum(1 for r in results if r.get("hands_pass")),
            "feet_pass": sum(1 for r in results if r.get("feet_pass")),
            "total": len(results),
        },
    }
    # Only overwrite full arm summary when running all poses in one worker
    if not pose_filter.strip():
        (out_arm / "arm_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print("ARM SUMMARY", json.dumps(summary["counts"]))
    else:
        (out_arm / f"partial_{poses[0]}.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )
        print("POSE SUMMARY", json.dumps(summary["counts"]))
    return 0


def _pick_hand_winner(arm_summaries: dict[str, dict]) -> str | None:
    """Prefer arm with best hand pass rate among arm_a/arm_b; tie → arm_a (smaller)."""
    candidates = []
    for arm in ("arm_a", "arm_b"):
        s = arm_summaries.get(arm)
        if not s:
            continue
        rows = [r for r in s.get("results", []) if r.get("pose_id") in HAND_POSES]
        if not rows:
            continue
        hp = sum(1 for r in rows if r.get("hands_pass"))
        candidates.append((hp / len(rows), -len(rows), arm))
    if not candidates:
        return None
    candidates.sort(reverse=True)
    return candidates[0][2]


def run_combo_worker(
    run_id: str,
    hand_arm: str,
    seed: int,
    steps: int,
    cfg: float,
    char_lora: float,
) -> int:
    """Optional arm_d: best hand LoRA + RealFeet on one foot-visible pose."""
    known = known_specialty_loras()
    specs = _arm_specs(known)
    if hand_arm not in specs or "real_feet" not in known:
        print("combo skipped: missing hand winner or real_feet")
        return 0
    hand_extra = list(specs[hand_arm]["extras"])
    feet_extra = ExtraLora(
        path=known["real_feet"],
        name="real_feet",
        weight=0.40,
        trigger="feet",
    )
    extras = hand_extra + [feet_extra]
    trigger = ", ".join(
        t for t in [specs[hand_arm].get("prompt_trigger", ""), "feet"] if t
    )
    pose_id = "foot_seated_ankle_soft"  # often shows soft hands + feet
    catalog = load_pose_catalog()
    arm = "arm_d"
    out_arm = OUT_ROOT / run_id / arm
    print(f"WORKER arm={arm} combo hand={hand_arm}+real_feet pose={pose_id}")
    pipe = load_txt2img(lora_scale=char_lora, extra_loras=extras)
    t0 = time.time()
    results: list[dict] = []
    try:
        pose = catalog.get(pose_id)
        pose_seed = seed + ALL_POSES.index(pose_id) * 97
        image, prompt, negative, used_seed = generate_base(
            pipe, catalog, pose, pose_seed, steps, cfg, trigger
        )
        path = out_arm / pose_id / "m1_catalog.jpg"
        meta = {
            "run_id": run_id,
            "arm": arm,
            "arm_label": f"{hand_arm}+real_feet @ 0.4",
            "hand_source_arm": hand_arm,
            "method": "1_pose_catalog_lora_ab",
            "pose_id": pose_id,
            "seed": used_seed,
            "character_lora": char_lora,
            "extra_loras": [
                {"name": e.name, "weight": e.weight, "path": str(e.path)} for e in extras
            ],
            "prompt": prompt,
            "negative": negative,
        }
        _save(image, path, meta)
        qc = evaluate_anatomy_image(
            path,
            pose_expected_hands=pose.expected_visible_hands,
            track="sfw",
            require_feet=True,
            require_genitals=False,
        )
        (out_arm / pose_id / "m1_qc.json").write_text(json.dumps(qc, indent=2), encoding="utf-8")
        results.append(
            {
                "arm": arm,
                "pose_id": pose_id,
                "path": str(path),
                "overall_pass": qc["overall_auto_pass"],
                "hands_pass": qc["hands"]["pass"],
                "feet_pass": qc["feet"]["pass"],
                "identity_pass": qc["identity"]["pass"],
            }
        )
        print(f"OK combo pass={qc['overall_auto_pass']}")
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR combo: {type(exc).__name__}: {exc}")
        results.append({"arm": arm, "error": str(exc), "overall_pass": False})
    finally:
        unload(pipe)
    summary = {
        "arm": arm,
        "label": f"{hand_arm}+real_feet",
        "elapsed_sec": round(time.time() - t0, 1),
        "results": results,
        "counts": {
            "overall_pass": sum(1 for r in results if r.get("overall_pass")),
            "total": len(results),
        },
    }
    out_arm.mkdir(parents=True, exist_ok=True)
    (out_arm / "arm_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return 0


def write_results_md(run_id: str, arm_summaries: dict[str, dict], winner: str | None) -> Path:
    out_dir = OUT_ROOT / run_id
    lines = [
        "# Hand / Foot LoRA A/B Results",
        "",
        f"Date: 2026-07-29",
        f"Run: `{run_id}`",
        f"Stack base: RealVisXL V5 + `rina_park_person` @ {CHAR_LORA}",
        f"Seeds: fixed from base {SEED} (+ pose offset ×97); steps={STEPS} cfg={CFG}",
        f"Poses: {', '.join(ALL_POSES)}",
        f"Output: `out/anatomy_lock/{run_id}/`",
        "",
        "## Design",
        "",
        "| Arm | Stack | Poses |",
        "|-----|-------|-------|",
        "| baseline | character only | all |",
        "| arm_a | + detailed_hands @ 0.45 | all |",
        "| arm_b | + Better Hands @ 0.40 | all |",
        "| arm_c | + RealFeet @ 0.40 | foot only |",
        "| arm_d | best hand + RealFeet @ 0.40 | foot_seated (optional) |",
        "",
        "Method: catalog base gen only (no FaceDetailer / 2-pass / ControlNet).",
        "",
        "## Auto QC pass rates",
        "",
        "| Arm | Overall pass | Hands pass | Feet pass | N | Elapsed |",
        "|-----|--------------|------------|-----------|---|---------|",
    ]
    for arm in ("baseline", "arm_a", "arm_b", "arm_c", "arm_d"):
        s = arm_summaries.get(arm)
        if not s:
            continue
        c = s.get("counts", {})
        n = c.get("total", len(s.get("results", [])))
        lines.append(
            f"| {arm} | {c.get('overall_pass', 0)}/{n} | "
            f"{c.get('hands_pass', sum(1 for r in s.get('results', []) if r.get('hands_pass')))}/{n} | "
            f"{c.get('feet_pass', sum(1 for r in s.get('results', []) if r.get('feet_pass')))}/{n} | "
            f"{n} | {s.get('elapsed_sec', '?')}s |"
        )

    lines.extend(["", "## Per pose", ""])
    lines.append("| Arm | Pose | Overall | Hands | Feet | Identity | Path |")
    lines.append("|-----|------|---------|-------|------|----------|------|")
    for arm in ("baseline", "arm_a", "arm_b", "arm_c", "arm_d"):
        s = arm_summaries.get(arm)
        if not s:
            continue
        for r in s.get("results", []):
            if r.get("error"):
                lines.append(
                    f"| {arm} | {r.get('pose_id', '?')} | ERR | — | — | — | {r['error'][:60]} |"
                )
                continue
            rel = r.get("path", "")
            if rel.startswith(str(RINA)):
                rel = rel[len(str(RINA)) + 1 :]
            lines.append(
                f"| {arm} | {r.get('pose_id')} | {r.get('overall_pass')} | "
                f"{r.get('hands_pass')} | {r.get('feet_pass')} | {r.get('identity_pass')} | `{rel}` |"
            )

    # Hand-focused comparison on HAND_POSES
    lines.extend(["", "## Hand arms (resting + pockets only)", ""])
    lines.append("| Arm | Hands pass / N | Notes |")
    lines.append("|-----|----------------|-------|")
    for arm in ("baseline", "arm_a", "arm_b"):
        s = arm_summaries.get(arm)
        if not s:
            continue
        rows = [r for r in s.get("results", []) if r.get("pose_id") in HAND_POSES]
        hp = sum(1 for r in rows if r.get("hands_pass"))
        lines.append(f"| {arm} | {hp}/{len(rows)} | {s.get('label', '')} |")

    win_label = {
        "arm_a": "detailed_hands @ **0.45** (trigger: `hand` / `detailed hands`)",
        "arm_b": "Better Hands @ **0.40** (trigger: `Perfect hand,` / `Detailed hand,`)",
        None: "none clear — keep character-only + pose whitelist",
    }.get(winner, winner or "undetermined")

    lines.extend(
        [
            "",
            "## Recommendation",
            "",
            f"**Hand LoRA winner:** {win_label}",
            "",
            "Production stack (tentative from this auto-QC run — confirm visually):",
            "",
            "```",
            f"rina_park_person_sdxl_lora.safetensors  @ {CHAR_LORA}",
        ]
    )
    if winner == "arm_a":
        lines.append("detailed_hands-000002.safetensors     @ 0.45")
    elif winner == "arm_b":
        lines.append("better_hands_sdxl_v1.safetensors      @ 0.40  # symlink → Better Hands SDXL v1.0")
    else:
        lines.append("# no specialty hand LoRA until visual review")
    lines.extend(
        [
            "# feet poses only:",
            "real_feet_xl_v1.safetensors               @ 0.40  # if arm_c/arm_d helps",
            "```",
            "",
            "## Next stage gate",
            "",
            "- If hand winner clearly beats baseline on `hand_resting_lap_seated` → promote to default stack.",
            "- Hard grips (cup/tote) still out of production default until winner proven there.",
            "- Do not stack both hand LoRAs.",
            "",
        ]
    )
    path = OPS / "HAND_FOOT_LORA_AB_RESULTS.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    # also copy under run dir
    (out_dir / "HAND_FOOT_LORA_AB_RESULTS.md").write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
    return path


def _merge_arm_summary(out_root: Path, arm: str, label: str) -> dict:
    arm_dir = out_root / arm
    rows: list[dict] = []
    if arm_dir.exists():
        for pose_dir in sorted(arm_dir.iterdir()):
            if not pose_dir.is_dir():
                continue
            pr = pose_dir / "pose_result.json"
            if pr.exists():
                rows.append(json.loads(pr.read_text(encoding="utf-8")))
    summary = {
        "arm": arm,
        "label": label,
        "results": rows,
        "counts": {
            "overall_pass": sum(1 for r in rows if r.get("overall_pass")),
            "hands_pass": sum(1 for r in rows if r.get("hands_pass")),
            "feet_pass": sum(1 for r in rows if r.get("feet_pass")),
            "total": len(rows),
        },
    }
    arm_dir.mkdir(parents=True, exist_ok=True)
    (arm_dir / "arm_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def run_orchestrator(args: argparse.Namespace) -> int:
    known = known_specialty_loras()
    if len(known) < 3:
        print("WARN missing specialty LoRAs:", known)
    run_id = args.run_id or _run_id()
    out_root = OUT_ROOT / run_id
    out_root.mkdir(parents=True, exist_ok=True)
    (OPS / "scorecards" / run_id).mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env.setdefault("SSL_CERT_FILE", str(Path.home() / "combined-cert.pem"))
    env.setdefault("REQUESTS_CA_BUNDLE", env["SSL_CERT_FILE"])
    env.setdefault("NODE_EXTRA_CA_CERTS", env["SSL_CERT_FILE"])
    env.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    env["PYTHONPATH"] = str(RINA) + (
        os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""
    )

    py = str(ROOT / ".venv" / "bin" / "python")
    script = str(Path(__file__).resolve())
    arms = ["baseline", "arm_a", "arm_b", "arm_c"]
    specs = _arm_specs(known)
    arms = [a for a in arms if a in specs]
    print(f"ORCHESTRATOR run_id={run_id} arms={arms} (process-isolate per pose)")
    t0 = time.time()
    for arm in arms:
        for pose_id in specs[arm]["poses"]:
            # Skip if already have a successful image (resume-friendly)
            existing = out_root / arm / pose_id / "m1_catalog.jpg"
            if existing.exists() and existing.stat().st_size > 10_000:
                print(f"skip existing {arm}/{pose_id}")
                continue
            cmd = [
                py,
                "-u",
                script,
                "--worker",
                "--arm",
                arm,
                "--pose",
                pose_id,
                "--run-id",
                run_id,
                "--seed",
                str(args.seed),
                "--steps",
                str(args.steps),
                "--cfg",
                str(args.cfg),
                "--lora",
                str(args.lora),
            ]
            print(f"\n>>>> spawn {arm}/{pose_id}")
            proc = subprocess.run(cmd, cwd=str(ROOT), env=env, check=False)
            if proc.returncode != 0:
                print(f"WARN {arm}/{pose_id} exited {proc.returncode}")

    arm_summaries: dict[str, dict] = {}
    for arm in arms:
        arm_summaries[arm] = _merge_arm_summary(out_root, arm, specs[arm]["label"])

    winner = _pick_hand_winner(arm_summaries)
    print(f"hand winner (auto): {winner}")

    if winner and "real_feet" in known and not args.skip_combo:
        existing = out_root / "arm_d" / "foot_seated_ankle_soft" / "m1_catalog.jpg"
        if not (existing.exists() and existing.stat().st_size > 10_000):
            cmd = [
                py,
                "-u",
                script,
                "--worker",
                "--arm",
                "arm_d",
                "--combo-hand-arm",
                winner,
                "--run-id",
                run_id,
                "--seed",
                str(args.seed),
                "--steps",
                str(args.steps),
                "--cfg",
                str(args.cfg),
                "--lora",
                str(args.lora),
            ]
            print(f"\n>>>> spawn arm_d combo with {winner}")
            subprocess.run(cmd, cwd=str(ROOT), env=env, check=False)
        arm_summaries["arm_d"] = _merge_arm_summary(
            out_root, "arm_d", f"{winner}+real_feet"
        )
        # combo worker writes arm_summary itself; merge pose_result if present
        p = out_root / "arm_d" / "arm_summary.json"
        if p.exists() and arm_summaries["arm_d"]["counts"]["total"] == 0:
            arm_summaries["arm_d"] = json.loads(p.read_text(encoding="utf-8"))

    md = write_results_md(run_id, arm_summaries, winner)
    hand_md = OPS / "HAND_FOOT_LORA.md"
    if hand_md.exists():
        text = hand_md.read_text(encoding="utf-8")
        note = (
            f"\n\n---\n\n## A/B run completed\n\n"
            f"- Run id: `{run_id}`\n"
            f"- Results: `ops/anatomy_lock/HAND_FOOT_LORA_AB_RESULTS.md`\n"
            f"- Auto hand winner: `{winner}`\n"
            f"- Outputs: `out/anatomy_lock/{run_id}/`\n"
        )
        if "## A/B run completed" not in text:
            hand_md.write_text(text.rstrip() + note, encoding="utf-8")
        else:
            head = text.split("## A/B run completed")[0].rstrip()
            hand_md.write_text(head + note, encoding="utf-8")

    full = {
        "run_id": run_id,
        "elapsed_sec": round(time.time() - t0, 1),
        "hand_winner_auto": winner,
        "arms": arm_summaries,
        "outputs": str(out_root),
        "results_md": str(md),
    }
    (OPS / f"run_summary_{run_id}.json").write_text(json.dumps(full, indent=2), encoding="utf-8")
    (out_root / "run_summary.json").write_text(json.dumps(full, indent=2), encoding="utf-8")
    print("\nDONE", json.dumps({"run_id": run_id, "winner": winner, "md": str(md)}, indent=2))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", action="store_true")
    ap.add_argument("--arm", type=str, default="")
    ap.add_argument("--pose", type=str, default="", help="Comma-separated pose filter for worker")
    ap.add_argument("--combo-hand-arm", type=str, default="")
    ap.add_argument("--run-id", type=str, default="")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--steps", type=int, default=STEPS)
    ap.add_argument("--cfg", type=float, default=CFG)
    ap.add_argument("--lora", type=float, default=CHAR_LORA)
    ap.add_argument("--skip-combo", action="store_true")
    args = ap.parse_args()

    if args.worker:
        run_id = args.run_id or _run_id()
        if args.arm == "arm_d":
            if not args.combo_hand_arm:
                print("--combo-hand-arm required for arm_d")
                return 2
            rc = run_combo_worker(
                run_id, args.combo_hand_arm, args.seed, args.steps, args.cfg, args.lora
            )
            # Ensure pose_result exists for merge
            arm_dir = OUT_ROOT / run_id / "arm_d"
            summary_path = arm_dir / "arm_summary.json"
            if summary_path.exists():
                s = json.loads(summary_path.read_text(encoding="utf-8"))
                for r in s.get("results", []):
                    pid = r.get("pose_id")
                    if pid:
                        (arm_dir / pid).mkdir(parents=True, exist_ok=True)
                        (arm_dir / pid / "pose_result.json").write_text(
                            json.dumps(r, indent=2), encoding="utf-8"
                        )
            return rc
        if not args.arm:
            print("--arm required with --worker")
            return 2
        return run_worker(
            args.arm,
            run_id,
            args.seed,
            args.steps,
            args.cfg,
            args.lora,
            pose_filter=args.pose,
        )
    return run_orchestrator(args)


if __name__ == "__main__":
    raise SystemExit(main())
