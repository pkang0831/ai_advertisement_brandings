#!/usr/bin/env python3
"""Promote I2V hero stills into out/i2v_heroes/current/ after human QC.

Examples:
  # Interactive: print candidates
  PYTHONPATH=rina_park .venv/bin/python rina_park/scripts/promote_i2v_heroes.py --run-id 20260730T...

  # Promote specific picks (pose_id:tag without .jpg)
  PYTHONPATH=rina_park .venv/bin/python rina_park/scripts/promote_i2v_heroes.py \\
    --run-id 20260730T... \\
    --pick hand_resting_lap_seated:s00_seed30072026 \\
    --pick hand_pockets_3q:s01_seed... \\
    --pick fitness_mat_soft_seated:s02_seed...

  # Auto: best auto_pass per pose (human scores optional in scorecard)
  PYTHONPATH=rina_park .venv/bin/python rina_park/scripts/promote_i2v_heroes.py --run-id ... --auto-best
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

RINA = Path(__file__).resolve().parents[1]
OUT = RINA / "out" / "i2v_heroes"
CURRENT = OUT / "current"
MAP = RINA / "ops" / "i2v" / "HERO_MOTION_MAP.yml"

DEFAULT_MOTION = {
    "hand_resting_lap_seated": "A2",
    "hand_pockets_3q": "A1",
    "fitness_mat_soft_seated": "A3",
}


def _human_ok(card: dict, pose_id: str) -> bool:
    h = card.get("human") or {}
    # If human not filled, only allow via --auto-best (caller checks)
    if any(h.get(k) is None for k in ("skin", "hands", "identity", "i2v_ready")):
        return False
    skin = int(h["skin"])
    hands = int(h["hands"])
    identity = int(h["identity"])
    i2v = int(h["i2v_ready"])
    if pose_id == "hand_pockets_3q":
        return skin >= 2 and identity >= 2 and i2v >= 2 and hands >= 1
    return skin >= 2 and hands >= 2 and identity >= 2 and i2v >= 2


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--pick", action="append", default=[], help="pose_id:tag")
    ap.add_argument("--auto-best", action="store_true", help="Pick first auto_pass per pose")
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()

    run_dir = OUT / args.run_id
    if not run_dir.is_dir():
        raise SystemExit(f"missing run dir {run_dir}")
    CURRENT.mkdir(parents=True, exist_ok=True)

    candidates: list[tuple[str, Path, dict]] = []
    for score in sorted(run_dir.glob("*/*_scorecard.json")):
        card = json.loads(score.read_text(encoding="utf-8"))
        img = Path(card["image"])
        if not img.is_file():
            continue
        candidates.append((card["pose_id"], img, card))

    if args.list or (not args.pick and not args.auto_best):
        print(f"run={args.run_id} candidates={len(candidates)}")
        for pose_id, img, card in candidates:
            print(
                f"  {pose_id}  auto={card.get('auto_pass')}  "
                f"human={card.get('human')}  {img.name}"
            )
        if not args.pick and not args.auto_best:
            return 0

    picks: dict[str, Path] = {}
    if args.auto_best:
        for pose_id, img, card in candidates:
            if pose_id in picks:
                continue
            if card.get("auto_pass"):
                picks[pose_id] = img
    for spec in args.pick:
        if ":" not in spec:
            raise SystemExit(f"--pick needs pose_id:tag got {spec}")
        pose_id, tag = spec.split(":", 1)
        img = run_dir / pose_id / f"{tag}.jpg"
        if not img.is_file():
            raise SystemExit(f"missing {img}")
        card_path = run_dir / pose_id / f"{tag}_scorecard.json"
        if card_path.is_file():
            card = json.loads(card_path.read_text(encoding="utf-8"))
            if card.get("human", {}).get("skin") is not None and not _human_ok(card, pose_id):
                print(f"WARN human scores below gate for {spec}; promoting anyway (explicit --pick)")
        picks[pose_id] = img

    if not picks:
        raise SystemExit("no picks")

    # Write motion map
    lines = [
        "# Auto-updated by promote_i2v_heroes.py",
        f"# run_id: {args.run_id}",
        "heroes:",
    ]
    for pose_id, img in sorted(picks.items()):
        dest = CURRENT / f"{pose_id}.jpg"
        if dest.exists() or dest.is_symlink():
            dest.unlink()
        dest.symlink_to(img.resolve())
        # also copy json sidecar if present
        side = img.with_suffix(".jpg.json")
        if side.is_file():
            dest_side = CURRENT / f"{pose_id}.jpg.json"
            if dest_side.exists() or dest_side.is_symlink():
                dest_side.unlink()
            dest_side.symlink_to(side.resolve())
        motion = DEFAULT_MOTION.get(pose_id, "A2")
        lines.append(f"  - pose_id: {pose_id}")
        lines.append(f"    still: {dest}")
        lines.append(f"    source: {img}")
        lines.append(f"    motion_prompt: {motion}  # see MOTION_PROMPTS.md")
        print(f"promoted {pose_id} -> {dest} -> {img}")

    MAP.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("wrote", MAP)
    print("current/", list(CURRENT.glob("*.jpg")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
