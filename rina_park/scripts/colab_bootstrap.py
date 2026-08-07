#!/usr/bin/env python3
"""Wire Colab session to Drive-backed rina_park assets.

Assumes Google Drive is already mounted at /content/drive.
Creates Drive layout, symlinks rina_park/{models,identity,moodboard,private,out}
to Drive, and prints a preflight summary.

Safety: never renames/moves a non-empty local directory unless --force.
On a fresh Colab clone, gitignored dirs are usually missing → safe to link.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

DRIVE_ROOT_DEFAULT = Path("/content/drive/MyDrive/rina_park_colab")
REPO_DEFAULT = Path("/content/ai_influencer")
LINK_NAMES = ("models", "identity", "moodboard", "private", "out")


def _is_empty_dir(path: Path) -> bool:
    if not path.is_dir() or path.is_symlink():
        return False
    try:
        next(path.iterdir())
    except StopIteration:
        return True
    return False


def _symlink(link: Path, target: Path, *, force: bool) -> str:
    target.mkdir(parents=True, exist_ok=True)
    if link.is_symlink():
        current = link.resolve()
        if current == target.resolve():
            return f"ok symlink {link} → {target}"
        if not force:
            raise SystemExit(
                f"Existing symlink {link} → {current} differs from {target}. "
                "Pass --force to replace."
            )
        link.unlink()
    elif link.exists():
        if _is_empty_dir(link):
            link.rmdir()
        elif force:
            backup = link.with_name(link.name + ".pre_drive_bak")
            if backup.exists():
                raise SystemExit(f"Backup already exists: {backup}")
            link.rename(backup)
            return f"backed up {link} → {backup}; linking → {target}"
        else:
            raise SystemExit(
                f"Refusing to replace non-empty path {link}. "
                "On Colab this is unexpected after a fresh clone. "
                "Pass --force only if you intend to move it aside."
            )
    link.symlink_to(target, target_is_directory=True)
    return f"linked {link} → {target}"


def bootstrap(repo: Path, drive_root: Path, *, force: bool) -> dict:
    rina = repo / "rina_park"
    if not rina.is_dir():
        raise SystemExit(f"Missing rina_park at {rina} — clone the repo first")

    drive_root.mkdir(parents=True, exist_ok=True)
    messages = []
    for name in LINK_NAMES:
        msg = _symlink(rina / name, drive_root / name, force=force)
        messages.append(msg)

    repo_s = str(repo.resolve())
    rina_s = str(rina.resolve())
    path_parts = [p for p in os.environ.get("PYTHONPATH", "").split(":") if p]
    for p in (repo_s, rina_s):
        if p not in path_parts:
            path_parts.insert(0, p)
    os.environ["PYTHONPATH"] = ":".join(path_parts)

    models = rina / "models"
    checks = {
        "repo": str(repo),
        "drive_root": str(drive_root),
        "models_dir": str(models),
        "models_is_symlink": models.is_symlink(),
        "realvis": (models / "checkpoints" / "RealVisXL_V5.0_fp16.safetensors").exists(),
        "character_lora": (models / "loras" / "rina_park_person_sdxl_lora.safetensors").exists(),
        "cuda": False,
        "mps": False,
        "device": "cpu",
    }
    try:
        import torch

        checks["cuda"] = bool(torch.cuda.is_available())
        checks["mps"] = bool(
            getattr(torch.backends, "mps", None)
            and torch.backends.mps.is_available()
        )
        if checks["cuda"]:
            checks["device"] = "cuda"
        elif checks["mps"]:
            checks["device"] = "mps"
    except ImportError:
        checks["torch"] = False

    return {"messages": messages, "checks": checks, "PYTHONPATH": os.environ["PYTHONPATH"]}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo", type=Path, default=REPO_DEFAULT)
    ap.add_argument("--drive-root", type=Path, default=DRIVE_ROOT_DEFAULT)
    ap.add_argument(
        "--force",
        action="store_true",
        help="Allow replacing existing non-empty dirs (moves to *.pre_drive_bak)",
    )
    ap.add_argument(
        "--allow-local",
        action="store_true",
        help="Allow running outside Colab (still refuses non-empty dirs unless --force)",
    )
    args = ap.parse_args()

    on_colab = Path("/content/drive").exists() or Path("/content").exists()
    if not on_colab and not args.allow_local:
        raise SystemExit(
            "Not on Colab (/content missing). Refusing to touch local dirs. "
            "Use Colab, or pass --allow-local for an intentional dry-run."
        )

    if not args.repo.exists():
        local = Path(__file__).resolve().parents[2]
        if (local / "rina_park").is_dir():
            args.repo = local
            print(f"repo fallback → {args.repo}")

    result = bootstrap(args.repo, args.drive_root, force=args.force)
    for m in result["messages"]:
        print(m)
    print("PYTHONPATH=", result["PYTHONPATH"])
    print("preflight:")
    for k, v in result["checks"].items():
        print(f"  {k}: {v}")

    if not result["checks"].get("realvis"):
        print(
            "\nNOTE: RealVis checkpoint missing — run colab_download_hf_models.py --tier sdxl",
            file=sys.stderr,
        )
    if not result["checks"].get("character_lora"):
        print(
            "\nNOTE: character LoRA missing — run sync_local_only_to_drive.sh from Mac",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
