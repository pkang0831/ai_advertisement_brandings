#!/usr/bin/env python3
"""Download HF models from colab_model_manifest.yml into Drive models root.

IMPORTANT (Colab): HF hub cache defaults to ~/.cache and fills the local disk.
This script forces cache + destination onto Google Drive.

Usage:
  python rina_park/scripts/colab_download_hf_models.py --tier sdxl
  python rina_park/scripts/colab_download_hf_models.py --tier wan \\
      --models-root /content/drive/MyDrive/rina_park_colab/models

Do NOT use --tier all on a small Colab disk unless Drive cache is set (default here).
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None  # type: ignore


SCRIPT = Path(__file__).resolve()
REPO = SCRIPT.parents[2]
DEFAULT_MANIFEST = REPO / "rina_park" / "ops" / "colab_model_manifest.yml"
DEFAULT_DRIVE_ROOT = Path("/content/drive/MyDrive/rina_park_colab")
DEFAULT_DRIVE_MODELS = DEFAULT_DRIVE_ROOT / "models"
DEFAULT_DRIVE_HF_HOME = DEFAULT_DRIVE_ROOT / ".hf_home"

TIER_EXPAND = {
    "sdxl": {"sdxl"},
    "wan": {"wan"},
    "qwen_cuda": {"qwen_cuda"},
    "all": {"sdxl", "wan", "qwen_cuda"},
}

# Rough sizes for warnings (GiB)
TIER_SIZE_GIB = {
    "sdxl": 15,
    "wan": 80,
    "qwen_cuda": 100,
    "all": 200,
}


def _load_manifest(path: Path) -> dict[str, Any]:
    if yaml is None:
        raise SystemExit("PyYAML required: pip install pyyaml")
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or "entries" not in data:
        raise SystemExit(f"Invalid manifest: {path}")
    return data


def _ensure_token() -> None:
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if token:
        os.environ.setdefault("HUGGING_FACE_HUB_TOKEN", token)
        print("HF token: present")
    else:
        print("HF token: not set (public repos only)")


def _force_hf_cache_on_drive(hf_home: Path) -> None:
    """Put Hugging Face cache on Drive so /content does not fill up."""
    hf_home.mkdir(parents=True, exist_ok=True)
    hub = hf_home / "hub"
    hub.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(hf_home)
    os.environ["HUGGINGFACE_HUB_CACHE"] = str(hub)
    os.environ["HF_HUB_CACHE"] = str(hub)
    # Avoid writing XDG cache under /root
    os.environ.setdefault("XDG_CACHE_HOME", str(hf_home / "xdg_cache"))
    print("HF_HOME:", hf_home)
    print("HF_HUB_CACHE:", hub)


def _is_under_drive(path: Path) -> bool:
    try:
        resolved = path.resolve()
    except OSError:
        resolved = path
    parts = resolved.parts
    return "drive" in parts or str(resolved).startswith("/content/drive")


def _dest(models_root: Path, local_path: str) -> Path:
    return models_root / local_path


def _download_file(
    *,
    repo_id: str,
    revision: str | None,
    filename: str,
    dest: Path,
    cache_dir: Path,
) -> None:
    from huggingface_hub import hf_hub_download

    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        print(f"  skip existing file: {dest}")
        return

    def _one(name: str) -> Path:
        kwargs: dict[str, Any] = {
            "repo_id": repo_id,
            "filename": name,
            "cache_dir": str(cache_dir),
            "local_dir": str(dest.parent),
            "local_dir_use_symlinks": False,
        }
        if revision:
            kwargs["revision"] = revision
        # Newer hub: prefer local_dir only; older still accepts cache_dir
        try:
            return Path(hf_hub_download(**kwargs))
        except TypeError:
            kwargs.pop("local_dir_use_symlinks", None)
            return Path(hf_hub_download(**kwargs))

    try:
        cached = _one(filename)
    except Exception as first_err:
        alts = []
        if filename.endswith("_fp16.safetensors"):
            alts.append(filename.replace("_fp16.safetensors", ".safetensors"))
        if filename == "RealESRGAN_x2plus.pth":
            alts.append("weights/RealESRGAN_x2plus.pth")
        last = first_err
        cached = None
        for alt in alts:
            try:
                print(f"  retry filename={alt}")
                cached = _one(alt)
                # If downloaded under alt name, rename to expected dest
                if cached.exists() and cached.name != dest.name:
                    shutil.move(str(cached), str(dest))
                    cached = dest
                break
            except Exception as e:  # noqa: BLE001
                last = e
        if cached is None:
            raise last from first_err

    # If hub wrote a different path in local_dir, ensure dest exists
    if cached.exists() and dest.resolve() != cached.resolve():
        if not dest.exists():
            shutil.copy2(cached, dest)
    if not dest.exists():
        # Fallback: file may be named as filename inside parent
        candidate = dest.parent / Path(filename).name
        if candidate.exists():
            if candidate != dest:
                shutil.move(str(candidate), str(dest))
        else:
            raise FileNotFoundError(f"download finished but missing {dest}")
    print(f"  wrote {dest} ({dest.stat().st_size} bytes)")


def _download_snapshot(
    *,
    repo_id: str,
    revision: str | None,
    dest: Path,
    allow_patterns: list[str] | None,
    cache_dir: Path,
) -> None:
    from huggingface_hub import snapshot_download

    dest.mkdir(parents=True, exist_ok=True)
    marker = dest / ".hf_download_complete"
    if marker.exists():
        print(f"  skip existing snapshot: {dest}")
        return
    kwargs: dict[str, Any] = {
        "repo_id": repo_id,
        "local_dir": str(dest),
        "cache_dir": str(cache_dir),
        "local_dir_use_symlinks": False,
        "resume_download": True,
    }
    if revision:
        kwargs["revision"] = revision
    if allow_patterns:
        kwargs["allow_patterns"] = allow_patterns
    print(f"  snapshot_download → {dest}")
    try:
        snapshot_download(**kwargs)
    except TypeError:
        kwargs.pop("local_dir_use_symlinks", None)
        snapshot_download(**kwargs)
    marker.write_text(repo_id + "\n", encoding="utf-8")
    print(f"  done {dest}")


def download_entry(entry: dict[str, Any], models_root: Path, cache_dir: Path) -> None:
    eid = entry["id"]
    kind = entry["kind"]
    dest = _dest(models_root, entry["local_path"])
    print(f"\n[{eid}] {kind} {entry['repo_id']} → {dest}")
    if not _is_under_drive(dest) and Path("/content/drive").exists():
        print(
            "  WARNING: destination is NOT under Google Drive — may fill Colab disk",
            file=sys.stderr,
        )
    if kind == "file":
        _download_file(
            repo_id=entry["repo_id"],
            revision=entry.get("revision"),
            filename=entry["filename"],
            dest=dest,
            cache_dir=cache_dir,
        )
    elif kind == "snapshot":
        _download_snapshot(
            repo_id=entry["repo_id"],
            revision=entry.get("revision"),
            dest=dest,
            allow_patterns=entry.get("allow_patterns"),
            cache_dir=cache_dir,
        )
    else:
        raise SystemExit(f"Unknown kind={kind} for {eid}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--tier",
        choices=sorted(TIER_EXPAND),
        default="sdxl",
        help="Which manifest tiers to download (prefer sdxl first; all is huge)",
    )
    ap.add_argument(
        "--models-root",
        type=Path,
        default=None,
        help="Destination models root (default: Drive path if present)",
    )
    ap.add_argument(
        "--hf-home",
        type=Path,
        default=None,
        help="HF cache home on Drive (default: .../rina_park_colab/.hf_home)",
    )
    ap.add_argument(
        "--require-drive",
        action="store_true",
        default=True,
        help="Fail if models-root is not under /content/drive (default: on)",
    )
    ap.add_argument(
        "--allow-local-disk",
        action="store_true",
        help="Allow downloading to Colab ephemeral disk (dangerous)",
    )
    ap.add_argument(
        "--confirm-large",
        action="store_true",
        help="Required for --tier wan|qwen_cuda|all",
    )
    ap.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    ap.add_argument("--only", nargs="*", default=None, help="Optional entry id filter")
    ap.add_argument("--list", action="store_true", help="List entries and exit")
    args = ap.parse_args()

    manifest = _load_manifest(args.manifest)
    wanted = TIER_EXPAND[args.tier]
    entries = [e for e in manifest["entries"] if e.get("tier") in wanted]
    if args.only:
        only = set(args.only)
        entries = [e for e in entries if e["id"] in only]

    if args.list:
        for e in entries:
            print(f"{e['id']}\ttier={e['tier']}\t{e['kind']}\t{e['repo_id']}\t→ {e['local_path']}")
        return

    if args.tier in {"wan", "qwen_cuda", "all"} and not args.confirm_large:
        approx = TIER_SIZE_GIB.get(args.tier, "?")
        raise SystemExit(
            f"--tier {args.tier} is ~{approx} GiB. "
            "Re-run with --confirm-large after Drive is mounted, "
            "or use --tier sdxl first."
        )

    if args.models_root is not None:
        models_root = args.models_root
    elif DEFAULT_DRIVE_MODELS.parent.exists():
        models_root = DEFAULT_DRIVE_MODELS
    else:
        models_root = REPO / "rina_park" / "models"

    require_drive = args.require_drive and not args.allow_local_disk
    if require_drive and not _is_under_drive(models_root):
        raise SystemExit(
            f"models_root is not on Google Drive: {models_root}\n"
            "Mount Drive and pass:\n"
            "  --models-root /content/drive/MyDrive/rina_park_colab/models\n"
            "Or pass --allow-local-disk (will fill Colab disk)."
        )

    hf_home = args.hf_home or (
        DEFAULT_DRIVE_HF_HOME
        if DEFAULT_DRIVE_ROOT.exists()
        else (models_root.parent / ".hf_home")
    )
    if require_drive and not _is_under_drive(hf_home):
        # Keep cache next to models on Drive when possible
        hf_home = models_root.parent / ".hf_home"

    _force_hf_cache_on_drive(hf_home)
    cache_dir = Path(os.environ["HF_HUB_CACHE"])

    models_root.mkdir(parents=True, exist_ok=True)
    print("models_root:", models_root.resolve())
    print("on_drive:", _is_under_drive(models_root))
    print("tier:", args.tier, "approx_gib:", TIER_SIZE_GIB.get(args.tier), "entries:", len(entries))
    _ensure_token()

    errors: list[str] = []
    for entry in entries:
        try:
            download_entry(entry, models_root, cache_dir)
        except Exception as e:  # noqa: BLE001
            msg = f"{entry['id']}: {type(e).__name__}: {e}"
            print("ERROR", msg, file=sys.stderr)
            errors.append(msg)

    if errors:
        print(f"\n{len(errors)} download(s) failed:", file=sys.stderr)
        for m in errors:
            print(" -", m, file=sys.stderr)
        raise SystemExit(1)
    print("\nAll downloads OK (on Drive).")
    print("Tip: free Colab local cache if needed: rm -rf /root/.cache/huggingface")


if __name__ == "__main__":
    main()
