#!/usr/bin/env python3
"""Download HF models from colab_model_manifest.yml into Drive (or local) models root.

Usage (Colab):
  python rina_park/scripts/colab_download_hf_models.py --tier sdxl
  python rina_park/scripts/colab_download_hf_models.py --tier all --models-root /content/drive/MyDrive/rina_park_colab/models
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
DEFAULT_DRIVE_MODELS = Path("/content/drive/MyDrive/rina_park_colab/models")

TIER_EXPAND = {
    "sdxl": {"sdxl"},
    "wan": {"wan"},
    "qwen_cuda": {"qwen_cuda"},
    "all": {"sdxl", "wan", "qwen_cuda"},
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


def _dest(models_root: Path, local_path: str) -> Path:
    return models_root / local_path


def _download_file(
    *,
    repo_id: str,
    revision: str | None,
    filename: str,
    dest: Path,
) -> None:
    from huggingface_hub import hf_hub_download

    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        print(f"  skip existing file: {dest}")
        return
    kwargs: dict[str, Any] = {"repo_id": repo_id, "filename": filename}
    if revision:
        kwargs["revision"] = revision
    try:
        cached = hf_hub_download(**kwargs)
    except Exception as first_err:
        # RealVis / RealESRGAN filename variants
        alts = []
        if filename.endswith("_fp16.safetensors"):
            alts.append(filename.replace("_fp16.safetensors", ".safetensors"))
        if filename == "RealESRGAN_x2plus.pth":
            # xinntao/Real-ESRGAN stores weights under weights/
            alts.append("weights/RealESRGAN_x2plus.pth")
        last = first_err
        cached = None
        for alt in alts:
            try:
                print(f"  retry filename={alt}")
                kwargs["filename"] = alt
                cached = hf_hub_download(**kwargs)
                break
            except Exception as e:  # noqa: BLE001
                last = e
        if cached is None:
            raise last from first_err
    src = Path(cached)
    if dest.resolve() != src.resolve():
        shutil.copy2(src, dest)
    print(f"  wrote {dest} ({dest.stat().st_size} bytes)")


def _download_snapshot(
    *,
    repo_id: str,
    revision: str | None,
    dest: Path,
    allow_patterns: list[str] | None,
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
        "local_dir_use_symlinks": False,
        "resume_download": True,
    }
    if revision:
        kwargs["revision"] = revision
    if allow_patterns:
        kwargs["allow_patterns"] = allow_patterns
    print(f"  snapshot_download → {dest}")
    snapshot_download(**kwargs)
    marker.write_text(repo_id + "\n", encoding="utf-8")
    print(f"  done {dest}")


def download_entry(entry: dict[str, Any], models_root: Path) -> None:
    eid = entry["id"]
    kind = entry["kind"]
    dest = _dest(models_root, entry["local_path"])
    print(f"\n[{eid}] {kind} {entry['repo_id']} → {dest}")
    if kind == "file":
        _download_file(
            repo_id=entry["repo_id"],
            revision=entry.get("revision"),
            filename=entry["filename"],
            dest=dest,
        )
    elif kind == "snapshot":
        _download_snapshot(
            repo_id=entry["repo_id"],
            revision=entry.get("revision"),
            dest=dest,
            allow_patterns=entry.get("allow_patterns"),
        )
    else:
        raise SystemExit(f"Unknown kind={kind} for {eid}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--tier",
        choices=sorted(TIER_EXPAND),
        default="sdxl",
        help="Which manifest tiers to download",
    )
    ap.add_argument(
        "--models-root",
        type=Path,
        default=None,
        help="Destination models root (default: Drive path if present, else rina_park/models)",
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

    if args.models_root is not None:
        models_root = args.models_root
    elif DEFAULT_DRIVE_MODELS.parent.exists():
        models_root = DEFAULT_DRIVE_MODELS
    else:
        models_root = REPO / "rina_park" / "models"

    models_root.mkdir(parents=True, exist_ok=True)
    print("models_root:", models_root)
    print("tier:", args.tier, "entries:", len(entries))
    _ensure_token()

    errors: list[str] = []
    for entry in entries:
        try:
            download_entry(entry, models_root)
        except Exception as e:  # noqa: BLE001
            msg = f"{entry['id']}: {type(e).__name__}: {e}"
            print("ERROR", msg, file=sys.stderr)
            errors.append(msg)

    if errors:
        print(f"\n{len(errors)} download(s) failed:", file=sys.stderr)
        for m in errors:
            print(" -", m, file=sys.stderr)
        raise SystemExit(1)
    print("\nAll downloads OK.")


if __name__ == "__main__":
    main()
