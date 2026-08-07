#!/usr/bin/env python3
"""Zip Kohya-ready train/val folders for Colab Drive upload."""

from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

CHARACTER_ID = "rina_character_v1"
ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "intake" / CHARACTER_ID / "dataset"
DEFAULT_OUT = ROOT / "intake" / CHARACTER_ID / f"{CHARACTER_ID}_kohya_dataset.zip"


def pack(dataset_dir: Path = DATASET, out_zip: Path = DEFAULT_OUT) -> Path:
    if not dataset_dir.is_dir():
        raise FileNotFoundError(f"dataset missing: {dataset_dir}")
    out_zip.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(dataset_dir.rglob("*")):
            if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg", ".txt", ".json"}:
                # Isolate under character id prefix inside the zip
                arc = Path(CHARACTER_ID) / path.relative_to(dataset_dir)
                zf.write(path, arcname=str(arc))
    return out_zip


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=DATASET)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    out = pack(args.dataset, args.out)
    print(out)
    print(f"bytes={out.stat().st_size}")


if __name__ == "__main__":
    main()
