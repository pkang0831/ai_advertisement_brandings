"""Local hashes, near-duplicate checks, and provenance completeness."""

from __future__ import annotations

import hashlib
from pathlib import Path
import re
from typing import Any, Iterable

from PIL import Image


SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
DEFAULT_LINEAGE_FIELDS = (
    "source_path",
    "source_type",
    "source_parent_sha256",
    "capture_or_generation_record",
    "transform_history",
    "reviewer",
    "reviewed_at_utc",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dhash(path: Path, hash_size: int = 8) -> str:
    """Return a dependency-light difference hash; it is not identity recognition."""
    with Image.open(path) as image:
        pixels = list(
            image.convert("L")
            .resize((hash_size + 1, hash_size), Image.Resampling.LANCZOS)
            .getdata()
        )
    bits = [
        pixels[row * (hash_size + 1) + column]
        > pixels[row * (hash_size + 1) + column + 1]
        for row in range(hash_size)
        for column in range(hash_size)
    ]
    value = sum(int(bit) << index for index, bit in enumerate(bits))
    return f"{value:0{hash_size * hash_size // 4}x}"


def hamming_distance(left: str, right: str) -> int:
    if len(left) != len(right):
        raise ValueError("perceptual hashes must have equal length")
    return (int(left, 16) ^ int(right, 16)).bit_count()


def build_integrity_report(
    image_path: Path,
    *,
    recorded_sha256: str | None,
    known_assets: Iterable[dict[str, Any]] = (),
    near_duplicate_distance: int = 5,
) -> dict[str, Any]:
    source_hash = sha256_file(image_path)
    perceptual_hash = dhash(image_path)
    matches: list[dict[str, Any]] = []
    for candidate in known_assets:
        exact = candidate.get("source_sha256") == source_hash
        candidate_phash = candidate.get("perceptual_hash")
        distance = (
            hamming_distance(perceptual_hash, candidate_phash)
            if isinstance(candidate_phash, str) and len(candidate_phash) == len(perceptual_hash)
            else None
        )
        if exact or (distance is not None and distance <= near_duplicate_distance):
            matches.append(
                {
                    "asset_id": candidate.get("asset_id"),
                    "exact": exact,
                    "perceptual_distance": distance,
                }
            )
    return {
        "status": "complete",
        "algorithm": "sha256+dhash64",
        "source_sha256": source_hash,
        "perceptual_hash": perceptual_hash,
        "source_hash_matches": recorded_sha256 == source_hash,
        "near_duplicate_distance": near_duplicate_distance,
        "duplicate": bool(matches),
        "matches": matches,
    }


def inspect_provenance(
    asset: dict[str, Any],
    *,
    required_lineage_fields: Iterable[str] = DEFAULT_LINEAGE_FIELDS,
) -> dict[str, Any]:
    missing = [field for field in required_lineage_fields if not asset.get(field)]
    for field in ("source_sha256", "mask_sha256"):
        if not SHA256_RE.fullmatch(str(asset.get(field, ""))):
            missing.append(field)
    for field in ("mask_path", "caption", "license"):
        if not asset.get(field):
            missing.append(field)
    return {
        "status": "complete",
        "complete": not missing,
        "lineage_complete": not any(field in missing for field in required_lineage_fields),
        "hashes_complete": not any(field in missing for field in ("source_sha256", "mask_sha256")),
        "mask_complete": not any(field in missing for field in ("mask_path", "mask_sha256")),
        "missing": sorted(set(missing)),
    }
