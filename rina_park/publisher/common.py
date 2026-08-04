from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .attestation import PromotionVerifier, VerifiedPromotion

DISCLOSURE = "Rina is a fictional virtual character. Visuals are AI-generated."
ALLOWED_TIERS = frozenset({"A", "B", "C"})
MATURE_MARKERS = ("mature", "adult", "18+", "nsfw", "explicit")
PROHIBITED_TEXT = (
    "nude",
    "nudity",
    "nipple",
    "genital",
    "explicit",
    "lingerie",
    "sexualized wet-look",
    "reveal",
    "sexual service",
)


class PublisherError(ValueError):
    """An input failed a publisher safety or integrity gate."""


@dataclass(frozen=True)
class ApprovedAsset:
    asset_id: str
    path: Path
    sha256: str
    media_type: str


@dataclass(frozen=True)
class ApprovedExport:
    post_id: str
    manifest: dict[str, Any]
    assets: tuple[ApprovedAsset, ...]
    promotion: VerifiedPromotion | None = None


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def validate_platform_text(manifest: dict[str, Any]) -> None:
    fields = ("title", "body", "cta", "alt_text", "hashtags")
    text = " ".join(str(manifest.get(field, "")) for field in fields).lower()
    for term in PROHIBITED_TEXT:
        if re.search(rf"(?<!\w){re.escape(term)}(?!\w)", text):
            raise PublisherError(f"prohibited SFW term in platform copy: {term}")
    if DISCLOSURE not in str(manifest.get("disclosure", "")):
        raise PublisherError("required English AI disclosure is missing")


def validate_audience_tiers(platform: str, tiers: Any) -> tuple[str, ...]:
    if not isinstance(tiers, list) or not tiers:
        raise PublisherError("audience_tiers must be a non-empty list")
    normalized = tuple(str(tier).upper() for tier in tiers)
    if len(set(normalized)) != len(normalized) or not set(normalized) <= ALLOWED_TIERS:
        raise PublisherError("audience_tiers contains invalid or duplicate values")
    if platform == "instagram" and normalized != ("A", "B", "C"):
        raise PublisherError("Instagram exports must target all audience tiers")
    if platform == "patreon" and normalized not in (
        ("A", "B", "C"),
        ("B", "C"),
        ("C",),
    ):
        raise PublisherError("Patreon audience_tiers violates cumulative tier access")
    return normalized


class ApprovedExportStore:
    """Reads only immutable files explicitly allowlisted by approved_exports/index.json."""

    def __init__(self, root: Path | str):
        self.root = Path(root).expanduser().absolute()
        if not self.root.is_dir() or self.root.is_symlink():
            raise PublisherError("approved export root must be a real directory")
        self.root_real = self.root.resolve(strict=True)
        forbidden_root_parts = {
            "mature",
            "mature_non_explicit",
            "private_media",
            "nsfw_test",
            "nsfw",
        }
        if any(
            part.lower() in forbidden_root_parts
            or part.lower().startswith("mne_")
            for part in self.root_real.parts
        ):
            raise PublisherError("mature roots are forbidden for approved exports")
        self.index_path = self._secure_file(self.root / "index.json")
        self.index = self._read_json(self.index_path)
        if self.index.get("version") != 1 or not isinstance(self.index.get("posts"), dict):
            raise PublisherError("unsupported or malformed approved export index")

    @staticmethod
    def _read_json(path: Path) -> dict[str, Any]:
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise PublisherError(f"invalid JSON: {path.name}") from exc
        if not isinstance(value, dict):
            raise PublisherError(f"JSON object required: {path.name}")
        return value

    def _secure_file(self, path: Path) -> Path:
        try:
            relative = path.absolute().relative_to(self.root)
        except ValueError as exc:
            raise PublisherError("path is outside approved export root") from exc
        if ".." in relative.parts or any(marker in part.lower() for part in relative.parts for marker in MATURE_MARKERS):
            raise PublisherError("mature or traversal path rejected")
        cursor = self.root
        for part in relative.parts:
            cursor = cursor / part
            if cursor.is_symlink():
                raise PublisherError("symlinks are forbidden")
        try:
            resolved = path.resolve(strict=True)
            resolved.relative_to(self.root_real)
            stat = resolved.stat()
        except (OSError, ValueError) as exc:
            raise PublisherError("file escapes or is missing from approved export root") from exc
        if not resolved.is_file() or stat.st_nlink != 1:
            raise PublisherError("only regular, non-hardlinked files are accepted")
        return resolved

    def _indexed_path(self, relative: Any) -> Path:
        if not isinstance(relative, str) or not relative or os.path.isabs(relative):
            raise PublisherError("index paths must be non-empty relative strings")
        return self._secure_file(self.root / relative)

    def load(
        self,
        post_id: str,
        platform: str,
        *,
        verifier: PromotionVerifier | None = None,
        action: str | None = None,
    ) -> ApprovedExport:
        if verifier is None or action not in {"package", "publish"}:
            raise PublisherError(
                "trusted promotion attestation verifier and action are required"
            )
        return self._load_attested(post_id, platform, verifier, action)

    def _load_attested(
        self,
        post_id: str,
        platform: str,
        verifier: PromotionVerifier,
        action: str,
    ) -> ApprovedExport:
        export, entry, manifest_path = self._load_for_promotion(post_id, platform)
        manifest = export.manifest
        assets = export.assets
        attestation_path = self._indexed_path(entry.get("attestation"))
        expected_attestation_hash = entry.get("attestation_sha256")
        if (
            not isinstance(expected_attestation_hash, str)
            or sha256_file(attestation_path) != expected_attestation_hash
        ):
            raise PublisherError("detached attestation checksum mismatch")
        envelope = self._read_json(attestation_path)
        promotion = verifier.verify(
            envelope=envelope,
            root=self.root_real,
            manifest_path=manifest_path,
            manifest=manifest,
            assets=assets,
            platform=platform,
            action=action,
        )
        return ApprovedExport(post_id, manifest, assets, promotion)

    def _load_for_promotion(
        self,
        post_id: str,
        platform: str,
    ) -> tuple[ApprovedExport, dict[str, Any], Path]:
        """Structural read used only before an interactive human signature."""
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", post_id):
            raise PublisherError("invalid post_id")
        entry = self.index["posts"].get(post_id)
        if not isinstance(entry, dict):
            raise PublisherError("post is not allowlisted")
        manifest_path = self._indexed_path(entry.get("manifest"))
        expected_manifest_hash = entry.get("manifest_sha256")
        if not isinstance(expected_manifest_hash, str) or sha256_file(manifest_path) != expected_manifest_hash:
            raise PublisherError("approved manifest checksum mismatch")
        manifest = self._read_json(manifest_path)
        if manifest.get("post_id") != post_id or manifest.get("platform") != platform:
            raise PublisherError("manifest identity or platform mismatch")
        if manifest.get("content_approval") != "approved" or manifest.get("schedule_approval") != "approved":
            raise PublisherError("content and schedule approvals are required")
        validate_audience_tiers(platform, manifest.get("audience_tiers"))
        validate_platform_text(manifest)

        allowlisted = entry.get("asset_ids")
        manifest_assets = manifest.get("assets")
        if not isinstance(allowlisted, list) or not isinstance(manifest_assets, list):
            raise PublisherError("asset allowlist and manifest assets are required")
        if [asset.get("asset_id") for asset in manifest_assets] != allowlisted:
            raise PublisherError("manifest assets do not exactly match the allowlist")

        assets: list[ApprovedAsset] = []
        for record in manifest_assets:
            if not isinstance(record, dict):
                raise PublisherError("malformed asset record")
            asset_id = record.get("asset_id")
            checksum = record.get("sha256")
            if not isinstance(asset_id, str) or not isinstance(checksum, str):
                raise PublisherError("asset ID and checksum are required")
            if any(marker in asset_id.lower() for marker in MATURE_MARKERS):
                raise PublisherError("mature asset injection rejected")
            path = self._indexed_path(record.get("path"))
            if sha256_file(path) != checksum:
                raise PublisherError(f"asset checksum mismatch: {asset_id}")
            assets.append(
                ApprovedAsset(asset_id, path, checksum, str(record.get("media_type", "")))
            )
        return (
            ApprovedExport(post_id, manifest, tuple(assets)),
            entry,
            manifest_path,
        )
