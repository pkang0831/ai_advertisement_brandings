from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from .attestation import PromotionVerifier, copy_attested_asset
from .common import ApprovedExportStore, DISCLOSURE, PublisherError
from .validators import load_capabilities, validate_asset


def build_instagram_package(
    approved_exports: Path | str,
    post_id: str,
    output_root: Path | str,
    *,
    verifier: PromotionVerifier | None = None,
) -> Path:
    verifier = verifier or PromotionVerifier()
    export = ApprovedExportStore(approved_exports).load(
        post_id,
        "instagram",
        verifier=verifier,
        action="package",
    )
    manifest = export.manifest
    post_format = str(manifest.get("format", "")).lower()
    if post_format not in {"image", "carousel", "reel"}:
        raise PublisherError("unsupported Instagram format")
    capabilities = load_capabilities()["instagram"]
    if post_format == "carousel":
        count = len(export.assets)
        carousel = capabilities["carousel"]
        if not carousel["min_items"] <= count <= carousel["max_items"]:
            raise PublisherError("carousel item count violates capabilities")
    if post_format in {"image", "reel"} and len(export.assets) != 1:
        raise PublisherError(f"{post_format} requires exactly one asset")

    validation: list[dict[str, Any]] = []
    for asset in export.assets:
        expected_type = "reel" if post_format == "reel" else "image"
        if asset.media_type != expected_type:
            raise PublisherError("asset type does not match Instagram format")
        validation.append(validate_asset(asset.path, asset.media_type, "instagram"))

    destination = Path(output_root).expanduser().absolute() / post_id
    destination.mkdir(parents=True, exist_ok=False)
    media_dir = destination / "media"
    media_dir.mkdir(mode=0o700)
    package_assets = []
    try:
        for order, (asset, details) in enumerate(zip(export.assets, validation), start=1):
            target = media_dir / f"{order:02d}_{asset.asset_id}{asset.path.suffix.lower()}"
            copy_attested_asset(asset, target)
            package_assets.append(
                {
                    "order": order,
                    "asset_id": asset.asset_id,
                    "file": f"media/{target.name}",
                    "sha256": asset.sha256,
                    "validation": details,
                }
            )
        package = {
            "schema_version": 1,
            "platform": "instagram",
            "post_id": post_id,
            "format": post_format,
            "audience_tiers": manifest["audience_tiers"],
            "caption": "\n\n".join(
                part
                for part in (
                    str(manifest.get("body", "")).strip(),
                    str(manifest.get("cta", "")).strip(),
                    str(manifest.get("hashtags", "")).strip(),
                    DISCLOSURE,
                )
                if part
            ),
            "alt_text": manifest.get("alt_text", ""),
            "location_label": manifest.get("location_label", ""),
            "location_instruction": "UI-only; never send location_label to the Instagram API.",
            "ai_disclosure": DISCLOSURE,
            "is_ai_generated": True,
            "assets": package_assets,
            "ui_checklist": [
                "Use only the official Instagram app or Meta Business Suite.",
                "Confirm every asset and its order against this package.",
                "Keep the AI disclosure and platform AI label enabled.",
                "Apply location_label manually only if still accurate and broad.",
                "Schedule only after final human review.",
            ],
        }
        (destination / "package.json").write_text(
            json.dumps(package, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        if export.promotion is None:
            raise PublisherError("verified promotion is missing")
        verifier.consume(export.promotion)
    except Exception:
        shutil.rmtree(destination, ignore_errors=True)
        raise
    return destination
