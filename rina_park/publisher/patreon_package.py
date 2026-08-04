from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from .attestation import PromotionVerifier, copy_attested_asset
from .common import ApprovedExportStore, DISCLOSURE, PublisherError
from .validators import validate_asset


def build_patreon_package(
    approved_exports: Path | str,
    post_id: str,
    output_root: Path | str,
    *,
    verifier: PromotionVerifier | None = None,
) -> Path:
    verifier = verifier or PromotionVerifier()
    export = ApprovedExportStore(approved_exports).load(
        post_id,
        "patreon",
        verifier=verifier,
        action="package",
    )
    manifest = export.manifest
    tiers = tuple(manifest["audience_tiers"])
    expected_counts = {("A", "B", "C"): 4, ("B", "C"): 6, ("C",): 8}
    if len(export.assets) != expected_counts[tiers]:
        raise PublisherError("Patreon asset count does not match cumulative audience_tiers")

    validation: list[dict[str, Any]] = []
    for asset in export.assets:
        if asset.media_type not in {"image", "video"}:
            raise PublisherError("unsupported Patreon asset type")
        validation.append(validate_asset(asset.path, asset.media_type, "patreon"))

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
                    "sfw_preview": order == 1,
                }
            )
        package = {
            "schema_version": 1,
            "platform": "patreon",
            "mode": "manual_official_web",
            "post_id": post_id,
            "title": manifest.get("title", ""),
            "body": manifest.get("body", ""),
            "tags": manifest.get("hashtags", ""),
            "audience_tiers": list(tiers),
            "ai_disclosure": DISCLOSURE,
            "assets": package_assets,
            "policy_checklist": [
                "Safe for All Audiences only; no mature or adult content.",
                "Upload and schedule manually on the official Patreon website.",
                "Do not use browser automation, saved cookies, or unofficial posting APIs.",
                "Confirm cumulative audience access exactly matches audience_tiers.",
                "Keep the AI disclosure visible and complete final human review.",
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
