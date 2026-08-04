from __future__ import annotations

import secrets
import urllib.parse
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from .attestation import PromotionVerifier
from .common import ApprovedExport, PublisherError


class EphemeralProvider(Protocol):
    """Provider spike contract. Implementations must not expose a browsable root."""

    def expose(self, source: Path, opaque_name: str) -> str: ...
    def probe(self, url: str) -> tuple[int, str, int]: ...
    def close(self) -> None: ...


@dataclass
class ApprovedOnlyStaging:
    """Maps approved assets to opaque HTTPS URLs and tears the provider down on exit."""

    provider: EphemeralProvider
    export: ApprovedExport
    verifier: PromotionVerifier
    closed: bool = False

    def __enter__(self) -> "ApprovedOnlyStaging":
        return self

    def stage(self) -> dict[str, str]:
        if self.closed:
            raise PublisherError("staging session is closed")
        if (
            self.export.promotion is None
            or self.export.promotion.action != "publish"
        ):
            raise PublisherError("publish promotion attestation is required for staging")
        self.verifier.rehash_immediately(self.export)
        urls: dict[str, str] = {}
        for asset in self.export.assets:
            opaque_name = f"{secrets.token_urlsafe(32)}{asset.path.suffix.lower()}"
            url = self.provider.expose(asset.path, opaque_name)
            parsed = urllib.parse.urlparse(url)
            if (
                parsed.scheme != "https"
                or not parsed.netloc
                or parsed.username
                or parsed.query
                or parsed.fragment
                or parsed.path.endswith("/")
                or opaque_name not in parsed.path
            ):
                self.close()
                raise PublisherError("staging provider did not return an opaque public HTTPS asset URL")
            status, mime, size = self.provider.probe(url)
            allowed_mime = "image/" if asset.media_type == "image" else "video/"
            if status != 200 or not mime.startswith(allowed_mime) or size != asset.path.stat().st_size:
                self.close()
                raise PublisherError("staged asset failed HEAD/MIME/size verification")
            urls[asset.asset_id] = url
        return urls

    def close(self) -> None:
        if not self.closed:
            self.provider.close()
            self.closed = True

    def __exit__(self, *_: object) -> None:
        self.close()
