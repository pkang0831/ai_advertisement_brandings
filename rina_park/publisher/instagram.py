from __future__ import annotations

import json
import hashlib
import ipaddress
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Protocol

from .attestation import PromotionVerifier
from .common import ApprovedExport, ApprovedExportStore, DISCLOSURE, PublisherError

GRAPH_ORIGIN = "https://graph.instagram.com"
REQUIRED_SCOPES = frozenset(
    {"instagram_business_basic", "instagram_business_content_publish"}
)
CONTAINER_TTL_SECONDS = 24 * 60 * 60


class RetryClass(str, Enum):
    NEVER = "never"
    BACKOFF = "backoff"
    RECONCILE = "reconcile"


class InstagramError(RuntimeError):
    def __init__(self, message: str, retry: RetryClass, status: int | None = None):
        super().__init__(message)
        self.retry = retry
        self.status = status


class ReconciliationRequired(InstagramError):
    def __init__(self, message: str):
        super().__init__(message, RetryClass.RECONCILE)


@dataclass(frozen=True)
class Token:
    value: str
    expires_at: datetime
    scopes: frozenset[str]

    def warnings(self, now: datetime | None = None) -> tuple[int, ...]:
        now = now or datetime.now(timezone.utc)
        days = (self.expires_at - now).total_seconds() / 86400
        return tuple(threshold for threshold in (30, 14, 7) if 0 < days <= threshold)

    def assert_usable(self, now: datetime | None = None) -> None:
        now = now or datetime.now(timezone.utc)
        if self.expires_at <= now:
            raise InstagramError("Instagram User token is expired", RetryClass.NEVER, 401)
        if not REQUIRED_SCOPES <= self.scopes:
            raise InstagramError("required Instagram Login scopes are missing", RetryClass.NEVER, 403)


class KeychainTokenSource:
    """Reads a token from macOS Keychain without accepting secrets via env, files, or CLI."""

    def __init__(self, service: str, account: str):
        self.service = service
        self.account = account

    def read(self, expires_at: datetime, scopes: frozenset[str]) -> Token:
        try:
            result = subprocess.run(
                [
                    "/usr/bin/security",
                    "find-generic-password",
                    "-w",
                    "-s",
                    self.service,
                    "-a",
                    self.account,
                ],
                check=True,
                capture_output=True,
                text=True,
                timeout=10,
            )
        except subprocess.SubprocessError as exc:
            raise InstagramError("Instagram token unavailable in macOS Keychain", RetryClass.NEVER) from exc
        value = result.stdout.rstrip("\n")
        if not value:
            raise InstagramError("Instagram token unavailable in macOS Keychain", RetryClass.NEVER)
        return Token(value, expires_at, scopes)


def redact(value: Any, secrets: tuple[str, ...]) -> Any:
    if isinstance(value, dict):
        return {
            key: "[REDACTED]" if key.lower() in {"access_token", "app_secret", "authorization"} else redact(item, secrets)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [redact(item, secrets) for item in value]
    if isinstance(value, str):
        redacted = value
        for secret in secrets:
            if secret:
                redacted = redacted.replace(secret, "[REDACTED]")
        return redacted
    return value


@dataclass(frozen=True)
class Response:
    status: int
    body: dict[str, Any]


class Transport(Protocol):
    def request(
        self, method: str, url: str, data: dict[str, Any] | None, timeout: float
    ) -> Response: ...


class UrlLibTransport:
    def request(
        self, method: str, url: str, data: dict[str, Any] | None, timeout: float
    ) -> Response:
        encoded = None
        if data and method == "GET":
            url = f"{url}?{urllib.parse.urlencode(data, doseq=True)}"
        elif data:
            encoded = urllib.parse.urlencode(data, doseq=True).encode()
        request = urllib.request.Request(url, data=encoded, method=method)
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                body = json.loads(response.read())
                return Response(response.status, body)
        except urllib.error.HTTPError as exc:
            try:
                body = json.loads(exc.read())
            except json.JSONDecodeError:
                body = {"error": {"message": "non-JSON Instagram API error"}}
            return Response(exc.code, body)
        except (TimeoutError, urllib.error.URLError) as exc:
            raise ReconciliationRequired("Instagram network outcome is unknown") from exc


@dataclass
class PublicationState:
    post_id: str
    status: str = "ready"
    container_id: str | None = None
    container_created_at: datetime | None = None
    remote_media_id: str | None = None
    remote_url: str | None = None
    child_container_ids: list[str] = field(default_factory=list)
    request_hash: str | None = None


class StateStore(Protocol):
    def get(self, post_id: str) -> PublicationState: ...
    def save(self, state: PublicationState) -> None: ...


class MemoryStateStore:
    def __init__(self) -> None:
        self.states: dict[str, PublicationState] = {}

    def get(self, post_id: str) -> PublicationState:
        return self.states.setdefault(post_id, PublicationState(post_id))

    def save(self, state: PublicationState) -> None:
        self.states[state.post_id] = state


def classify(status: int | None, *, timeout: bool = False, mutation: bool = False) -> RetryClass:
    if timeout and mutation:
        return RetryClass.RECONCILE
    if status == 429 or (status is not None and 500 <= status <= 599):
        return RetryClass.BACKOFF
    return RetryClass.NEVER


class InstagramClient:
    """Instagram API with Instagram Login client; it does not promise exactly-once delivery."""

    def __init__(
        self,
        instagram_user_id: str,
        token: Token,
        transport: Transport | None = None,
        api_version: str = "v24.0",
        timeout: float = 20,
        clock: callable = lambda: datetime.now(timezone.utc),
        sleeper: callable = time.sleep,
    ):
        if not instagram_user_id.isdigit():
            raise PublisherError("invalid Instagram User ID")
        self.user_id = instagram_user_id
        self.token = token
        self.transport = transport or UrlLibTransport()
        self.base = f"{GRAPH_ORIGIN}/{api_version}"
        self.timeout = timeout
        self.clock = clock
        self.sleeper = sleeper

    def _call(
        self, method: str, path: str, data: dict[str, Any] | None = None, *, mutation: bool = False
    ) -> dict[str, Any]:
        self.token.assert_usable(self.clock())
        payload = dict(data or {})
        payload["access_token"] = self.token.value
        try:
            response = self.transport.request(method, f"{self.base}/{path.lstrip('/')}", payload, self.timeout)
        except ReconciliationRequired:
            if mutation:
                raise
            raise InstagramError("Instagram read request timed out", RetryClass.BACKOFF)
        if not 200 <= response.status < 300:
            message = str(response.body.get("error", {}).get("message", "Instagram API request failed"))
            raise InstagramError(message, classify(response.status, mutation=mutation), response.status)
        return response.body

    def assert_capacity(self) -> None:
        body = self._call(
            "GET",
            f"{self.user_id}/content_publishing_limit",
            {"fields": "config,quota_usage"},
        )
        record = (body.get("data") or [{}])[0]
        quota = record.get("quota_usage")
        total = (record.get("config") or {}).get("quota_total")
        if not isinstance(quota, int) or not isinstance(total, int):
            raise InstagramError("publishing limit response is malformed", RetryClass.NEVER)
        if quota >= total:
            raise InstagramError("Instagram content publishing limit reached", RetryClass.NEVER, 429)

    def create_container(self, fields: dict[str, Any]) -> str:
        try:
            body = self._call("POST", f"{self.user_id}/media", fields, mutation=True)
        except InstagramError as exc:
            if exc.retry is RetryClass.BACKOFF:
                raise ReconciliationRequired("container creation outcome requires reconciliation") from exc
            raise
        container_id = body.get("id")
        if not isinstance(container_id, str):
            raise ReconciliationRequired("container creation returned no stable ID")
        return container_id

    def container_status(self, container_id: str) -> str:
        body = self._call("GET", container_id, {"fields": "status_code"})
        status = body.get("status_code")
        if not isinstance(status, str):
            raise InstagramError("container status response is malformed", RetryClass.NEVER)
        return status

    def publish_container(self, container_id: str) -> str:
        try:
            body = self._call(
                "POST",
                f"{self.user_id}/media_publish",
                {"creation_id": container_id},
                mutation=True,
            )
        except InstagramError as exc:
            if exc.retry is RetryClass.BACKOFF:
                raise ReconciliationRequired("publish outcome requires reconciliation") from exc
            raise
        media_id = body.get("id")
        if not isinstance(media_id, str):
            raise ReconciliationRequired("publish response returned no stable media ID")
        return media_id

    def media_permalink(self, media_id: str) -> str | None:
        try:
            body = self._call("GET", media_id, {"fields": "permalink"})
        except InstagramError:
            return None
        permalink = body.get("permalink")
        return permalink if isinstance(permalink, str) and permalink.startswith("https://") else None

    def publish(
        self,
        approved_exports: Path | str,
        post_id: str,
        public_urls: dict[str, str],
        states: StateStore,
        *,
        verifier: PromotionVerifier | None = None,
        poll_attempts: int = 12,
    ) -> PublicationState:
        verifier = verifier or PromotionVerifier()
        export = ApprovedExportStore(approved_exports).load(
            post_id,
            "instagram",
            verifier=verifier,
            action="publish",
        )
        self.token.assert_usable(self.clock())
        state = states.get(export.post_id)
        if state.remote_media_id or state.status == "published":
            return state
        if state.status == "needs_reconciliation":
            raise ReconciliationRequired("post requires human reconciliation before any retry")
        self.assert_capacity()
        for asset in export.assets:
            url = public_urls.get(asset.asset_id, "")
            parsed = urllib.parse.urlparse(url)
            hostname = parsed.hostname or ""
            try:
                address = ipaddress.ip_address(hostname)
                non_public_ip = not address.is_global
            except ValueError:
                non_public_ip = hostname == "localhost" or hostname.endswith((".local", ".localhost"))
            if (
                parsed.scheme != "https"
                or not parsed.netloc
                or parsed.username
                or parsed.fragment
                or non_public_ip
            ):
                raise InstagramError("all API media URLs must be public HTTPS URLs", RetryClass.NEVER)

        verifier.rehash_immediately(export)
        if export.promotion is None:
            raise PublisherError("verified promotion is missing")
        verifier.consume(export.promotion)

        now = self.clock()
        if state.container_id and state.container_created_at:
            age = (now - state.container_created_at).total_seconds()
            try:
                existing_status = self.container_status(state.container_id)
            except InstagramError:
                state.status = "needs_reconciliation"
                states.save(state)
                raise ReconciliationRequired("stored container status could not be reconciled")
            if existing_status == "PUBLISHED":
                state.status = "needs_reconciliation"
                states.save(state)
                raise ReconciliationRequired("container is published but remote media ID is unknown")
            if age >= CONTAINER_TTL_SECONDS or existing_status == "EXPIRED":
                state.container_id = None
                state.child_container_ids.clear()

        caption = "\n\n".join(
            part
            for part in (
                str(export.manifest.get("body", "")).strip(),
                str(export.manifest.get("cta", "")).strip(),
                str(export.manifest.get("hashtags", "")).strip(),
                DISCLOSURE,
            )
            if part
        )
        try:
            if not state.container_id:
                post_format = str(export.manifest.get("format", "")).lower()
                if post_format == "carousel":
                    children = []
                    for asset in export.assets:
                        children.append(
                            self.create_container(
                                {"image_url": public_urls[asset.asset_id], "is_carousel_item": "true"}
                            )
                        )
                        state.child_container_ids = list(children)
                        state.status = "creating_carousel"
                        states.save(state)
                    state.container_id = self.create_container(
                        {
                            "media_type": "CAROUSEL",
                            "children": ",".join(children),
                            "caption": caption,
                            "is_ai_generated": "true",
                        }
                    )
                elif post_format == "reel":
                    asset = export.assets[0]
                    state.container_id = self.create_container(
                        {
                            "media_type": "REELS",
                            "video_url": public_urls[asset.asset_id],
                            "caption": caption,
                            "is_ai_generated": "true",
                        }
                    )
                elif post_format == "image":
                    asset = export.assets[0]
                    state.container_id = self.create_container(
                        {
                            "image_url": public_urls[asset.asset_id],
                            "caption": caption,
                            "is_ai_generated": "true",
                        }
                    )
                else:
                    raise InstagramError("unsupported Instagram format", RetryClass.NEVER)
                state.container_created_at = now
                state.request_hash = hashlib.sha256(
                    json.dumps(
                        {
                            "post_id": export.post_id,
                            "format": export.manifest.get("format"),
                            "assets": [
                                {"asset_id": asset.asset_id, "sha256": asset.sha256}
                                for asset in export.assets
                            ],
                            "caption": caption,
                            "is_ai_generated": True,
                        },
                        sort_keys=True,
                        separators=(",", ":"),
                    ).encode()
                ).hexdigest()
                state.status = "container_created"
                states.save(state)

            status = ""
            for attempt in range(poll_attempts):
                status = self.container_status(state.container_id)
                if status == "FINISHED":
                    break
                if status in {"ERROR", "EXPIRED"}:
                    raise InstagramError(f"container reached terminal state {status}", RetryClass.NEVER)
                if attempt + 1 < poll_attempts:
                    self.sleeper(min(2**attempt, 30))
            if status != "FINISHED":
                raise InstagramError("container processing did not finish in time", RetryClass.BACKOFF)
            state.status = "publishing"
            states.save(state)
            state.remote_media_id = self.publish_container(state.container_id)
            state.status = "published"
            states.save(state)
            state.remote_url = self.media_permalink(state.remote_media_id)
            states.save(state)
            return state
        except ReconciliationRequired:
            state.status = "needs_reconciliation"
            states.save(state)
            raise
