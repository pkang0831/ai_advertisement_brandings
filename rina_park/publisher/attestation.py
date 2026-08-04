from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import shutil
import sqlite3
import stat
import subprocess
import tempfile
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Iterable

from .common import ApprovedAsset, ApprovedExport, PublisherError, sha256_file

UTC = timezone.utc
SCHEMA_VERSION = 1
ALGORITHM = "ECDSA_P256_SHA256"
NONCE_PATTERN = re.compile(r"[a-f0-9]{64}")
HASH_PATTERN = re.compile(r"[a-f0-9]{64}")
MATURE_MARKERS = ("mature", "18+", "nsfw", "explicit", "mne_")
DEFAULT_PUBLIC_KEY = (
    Path(__file__).resolve().parents[1]
    / "ops/attestation/trusted_public_key.pem"
)
DEFAULT_LEDGER = (
    Path(__file__).resolve().parents[1]
    / "ops/attestation/promotion_consumptions.db"
)
DEFAULT_REVOCATIONS = (
    Path(__file__).resolve().parents[1]
    / "ops/attestation/revocations.json"
)


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


def _parse_utc(value: Any, field: str) -> datetime:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise PublisherError(f"attestation {field} must be UTC RFC3339")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as exc:
        raise PublisherError(f"attestation {field} is invalid") from exc
    return parsed.astimezone(UTC)


def _safe_regular_file(path: Path, label: str) -> Path:
    if path.is_symlink():
        raise PublisherError(f"{label} must not be a symlink")
    try:
        resolved = path.resolve(strict=True)
        stat = resolved.stat()
    except OSError as exc:
        raise PublisherError(f"{label} is missing") from exc
    if not resolved.is_file() or stat.st_nlink != 1:
        raise PublisherError(f"{label} must be a regular non-hardlinked file")
    return resolved


def public_key_id(public_key_path: Path) -> str:
    public_key = _safe_regular_file(public_key_path, "trusted public key")
    try:
        result = subprocess.run(
            [
                "/usr/bin/openssl",
                "pkey",
                "-pubin",
                "-in",
                str(public_key),
                "-outform",
                "DER",
            ],
            check=True,
            capture_output=True,
            timeout=10,
        )
    except subprocess.SubprocessError as exc:
        raise PublisherError("trusted public key is invalid") from exc
    return hashlib.sha256(result.stdout).hexdigest()


def copy_attested_asset(asset: ApprovedAsset, destination: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(asset.path, flags)
    except OSError as exc:
        raise PublisherError("attested asset cannot be opened safely") from exc
    try:
        stat = os.fstat(descriptor)
        if stat.st_nlink != 1:
            raise PublisherError("attested asset became hardlinked")
        with os.fdopen(descriptor, "rb", closefd=False) as source:
            digest = hashlib.sha256()
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
            if digest.hexdigest() != asset.sha256:
                raise PublisherError("attested asset changed immediately before copy")
            source.seek(0)
            with destination.open("xb") as target:
                shutil.copyfileobj(source, target, length=1024 * 1024)
        if sha256_file(destination) != asset.sha256:
            raise PublisherError("copied attested asset hash mismatch")
    finally:
        os.close(descriptor)


@dataclass(frozen=True)
class VerifiedPromotion:
    attestation_id: str
    nonce: str
    key_id: str
    action: str
    content_id: str
    queue_id: str
    platform: str
    track: str
    issued_at: str
    expires_at: str
    payload_sha256: str


class ConsumptionLedger:
    def __init__(self, path: Path | str):
        self.path = Path(path).expanduser().absolute()

    def _connect(self) -> sqlite3.Connection:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.path, isolation_level=None)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA synchronous=FULL")
        connection.execute("PRAGMA busy_timeout=5000")
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS promotion_consumptions (
                sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                attestation_id TEXT NOT NULL UNIQUE,
                nonce TEXT NOT NULL UNIQUE,
                action TEXT NOT NULL,
                content_id TEXT NOT NULL,
                payload_sha256 TEXT NOT NULL,
                previous_entry_hash TEXT NOT NULL,
                entry_hash TEXT NOT NULL UNIQUE,
                consumed_at TEXT NOT NULL
            );
            CREATE TRIGGER IF NOT EXISTS promotion_consumptions_no_update
            BEFORE UPDATE ON promotion_consumptions
            BEGIN SELECT RAISE(ABORT, 'promotion ledger is append-only'); END;
            CREATE TRIGGER IF NOT EXISTS promotion_consumptions_no_delete
            BEFORE DELETE ON promotion_consumptions
            BEGIN SELECT RAISE(ABORT, 'promotion ledger is append-only'); END;
            """
        )
        try:
            os.chmod(self.path, 0o600)
        except OSError:
            pass
        return connection

    @staticmethod
    def _entry_hash(
        previous: str,
        attestation_id: str,
        nonce: str,
        action: str,
        content_id: str,
        payload_sha256: str,
        consumed_at: str,
    ) -> str:
        return hashlib.sha256(
            canonical_json_bytes(
                {
                    "previous_entry_hash": previous,
                    "attestation_id": attestation_id,
                    "nonce": nonce,
                    "action": action,
                    "content_id": content_id,
                    "payload_sha256": payload_sha256,
                    "consumed_at": consumed_at,
                }
            )
        ).hexdigest()

    def _verify_chain(self, connection: sqlite3.Connection) -> str:
        previous = "0" * 64
        for row in connection.execute(
            "SELECT * FROM promotion_consumptions ORDER BY sequence"
        ):
            expected = self._entry_hash(
                previous,
                row["attestation_id"],
                row["nonce"],
                row["action"],
                row["content_id"],
                row["payload_sha256"],
                row["consumed_at"],
            )
            if row["previous_entry_hash"] != previous or row["entry_hash"] != expected:
                raise PublisherError("promotion consumption ledger integrity failure")
            previous = row["entry_hash"]
        return previous

    def consume(self, promotion: VerifiedPromotion, now: datetime) -> None:
        consumed_at = (
            now.astimezone(UTC)
            .isoformat(timespec="milliseconds")
            .replace("+00:00", "Z")
        )
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            previous = self._verify_chain(connection)
            entry_hash = self._entry_hash(
                previous,
                promotion.attestation_id,
                promotion.nonce,
                promotion.action,
                promotion.content_id,
                promotion.payload_sha256,
                consumed_at,
            )
            connection.execute(
                "INSERT INTO promotion_consumptions("
                "attestation_id,nonce,action,content_id,payload_sha256,"
                "previous_entry_hash,entry_hash,consumed_at"
                ") VALUES(?,?,?,?,?,?,?,?)",
                (
                    promotion.attestation_id,
                    promotion.nonce,
                    promotion.action,
                    promotion.content_id,
                    promotion.payload_sha256,
                    previous,
                    entry_hash,
                    consumed_at,
                ),
            )
            connection.execute("COMMIT")
        except sqlite3.IntegrityError as exc:
            connection.execute("ROLLBACK")
            raise PublisherError("promotion attestation replay rejected") from exc
        except BaseException:
            connection.execute("ROLLBACK")
            raise
        finally:
            connection.close()

    def assert_unused(self, attestation_id: str, nonce: str) -> None:
        connection = self._connect()
        try:
            self._verify_chain(connection)
            if connection.execute(
                "SELECT 1 FROM promotion_consumptions "
                "WHERE attestation_id=? OR nonce=?",
                (attestation_id, nonce),
            ).fetchone():
                raise PublisherError("promotion attestation replay rejected")
        finally:
            connection.close()


class PromotionVerifier:
    def __init__(
        self,
        public_key_path: Path | str = DEFAULT_PUBLIC_KEY,
        ledger_path: Path | str = DEFAULT_LEDGER,
        revocations_path: Path | str = DEFAULT_REVOCATIONS,
        *,
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
        max_age: timedelta = timedelta(hours=24),
        future_skew: timedelta = timedelta(minutes=5),
    ):
        self.public_key_path = Path(public_key_path).expanduser().absolute()
        self.ledger = ConsumptionLedger(ledger_path)
        self.revocations_path = Path(revocations_path).expanduser().absolute()
        self.clock = clock
        self.max_age = max_age
        self.future_skew = future_skew

    def readiness(self) -> dict[str, Any]:
        reasons: list[str] = []
        key_path = Path(self.public_key_path)
        key_mode: str | None = None
        key_owner_is_current_user = False
        try:
            key_id = public_key_id(key_path)
            key_stat = key_path.lstat()
            key_mode = oct(stat.S_IMODE(key_stat.st_mode))
            key_owner_is_current_user = key_stat.st_uid == os.getuid()
            if stat.S_IMODE(key_stat.st_mode) & 0o022:
                reasons.append("trusted public key must not be group/world writable")
            if not key_owner_is_current_user:
                reasons.append("trusted public key must be owned by the current user")
        except PublisherError as exc:
            key_id = None
            reasons.append(str(exc))
        if not Path("/usr/bin/openssl").is_file():
            reasons.append("system OpenSSL verifier is unavailable")
        return {
            "ready": not reasons,
            "production_promotion_allowed": not reasons,
            "key_id": key_id,
            "public_key_mode": key_mode,
            "public_key_owner_is_current_user": key_owner_is_current_user,
            "reasons": reasons,
            "private_key_loaded_by_python": False,
        }

    def _revocations(self) -> tuple[set[str], set[str], set[str]]:
        if not self.revocations_path.exists():
            return set(), set(), set()
        path = _safe_regular_file(self.revocations_path, "revocations file")
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise PublisherError("revocations file is invalid") from exc
        if not isinstance(value, dict):
            raise PublisherError("revocations file must be an object")
        return (
            set(value.get("revoked_key_ids", [])),
            set(value.get("revoked_nonces", [])),
            set(value.get("revoked_attestation_ids", [])),
        )

    def _verify_signature(self, payload: dict[str, Any], signature: bytes) -> None:
        public_key = _safe_regular_file(
            self.public_key_path, "trusted public key"
        )
        payload_bytes = canonical_json_bytes(payload)
        with tempfile.TemporaryDirectory(prefix="rina-attestation-verify-") as directory:
            root = Path(directory)
            payload_path = root / "payload.json"
            signature_path = root / "signature.der"
            payload_path.write_bytes(payload_bytes)
            signature_path.write_bytes(signature)
            try:
                result = subprocess.run(
                    [
                        "/usr/bin/openssl",
                        "dgst",
                        "-sha256",
                        "-verify",
                        str(public_key),
                        "-signature",
                        str(signature_path),
                        str(payload_path),
                    ],
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
            except subprocess.SubprocessError as exc:
                raise PublisherError("attestation signature verification failed") from exc
        if result.returncode != 0 or "Verified OK" not in result.stdout:
            raise PublisherError("attestation signature is invalid")

    def verify(
        self,
        *,
        envelope: dict[str, Any],
        root: Path,
        manifest_path: Path,
        manifest: dict[str, Any],
        assets: Iterable[ApprovedAsset],
        platform: str,
        action: str,
    ) -> VerifiedPromotion:
        if action not in {"package", "publish"}:
            raise PublisherError("unsupported attestation action")
        if envelope.get("schema_version") != SCHEMA_VERSION:
            raise PublisherError("unsupported attestation envelope")
        payload = envelope.get("payload")
        encoded_signature = envelope.get("signature_der_base64")
        if not isinstance(payload, dict) or not isinstance(encoded_signature, str):
            raise PublisherError("malformed detached attestation")
        try:
            signature = base64.b64decode(encoded_signature, validate=True)
        except ValueError as exc:
            raise PublisherError("attestation signature encoding is invalid") from exc
        if payload.get("schema_version") != SCHEMA_VERSION:
            raise PublisherError("unsupported attestation payload")
        if payload.get("algorithm") != ALGORITHM:
            raise PublisherError("unsupported attestation algorithm")

        key_id = public_key_id(self.public_key_path)
        if payload.get("key_id") != key_id:
            raise PublisherError("attestation key is not trusted")
        self._verify_signature(payload, signature)

        now = self.clock().astimezone(UTC)
        issued = _parse_utc(payload.get("issued_at"), "issued_at")
        expires = _parse_utc(payload.get("expires_at"), "expires_at")
        if issued > now + self.future_skew:
            raise PublisherError("attestation is from the future")
        if now > expires or now - issued > self.max_age:
            raise PublisherError("attestation is expired")
        if expires <= issued or expires - issued > self.max_age:
            raise PublisherError("attestation validity window is invalid")

        nonce = payload.get("nonce")
        attestation_id = payload.get("attestation_id")
        if not isinstance(nonce, str) or not NONCE_PATTERN.fullmatch(nonce):
            raise PublisherError("attestation nonce is invalid")
        if not isinstance(attestation_id, str) or not HASH_PATTERN.fullmatch(
            attestation_id
        ):
            raise PublisherError("attestation ID is invalid")
        expected_id = hashlib.sha256(
            canonical_json_bytes(
                {key: value for key, value in payload.items() if key != "attestation_id"}
            )
        ).hexdigest()
        if attestation_id != expected_id:
            raise PublisherError("attestation ID does not match payload")

        revoked_keys, revoked_nonces, revoked_ids = self._revocations()
        if key_id in revoked_keys or nonce in revoked_nonces or attestation_id in revoked_ids:
            raise PublisherError("attestation or signing key is revoked")

        content_id = payload.get("content_id")
        queue_id = payload.get("queue_id")
        track = payload.get("track")
        if content_id != manifest.get("post_id"):
            raise PublisherError("attestation content ID mismatch")
        if queue_id != manifest.get("queue_id"):
            raise PublisherError("attestation queue ID mismatch")
        if payload.get("platform") != platform or manifest.get("platform") != platform:
            raise PublisherError("cross-platform attestation rejected")
        if track != manifest.get("track"):
            raise PublisherError("cross-track attestation rejected")
        if payload.get("action") != action:
            raise PublisherError("attestation action mismatch")
        if payload.get("policy_version") != manifest.get("policy_version"):
            raise PublisherError("attestation policy version mismatch")
        if payload.get("qc_report_sha256") != manifest.get("qc_report_sha256"):
            raise PublisherError("attestation QC report hash mismatch")
        root_real = root.resolve(strict=True)
        if payload.get("source_root") != str(root_real):
            raise PublisherError("attestation source root changed")
        qc_relative = manifest.get("qc_report_path")
        if (
            not isinstance(qc_relative, str)
            or not qc_relative
            or os.path.isabs(qc_relative)
        ):
            raise PublisherError("QC report path must be relative")
        qc_candidate = root.resolve(strict=True) / qc_relative
        qc_path = _safe_regular_file(qc_candidate, "QC report")
        try:
            qc_path.relative_to(root.resolve(strict=True))
        except ValueError as exc:
            raise PublisherError("QC report escapes attested root") from exc
        qc_hash = sha256_file(qc_path)
        if (
            payload.get("qc_report_path") != str(qc_path)
            or payload.get("qc_report_sha256") != qc_hash
        ):
            raise PublisherError("attested QC report path or hash changed")
        if manifest.get("review_status") != "approved":
            raise PublisherError("non-approved review status rejected")
        if manifest.get("production_status") != "approved":
            raise PublisherError("non-approved production status rejected")

        if payload.get("manifest_sha256") != sha256_file(manifest_path):
            raise PublisherError("attested manifest hash changed")
        boundary_values = " ".join(
            str(manifest.get(field, "")).lower()
            for field in ("post_id", "queue_id", "track", "platform")
        )
        if any(marker in boundary_values for marker in MATURE_MARKERS):
            raise PublisherError("mature attestation input rejected")

        actual_assets = tuple(assets)
        expected_media = payload.get("media")
        if not isinstance(expected_media, list) or len(expected_media) != len(
            actual_assets
        ):
            raise PublisherError("attested media inventory mismatch")
        for expected, asset in zip(expected_media, actual_assets):
            if not isinstance(expected, dict):
                raise PublisherError("attested media entry is malformed")
            actual = {
                "asset_id": asset.asset_id,
                "source_path": str(asset.path.resolve(strict=True)),
                "sha256": sha256_file(asset.path),
                "media_type": asset.media_type,
            }
            if expected != actual or actual["sha256"] != asset.sha256:
                raise PublisherError("attested media path or hash changed")

        payload_hash = hashlib.sha256(canonical_json_bytes(payload)).hexdigest()
        self.ledger.assert_unused(attestation_id, nonce)
        return VerifiedPromotion(
            attestation_id=attestation_id,
            nonce=nonce,
            key_id=key_id,
            action=action,
            content_id=str(content_id),
            queue_id=str(queue_id),
            platform=platform,
            track=str(track),
            issued_at=str(payload["issued_at"]),
            expires_at=str(payload["expires_at"]),
            payload_sha256=payload_hash,
        )

    def consume(self, promotion: VerifiedPromotion) -> None:
        self.ledger.consume(promotion, self.clock())

    @staticmethod
    def rehash_immediately(export: ApprovedExport) -> None:
        for asset in export.assets:
            if asset.path.is_symlink() or asset.path.stat().st_nlink != 1:
                raise PublisherError("attested asset link state changed")
            if sha256_file(asset.path) != asset.sha256:
                raise PublisherError("attested asset changed before use")
