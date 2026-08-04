from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import secrets
import subprocess
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from .attestation import (
    ALGORITHM,
    DEFAULT_PUBLIC_KEY,
    MATURE_MARKERS,
    PromotionVerifier,
    canonical_json_bytes,
    public_key_id,
)
from .common import ApprovedExportStore, PublisherError, sha256_file

UTC = timezone.utc
RINA_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SIGNER = (
    RINA_ROOT / "ops/attestation/bin/rina-promotion-signer"
)


def _utc(moment: datetime) -> str:
    return (
        moment.astimezone(UTC)
        .isoformat(timespec="milliseconds")
        .replace("+00:00", "Z")
    )


def _required_hash(value: Any, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise PublisherError(f"{field} must be a lowercase SHA-256")
    return value


def build_payload(
    root: Path | str,
    post_id: str,
    platform: str,
    action: str,
    public_key_path: Path | str,
    *,
    now: datetime | None = None,
    nonce: str | None = None,
    validity: timedelta = timedelta(hours=1),
) -> tuple[dict[str, Any], ApprovedExportStore]:
    if action not in {"package", "publish"}:
        raise PublisherError("promotion action must be package or publish")
    store = ApprovedExportStore(root)
    export, _, manifest_path = store._load_for_promotion(post_id, platform)
    manifest = export.manifest
    if manifest.get("review_status") != "approved":
        raise PublisherError("human promotion requires approved review status")
    if manifest.get("production_status") != "approved":
        raise PublisherError("human promotion requires approved production status")
    queue_id = manifest.get("queue_id")
    track = manifest.get("track")
    policy_version = manifest.get("policy_version")
    if not all(
        isinstance(value, str) and value.strip()
        for value in (queue_id, track, policy_version)
    ):
        raise PublisherError("queue_id, track and policy_version are required")
    boundary_values = " ".join(
        str(value).lower()
        for value in (post_id, queue_id, platform, track)
    )
    if any(marker in boundary_values for marker in MATURE_MARKERS):
        raise PublisherError("mature input cannot be human-promoted")
    qc_hash = _required_hash(
        manifest.get("qc_report_sha256"), "qc_report_sha256"
    )
    qc_path = store._indexed_path(manifest.get("qc_report_path"))
    if sha256_file(qc_path) != qc_hash:
        raise PublisherError("QC report hash mismatch before promotion")
    issued = (now or datetime.now(UTC)).astimezone(UTC)
    expires = issued + validity
    nonce_value = nonce or secrets.token_hex(32)
    if len(nonce_value) != 64:
        raise PublisherError("promotion nonce must be 32 bytes")
    unsigned: dict[str, Any] = {
        "schema_version": 1,
        "algorithm": ALGORITHM,
        "key_id": public_key_id(Path(public_key_path)),
        "action": action,
        "content_id": post_id,
        "queue_id": queue_id,
        "platform": platform,
        "track": track,
        "policy_version": policy_version,
        "qc_report_path": str(qc_path),
        "qc_report_sha256": qc_hash,
        "source_root": str(store.root_real),
        "manifest_sha256": sha256_file(manifest_path),
        "media": [
            {
                "asset_id": asset.asset_id,
                "source_path": str(asset.path.resolve(strict=True)),
                "sha256": sha256_file(asset.path),
                "media_type": asset.media_type,
            }
            for asset in export.assets
        ],
        "issued_at": _utc(issued),
        "expires_at": _utc(expires),
        "nonce": nonce_value,
    }
    payload = dict(unsigned)
    payload["attestation_id"] = hashlib.sha256(
        canonical_json_bytes(unsigned)
    ).hexdigest()
    return payload, store


def promote(
    root: Path | str,
    post_id: str,
    platform: str,
    action: str,
    *,
    signer_path: Path | str = DEFAULT_SIGNER,
    public_key_path: Path | str = DEFAULT_PUBLIC_KEY,
) -> Path:
    signer = Path(signer_path).expanduser().resolve(strict=True)
    if signer.is_symlink() or not os.access(signer, os.X_OK):
        raise PublisherError("trusted signer helper is unavailable")
    payload, store = build_payload(
        root,
        post_id,
        platform,
        action,
        public_key_path,
    )
    summary = (
        f"Promote {post_id} to {platform}/{payload['track']} for {action}; "
        f"media={len(payload['media'])}; "
        f"manifest={payload['manifest_sha256'][:12]}"
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    with tempfile.TemporaryDirectory(prefix="rina-promotion-sign-") as directory:
        payload_path = Path(directory) / "payload.json"
        payload_path.write_bytes(canonical_json_bytes(payload))
        try:
            result = subprocess.run(
                [
                    str(signer),
                    "sign",
                    "--payload",
                    str(payload_path),
                    "--reason",
                    summary,
                ],
                check=True,
                capture_output=True,
                text=True,
                timeout=120,
            )
        except subprocess.SubprocessError as exc:
            raise PublisherError(
                "human promotion signature was not completed"
            ) from exc
    response = json.loads(result.stdout)
    signature = response.get("signature_der_base64")
    helper_public = response.get("public_key_der_base64")
    if not isinstance(signature, str) or not isinstance(helper_public, str):
        raise PublisherError("signer helper response is invalid")
    try:
        helper_key_id = hashlib.sha256(
            base64.b64decode(helper_public, validate=True)
        ).hexdigest()
    except ValueError as exc:
        raise PublisherError("signer public key response is invalid") from exc
    if helper_key_id != payload["key_id"]:
        raise PublisherError("signer key does not match trusted public key")

    envelope = {
        "schema_version": 1,
        "payload": payload,
        "signature_der_base64": signature,
    }
    post_dir = store.root_real / post_id
    destination = post_dir / f"promotion-{action}.attestation.json"
    with destination.open("x", encoding="utf-8") as handle:
        json.dump(envelope, handle, indent=2, sort_keys=True)
        handle.write("\n")
    index = store.index
    entry = index["posts"][post_id]
    entry["attestation"] = str(destination.relative_to(store.root_real))
    entry["attestation_sha256"] = sha256_file(destination)
    temporary = store.index_path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(index, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, store.index_path)
    return destination


def readiness(
    signer_path: Path | str = DEFAULT_SIGNER,
    public_key_path: Path | str = DEFAULT_PUBLIC_KEY,
) -> dict[str, Any]:
    signer = Path(signer_path).expanduser().absolute()
    reasons: list[str] = []
    signer_report: dict[str, Any] = {}
    if not signer.is_file() or signer.is_symlink() or not os.access(signer, os.X_OK):
        reasons.append("compiled Secure Enclave signer helper is unavailable")
    else:
        result = subprocess.run(
            [
                str(signer),
                "readiness",
                "--public-key",
                str(Path(public_key_path).expanduser().absolute()),
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            signer_report = json.loads(result.stdout)
        else:
            reasons.append("Secure Enclave signer readiness failed")
    verifier_report = PromotionVerifier(
        public_key_path=public_key_path
    ).readiness()
    reasons.extend(verifier_report["reasons"])
    if not signer_report.get("production_ready", False):
        reasons.append("explicit Secure Enclave enrollment has not been completed")
    return {
        "ready": not reasons,
        "production_promotion_allowed": not reasons,
        "live_signed_attestation_required": True,
        "package_publish_ready_without_attestation": False,
        "key_pair_match_verified": (
            signer_report.get("key_pair_match") == "verified_by_live_signature"
        ),
        "signer": signer_report,
        "verifier": verifier_report,
        "reasons": sorted(set(reasons)),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Secure-Enclave-backed human promotion boundary"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    ready = subparsers.add_parser("readiness")
    ready.add_argument("--signer", default=str(DEFAULT_SIGNER))
    ready.add_argument("--public-key", default=str(DEFAULT_PUBLIC_KEY))
    promote_parser = subparsers.add_parser("promote")
    promote_parser.add_argument("--root", required=True)
    promote_parser.add_argument("--post-id", required=True)
    promote_parser.add_argument(
        "--platform", required=True, choices=("instagram", "patreon")
    )
    promote_parser.add_argument(
        "--action", required=True, choices=("package", "publish")
    )
    promote_parser.add_argument("--signer", default=str(DEFAULT_SIGNER))
    promote_parser.add_argument("--public-key", default=str(DEFAULT_PUBLIC_KEY))
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.command == "readiness":
        report = readiness(args.signer, args.public_key)
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0 if report["ready"] else 2
    destination = promote(
        args.root,
        args.post_id,
        args.platform,
        args.action,
        signer_path=args.signer,
        public_key_path=args.public_key,
    )
    print(destination)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
