# Attested approved export contract

Publishers accept only a root directory containing `index.json` and a detached, valid,
single-use human-promotion attestation. Approval strings and checksums without a signature are
not trusted.

```json
{
  "version": 1,
  "posts": {
    "ig-0001": {
      "manifest": "ig-0001/manifest.json",
      "manifest_sha256": "<sha256 of exact manifest bytes>",
      "asset_ids": ["asset-01", "asset-02"],
      "attestation": "ig-0001/promotion-package.attestation.json",
      "attestation_sha256": "<sha256 of detached attestation bytes>"
    }
  }
}
```

The referenced manifest must contain the same `post_id`, a non-empty `queue_id`, target
`platform` and `track`, policy version, approved content/schedule/review/production status, a
relative QC report path and hash, audience tiers, required AI disclosure, SFW copy, and an
ordered `assets` array:

```json
{
  "asset_id": "asset-01",
  "path": "ig-0001/asset-01.jpg",
  "sha256": "<sha256 of exact asset bytes>",
  "media_type": "image"
}
```

The asset IDs must exactly equal the ordered index allowlist. Files are opened only after
real-path containment and checksum checks. Absolute/traversal paths, symlinks, hardlinks,
unregistered files, checksum changes, mature markers in paths or IDs, and unapproved copy fail
closed. Regenerate the immutable export and its index after any approved change; never patch an
existing approved export in place.

The detached ECDSA P-256 attestation binds:

- exact canonical export root, media paths, media hashes, asset IDs, and media types;
- exact manifest hash and QC report canonical path/hash;
- content ID, queue ID, platform, track, policy version, action (`package` or `publish`);
- issuing/expiry timestamps, a 256-bit nonce, signing-key ID, and attestation ID.

Verification uses only the enrolled public key. The private key is generated in Secure Enclave,
is non-exportable, and requires macOS `.userPresence` for every signature. Each attestation is
consumed once in the append-only hash-chained SQLite ledger. A package and an API publication
require separate attestations.

## Build and readiness (non-interactive)

```bash
/bin/sh rina_park/ops/attestation/build_signer.sh
.venv/bin/python -m rina_park.publisher.promotion readiness
```

Readiness remains fail-closed until explicit enrollment creates the Secure Enclave key and
`rina_park/ops/attestation/trusted_public_key.pem`.

## Explicit enrollment (run later)

This command invokes Touch ID or the macOS credential UI. Run it once, interactively, only after
reviewing `RinaPromotionSigner.swift`:

```bash
rina_park/ops/attestation/bin/rina-promotion-signer enroll \
  --public-key /Users/RBIPK031/ai_influencer/rina_park/ops/attestation/trusted_public_key.pem
```

Enrollment is not part of builds, tests, imports, or readiness checks.

## Human promotion

The command prints the exact payload before invoking the macOS user-presence prompt:

```bash
.venv/bin/python -m rina_park.publisher.promotion promote \
  --root /absolute/path/to/approved_exports \
  --post-id CONTENT_ID \
  --platform instagram \
  --action package
```

Use `--action publish` for a separate one-time Instagram API authorization. Patreon remains a
manual official-website package path.
