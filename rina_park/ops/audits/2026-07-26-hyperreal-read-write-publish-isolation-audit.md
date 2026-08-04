# Hyperreal read/write regression and publish-isolation audit

Date: 2026-07-26  
Scope: `/Users/RBIPK031/ai_influencer/rina_park`  
Result: regression suite passes; cryptographic promotion boundary implemented; production remains fail-closed pending explicit Secure Enclave enrollment.

## Executive result

- 177 pytest cases passed: the prior 152-case suite plus 25 focused promotion-attestation cases.
- 8 unittest subtests also passed.
- 54 warnings remain, all from two Pillow `Image.getdata()` deprecations in `qc/engine.py`.
- Compilation, Streamlit import, review-app import, and all three launchd plist validations passed.
- SQLite migrations 001 and 002 are idempotent, `user_version=2`, WAL/full durability settings are active, foreign keys are enabled, and `integrity_check=ok`.
- The calendar imports exactly 56 platform rows and contains no mature identifiers or hyperreal/review output paths.
- 32 rejected, pending, provisional, Phase v1-v4-related output files were hash/inode checked against 24 files in seven checked-in dry-run approved-export post directories. No overlap was found.
- No CivitAI token-shaped value remains anywhere under the repository.
- No image generation, model download, publication, account access, output deletion, model modification, commit, push, or git configuration change occurred.

## Severity-ranked findings

### Critical — fixed: plaintext CivitAI credential in `rina_park/.env`

The audit found a non-empty CivitAI API token committed/stored as plaintext in `rina_park/.env`. A new system test reproduced the failure. The value was removed and replaced with a no-secrets notice.

Residual action: removal does not revoke a potentially exposed credential. Rotate/revoke it in CivitAI manually before reuse. This audit did not access the account.

### High — mitigated: enrollment verified; live key-pair proof remains

The self-asserted manifest path is no longer accepted by package builders, staging, or the Instagram API publisher. A detached ECDSA P-256 human-promotion attestation now binds the exact canonical export root, media paths and hashes, manifest hash, content/queue ID, platform, track, action, policy version, QC report path/hash, issue/expiry timestamps, nonce, key ID, and attestation ID.

The private key path is implemented by a compiled Swift CryptoKit helper:

- `SecureEnclave.P256.Signing.PrivateKey`;
- `SecAccessControl` with `.privateKeyUsage` and `.userPresence`;
- explicit Touch ID/macOS credential presence during enrollment;
- Touch ID/macOS credential presence for every signature;
- only the Secure Enclave encrypted key handle is stored in Keychain;
- Python receives no private key and verifies with the public key only.

The user completed explicit enrollment. The noninteractive readiness probe now returns exit code 0 and:

- `secure_enclave_available=true`;
- Keychain enrollment-record metadata present for service
  `com.devenira.rina.promotion.secure-enclave`, account
  `production-p256-v1`;
- `enrolled=true`;
- `public_key_exists=true` and `public_key_valid=true`;
- P-256 SPKI SHA-256/key ID
  `f0eaebaac3da75a4cce5fa711d3b40651951ec77f598154920b4cfa53f62d558`;
- public-key mode `0644`, owned by the current user, not group/world writable;
- `production_promotion_allowed=true`, meaning the interactive promotion
  command may now be attempted;
- `package_publish_ready_without_attestation=false` and
  `live_signed_attestation_required=true`.

The enrollment helper originally emitted a 177-byte malformed PEM because its
footer lacked a preceding newline. The exact existing public bytes were
reformatted into a valid 178-byte PEM, and the helper was fixed for future
enrollments. No key was regenerated and no enrollment command was rerun.

Keychain metadata can be checked without requesting secret data. Reading the
opaque key handle after rebuilding the unsigned helper is authorization-gated,
so a non-signing exact key-pair comparison is not available without potential
credential UI on this installation. Readiness therefore reports
`key_pair_match=pending_live_signature_verification` and never claims a match.
The code path that generated the public file from the newly enrolled key is
strong provenance evidence, but independent operational proof requires one
explicit user-presence test signature verified by this public key.

Therefore the original forge path is closed in code and enrollment is
operationally ready, but the High finding remains mitigated—not fully closed—
until the one-shot harmless live signature rehearsal succeeds. No production
attestation has been created or consumed.

No plaintext, HMAC, software production private key, cloud signer, or unsigned compatibility bypass was added.

### Medium — fixed: mature root itself was not rejected early

Nested mature paths and mature asset IDs were already rejected, but an approved-export root named `mature`, `mature_non_explicit`, `private_media`, or `mne_*` was not rejected at construction. A concrete failing test proved this. `ApprovedExportStore` now rejects those root components before reading an index.

### Low — open: Pillow deprecation warnings

`qc/engine.py` uses `Image.getdata()` at lines 114 and 153. Pillow reports removal in version 14 (2027-10-15). This is not a current correctness failure.

### Informational — lint runner availability

No Ruff installation/configuration is available in the project virtual environment. Python compilation and IDE diagnostics passed with no errors. This audit did not install dependencies.

## Promotion architecture and threat model

The human promotion flow is:

1. An approved-export tree contains exact media, manifest, and QC report bytes.
2. The promotion CLI structurally rejects mature, rejected, pending, or incomplete input.
3. It computes the canonical payload and displays it before signing.
4. The Swift helper requires macOS user presence and returns only a DER ECDSA signature plus public key.
5. Python verifies the detached signature and every bound field with the enrolled public key.
6. Package/publish code rehashes immediately before use and copies media from a no-follow file descriptor.
7. Successful use consumes the nonce in an append-only, immutable-triggered, hash-chained SQLite ledger.

Package and publish are separate signed actions. Missing, invalid, replayed, expired, revoked, cross-platform, cross-track, cross-action, moved-root, changed-path, changed-hash, changed-policy, changed-QC, rejected, pending, and mature inputs fail closed.

Threat model:

- Protects against forged manifests, copied rejected outputs, replay, path substitution, and an ordinary unmodified local Python process attempting to sign without user presence.
- Assumes the checked verifier/helper code, enrolled public-key file, revocation file, and consumption ledger are not replaced by an attacker who already has arbitrary same-user code execution and can rewrite the application itself. Such an attacker can bypass any local Python publisher and call a network API directly; resisting that requires a separately signed/hardened app, OS sandbox, and administrator-managed ACLs.
- Ledger deletion by an arbitrary same-user filesystem attacker is outside this local application boundary. Within the application, update/delete triggers and the hash chain detect mutation and every nonce is unique.
- Enrollment and key rotation are explicit interactive operations, never build/import/test side effects.

## Publish-isolation evidence

- Review v1: run-level `review_status=rejected`.
- Review v2: platform status `rejected_needs_regeneration`.
- Review v3: platform status `rejected_needs_identity_reference_diversity`.
- Review v4 and repairs: selected items remain `selected_not_published`, `not_approved_not_published`, or pending user approval; rejected items remain `rejected_not_promoted`.
- Phase0: manifest and run summary both disable publication; all 12 blind outputs are production-ineligible; aggregate decision has no promotions.
- Identity pilot: private, `production_approved=false`, failed closed, and created no edited output.
- Rejected/pending/blank/invalidated approvals were exercised for both platforms. Store loading and both package builders failed closed and created no package directory.
- Calendar rows contain no mature IDs, review-sample paths, or hyperreal paths.
- Patreon remains manual-package only. Instagram Graph routing remains gate-disabled unless every graph gate passes.

## Integrity and recovery evidence

- Path traversal, symlink, hardlink, manifest checksum, and asset checksum tampering fail closed.
- Mature DB/root/assets cannot register in the platform manifest or approved-export reader.
- Stale generation leases are reclaimed by a new worker.
- Stale heartbeat leases are reclaimed after expiry.
- A simulated power-loss exception rolls back the SQLite transaction.
- Duplicate `(platform, post_id)` publication records are rejected by SQLite uniqueness constraints.
- Existing Instagram tests prove published-state replay makes no network call and ambiguous mutation timeouts move to `needs_reconciliation` without recreation.
- Model registry gates reject non-commercial, unapproved, unpinned, unhashed, or unknown-license entries.

## Secrets and launchd

- Structured logger tests redact access tokens and bearer values.
- Text files, JSON/JSONL sidecars, logs, plists, and SQLite files were scanned for credential material.
- No CivitAI token-shaped value remains.
- Publisher token retrieval uses `/usr/bin/security find-generic-password -w -s ... -a ...`; no environment/file/CLI token input path is accepted.
- All launchd templates parse successfully. They contain no token or secret values. The orchestrator interval is exactly 300 seconds.

## Commands and exact results

- Full audited pytest after enrollment-readiness updates:
  `184 passed, 54 warnings, 8 subtests passed`.
- Focused promotion-attestation suite: `27 passed`.
- Focused attestation/publisher/dry-run/isolation run: `71 passed`.
- Pre-enrollment full-suite result: `177 passed, 54 warnings, 8 subtests passed`.
- Previous audit baseline: `152 passed`.
- `python -m compileall`: passed for analytics, factory, hyperreal, orchestrator, publisher, QC, review, tests, and scripts.
- Swift CryptoKit helper build: passed with Apple Swift 6.3.3 on arm64 macOS.
- Promotion readiness after enrollment: exit code 0; signer and public-only
  verifier ready; live signed attestation remains mandatory.
- Enrollment readiness probe performed no signature and created/consumed no
  production attestation.
- `plutil -lint`: all three templates reported `OK`.
- `import streamlit; import rina_park.review.app`: passed.
- IDE lint diagnostics on changed Python files: no errors.

## Files changed by this audit

- Added `rina_park/tests/system/test_publish_isolation_audit.py`.
- Added this audit report.
- Fixed `rina_park/publisher/common.py` to reject mature approved-export roots.
- Removed the plaintext credential from `rina_park/.env`.
- Added `publisher/attestation.py` and `publisher/promotion.py`.
- Added and compiled `ops/attestation/RinaPromotionSigner.swift`.
- Added test-only deterministic P-256 fixtures under `tests/system/fixtures`.
- Refactored package, staging, API publisher, and dry-run paths to require verified one-time promotion.
