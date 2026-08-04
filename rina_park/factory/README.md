# Rina production factory

`manifest.db` is the only platform production manifest after calendar import.
Every connection enables WAL and foreign keys. Public methods use
`BEGIN IMMEDIATE` transactions. Apply migrations with `Manifest.migrate()`.

## Post state machine

Normal flow:

`draft → generating → assets_ready → content_approved → schedule_approved → packaged → publishing → published`

Explicit exception flow:

- Active non-published states may enter `needs_review` or `failed`.
- `publishing` may enter `needs_reconciliation`.
- `needs_review` may restart at `generating` or `assets_ready`.
- `failed` may restart at `generating`.
- `needs_reconciliation` may return to `publishing` or resolve to `published`.

SQLite rejects every other transition. Approved content hash or asset-set
changes append an invalidation approval and return the post to
`needs_review`. Approval and audit rows cannot be updated or deleted.

Generation work is unique by
`(post_id, asset_slot, generation_version, candidate_index)` and is claimed
with an expiring lease. Publication records and attempts use separate tables
so remote container creation can be reconciled without blind reposting.

## Readiness gates

`readiness.py` fails closed on missing revision, SHA-256, license,
commercial-use approval, local file or benchmark evidence. InsightFace code
licensing does not make its pretrained Antelope/Buffalo weights commercially
usable; those weights are always rejected for platform identity locking.
Identity and ControlNet require 12/12 benchmark evidence. Wan/VACE require
three end-to-end clips and the 81-frame gate; VACE additionally requires a
pinned control-video format and MPS result. No readiness check downloads a
model.

`pipeline.py` provides the production V0 FFmpeg 5–8 second, 1080×1920,
silent H.264 fallback.

## Mature local profile

`mature_non_explicit` uses a separate database and mode-`0700` media root.
Schema/presets live under `private/factory/mature_non_explicit/` (gitignored).
Its API has no publication operation and rejects any platform DB or publisher
secret context. Registration and every subsequent read reject symlinks,
hardlinks, root escapes, unregistered IDs and changed hashes. No mature ID or
path is accepted by the platform manifest.
