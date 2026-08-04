# Rina Park local content operations

Two-track content (Instagram + Patreon A/B/C) with local Apple Silicon pipelines.

| Path | Purpose |
|------|---------|
| `identity/bible.md` | Character + track rules |
| `moodboard/` | Visual references |
| `models/CIVITAI_DOWNLOADS.md` | **Download checklist** |
| `workflows/` | ComfyUI + prompt presets |
| `queues/` | IG / Patreon CSV queues |
| `out/` | Generated stills + reels |
| `SETUP.md` | ComfyUI + Draw Things setup |
| `factory/manifest.db` | Sole runtime calendar/job/publication source |
| `orchestrator/` | Five-minute local heartbeat and dry-run tooling |
| `analytics/metrics.db` | Manually imported weekly metrics |

## Safe operating sequence

The CSV is a one-time seed and read-only export. After import, never use it as
the runtime schedule:

```bash
cd /Users/RBIPK031/ai_influencer
.venv/bin/python -m rina_park.orchestrator.cli import-seed
.venv/bin/python -m rina_park.orchestrator.cli heartbeat
```

Install the launchd template only after replacing/confirming every absolute
path and running `plutil -lint`. It uses `RunAtLoad` plus a 300-second
heartbeat; the runner performs UTC catch-up and holds a SQLite single-instance
lease. Generation remains disabled until licensed models and the production
generation hook pass readiness gates.

Instagram Graph publishing is disabled by default. A manual official
Instagram/Meta Business Suite package is the fallback until capability, auth,
public HTTPS transport, reconciliation, and explicit-enable gates all pass.
Patreon is always a manual official-web package. Neither route accepts a post
without current content and schedule approval hashes.

Run the network-free one-week rehearsal:

```bash
cd /Users/RBIPK031/ai_influencer
.venv/bin/python -m rina_park.orchestrator.dry_run
```

Import weekly manually exported metrics:

```bash
cp analytics/weekly_metrics_template.csv /tmp/rina_metrics.csv
.venv/bin/python -m rina_park.orchestrator.cli import-metrics /tmp/rina_metrics.csv
```

Structured JSONL logs redact common secret fields and Bearer credentials and
retain 14 days by default. `mature_non_explicit` remains a separate DB/root;
it has no calendar, export, package, metrics, or publisher capability.

## Quick smoke

```bash
cd /Users/RBIPK031/ai_influencer
.venv/bin/python rina_park/scripts/generate_track_smoke.py --track face_lock --limit 6
.venv/bin/python rina_park/scripts/generate_track_smoke.py --track ig --limit 1
.venv/bin/python rina_park/scripts/generate_track_smoke.py --track patreon_b --limit 1
rina_park/scripts/smoke_i2v_placeholder.sh rina_park/out/ig/rina_ig_0001.jpg
```
