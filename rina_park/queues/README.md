# Queue schema — Rina IG / Patreon

Diet carousel `image_prompt_queue.csv` is **out of scope** for this debut feed.

## Files

| File | Track |
|------|-------|
| `rina_ig_queue.csv` | Instagram |
| `rina_patreon_a_queue.csv` | Patreon A |
| `rina_patreon_b_queue.csv` | Patreon B |
| `rina_patreon_c_queue.csv` | Patreon C |

## Columns

`asset_id, track, tier, production_id, slide_role, location, outfit, shot, phone_real_notes, seed, width, height, steps, cfg, prompt, negative_prompt, output_subdir, image_filename, status`

- `track`: `ig` | `patreon`
- `tier`: empty for IG; `a`|`b`|`c` for Patreon
- `status`: `queued` | `generated` | `retry` | `rejected`
- Prompts stay visual-only (CLIP ~77 tokens). Captions/education copy live elsewhere.

## Generate

```bash
cd /Users/RBIPK031/ai_influencer
.venv/bin/python rina_park/scripts/generate_track_smoke.py --track ig --limit 1
.venv/bin/python rina_park/scripts/generate_track_smoke.py --track patreon_b --limit 1
```
