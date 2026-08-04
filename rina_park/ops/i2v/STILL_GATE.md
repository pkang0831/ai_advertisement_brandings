# I2V Hero Still Gate

I2V 입력 품질이 영상 천장이다. 아래를 **전부** 통과한 스틸만 Wan A14B에 넣는다.

## Generation path (native → I2V pack)

1. **Generate** at **832×1216** (anatomy_lock settings). Prompt order:  
   identity+beauty → pose/framing (hands/feet hidden) → short scene glue (≤ ~75 CLIP tokens).
2. **Upscale / pack** to **1080×1920** (center-crop to 9:16 + LANCZOS interim).
3. **`auto_pass` ≠ promote** — auto gate is a filter only. Human QC before `current/`.

## Default hero strategy

- **Hands/feet hidden** — default poses: `hand_pockets_3q`, `hand_cropped_out` (no finger farm).
- **Beauty ON** — short soft-glam LOOK_POS + FaceDetailer (looking-at-camera face prompt).
- **Framing** — three-quarter / upper-body; reject extreme face close-ups (face area &gt; ~0.18).

## Pass criteria (must)

1. **Resolution** ≥ **1080×1920**. Native `*_gen832x1216.jpg` are debug — Wan reads upscaled `sXX_seed*.jpg`.
2. **Photoreal soft glam** — no painterly, plastic beauty-filter, horror highlights, CGI sheen.
3. **Hands** — whitelist pose only ([`pose_catalog.yml`](../anatomy_lock/pose_catalog.yml)):
   - Hidden/pockets/cropped → character LoRA only (no hand LoRA)
   - Hand-visible (optional `--poses`) → `detailed_hands@0.45`; auto requires `hand_count == expected`
3b. **FaceDetailer** — **ON by default**. Auto-reverts if upper-frame black blob &gt; 8% (`--skip-face-detailer` to disable).
3c. **Framing** — reject extreme face close-ups; avoid mannequin/hand-near-face for pocket heroes.
4. **Identity** — Rina face lock; prefer soft eye contact for IG heroes.
5. **I2V-friendly** — no complex grips, finger weaves, phone typing, UI/text. Motion = blink / breath / hair / micro smile.

## Hard rejects

- `identity/face_lock` and illustration / screenshot-glam sources
- Cup / tote / typing / interlocked fingers (banlist)
- Blue-noise / NaN FaceDetailer blackouts (revert → base still; if base bad, reject)
- Headless / face-cropped “identity fail”
- NSFW into IG/Patreon packs (private track stays local)

## Folder layout

```
out/i2v_heroes/
  <run_id>/           # raw multi-seed candidates + scorecards
  <run_id>/_reject/   # human or auto rejects
  current/            # promoted passers only — human QC only
```

## Promotion

1. `scripts/gen_i2v_heroes.py` (do **not** use `--promote-auto` for production)
2. Human scorecard: skin, hands, identity, i2v_ready ∈ 0|1|2
3. Promote only scores ≥ 2 (pocket poses: hands ≥ 1 OK if hidden) into `current/`
4. Map `current/` → [`HERO_MOTION_MAP.yml`](HERO_MOTION_MAP.yml)

## Next step after stills pass

```bash
# only after current/ has ≥1 passer
screen -dmS wan_i2v /path/to/run_i2v_from_hero.sh
```

Do **not** start I2V if `current/` is empty.
