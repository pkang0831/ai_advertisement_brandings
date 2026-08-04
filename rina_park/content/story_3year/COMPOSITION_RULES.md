# Composition rules (locked)

Applies to IG SFW and most home NSFW. See also `scene_presets.yml` / `theme_pillars.yml`.

## Gaze (locked)

- **Default:** looking away / candid / engaged with activity / gaze off-camera
- **Avoid:** looking at camera, eye contact with viewer, selfie stare
- **Selfie beats:** rare optional; `intentional_selfie: true` only; **Week1 OFF**

## Hands / feet (locked)

### Generation (new shots)

**Prefer hiding via composition** — not ADetailer/inpaint by default.

| Prefer (positive) | Avoid (negative) |
|-------------------|------------------|
| hands in pockets | extra fingers |
| holding cup / bag / tote | deformed hands |
| hands cropped out of frame | bad anatomy hands/feet |
| soft out-of-focus hands | bare feet close-up (unless scene needs) |
| feet out of frame or obscured | detailed hands in foreground |

### Remediation (existing keepers) — separate rule

**Do NOT default to hide / crop / occlusion** to fix bad hands.  
Only hide when the composition already naturally supports it.  
Otherwise: **local hand inpaint** (+ contact physics) while preserving face/pose/framing/body proportions.  
Edit order: hands → hair → skin → fabric → lighting/contact shadows.  
See `HYPERREAL_REMEDIATION_PLAN.md` §0 / §3.

### `hands_hero` exception

- Flag: `hands_hero: true` only for rare grip / workout hero beats
- Then optional hand refine / ADetailer is allowed
- **Default `false`** — Week1 all scenes `hands_hero: false`
- FaceDetailer **face** stays ON; **hand detailer OFF** unless `hands_hero`

## Eyes (soft preference — applied)

Applies to Week1 regen (`eyes_slightly_larger_preferred`) and onward.
Prior approved `20260729T020649Z` runs kept as superseded (not deleted).

| Prefer (soft) | Avoid | Never |
|---------------|-------|-------|
| slightly larger eyes | small eyes | anime eyes |
| subtle doe eyes (photoreal) | squinting | exaggerated / oversized eyes |
| soft open eyes | beady eyes | doll eyes that break realism |

Keep photoreal Korean-Canadian adult face — gentle bias only, not cartoon.

## Finishing

- FaceDetailer face: ON (default)
- Hand detailer / hand ADetailer: OFF by default
- LoRA soft glam ~0.90; serialize MPS (one generator at a time)
- FaceDetailer soft prompts include subtle larger/doe eyes + negatives for small/squinting/beady (and anime/exaggerated)

## Hyperreal photoreal (skin / hands / hair / lighting)

Week1 ChatGPT feedback (~85–90% AI-looking) → remediation plan (no gen until approved):

→ **`HYPERREAL_REMEDIATION_PLAN.md`**

| Priority | Fix |
|----------|-----|
| 1 Hands | **Remediation:** local inpaint first (hide only if composition naturally supports). Generation still composition-hide preferred. |
| 2 Hair edges | Avoid melt / helmet silhouette |
| 3 Skin | **Photoreal soft glam**: faint pores/asymmetry, not plastic, not documentary acne |
| 4 Dark fabric | Cardigan/black cloth blobs, logo mush |
| 5 Lighting | Face vs environment + contact shadows |

Anti-pattern: upscale/sharpen-only; hand-hide as default remediation. Prefer per-image masked work on originals; identity-lock img2img denoise 0.15–0.3; ≤3 cands/region; SeedVR2 blocked/conservative. Identity cross-shot mandatory. QC: `hyperreal/qc/chatgpt_style_checklist.yml`.
