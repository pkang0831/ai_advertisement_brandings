# Hand / Foot specialty LoRAs (SDXL)

Date: 2026-07-29  
Base: RealVisXL V5 + `rina_park_person` @ **0.88–0.90**  
Save dir: `/Users/RBIPK031/ai_influencer/rina_park/models/loras/`

**Agent note:** Civitai/HF downloads are Netskope-blocked in this environment. User placed upstream filenames (not snake_case aliases). Wire stack to **On disk** names below (quote spaces for Better Hands).

---

## Chosen (download these)

### 1) Hand Detail XL v2.0 — **primary hand**

| | |
|--|--|
| Why | Focused hand-detail LoRA; high adoption; ~55MB; photoreal-friendly; low style drift |
| Model page | https://civitai.com/models/260852?modelVersionId=294259 |
| Direct download | https://civitai.com/api/download/models/294259 |
| Upstream file | `detailed_hands-000002.safetensors` |
| **On disk (verified 2026-07-29)** | `detailed_hands-000002.safetensors` (55M, SDXL, valid safetensors; `ss_output_name=detailed_hands`) |
| Optional alias | `hand_detail_xl_v2.safetensors` (not created — use on-disk name or symlink) |
| Trigger | `hand` (also works with natural “detailed hands” in prompt) |
| Start weight | **0.45** (range 0.4–0.5; raise to ~0.6 only if still soft) |
| License / use | Civitai: commercial Image/Rent allowed (check page before sell/redistribute weights) |
| HF mirror | No reliable official HF mirror found — use Civitai |

### 2) Better hands SDXL v1.0 — **secondary hand (A/B)**

| | |
|--|--|
| Why | Explicit photoreal hand concept; clear triggers; v1 (~218MB) preferred over huge v2 (~870MB) |
| Model page | https://civitai.com/models/1584999?modelVersionId=1793605 |
| Direct download | https://civitai.com/api/download/models/1793605 |
| Upstream file | `Better Hands SDXL v1.0.safetensors` |
| **On disk (verified 2026-07-29)** | `Better Hands SDXL v1.0.safetensors` (218M, SDXL, valid; spaces in name — quote paths) |
| Optional alias | `better_hands_sdxl_v1.safetensors` (not created — rename/symlink recommended for scripts) |
| Trigger | `Perfect hand,` / `Detailed hand,` |
| Start weight | **0.4** (author tip 0.6–0.8 alone; stack under character → start lower) |
| License / use | Civitai commercial Image/Rent — verify page |
| HF mirror | None known — use Civitai |

### 3) real_feet_xl v1.0 — **optional foot** (skip if you only want hands)

| | |
|--|--|
| Why | Small (~41MB) SDXL feet concept; less fetish-heavy than “Feet XL” mega pack |
| Model page | https://civitai.com/models/127238?modelVersionId=139194 |
| Direct download | https://civitai.com/api/download/models/139194 |
| Upstream file | `RealFeet_xl_v1.safetensors` |
| **On disk (verified 2026-07-29)** | `RealFeet_xl_v1.safetensors` (41M, SDXL, valid; `ss_output_name=RealFeet_xl_v1`) |
| Optional alias | `real_feet_xl_v1.safetensors` (not created — close enough; casing differs only) |
| Trigger | `feet` |
| Start weight | **0.4** |
| Skip instead | https://civitai.com/models/200251 (Feet XL) — larger, style/fetish drift risk |

---

## HuggingFace alternative (if Civitai blocked for you too)

| | |
|--|--|
| Name | SDXL LoRA slider: nice hands |
| Why | Small HF-hosted hand slider; weaker than Hand Detail XL but downloadable if HF works |
| HF page | https://huggingface.co/ntc-ai/SDXL-LoRA-slider.nice-hands |
| Direct | https://huggingface.co/ntc-ai/SDXL-LoRA-slider.nice-hands/resolve/main/nice%20hands.safetensors |
| **Save as** | `nice_hands_slider_sdxl.safetensors` |
| Trigger | slider / prompt “nice hands” (check card) |
| Start weight | **0.4–0.5** |

---

## Production stack (recommended)

```
rina_park_person_sdxl_lora.safetensors              @ 0.88–0.90
detailed_hands-000002.safetensors                   @ 0.45   (+ trigger "hand" / "detailed hands")
# optional A/B instead of primary hand:
# Better Hands SDXL v1.0.safetensors                 @ 0.40   (+ "Perfect hand," / "Detailed hand,")
# optional feet poses only:
# RealFeet_xl_v1.safetensors                         @ 0.40   (+ "feet")
```

**HF fallback:** `nice_hands_slider_sdxl.safetensors` — **not present** (not needed; Civitai trio verified).

Do **not** stack both hand LoRAs at once for production (pick one winner after A/B).

---

## Skipped

| Model | Why skipped |
|-------|-------------|
| ClearHandsXL (https://civitai.com/models/132884) | ~751MB; oversized for specialty stack |
| Better hands SDXL **v2** | ~870MB; prefer v1 |
| Feet XL mega | Fetish/pose drift; use real_feet_xl if needed |

---

## On-disk status (verified 2026-07-29)

| Slot | Exact path | Size | Header |
|------|------------|------|--------|
| 1 primary hand | `.../loras/detailed_hands-000002.safetensors` | 55M | OK (SDXL LoRA) |
| 2 A/B hand | `.../loras/Better Hands SDXL v1.0.safetensors` | 218M | OK (SDXL LoRA; quote spaces) |
| 3 feet | `.../loras/RealFeet_xl_v1.safetensors` | 41M | OK (SDXL LoRA) |
| HF fallback | `nice_hands_slider_sdxl.safetensors` | — | missing |

```bash
ls -lh \
  "/Users/RBIPK031/ai_influencer/rina_park/models/loras/detailed_hands-000002.safetensors" \
  "/Users/RBIPK031/ai_influencer/rina_park/models/loras/Better Hands SDXL v1.0.safetensors" \
  "/Users/RBIPK031/ai_influencer/rina_park/models/loras/RealFeet_xl_v1.safetensors"
```

Optional rename (only if pipeline wants snake_case, no spaces):

```bash
cd /Users/RBIPK031/ai_influencer/rina_park/models/loras
# ln -sf "detailed_hands-000002.safetensors" hand_detail_xl_v2.safetensors
# ln -sf "Better Hands SDXL v1.0.safetensors" better_hands_sdxl_v1.safetensors
# ln -sf "RealFeet_xl_v1.safetensors" real_feet_xl_v1.safetensors
```

**A/B completed 2026-07-29** — see `HAND_FOOT_LORA_AB_RESULTS.md`.

Human winner: **`detailed_hands` @ 0.45** on hand-visible resting; avoid hand LoRA on pocket/hidden poses.  
Run: `out/anatomy_lock/20260729T201141Z_hand_lora_ab/`

---

## A/B run completed

- Run id: `20260729T201141Z_hand_lora_ab`
- Results: `ops/anatomy_lock/HAND_FOOT_LORA_AB_RESULTS.md`
- Auto QC: all hand poses pass (tie) → **human winner `arm_a` / detailed_hands @ 0.45**
- Outputs: `out/anatomy_lock/20260729T201141Z_hand_lora_ab/`
- Symlinks created: `hand_detail_xl_v2`, `better_hands_sdxl_v1`, `real_feet_xl_v1`
