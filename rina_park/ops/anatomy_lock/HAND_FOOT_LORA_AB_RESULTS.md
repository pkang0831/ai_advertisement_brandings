# Hand / Foot LoRA A/B Results

Date: 2026-07-29  
Run: `20260729T201141Z_hand_lora_ab`  
Stack base: RealVisXL V5 + `rina_park_person` @ **0.90**  
Seeds: fixed from base `29072026` (+ pose offset ×97); steps=32 cfg=4.2  
Poses: `hand_resting_lap_seated`, `hand_pockets_3q`, `foot_standing_flat_sneakers`, `foot_seated_ankle_soft`  
Output: `out/anatomy_lock/20260729T201141Z_hand_lora_ab/`  
Elapsed (resume run): ~429s (process-isolated per pose)

## Design

| Arm | Stack | Poses |
|-----|-------|-------|
| baseline | character only | all |
| arm_a | + detailed_hands @ 0.45 (+ trigger `hand, detailed hands`) | all |
| arm_b | + Better Hands @ 0.40 (+ `Perfect hand, Detailed hand`) | all |
| arm_c | + RealFeet @ 0.40 (+ `feet`) | foot only |
| arm_d | Better Hands + RealFeet (auto-tiepick; superseded by human) | foot_seated |

Method: catalog **base gen only** (no FaceDetailer / 2-pass / ControlNet).  
Multi-LoRA: specialty adapters loaded **UNet-only** (TE keys break PEFT multi-adapter under character LoRA).

## Auto QC pass rates

| Arm | Overall pass | Hands pass | Feet pass | N |
|-----|--------------|------------|-----------|---|
| baseline | 3/4 | 4/4 | 4/4 | 4 |
| arm_a | 3/4 | 4/4 | 3/4 | 4 |
| arm_b | 3/4 | 4/4 | 3/4 | 4 |
| arm_c | 1/2 | 2/2 | 1/2 | 2 |
| arm_d | 1/1 | 1/1 | 1/1 | 1 |

Hand-pose auto: **all arms 2/2** on resting+pockets → auto cannot rank; **human visual decides**.

`foot_standing_flat_sneakers` often fails **identity** (headless / face crop) across arms — known framing issue, not LoRA-specific.

## Human visual scorecard (primary)

Rubric: 0 fail / 1 borderline / 2 pass. Focus: `hand_resting_lap_seated` (hand-visible).

| Arm | Resting hands | Pockets | Notes |
|-----|---------------|---------|-------|
| baseline | **0–1** | **2** | Resting drifted to phone grip; soft/fused digits. Pockets correctly hidden. |
| **arm_a** | **2** | **1** | Resting: chest+lap soft pose, clearer finger separation. Pockets drifted to phone (trigger pulls hands visible). |
| arm_b | **0–1** | **1** | Resting: clasped/interlock with melt/fusion. Pockets → phone; soft joints. |

Feet (`foot_seated_ankle_soft`): baseline sneakers clean parallel plant (**2**). arm_c one-foot-up is more complex, lace clutter (**1–2**) — **not a clear win** over baseline sneakers.

## Recommendation (human)

**Hand LoRA winner: `detailed_hands` @ 0.45** (`detailed_hands-000002.safetensors` / symlink `hand_detail_xl_v2.safetensors`)

- Best improvement on **hand-visible resting** vs baseline and vs Better Hands.  
- Do **not** use hand LoRA (+ hand triggers) on **pocket / hands-hidden** poses — encourages phone/grip drift.  
- Better Hands @ 0.4: no win this pass; keep for optional retest on grip poses later.  
- RealFeet @ 0.4: optional; shod sneakers already strong without it.

### Production stack

```
rina_park_person_sdxl_lora.safetensors     @ 0.88–0.90
detailed_hands-000002.safetensors          @ 0.45   # hand-visible whitelist only
# Better Hands — not default after this A/B
# RealFeet_xl_v1 @ 0.40 — optional feet-only; skip if sneakers whitelist holds
```

Wire via:

```bash
PYTHONPATH=rina_park .venv/bin/python rina_park/scripts/run_anatomy_lock.py \
  --extra-lora detailed_hands:0.45 --skip-method2 --poses hand_resting_lap_seated
```

## Next stage gate

| Question | Verdict |
|----------|---------|
| Hands improved enough for next stage? | **Yes, conditionally** — promote `detailed_hands@0.45` on **resting / soft-visible** whitelist; keep pockets **without** hand LoRA |
| Hard grips (cup/tote)? | Still out of default — retest winner there before promote |
| Stack both hand LoRAs? | **No** |

## Output paths

| Arm | Resting | Pockets |
|-----|---------|---------|
| baseline | `out/anatomy_lock/20260729T201141Z_hand_lora_ab/baseline/hand_resting_lap_seated/m1_catalog.jpg` | `…/baseline/hand_pockets_3q/m1_catalog.jpg` |
| arm_a ★ | `…/arm_a/hand_resting_lap_seated/m1_catalog.jpg` | `…/arm_a/hand_pockets_3q/m1_catalog.jpg` |
| arm_b | `…/arm_b/hand_resting_lap_seated/m1_catalog.jpg` | `…/arm_b/hand_pockets_3q/m1_catalog.jpg` |
| arm_c feet | `…/arm_c/foot_seated_ankle_soft/m1_catalog.jpg` | |
| arm_d combo | `…/arm_d/foot_seated_ankle_soft/m1_catalog.jpg` | |

Full JSON: `ops/anatomy_lock/run_summary_20260729T201141Z_hand_lora_ab.json`
