# Anatomy Lock Results — Rina Park

Date: 2026-07-29  
Primary run: `20260729T190311Z` (~510s, 9 poses × methods 1/2/3)  
Guard rerun: `20260729T191300Z` (NaN crop guards; MPS black-frame after CN load)

## Mandate

Try **all three** approaches with evidence so hands / feet / genital detail can be production-gated:

1. Pose catalog whitelist  
2. Dedicated 2-pass (narrow crop img2img / ADetailer-style)  
3. ControlNet on **hand/foot crop** + verified pose ref (not full-frame OpenPose retry)

Stack: RealVisXL V5 + `rina_park_person` LoRA ~0.88–0.90, FaceDetailer face (with NaN revert), strong anatomy negatives.

## Deliverable map

| Artifact | Path |
|----------|------|
| Pose catalog | `ops/anatomy_lock/pose_catalog.yml` |
| Verified pose refs | `ops/anatomy_lock/pose_refs/` |
| Modules | `hyperreal/anatomy/` |
| Runner | `scripts/run_anatomy_lock.py` |
| SFW outs | `out/anatomy_lock/<run_id>/` |
| NSFW outs (local only) | `private/nsfw_test/private_media/anatomy_lock/<run_id>/` |
| Auto scorecards | `ops/anatomy_lock/scorecards/<run_id>/` |
| Run JSON | `ops/anatomy_lock/run_summary_<run_id>.json` |

```bash
cd /Users/RBIPK031/ai_influencer
PYTHONPATH=rina_park .venv/bin/python -u rina_park/scripts/run_anatomy_lock.py
```

---

## Auto QC pass rates (run `20260729T190311Z`)

| Method | Auto overall pass | Notes |
|--------|-------------------|-------|
| **1 Pose catalog** | **5 / 9** | Best raw rate |
| **2 Dedicated 2-pass** | **0 / 7 applied** | Many crops → MPS NaN/black paste |
| **3 Crop ControlNet** | **4 / 8 applied** | Auto topology often cleaner; visual mixed |

Per-region (auto):

| Region | M1 | M2 | M3 | Notes |
|--------|----|----|----|-------|
| Hands | Often pass when hidden/simple; cup/tote grip flaky | Failed (black voids / lost landmarks) | Several auto-pass | Auto ≠ human OK |
| Feet | `foot_seated` pass; `foot_standing` fail (no face / headless crop) | Mixed | Pass when applied | Sneakers help |
| Genitals (NSFW) | Proxy pass on 3/3 NSFW | Proxy pass but image damage | N/A (hand/foot crops) | Human: soft detail, not locked |

---

## Human visual scorecard (selected)

Rubric: 0 fail / 1 borderline / 2 pass.

### SFW hands

| Image | Method | Hands | Notes |
|-------|--------|-------|-------|
| `…/190311Z/hand_holding_cup_soft/m1_catalog.jpg` | 1 | **0** | Extra/fused digits on cup grip; auto falsely passed |
| `…/191300Z/hand_holding_cup_soft/m2_2pass.jpg` | 2 | **1** | No black void after guard; nails weird, joints soft |
| `…/190311Z/hand_resting_lap_seated/m3_crop_cn.jpg` | 3 | **1–2** | Soft resting; usable lifestyle |
| `…/190311Z/hand_holding_tote_side/m3_crop_cn.jpg` | 3 | **0** | Black oval on grip + melted free hand |
| `…/190311Z/hand_pockets_3q/m1_catalog.jpg` | 1 | **0** | Prompt ignored pockets → phone grip (banned class) |
| `…/190311Z/foot_seated_ankle_soft/m1_catalog.jpg` | 1 | **1** | Phone hands soft; sneakers/feet **2** |

### SFW feet

| Image | Method | Feet | Notes |
|-------|--------|------|-------|
| `…/foot_seated_ankle_soft/m1_catalog.jpg` | 1 | **2** | Flat sneakers, clear ankles |
| `…/foot_standing_flat_sneakers/m1_catalog.jpg` | 1 | **2** | Good feet; **identity fail** (headless framing) |

### NSFW genitals (local only)

| Image | Method | Genitals | Hands | Notes |
|-------|--------|----------|-------|-------|
| `private/…/nsfw_standing_mirror_hip_angle/m1_catalog.jpg` | 1 | **1** | **1** | Visible coherent region; soft/airbrushed; left hand mild fuse |
| `private/…/nsfw_standing_mirror_hip_angle/m2_2pass.jpg` | 2 | **1** | — | Genital refine attempted; **face/torso black NaN damage** |
| `private/…/nsfw_reclined_knee_up_partial/m1_catalog.jpg` | 1 | **1** | 0 | Partial occlusion angle; hand detect miss |
| `private/…/nsfw_side_lying_tucked/m1_catalog.jpg` | 1 | **1** | 1–2 | Tucked angle more stable; feet gate flaky |

---

## Method verdicts

### 1 — Pose catalog — **PARTIAL PASS (best default)**

- Whitelist + banlist + short anatomy negatives **implemented and enforced**.
- High-pass compositions work when the model **obeys** (resting / sneakers).
- Failures: complex grips (cup/phone), prompt drift away from pockets.
- **Production:** default generator must sample only whitelist IDs; reject banned keywords.

### 2 — Dedicated 2-pass — **FAIL as reliable MPS path (attempted)**

- Narrow crop img2img + fixed seed offsets **implemented**.
- On this Mac, second-pass crops frequently decode to **NaN/black** → destroys hands/face.
- After guards: can preserve base when refine is void; does not yet *fix* anatomy reliably.
- **Next:** isolate one image per process; lower denoise (0.25–0.32); consider SDXL inpaint UNet on CUDA.

### 3 — Crop + verified pose ControlNet — **PARTIAL (better than full-frame CN)**

- OpenPoseXL2 on **crops** + template pose refs **ran** (not full-frame retry).
- Auto hand gates improved on several shots; visual still uneven; NaN crops possible.
- Static verified skeletons ≠ instance-matched OpenPose from the crop → weak grip alignment.
- **Next:** draw control from MediaPipe landmarks of the crop itself; try depth-CN on crop; DWPose once `controlnet_aux`/`mediapipe.solutions` conflict fixed.

---

## Recommended production path

**Ship path (honest):**

1. **Primary:** Method 1 only — prefer `hand_pockets_3q` / `hand_cropped_out` / `hand_resting_lap_seated` / shod-feet poses.  
   - Demote or ban **two-hand cup** and **phone typing** from default.  
2. **Optional repair:** Method 3 crop-CN **only** when MediaPipe sees a hand box and refine patch passes mean/finite guard.  
3. **Do not** chain FaceDetailer + M2 + M3 in one long MPS session without process recycle (black-frame cascade observed after ControlNet load).  
4. NSFW genital: Method 1 whitelist angles only for now; M2 genital pass needs stable decode (CUDA preferred). Keep under `private/` forever.

### Best current keepers

| Use | Path |
|-----|------|
| SFW feet keeper | `out/anatomy_lock/20260729T190311Z/foot_seated_ankle_soft/m1_catalog.jpg` |
| SFW soft hands (borderline) | `out/anatomy_lock/20260729T190311Z/hand_resting_lap_seated/m3_crop_cn.jpg` |
| SFW feet-only framing | `out/anatomy_lock/20260729T190311Z/foot_standing_flat_sneakers/m1_catalog.jpg` |
| NSFW genital (local) | `private/nsfw_test/private_media/anatomy_lock/20260729T190311Z/nsfw_standing_mirror_hip_angle/m1_catalog.jpg` |
| Cleaner cup triple (auto) | `out/anatomy_lock/20260729T191300Z/hand_holding_cup_soft/{m1,m2,m3}_*.jpg` |

---

## Blockers / next changes

1. **MPS VAE NaN** after multi-pass / ControlNet — #1 reliability blocker. Mitigate: new process per pose; NaN guards (landed); float32 VAE already on.  
2. **`controlnet_aux` broken** (`mediapipe` has no `.solutions`) — cannot use stock OpenposeDetector; using drawn verified refs + MediaPipe Tasks instead.  
3. **Auto hand gate ≠ human** — cup grip auto-passed while visually fused; tighten fusion thresholds + optional CLIP/finger-count model.  
4. **Hand LoRA A/B done** — winner `detailed_hands@0.45` on resting (see `HAND_FOOT_LORA_AB_RESULTS.md`); pocket poses without hand LoRA.  
5. **Prompt obedience** — pockets → phone; consider stronger negatives for phone/keyboard + img2img from pocket reference.  
6. **True SDXL inpaint UNet** not used (9-ch); ADetailer-style crop img2img only.

## Success criteria checklist

| Criterion | Status |
|-----------|--------|
| Clear pass/fail per method for hands/feet/genitals | **Yes** (auto + human above) |
| Recommended production path | **Method 1 primary; M3 optional; M2 not production on MPS yet** |
| Paths to best outputs | **Yes** |
| Honest if still failing + next change | **Yes — not stage-ready for hero hand grips; feet/sneakers OK; NSFW genital soft-pass only** |
