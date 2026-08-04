# Hyperreal v2 — Phase-0 foundation bake-off

## Phase status

**Phase 0 is complete.** The final three-reviewer aggregate selected
`qwen_image_2512` as the Phase-1 foundation: 84.944444 average and five of six
paired wins, versus 81.000000 and one paired win for `z_image`. Keep `z_image`
as the fast fallback and research comparator.

## Phase-1 identity gate

Direct Qwen-Image-2512 LoRA training is **deferred**. The current Apple trainer
path does not prove face/hair spatial mask-weighted loss plus export and loading
compatibility with the pinned MLX q8 runner.

The now-closed no-training gate was a three-image identity-region edit pilot:

- Qwen-Image-2512 supplies the complete realistic scene;
- Qwen-Image-Edit-2511 q8 may edit face/hair only;
- each call must use scene first, approved master second, and one internally
  approved matching-angle Qwen identity reference third;
- H03-A, H04-A, and H05-B are the only planned scene bases;
- moodboard faces, headshot-to-full-body expansion, single-reference whole-image
  fallback, and remote fallback are prohibited;
- pose, body, hands, wardrobe, background, lighting, lens, and sensor pixels
  outside the mask must be preserved; mask-exterior LPIPS and normalized diff
  must each be <=0.02;
- source and mask hashes, no source-face leakage, and ready Phase-1 QC are hard
  gates.

The lane is currently fail-closed because `mlx-gen==0.23.1` does not establish a
combined multi-reference-plus-mask route, the pinned q8 checkpoint cannot load
as a Diffusers checkpoint, masks and spatial LPIPS are not pinned, combined
memory is unmeasured. Phase-1 MediaPipe/OCR QC currently passes its readiness
probe and must be rechecked immediately before any future execution. Source of truth:
`hyperreal/identity/editing/pilot_plan.v1.json` and
`hyperreal/identity/editing/README.md`. Do not execute the pilot until its
readiness report is fully green and generation is separately authorized.

The mask readiness execution `20260726T151756Z` failed closed before Qwen model
load. MediaPipe found zero faces in H03-A and H04-A; H05-B produced a 103×151px
face bbox. The gate semantics were subsequently corrected: >=512px remains
mandatory for LoRA training sources, while environmental scene editing uses
>=96px plus mandatory HITL. No training gate was weakened.

The one authorized H05-B crop smoke `20260726T152322Z` passed its revised scene
gate and mask HITL, then proved the pinned MLX runtime cannot combine three
references with a mask. The first denoise blend failed because the concatenated
three-image latent shape `(1,6480,64)` cannot broadcast with the scene latent
shape `(1,2160,64)`. No edit or composite was created. Close the combined
identity-edit lane for `mlx-gen==0.23.1`; do not retry, patch the installed
runtime, drop references/mask, or use whole-image fallback. Reopen only with a
natively supported pinned combined-inpaint runtime. Direct Qwen LoRA remains
deferred.

The separately authorized external-alpha mode also closed after power-loss
recovery. Run `20260726T160610Z` successfully executed one deterministic
three-image edit without a native mask, but mandatory visual QC rejected the
crop before compositing: Rina identity transfer was insufficient, face/eye
geometry drifted, and the crop background/scale regenerated. No composite was
created. Do not retry this external-alpha contract; wait for a natively
supported region-constrained identity runtime. Direct Qwen LoRA remains
deferred.

The next safest gate is dataset acquisition, not generation or training. The
read-only identity audit inventories 192 image assets, including 18 moodboard
references that are prohibited as identity data. It quantitatively scored 110
single-face synthetic/anchor assets with uncalibrated MediaPipe geometry and
appearance diagnostics; these metrics cannot approve identity or age. Nine Qwen
assets are approved only as internal references, zero assets are
training-approved, and only two assets meet the unchanged 512px face-resolution
gate—both are explicitly rejected PhotoMaker outputs. Exact duplicates affect
73 inventory records, while the conservative dHash<=5 review queue contains 85
records (including those exact duplicates). Source of truth:
`hyperreal/identity/audit/candidate_dataset_manifest.v1.json` and
`hyperreal/identity/audit/audit_report.v1.json`.

Training remains blocked until 32 unique owner-controlled or separately
licensed, high-resolution sources fill the specified 24-train/8-validation
angle, expression, gaze, and distance slots. Rejected outputs cannot be
reclassified by resemblance or proxy score, and moodboard people can never be
used. Each source, rights record, face/hair mask, caption, deduplicated split,
trainer/runtime validation, and eventual training start requires the explicit
approvals defined in `phase1_identity_dataset_spec.v1.md`.

No Phase-0 image is approved for production. Four Qwen outputs reached aggregate
score >=85, but no output reached the photography-physics threshold of 85.
Foundation selection is a model-level decision and does not imply approval of
any sample. Final records are
`out/hyperreal_phase0_bakeoff/20260726T134754Z/review/aggregate_scores.v1.json`
and `out/hyperreal_phase0_bakeoff/20260726T134754Z/review/phase0_results.v1.md`.

## Decision and scope

Phase 0 compares foundation realism only: Qwen-Image-2512 versus Z-Image on six
shared camera hypotheses. It does not condition on Rina,
an identity image, a face embedding, a LoRA, pose pixels, or a moodboard image.
The same generic adult Korean-Canadian recreational-swimmer description and the
same seed are used for both models in each pair.

No Phase-0 output is publishable. The harness creates private evaluation
artifacts only and has no publisher integration.

Source of truth:

- `hyperreal/phase0_manifest.v1.json`: six camera hypotheses and deterministic seeds
- `hyperreal/rubric.v1.json`: blinded 100-point human rubric and calibration plan
- `hyperreal/sidecar.schema.v1.json`: private output provenance/QC contract
- `hyperreal/spec.py`: manifest validation and 12-slot blinded plan
- `hyperreal/orchestration.py`: readiness gates, serialization, logs, and adapter calls
- `hyperreal/prescreen.py`: automatic non-generative checks

## Reference abstraction record

All nine moodboard references were visually inspected read-only. They are
screenshots containing real or purported people and UI, so their people,
identity, anatomy, exact pose coordinates, wardrobe details, architecture, text,
logos, and pixels are prohibited inputs. Only scene-level observations were
abstracted:

1. Indoor public pool candid — wide phone perspective, ordinary clutter, uneven
   daylight, medium-full occupancy.
2. Outdoor pool transition — ladder contact, low camera, wet/dry boundaries,
   cloud-filtered late-day light.
3. Private covered poolside note — side-seated environmental scale, blank
   notebook action, soft pool-side light, opaque cover layer.
4. Motion/walking scene — turn during a recovery walk, small timing error,
   foreground occlusion and motion-soft foot.
5. Low-light/night pool — practical warm key plus cyan water spill, high-ISO
   shadow noise and restrained bloom.
6. City/park recovery — very wide environmental framing, small subject,
   sunset direction and a generic invented skyline.

The output prompt text is self-contained and does not send moodboard files to a
model. Moodboards are read only after generation for exact-hash and high-SSIM
copy safeguards.

## Readiness gates

Generation remains blocked until both adapters return `ready: true` before the
run directory is created. Each model-specific adapter should report these checks:

- expected local model/revision is present and integrity checks pass;
- runtime imports and target hardware backend initialize;
- license and intended-use review is acknowledged;
- one adapter-owned readiness smoke completed successfully;
- requested dimensions, seed, and output-path contract are supported;
- adapter confirms it will not route to another model or remote fallback.

The shared harness requires adapter API `phase0-foundation-adapter-v1`, exact
model IDs `qwen_image_2512` and `z_image`, and Boolean readiness checks named
`artifact_integrity`, `runtime_backend`, `license`, `generation_smoke`,
`deterministic_request`, and `no_fallback`. Missing, additional, mismatched, or
unready adapters abort before any generation. Legacy `hidream_o1` or
`hidream-o1-image-dev-2604` identifiers are explicit incompatibility errors and
are never aliases or fallback routes. A generation exception aborts the run;
the harness never retries through the other model.

The shared `AdapterRequest` mirrors the completed Qwen runner's non-model-
specific inputs: blind/case ID, hypothesis ID, exact prompt, seed, width, height,
and the complete camera hypothesis. Steps, guidance, scheduler, revision, and
runtime remain adapter-owned and must be returned as adapter metadata.

## Run behavior

- The deterministic planner creates exactly six hypotheses × two models.
- Each pair receives one identical seed and prompt.
- A fixed randomization seed assigns models to anonymous `H##_A.png` and
  `H##_B.png` filenames.
- `review_sheet.json` contains no model ID.
- `private/mapping_key.json` stores the randomized model mapping separately.
- Private model-bearing sidecars are under `private/sidecars/`.
- A filesystem lock serializes all GPU/unified-memory work.
- Per-image wall timing and process memory snapshots are logged.
- No generated file is copied into old outputs, publisher staging, or social
  packages.

## Automatic pre-screen

Pre-screening is a rejection/flagging aid, not a realism score:

- dimensions, near-black frame, severe black/white clipping;
- optional MediaPipe pose/face/hands presence flags;
- exact SHA-256 duplicate and high-threshold global SSIM checks against all nine
  moodboards and earlier bake-off outputs;
- pluggable text/watermark detector (unavailable is recorded, never presented as
  a clean result);
- complete camera-hypothesis and sidecar metadata.

An exact or high-SSIM moodboard match is a critical failure. SSIM is deliberately
used only at a high similarity threshold as a pixel-copy safeguard, not as a
semantic target.

## Blind review and decision

Lock all reviewer sheets before opening the mapping key. Score:

- Anatomy: 25
- Lighting/reflection: 20
- Lens/DOF/motion: 15
- Skin/fabric/water: 10
- Sensor/color/compression: 10
- Background: 10
- Candid composition: 10

Pass requires at least 85/100 and no critical defect. Extra/missing anatomy,
severe hand/eye/teeth failures, impossible contact/shadows/reflections,
moodboard copying, unintended text/watermarks, identity leakage, wrong scene, or
unsafe wardrobe are critical failures.

Select a foundation only after paired blind scores, critical-defect rates,
runtime, and peak-memory observations are compared. A tie is acceptable; do not
force a winner from six pairs.

## Reference-distribution calibration

Before scoring, collect 24–40 licensed or self-shot real photographs spanning
the six hypotheses. Record broad distributions for occupancy, focal-length
class, horizon roll, luminance percentiles, blur/noise proxies, and color
temperature. Reserve eight as reviewer anchors, never model conditioning. Two
reviewers independently score mixed real anchors and synthetic pilots, reconcile
differences above ten points, then lock the anchors.

Do not fabricate EXIF, claim generated files are camera originals, or tune for
AI-detector evasion. Calibration targets photographic consistency, not
provenance concealment.

## Invocation contract

Run from the repository root. This entry point performs both mandatory smokes,
validates them, and starts the 12-slot run only if both pass:

```bash
PYTHONPATH=rina_park caffeinate -dimsu .venv/bin/python -m hyperreal.execute
```

The explicit adapter exports are
`hyperreal.runners.qwen_2512.runner:ADAPTER` and
`hyperreal.runners.z_image.runner:ADAPTER`. Do not run with one adapter, pass
either runner CLI directly, or substitute HiDream.
