# Phase-3 conservative finishing

This lane is private, offline, serialized, and fail-closed. It does not approve,
repair, or publish an image.

## Fixed route

- Model: `AbstractFramework/seedvr2-3b-8bit`
- Revision: `6e6b162d1e96c873ad93645508bbc4636814c0c7`
- Runtime: `mlx-gen==0.23.1` / MLX Metal (`mlx==0.31.2`)
- Scale: exactly `1.5x` or `2x`
- Model output blend: `0.15–0.45`, default `0.30`
- Default VAE tile: `768`, overlap `128`
- MLX cache limit: at most `8 GiB`
- Input/output: hashed approved lossless source to a new private lossless path

The runtime network is disabled before model construction. A non-blocking file
lock permits one unified-memory job. Failure or metric rejection deletes the
candidate image, clears the MLX cache, and leaves a JSON rejection/failure
sidecar.

## Hard policy

Rejected images, anatomy failures, critical defects, and sources not explicitly
marked `approved_for_finishing` cannot enter the runner. Restoration is not an
anatomy repair operation. Output paths containing publisher, social, calendar,
staging, or approved-export components are rejected.

The non-generative fallback additionally requires `source_media_class` to be
exactly `standard_sfw` and rejects source paths marked mature, NSFW, or
adult-only. It cannot run on rejected, mature, or unknown-class media.

A metric pass still receives only `pending_100_percent_human_review`;
automatic promotion and publication are always false.

## Validation

The caller must inject local, offline landmark and LPIPS providers. Validation
rejects face-count changes, normalized landmark drift, face-box drift,
face/global LPIPS changes, excessive or absent edge gain, added black/white
clipping, low-frequency shifts, and excessive high-frequency gain.

SeedVR2 does **not** have an authoritative identity-preservation guarantee.
The official model card warns that its generation strength can over-generate
detail on lightly degraded inputs. The LPIPS and non-Rina corpus readiness
requirements are now closed, but Rina inference remains blocked because neither
scale passed the conservative calibration gates.

## Calibration result

The isolated run `20260726T163200Z` processed three Pexels-licensed portraits
whose source captions explicitly identify adults. The local moodboard was not
used because its manifest does not explicitly establish adulthood; no age or
identity was inferred and no moodboard file was copied.

Official LPIPS `0.1.4` with its v0.1 Alex calibration head is pinned locally.
The provider explicitly loads the local AlexNet trunk and then runs offline on
PyTorch MPS. LPIPS code/head are BSD-2-Clause and TorchVision code is
BSD-3-Clause. TorchVision warns that pretrained weights may carry separate
training-data terms, so this provider is restricted to private metric
evaluation and is never shipped or redistributed.

Three images at `1.5x` and three at `2x` were run sequentially at blend strength
`0.30`. Both scales remain disabled:

- `1.5x`: one of three passed; two exceeded the high-frequency hallucination
  guardrail and one also added black clipping.
- `2x`: zero of three passed; all exceeded the high-frequency guardrail and one
  also added black clipping.

Landmark shape, face-box IoU, LPIPS, edge gain, white clipping, and low-frequency
preservation passed, but those passes do not override the failed detail/clipping
gates. Thresholds were not relaxed. Source of truth:
`calibration/calibration_report.v1.json`.

## Non-generative fallback

Run `20260726T180000Z` reused the same three Pexels-licensed, explicitly adult,
non-Rina portraits. It evaluated 15 sequential configurations: three images
across metadata-only original resolution, Lanczos `1.5x`/`2x`, and bounded
Lanczos plus unsharp/local-contrast/color presets. No Rina or moodboard image,
generative model, face enhancer, diffusion model, or learned image operation
was used.

The production default and only enabled preset is
`original_metadata_only`. It writes a validation sidecar and no replacement
pixels. All upscale presets remain blocked:

- Lanczos `1.5x`: `0/3`; mean ringing fraction `0.004259`.
- Lanczos `2x`: `0/3`; mean ringing fraction `0.005975`.
- Conservative `1.5x`: `0/3`; mean ringing `0.007211`, black clipping
  `0.004971`, and one face-IoU failure.
- Conservative `2x`: `0/3`; mean ringing `0.009426`, black clipping
  `0.004983`, and one face-IoU failure.

The ringing limit remains `0.001`; clipping and face geometry limits were also
not relaxed. Every result remains non-promotable and requires human review.
Source of truth:
`calibration/fallback_calibration_report.v1.json`.
