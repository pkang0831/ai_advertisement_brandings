# ControlNet img2img (pose / depth from moodboard)

Goal: copy **composition** from `moodboard/ref_XX.png` while keeping Rina face lock.

## Graph outline

1. LoadImage → moodboard ref (pose donor)
2. `comfyui_controlnet_aux` preprocessor:
   - OpenPose (full body swim poses)
   - Depth (environment layout)
   - Canny (hard edges / ladder rails)
3. ControlNetLoader (SDXL OpenPose / Depth / Canny from `models/controlnet/`)
4. ControlNetApply (strength 0.55–0.85)
5. Optional: img2img latent from ref at denoise 0.45–0.65 for phone-real structure
6. Positive prompt from the SFW `prompt_presets.json` track (ig / patreon_a|b|c)
7. Apply identity lock only after the commercial-license and 12/12 gates pass

## Per-track denoise tips

| Track | Denoise | ControlNet |
|-------|---------|------------|
| IG pool plog | 0.5–0.65 | OpenPose 0.7 |
| Patreon A home | 0.45–0.6 | Depth 0.6 |
| Patreon B alternate edit | 0.55–0.7 | OpenPose 0.75 |
| Patreon C archive edit | 0.55–0.7 | OpenPose/Depth mix |

Drop finished graphs here as `02_img2img_controlnet_rina.json` after first successful UI export (Load → Save).
Pose and depth production use remains disabled until each exact ControlNet and
preprocessor tuple has pinned revision, SHA-256, commercial license, MPS smoke
result, and a 12/12 fixed-scene benchmark. Do not download models from this
workflow.
