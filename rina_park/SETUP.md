# Tool setup — Rina Park (Apple Silicon)

## ComfyUI (txt2img / img2img)

Location: `/Users/RBIPK031/ai_influencer/ComfyUI`

```bash
export SSL_CERT_FILE=$HOME/combined-cert.pem
cd /Users/RBIPK031/ai_influencer/ComfyUI
.venv/bin/python main.py --force-fp16
# UI: http://127.0.0.1:8188
```

Custom nodes installed under `custom_nodes/`:

- `ComfyUI_IPAdapter_plus` — FaceID / IP-Adapter
- `comfyui_controlnet_aux` — OpenPose / Depth / Canny preprocessors
- `ComfyUI-Impact-Pack` — face detail / detection helpers

Model paths: drop downloads into `rina_park/models/{checkpoints,loras,ipadapter,controlnet,upscale}` — ComfyUI has symlinks into those folders. Checkpoint already linked: RealVisXL Lightning.

Workflows: `rina_park/workflows/*.json`

## Draw Things (img2vid / vid2vid)

**Status:** not installed on this Mac yet.

1. Install from Mac App Store: [Draw Things](https://apps.apple.com/app/draw-things-ai-generation/id6444050820)
2. In-app Models:
   - Wan 2.2 **5B** (smoke) then **14B** (quality)
   - **CausVid** LoRA for Wan (4–12 steps)
   - **VACE** pose/depth pack for vid2vid
3. i2v recipe: Rina still → subtle motion prompt → ~81 frames (~5s) → export to `rina_park/out/reels/`
4. v2v: driving swim/selfie clip → DWPose/Depth → Rina face lock still → VACE

Until Draw Things is installed, use `scripts/smoke_i2v_placeholder.sh` (Ken Burns fallback) for pipeline wiring only — replace with Wan ASAP.

## Diffusers batch (already working)

```bash
cd /Users/RBIPK031/ai_influencer
.venv/bin/python rina_park/scripts/generate_track_smoke.py --track ig
```
