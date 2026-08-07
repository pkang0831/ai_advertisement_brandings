# Rina Park — Google Colab + Drive setup

Colab에서 작업할 때 Drive를 **모델·출력·identity 저장소**로 쓰고, 코드는 매 세션 `git clone` 합니다.

## 아키텍처

| 위치 | 내용 |
|------|------|
| `MyDrive/rina_park_colab/models/` | 가중치 (HF는 Colab에서 다운로드, 캐릭터/CivitAI는 Mac rclone) |
| `MyDrive/rina_park_colab/{identity,moodboard,private,out}/` | 자산·출력 |
| `/content/ai_influencer` | git clone 코드; `rina_park/models` 등은 Drive로 symlink |

**중요:** 로컬 Qwen/SeedVR/Z-Image AbstractFramework 팩은 **MLX(맥 전용)** 입니다. Colab에는 매니페스트의 **CUDA upstream** (`Qwen/...`, `ByteDance-Seed/...`)을 받습니다.

## 1) Mac — 로컬 자산 Drive로

올리는 것: `loras/` 전체(캐릭터+CivitAI), Juggernaut/CyberRealistic/Lightning, ultrasharp, `identity/`, `moodboard/`, `private/`.  
(RealVis full 등 HF 공개 모델은 Colab `colab_download_hf_models.py`로 받아도 됨.)

```bash
cd /Users/RBIPK031/ai_influencer
chmod +x rina_park/scripts/sync_local_only_to_drive.sh
./rina_park/scripts/sync_local_only_to_drive.sh --dry-run   # 확인
./rina_park/scripts/sync_local_only_to_drive.sh             # 실행 (~24GB, 이어받기)
```

이미 올라간 LoRA는 rclone이 스킵하고, 체크포인트부터 이어갑니다.

## 2) Colab Secrets

런타임 → 연결 → 보안 비밀:

- `HF_TOKEN` (gated 모델용, 권장)
- `GIT_TOKEN` (private repo면)
- 선택: `CIVITAI_API_TOKEN`

## 3) Colab 부트스트랩

노트북: [`rina_park/notebooks/colab_bootstrap.ipynb`](../notebooks/colab_bootstrap.ipynb)

`colab_bootstrap.py`는 **Colab(`/content`)에서만** symlink를 만듭니다. Mac에서 실수로 실행하면 거부합니다. 로컬에 이미 `models/`가 있으면 non-empty 거부(덮어쓰지 않음). Colab fresh clone에서는 gitignore된 폴더가 없어 안전합니다.

또는 셀:

```python
from google.colab import drive, userdata
import os
drive.mount('/content/drive')

os.environ['HF_TOKEN'] = userdata.get('HF_TOKEN', '')
git_token = userdata.get('GIT_TOKEN', '')
repo = 'https://github.com/pkang0831/ai_advertisement_brandings.git'
if git_token:
    repo = f'https://{git_token}@github.com/pkang0831/ai_advertisement_brandings.git'

!git clone --depth 1 {repo} /content/ai_influencer
%cd /content/ai_influencer
!pip -q install -r requirements-colab.txt

!python rina_park/scripts/colab_bootstrap.py
!python rina_park/scripts/colab_download_hf_models.py --tier sdxl
```

티어:

- `--tier sdxl` — RealVis, IP-Adapter, ControlNet, PhotoMaker, RealESRGAN (~수십 GB)
- `--tier wan` — Wan TI2V-5B + I2V-A14B Diffusers (큼)
- `--tier qwen_cuda` — Qwen / SeedVR2 / Z-Image upstream
- `--tier all` — 전부

매니페스트: [`colab_model_manifest.yml`](colab_model_manifest.yml)

## 4) Smoke (SDXL 1장)

```bash
cd /content/ai_influencer
PYTHONPATH=/content/ai_influencer:/content/ai_influencer/rina_park \
  python rina_park/scripts/generate_ig_quality.py
```

디바이스는 `rina_park.runtime_device`가 **cuda > mps > cpu**로 고릅니다.

## 5) 이후

- Wan / Qwen CUDA 러너는 별도 포트 (mlx-gen 스크립트는 Colab에서 그대로 안 됨).
- ComfyUI는 1차 범위 밖 — Diffusers 스크립트 우선.
- 세션이 끊겨도 Drive `models/`는 유지; `colab_download_hf_models.py`는 resume / skip 지원.

## rclone 참고

이 Mac의 remote 이름은 `gdrive`입니다. rclone shared client_id 경고가 나면 [자체 client_id](https://rclone.org/drive/#making-your-own-client-id) 설정을 권장합니다.
