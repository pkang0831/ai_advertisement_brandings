# rina_park 디렉터리 지도

정리 일시: 2026-07-29 (콘텐츠 리셋 — Week1·테스트 산출물 삭제, 파이프라인 유지)

## 상태

- **콘텐츠 리셋 완료.** Week1 upload packs / IG·NSFW 생성물 / pilot·smoke·remediation outs / pytest / dry-run artifacts 삭제.
- **파이프라인 유지.** 생성 스크립트, `hyperreal/` 코드, `models/`(LoRA·체크포인트), identity 정의, story scaffolding, ops docs.
- 방향 재설정 후 새 calendar / week packs를 `out/`·`content/`에 다시 채우면 됨.

## 활성 진입점 (파이프라인)

| 용도 | 경로 |
|------|------|
| 생성·실험 스크립트 | `scripts/` |
| Anatomy lock (손/발/NSFW) | `scripts/run_anatomy_lock.py`, `hyperreal/anatomy/`, `ops/anatomy_lock/` |
| **I2V 히어로 스틸** | `scripts/gen_i2v_heroes.py`, `scripts/promote_i2v_heroes.py`, `ops/i2v/STILL_GATE.md` |
| I2V from hero | `scripts/run_i2v_from_hero.sh` (A14B; use detached `screen`) |
| Wan 가중치 | `models/wan/` (Comfy pack + TI2V-5B + I2V-A14B-8bit) |
| Hyperreal 오케스트레이션 | `hyperreal/` |
| 캐릭터 LoRA / 가중치 | `models/` (~118G, 미삭제) |
| Identity 레퍼런스 | `identity/` (master, face_lock, moodboard refs) |
| 스토리 스캐폴딩 | `content/story_3year/` (템플릿만; week01 결정본 삭제됨) |
| 생성 출력 | `out/` (`anatomy_lock/`, `i2v_heroes/`, `reels/`) |
| Private NSFW | `private/nsfw_test/`, `private/pose_catalog_nsfw.yml`, `private/factory/mature_non_explicit/` |
| Comfy / 루트 모델 | `../ComfyUI`, `../realvisxl…`, `../sd15_inpainting.tar.gz` |

## 레이아웃 (리셋 후)

```
rina_park/
  out/anatomy_lock/    # SFW anatomy-lock 런
  private/nsfw_test/private_media/anatomy_lock/  # NSFW local-only
  ops/anatomy_lock/    # pose_catalog.yml, RESULTS.md, scorecards, pose_refs
  hyperreal/anatomy/   # catalog / QC / 2-pass / crop-CN
  scripts/run_anatomy_lock.py
  …
```

Anatomy lock 결과 요약: `ops/anatomy_lock/RESULTS.md`

## 삭제된 주요 항목 (이번 정리)

- `out/**` 전체 (upload_packs week01, week01_ig, archive, hyperreal_* pilots, smokes, ig/patreon/reels samples)
- `private/nsfw_test` 미디어·archive·current 포인터; `private/mature_non_explicit/`; `private/factory/mature_non_explicit/`
- `tests/`, `.pytest_cache`, hyperreal `*/tests`
- `artifacts/dry_run_week_1`
- `content/story_3year/week_01_*`, `sample_week_01.*`, `HYPERREAL_REMEDIATION_PLAN.md`, `calendar_8_weeks.csv`
- identity smoke 출력 폴더; calibration `runs/` / `fallback_runs/`
- 채워진 `queues/*.csv`; `review/review.db`
- `ai_influencer` 루트: sample/screenshot PNGs, `local_gen/`, `image_prompt_queue.csv`

## 손대지 않은 것

- `models/` 전체 (LoRA safetensors 포함)
- `hyperreal/identity/training/intake` (캐릭터 학습 데이터)
- 파이프라인 소스 (`factory`, `orchestrator`, `publisher`, `scripts`, …)
- `.env`, `.gitignore`, README/SETUP
- git commit/push 없음

## 후속 정리 (스크립트)

- week01 / pilot / smoke 생성 스크립트·하네스 추가 삭제 완료 (`scripts/*week01*`, `pilot_*`, `*smoke*`, `nsfw_private_pilot`, `scene_pilot` 등).
- `hyperreal/identity/editing/pilot_plan.v1.json` + `readiness.py`는 아이덴티티 편집 readiness용으로 **유지**.
- 루트 `sd15_inpainting.tar.gz` / `realvisxl…` / `ComfyUI` — 파이프라인 자산으로 유지.
