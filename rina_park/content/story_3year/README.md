# Rina Park — 3년 일상 스토리 시스템

> **기획 패키지만.** 이미지 생성·학습·게시·커밋 없음.  
> 캐릭터: 가상 성인(27세) 한인 캐나다인 **Rina Park**. 실존 인물 닮음 금지. 무드보드 = 포즈/조명/분위기만.

## 한 줄 요약

1095일을 손으로 쓰지 않는다. **주간 리듬 템플릿 × 테마 필라 × 연도 아크 × 시즌 오버레이**로 하루를 생성하고, Instagram SFW 트랙과 **귀가 후 local-only NSFW 트랙**을 스키마로 분리한다.

## 두 트랙 (절대 교차 금지)

| 트랙 | 경로 / 필드 | 플랫폼 | 내용 |
|------|-------------|--------|------|
| **SFW Daily (IG)** | `ig_scenes[]` → `queues/`, `publisher/`, calendar | Instagram (및 별도 Patreon SFW) | 운동·카페·심부름·자케어·허구 친구·야외 등 일상 |
| **Home Mature (local)** | `home_mature_scenes[]` → `private/nsfw_test/` only | **게시 금지** | 상세 테마/프롬프트는 `private/content/`·`private/nsfw_test/themes/` (gitignored) |

- Home Mature는 Instagram / Patreon / `approved_exports` / packages / attestation에 **절대 진입하지 않음**.
- 기존 `mature_non_explicit` 레인과도 별도: home 트랙 사양은 `private/nsfw_test/themes/` + `private/content/` overlays.

## 생성 파이프라인 (개념)

```
year_arcs.md          → 연도·시즌 톤, 성장 비트
weekly_template.yml   → 요일별 슬롯 (gym / pilates / cafe …)
theme_pillars.yml     → 필라별 프롬프트 톤 + 하이퍼파라미터
scene_presets.yml     → scene_type → 프롬프트 블록 + 프리셋
day_schema.md         → 하루 JSON/YAML 스키마
seasonal overlay      → Toronto 계절 분위기 (허구, 실시간 위치 주장 금지)
        ↓
day_YYYYMMDD.yml      → ig_scenes[] + optional home_mature_scenes[]
```

일일 게시 목표: **최소 1–2 IG 포스트 개념/일** (최대 4–8 비트 중 선택 발행).  
연속 3년 무결석 = 템플릿이 비어 있는 날을 만들지 않음 (라이트 데이: 1컷 + 짧은 카피).

## 아이덴티티·추론 기본값 (SFW soft glam)

| 항목 | 값 |
|------|-----|
| Trigger | `rina_park_person` |
| 기본 `lora_scale` | **~0.90** (soft glam) |
| 프레이밍 | 환경형 medium-wide / 3/4 / full body; 얼굴·가슴 타이트 크롭 지양 |
| Hard angle | 뒤통수·극단 오버숄더 = **optional / rare** — 아이덴티티 리스크 |
| CFG / steps | 필라·scene_type별 프리셋 (`theme_pillars.yml`, `scene_presets.yml`) |
| 네거티브 (플랫폼) | `nude, explicit, nipples, genitals` 유지 |

## 카피·공개 게이트

- 게시 카피에는 **AI/가상 캐릭터 disclosure**가 필요 — 문구·채널 정책은 `ops/launch_pack/disclosure_candidates.md` 등 **사용자 결정**. 이 패키지는 `disclosure_gate: pending_user` 플래그만 둔다.
- 위치는 허구 분위기 (`Toronto-ish`, 계절감). **실시간 체크인·시설명·집 주소·현재 위치 주장 금지.** 카피는 ≥24h 회상 톤 권장 (`ops/strategy.md`와 정합).

## 파일 맵

| 파일 | 역할 |
|------|------|
| [year_arcs.md](./year_arcs.md) | Year 1–3 내러티브 아크 |
| [weekly_template.yml](./weekly_template.yml) | 반복 주간 리듬 + 씬 슬롯 |
| [theme_pillars.yml](./theme_pillars.yml) | 콘텐츠 필라 + 톤 + 하이퍼파라미터 |
| [day_schema.md](./day_schema.md) | 하루 레코드 스키마 |
| [sample_week_01.md](./sample_week_01.md) | Week 1 샘플 (7일 채움) |
| [sample_week_01.csv](./sample_week_01.csv) | 같은 주 표 형식 |
| [scene_presets.yml](./scene_presets.yml) | SFW scene_type → 프롬프트 블록 + 프리셋 |
| [load_overlays.py](./load_overlays.py) | private home_mature YAML soft-merge |
| `private/content/*_home_mature*.yml` | **LOCAL ONLY** home_mature presets / pillar / weekly slots |
| [../../private/nsfw_test/themes/home_mature_track.md](../../private/nsfw_test/themes/home_mature_track.md) | **LOCAL ONLY** 귀가 NSFW 테마 사다리 |

## 하지 않는 것 (이 패키지 범위)

- 이미지/영상 생성, LoRA 학습, 큐 발행, git commit/push
- 실존 인물 얼굴 학습·게시
- Home Mature를 공개 캘린더에 병합
