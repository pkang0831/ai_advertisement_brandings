# Day Schema — 하루 레코드

하루 = **스토리 비트 1개** + **IG 씬 1–8개(발행은 1–2개 선택)** + **optional home_mature(로컬 전용)**.  
파일 예: `days/y01_w01/day_20260803.yml` (생성기는 나중에; 지금은 스키마만).

## 필수 원칙

1. `ig_scenes[]`와 `home_mature_scenes[]`는 **같은 파일에 있어도 출력 루트가 다름**.
2. `home_mature_scenes[].local_only`는 항상 `true`. 플랫폼 큐로 export 금지.
3. `disclosure_gate`가 `approved`가 아니면 IG 카피 발행 보류 (문구는 사용자 결정).
4. `hard_angle` 씬은 `rarity: rare` + `identity_risk: true`.

---

## 필드 정의

| 필드 | 타입 | 필수 | 설명 |
|------|------|------|------|
| `day_id` | string | ✅ | `y{YY}_w{WW}_d{Dow}` 예: `y01_w01_mon` |
| `date` | date | ✅ | `YYYY-MM-DD` (스토리 캘린더; 실시간 아님) |
| `year_arc` | enum | ✅ | `1` \| `2` \| `3` |
| `week_index` | int | ✅ | 연내 1–52/53 |
| `weekday` | enum | ✅ | `mon`…`sun` |
| `story_beat` | string | ✅ | 그날 한 줄 내러티브 (한국어 또는 영문) |
| `seasonal_overlay` | object | ✅ | `year_arcs.md` 키 |
| `outfit_continuity` | object | ✅ | 아침→저녁 옷/소품 연속성 |
| `ig_scenes` | array | ✅ | 길이 ≥1 |
| `home_mature_scenes` | array | ❌ | 있으면 전부 `local_only: true` |
| `params_defaults` | object | ✅ | day-level LoRA/CFG 기본 |
| `disclosure_gate` | enum | ✅ | `pending_user` \| `approved` \| `blocked` |
| `status` | enum | ✅ | `planned` \| `assets_pending` \| `ready` \| `published_ig` \| `skipped_light` |

---

## YAML 예시 (구조만)

```yaml
day_id: y01_w01_mon
date: 2026-08-03
year_arc: 1
week_index: 1
weekday: mon
story_beat: "새 주 — 짐 가방을 문 앞에 두고 다시 시작하는 월요일."
seasonal_overlay:
  code: toronto_late_summer
  wardrobe_bias: [leggings, light_hoodie, sneakers]
  light_bias: soft_morning_gym
  copy_mood: warm_reflective_not_live
outfit_continuity:
  morning: "black leggings, heather tank, light gray hoodie, white sneakers"
  training: "hoodie off; tank + sports bra layered look still covered"
  evening: "same leggings, oversized hoodie back on, tote"
  recurring_props: [water_bottle, charcoal_tote, wireless_earbuds]
params_defaults:
  trigger: rina_park_person
  lora_scale: 0.90
  steps: 36
  cfg: 4.2
  size: "832x1216"
disclosure_gate: pending_user
status: planned

ig_scenes:
  - scene_id: y01_w01_mon_ig_01
    pillar: fitness
    scene_type: gym_arrival
    format: carousel  # still | carousel | reel
    publish_intent: primary  # primary | alternate | archive_only
    story_note: "도착 — 락커 복도 쪽 환경샷."
    prompt_ref: scene_presets.gym_arrival
    framing: environmental_medium_wide
    hard_angle: false
    hyperparams:
      lora_scale: 0.90
      steps: 36
      cfg: 4.0
    caption_seed: "어제 문 앞에 둔 가방이 오늘을 결정했다."
    location_label: "Greater Toronto Area"  # broad only
    local_only: false

  - scene_id: y01_w01_mon_ig_02
    pillar: selfcare
    scene_type: post_workout_stretch_home
    format: still
    publish_intent: primary
    story_note: "귀가 후 SFW 스트레칭 — 플랫폼 세이프."
    prompt_ref: scene_presets.post_workout_stretch_home
    hard_angle: false
    hyperparams:
      lora_scale: 0.90
      steps: 34
      cfg: 4.2
    caption_seed: "운동보다 어려운 건 스트레칭을 건너뛰지 않는 일."
    local_only: false

home_mature_scenes:
  # Theme ids / prompt bodies: private/nsfw_test/themes/ + private/content/
  - scene_id: y01_w01_mon_home_01
    theme: <see private home_mature_track>
    local_only: true
    never_enter_platform_publishers: true
    output_root: private/nsfw_test/private_media/
    story_note: "local-only home track — never publish"
    prompt_ref: home_mature_track.<theme>
    hyperparams:
      lora_scale: 0.90
      steps: 40
      cfg: 4.5
    adult_only: true
```

---

## `ig_scenes[]` 항목

| 필드 | 필수 | 노트 |
|------|------|------|
| `scene_id` | ✅ | 유일 |
| `pillar` | ✅ | `theme_pillars.yml` 키 |
| `scene_type` | ✅ | `scene_presets.yml` 키 |
| `format` | ✅ | still / carousel / reel |
| `publish_intent` | ✅ | 하루 중 `primary` ≥1 |
| `story_note` | ✅ | 내부용 |
| `prompt_ref` | ✅ | 프리셋 포인터 |
| `hyperparams` | ✅ | day 기본 override 가능 |
| `hard_angle` | ✅ | bool |
| `caption_seed` | ❌ | disclosure는 별도 게이트 |
| `location_label` | ❌ | broad만 |
| `local_only` | ✅ | IG는 항상 `false` |

## `home_mature_scenes[]` 항목

| 필드 | 필수 | 노트 |
|------|------|------|
| `scene_id` | ✅ | |
| `theme` | ✅ | `home_mature_track.md` 사다리 |
| `local_only` | ✅ | **항상 true** |
| `never_enter_platform_publishers` | ✅ | **항상 true** |
| `output_root` | ✅ | `private/nsfw_test/...` |
| `adult_only` | ✅ | true |
| `explicit_genital_ok` | ❌ | private 테마 사다리가 허용할 때만 |
| `hyperparams` | ✅ | |
| `prompt_ref` | ✅ | |

---

## 검증 체크리스트 (수동/스크립트용)

- [ ] `ig_scenes` 길이 ≥ 1, `publish_intent: primary` ≥ 1  
- [ ] 어떤 `home_mature_*`도 `queues/`, `publisher/`, Patreon CSV에 없음  
- [ ] `hard_angle: true`이면 주간 rare 예산 확인  
- [ ] `disclosure_gate != approved`면 publish status 금지  
- [ ] moodboard 실존 인물 얼굴이 identity로 쓰이지 않음  
