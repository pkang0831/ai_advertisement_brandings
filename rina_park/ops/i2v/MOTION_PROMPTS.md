# Wan I2V motion prompts — Rina Park

Skill convention: start from portrait still → prompt **subtle** motion → ~81 frames (~5s) → interpolate → stitch.  
Do **not** ask for big limb travel, camera whip-pans, or outfit changes.

Push sensuality via **breath, gaze, lips, hair, fabric tension, micro weight-shift** — not choreography.

**Pack rules**
- **IG default set** → Instagram / Patreon SFW packs only. Max sexy while platform-safe.
- **PRIVATE / LOCAL-ONLY** section → never copy into IG calendars, exports, or Patreon uploads.

**Hero still pairing (I2V)** — only after [`STILL_GATE.md`](STILL_GATE.md) pass + promote to `out/i2v_heroes/current/`.  
See [`HERO_MOTION_MAP.yml`](HERO_MOTION_MAP.yml) for live mapping. Defaults:

| Still pose | Motion | Notes |
|------------|--------|--------|
| `hand_resting_lap_seated` | **A2** | sofa soft portrait |
| `hand_pockets_3q` | **A1** | pockets; no hand LoRA on still |
| `fitness_mat_soft_seated` | **A3** | yoga-mat fitness loop |

Generate stills: `scripts/gen_i2v_heroes.py` → promote: `scripts/promote_i2v_heroes.py` → I2V: `scripts/run_i2v_from_hero.sh` (via `screen`).

---

## Shared negative (append to every run)

```
warp, morphing, melting face, identity drift, face swap, age change, different person,
extra fingers, missing fingers, fused fingers, morphing hands, deformed hands,
warped limbs, rubber body, liquid skin, body dissolve, clothing dissolve,
jitter, flicker, strobing, abrupt cut, teleport pose, camera shake heavy,
text, watermark, subtitle, UI overlay, logo, caption burn-in,
nude, nipples, genitals, explicit sexual act
```

한국어 메모: 손·얼굴 고정이 최우선. “sexy”는 **호흡·시선·입술·헤어·패브릭 텐션·미세 체중이동**으로만. 큰 동작 = 붕괴.

---

## A. IG-safe default set (public Reels) — max sensual within IG

### A1 — Soft breath + heated gaze (pair with `01_fit_smile_camera` / `04_clinic_soft_writing`)

```
slow sensual breathing, soft chest rise and fall under clothing,
heavy-lidded warm eye contact toward camera, lips slightly parted then soft close,
gentle head tilt, hair strands caressing cheek and collarbone,
skin glow soft glam, intimate but wholesome reel energy,
micro shoulder settle, loopable micro-motion,
realistic skin, stable face identity, calm seductive presence, keep clothing on
```

KR: 숨 + 반쯤 감은 눈 + 살짝 벌린 입. IG 기본 루프의 상한선.

### A2 — Smile bloom into soft heat (pair with `01` / `05_clinic_warm_laugh`)

```
soft smile deepening slowly into knowing warmth, eyes gently narrowing,
tiny shoulder roll, natural slow blink, hair micro-sway over neck,
breathing visible at collarbone, soft glam intimate light,
wholesome-sexy influencer vibe, no mouth morphing, stable identity, loopable
```

KR: 미소가 천천히 “아는 듯한” 온기로. 입 morph 주의.

### A3 — Cat-stretch micro-arch (pair with `02_fit_cat_stretch`)

```
slow subtle spinal micro-arch, sensual breath expanding ribs,
hips settle softly, athletic shorts soft fabric tension,
hair strands move lightly, eyes glance to camera with quiet heat,
keep pose almost still, fitness lifestyle reel, glamorous body line,
stable hands on mat almost still, loopable
```

KR: 피트니스 지식톤 + 실루엣 센슈얼. 큰 동작 금지.

### A4 — Side silhouette breath (pair with `03_fit_side_tabletop`)

```
side profile tabletop hold, deep soft breathing, subtle hip settle,
gentle back curve micro-motion, athletic wear hugging the body softly,
bright clean home studio light, sensual fitness atmosphere,
fabric tension on shorts and sports bra, no face morph needed,
stable body proportions, loopable
```

KR: 얼굴 최소 → 바디 실루엣·호흡·패브릭. identity drift 낮음.

### A5 — Clipboard / beauty-clinic soft look (pair with `04`)

```
soft glam close-up, slow downward glance then lift eyes with quiet allure,
gentle breath, lips soft and slightly parted, pen hand almost still,
hair fall soft across cheek, professional beauty-clinic mood,
warm intimate lighting, subtle sensual micro-motion only,
stable face and hands, keep uniform on, loopable
```

KR: beauty-as-medium 훅. 시선 업 + 숨. 손은 “거의 정지”.

### A6 — Warm laugh settle into intimacy (pair with `05`)

```
warm laugh settling into soft intimate smile, natural blink,
slight head bob then calm eye contact, hair shimmer on shoulders,
breathing soft at neckline, soft glam influencer close-up,
friendly sensual warmth, keep teeth and mouth structure stable, loopable
```

KR: 웃음→안착→눈맞춤. 치아/입 morph 위험 → strength 보수적.

---

## B. PRIVATE / LOCAL-ONLY — spicier (NOT for IG packs)

Prompt bodies live in gitignored  
`private/ops/i2v/MOTION_PROMPTS_PRIVATE.md` (soft-available when present).

Wan tip: if hands visible, keep strength low and emphasize “hands almost still / no finger motion”.
