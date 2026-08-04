# 계정 출시 준비 체크리스트

이 패키지는 로컬 파일만 검사합니다. 웹사이트 접속, 로그인, 게시, API 호출을 하지 않습니다.
`account_setup.md`의 기존 운영 정책은 그대로 유지합니다.

## 1. 사용자 작업 파일

확정된 선택을 반영한 `account_readiness.local.yaml`을 사용합니다. 이 파일은
`.gitignore` 대상이며 비밀·사적 신원·결제 정보를 넣지 않습니다. JSON 문법은 YAML
1.2의 유효한 부분집합이므로 따옴표, 쉼표, 중괄호를 유지해 작성합니다.

## 2. 수동 출시 필수 항목

다음 값만 사용자가 확인해 입력합니다.

- 공통
  - `timezone`: 반드시 `America/Toronto`
  - 출시 정책은 `next_monday_after_all_gates_pass`
  - 모든 품질·준비 게이트가 통과하면
    `quality_and_readiness_gates_passed: true`와 Toronto 현지 기준 통과일
    `gates_passed_on: YYYY-MM-DD`를 기록
  - 릴리스 후보 확정 전에는 `release_candidate_finalized: false`,
    `resolved_launch_date: null`
  - 릴리스 후보를 확정할 때만 `release_candidate_finalized: true`와
    `gates_passed_on` 다음 월요일을 `resolved_launch_date`에 기록
- 공개 AI 고지 결정
  - 현재 `status: undecided`, Instagram/Patreon의 `present`와 `wording`은 `null`
  - 사용자가 결정하기 전까지 공개 출시는 차단됨
  - 결정 후 `status: resolved`; 각 플랫폼 `present`를 `true/false`로 선택
  - `present: true`이면 사용자가 확정한 공개 문구를 `wording`에 입력
  - `present: false`이면 `wording: null`; 검증기는 문구나 존재 여부를 대신 결정하지 않음
- Instagram
  - `public_handle`: 공개 사용자명
  - `public_profile_url`: 공개 `https://www.instagram.com/.../` URL
  - `account_type`: `creator`
  - `professional_status_confirmed`: Professional/Creator이면 `true`
  - `is_public`: 공개 계정이면 `true`
  - `two_factor_enabled`: 2FA 완료 시 `true`
- Patreon
  - `public_handle`, `public_page_url`: 공개 핸들 및 페이지 URL
  - `safe_for_all_audiences`: 해당 카테고리 확인 시 `true`
  - `two_factor_enabled`: 2FA 완료 시 `true`
  - `legal_identity_completed_with_real_owner_info`: 실제 소유자 정보로 Patreon 내부
    확인을 끝냈는지만 `true/false`
  - `payout_setup_complete`, `tax_setup_complete`: 완료 여부만 `true/false`
  - `currency`: 기존 패키지 통화인 `CAD`
  - 각 `tiers[]`: 공개 tier `id`, `name`, `monthly_price_cad`,
    `audience_includes_tier_ids`
- 링크
  - `instagram_to_patreon_url`: Instagram에서 연결할 공개 Patreon URL
  - `patreon_to_instagram_url`: Patreon에서 연결할 공개 Instagram URL
  - `canonical_landing_url`: 사용 시 공개 HTTPS 랜딩 URL, 없으면 `null`

Tier는 정확히 A/B/C, CAD `$3/$8/$15`를 유지합니다. 상속은 저가→고가 순서로 A
콘텐츠가 `[A,B,C]`, B가 `[B,C]`, C가 `[C]`입니다. 배열에는 실제 공개 tier ID를
넣습니다.

## 3. 선택 사항: Instagram Graph API

수동 출시에 필요하지 않습니다. 사용하지 않으면 `intended: false`를 유지합니다.
사용할 때만 다음 비밀 아닌 상태를 기록합니다.

- `meta_app_created`
- `instagram_login_product_configured`
- `account_added_as_app_tester`
- `tester_invitation_accepted`
- `permissions`: 정확히 `instagram_business_basic`,
  `instagram_business_content_publish`
- `token_available_in_external_secret_store`: 토큰이 저장소 밖 보안 저장소에 있으면
  `true`
- `token_expires_on`: 비밀이 아닌 만료일만 `YYYY-MM-DD` 또는 `null`
- `public_https_media_transport_ready`
- `publishing_limit_probe_passed`
- `mocked_reconciliation_tests_passed`
- `dedicated_test_account_rehearsal_passed`

Graph API 게이트가 실패해도 공식 Instagram 앱/Meta Business Suite를 이용한 수동
출시는 차단하지 않습니다.

## 4. 절대 입력하지 않을 정보

법적 이름, 생년월일, 신분증/여권/면허 번호, SIN/SSN/세금 식별번호, 은행·카드·결제
정보, 비밀번호, 앱 시크릿, 토큰 값, 쿠키, 세션 값을 넣지 않습니다. 법적 신원·지급·
세금 정보는 해당 플랫폼 안에서만 처리하고 이 파일에는 완료 여부 불리언만 둡니다.

## 5. 로컬 검증과 테스트

```bash
.venv/bin/python -m rina_park.ops.account_readiness.validator \
  rina_park/ops/account_readiness/account_readiness.local.yaml

.venv/bin/python -m pytest \
  rina_park/ops/account_readiness/test_validator.py -q
```

결과의 `manual_launch.ready`가 `true`인지 확인합니다.
`graph_api_optional.ready`는 `intended: true`일 때만 의미가 있으며 수동 출시를 막지
않습니다. 검증기는 금지 필드와 토큰·개인키·카드/SIN 형태 값을 거부하고 출력 시
민감해 보이는 값을 `[REDACTED]`로 치환합니다.

`resolved_launch_date`는 사용자가 임의로 미리 정하지 않습니다. 검증기는 게이트
통과일이 월요일인 경우에도 그 날이 아닌 7일 뒤 월요일을 요구합니다.
