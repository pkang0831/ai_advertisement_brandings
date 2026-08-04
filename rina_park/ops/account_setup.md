# Rina account setup

## Shared safety gate

- Keep every platform output Safe for All Audiences.
- Use the bio: `Virtual creator • AI-generated • Toronto swim diaries`.
- Use `Rina is a fictional virtual character. Visuals are AI-generated.` on factual-looking posts.
- Never present Rina as a real person, use real-time/precise location, or accept DMs implying
  medical, sexual, or personal services.
- Rina's public fictional-character date of birth is `1999-03-18` (age 27 as of
  `2026-07-26`). Never enter this fictional metadata in Patreon or Meta legal identity, tax,
  payout, or age-verification fields; always use the real account owner's information.
- Enable 2FA with recovery codes stored outside this repository.

## Instagram

1. Create a public Professional **Creator** account and complete 2FA.
2. Use the official Instagram app or Meta Business Suite for the initial rolling schedule.
3. Create a Meta developer app only when the account is stable. Add the account as an app
   role/tester and complete the tester invitation before testing.
4. Select **Instagram API with Instagram Login**. This project uses
   `https://graph.instagram.com`, an Instagram User token, and only
   `instagram_business_basic` plus `instagram_business_content_publish`. It does not require a
   Facebook Page. If a Page is already linked, verify its Page Publishing Authorization state.
5. Put the token in macOS Keychain using the Keychain Access GUI as a generic-password item.
   Never paste it into Terminal, `.env`, a plist, source, a database, screenshots, or logs.
   Grant access only to the publishing process. Store no app secret in this project.
6. Record token expiry as non-secret operational metadata. At 30, 14, and 7 days remaining,
   pause and complete refresh/re-login with human review. An expired token disables API posting.
7. API posting stays disabled until a public HTTPS media transport passes the staging spike,
   content-publishing-limit probing, mocked reconciliation tests, and one dedicated test-account
   rehearsal. Images cannot be sent from localhost or a private/file URL.
8. The API sequence is container create, status polling until `FINISHED`, then publish. Containers
   expire after 24 hours. A timeout or ambiguous result requires reconciliation; never assume an
   idempotency key gives exactly-once delivery.
9. `is_ai_generated=true` is required. For a carousel it belongs on the parent only.
   `location_label` is for official UI entry only and must never enter an API payload.
10. If any capability, auth, quota, transport, policy, or reconciliation gate fails, use the
    generated UI package. Add Meta library music manually in the official UI.

## Patreon

1. Create a page fixed to **Safe for All Audiences** and enable 2FA.
2. Proposed tiers: A `Poolside Notes` at $3/month, B `Extended Cut` at $8/month, and C
   `Season Archive` at $15/month.
3. Configure cumulative access exactly as exported: A posts `[A,B,C]`, B posts `[B,C]`, and C
   posts `[C]`. Benefits differ by quantity, alternate editorial takes, production notes, and
   archive access—not exposure level.
4. Rehearse one SFW draft with the package's first image as its preview. Confirm title, body,
   image order, AI disclosure, audience access, and scheduled time in `America/Toronto`.
5. Upload and schedule only through Patreon's official website. Patreon creation/publishing is
   manual: do not use browser automation, unofficial write APIs, saved login cookies, or cookie
   exports.
6. Every Patreon post requires final human schedule approval. Any changed copy or asset checksum
   invalidates the package and must return to review.
