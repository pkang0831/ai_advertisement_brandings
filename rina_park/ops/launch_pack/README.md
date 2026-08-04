# Rina Park Launch Copy Pack

English launch copy for Rina Park, a fictional AI-generated virtual creator.

## Decision status

Public AI/virtual-character disclosure wording is intentionally unresolved. All variants are drafts; none is selected, default, approved, or publishable. The only later decision is for the user to choose one candidate ID from `disclosure_a`, `disclosure_b`, or `disclosure_c` and explicitly approve it.

This state references the non-secret `Shared safety gate` in `../account_setup.md`. It does not read or store account credentials, tokens, recovery codes, or other mutable secrets.

## Contents

- `instagram.md` — profile, link-in-bio, comment voice, boundaries, and three pinned posts
- `patreon.md` — public page copy, SFW disclosure, tiers, welcomes, preview, CTA, and expectations
- `two_week_copy_proof.md` — draft proof copy for the first 14 existing calendar post IDs
- `disclosure_candidates.md` — three compliant, unselected disclosure bundles
- `disclosure_decision.json` — user-owned selection and release-gate state
- `launch_policy.json` — machine-readable disclosures, tiers, post IDs, and location expectations
- `validate_launch_copy.py` — read-only draft and release-gate validator; it cannot select or apply copy
- `tests/test_validate_launch_copy.py` — validator unit tests

## Source boundary

This pack was prepared from `../../ops/strategy.md`, `../../identity/bible.md`, and `../../content/calendar_8_weeks.csv` without changing those files. It creates no media and performs no posting or account actions.

## Validation

From `rina_park/ops/launch_pack`:

```bash
python3 validate_launch_copy.py
python3 -m unittest discover -s tests -v
```

Draft-package validation must pass while the choice is unresolved. The release gate must fail:

```bash
python3 validate_launch_copy.py --release-gate
```

The release gate can pass only after the user deliberately records one valid candidate ID, sets `approved_by_user` to `true`, changes `status` to `user_approved`, and changes `release_gate` to `ready`. The validator never makes those edits.

The validator checks exact post IDs for the first two calendar weeks, explicit fictional/AI draft disclosures, broad approved location labels, tier names and draft prices, non-empty alt text, prohibited claims, and disclosure decision integrity.
