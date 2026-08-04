import shutil
import sys
import tempfile
import unittest
from pathlib import Path


PACKAGE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_DIR))

import validate_launch_copy as validator  # noqa: E402


class LaunchCopyValidatorTests(unittest.TestCase):
    def test_complete_package_passes(self):
        self.assertEqual(validator.validate_package(), [])

    def test_release_gate_fails_while_user_decision_is_unresolved(self):
        errors = validator.validate_package(release_gate=True)
        self.assertIn(
            "USER DECISION REQUIRED: choose one disclosure candidate ID "
            "and explicitly approve it.",
            errors,
        )

    def test_missing_mandatory_disclosure_fails(self):
        disclosure = validator.load_policy()["draft_placeholder_disclosure"]
        errors = validator.validate_disclosure_unit(
            "sample post", "A fictional chapter with AI-generated visuals.", disclosure
        )
        self.assertEqual(
            errors, ["Missing mandatory disclosure in: sample post"]
        )

    def test_package_detects_removed_pinned_post_disclosure(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary_path = Path(temporary_directory)
            for filename in (
                "instagram.md",
                "patreon.md",
                "two_week_copy_proof.md",
                "disclosure_candidates.md",
            ):
                shutil.copy2(PACKAGE_DIR / filename, temporary_path / filename)

            instagram_path = temporary_path / "instagram.md"
            instagram = instagram_path.read_text(encoding="utf-8")
            disclosure = validator.load_policy()["draft_placeholder_disclosure"]
            instagram_path.write_text(
                instagram.replace(disclosure, "", 1), encoding="utf-8"
            )

            errors = validator.validate_package(
                base_dir=temporary_path,
                policy_path=validator.DEFAULT_POLICY,
                calendar_path=None,
            )
            self.assertIn(
                "Missing mandatory disclosure in: Pinned post 1 — Meet Rina",
                errors,
            )

    def test_each_prohibited_claim_type_is_rejected(self):
        examples = {
            "real-time location claim": "I'm at Harbourfront right now.",
            "real bodily achievement claim": "I swam 1,000 metres.",
            "real weight or body result claim": "I lost 10 pounds.",
            "human identity claim": "I am a real person.",
            "firsthand product claim": "I personally recommend this swimsuit.",
            "meeting or live-location invitation": "Meet me at the pool.",
        }
        for expected_label, copy in examples.items():
            with self.subTest(expected_label=expected_label):
                errors = validator.validate_prohibited_claims(
                    (("example", copy),)
                )
                self.assertEqual(len(errors), 1)
                self.assertIn(expected_label, errors[0])

    def test_clear_negative_disclosure_is_allowed(self):
        copy = (
            "Rina Park is not a real person. "
            "Rina is a fictional virtual character. Visuals are AI-generated."
        )
        self.assertEqual(
            validator.validate_prohibited_claims((("example", copy),)), []
        )

    def test_decision_manifest_has_no_selection_or_auto_apply(self):
        decision = validator.load_decision()
        self.assertIsNone(decision["selected"])
        self.assertFalse(decision["approved_by_user"])
        self.assertFalse(decision["auto_apply"])
        self.assertEqual(decision["release_gate"], "blocked")

    def test_all_disclosure_candidates_remain_explicit(self):
        errors = validator.validate_disclosure_decision(
            validator.load_policy(),
            validator.load_decision(),
            (PACKAGE_DIR / "disclosure_candidates.md").read_text(
                encoding="utf-8"
            ),
        )
        self.assertEqual(errors, [])

    def test_proof_policy_contains_exact_first_two_weeks(self):
        policy = validator.load_policy()
        self.assertEqual(len(policy["proof_posts"]), 14)
        self.assertEqual(
            list(policy["proof_posts"])[0], "ig_20260810_carousel"
        )
        self.assertEqual(
            list(policy["proof_posts"])[-1], "ig_20260823_reel"
        )


if __name__ == "__main__":
    unittest.main()
