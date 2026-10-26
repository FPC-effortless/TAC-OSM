"""Constructed-only tests for fail-closed ecosystem interface; no source, fit or secrets."""
from __future__ import annotations

from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "ecosystem_boundary", ROOT / "scripts" / "validate_ecosystem_boundary.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
FIXTURE = ROOT / "contracts" / "TACOSM-ECOSYSTEM-BOUNDARY-v1.json"


class EcosystemBoundaryTests(unittest.TestCase):
    def setUp(self):
        self.b = json.loads(FIXTURE.read_text(encoding="utf-8"))

    def fail(self, changed):
        with self.assertRaises(MODULE.BoundaryError):
            MODULE.check_bundle(changed)

    def test_constructed_chain_is_structural_only(self):
        result = MODULE.check_bundle(self.b)
        self.assertEqual(result["structure"], "PASS")
        self.assertEqual(result["veritas_independent_outcome_attestation"], "NOT_PRESENT")
        self.assertEqual(result["model_scientific_qualification"], "NOT_EVALUATED")
        self.assertEqual(result["woyengi_governing_authority"], "NOT_GRANTED")

    def test_late_recorded_source_rejected_even_when_valid_past(self):
        self.b["observations"][0]["recorded_at"] = "2026-10-26T09:00:00+01:00"
        self.fail(self.b)

    def test_available_future_input_rejected(self):
        self.b["observations"][0]["available_at"] = "2026-10-26T09:10:00+01:00"
        self.fail(self.b)

    def test_offsets_compare_by_utc_not_lexicographic(self):
        self.b["observations"][0]["recorded_at"] = "2026-10-26T06:02:00Z"
        self.b["observations"][0]["available_at"] = "2026-10-26T06:03:00Z"
        self.assertEqual(MODULE.check_bundle(self.b)["structure"], "PASS")
        self.b["observations"][0]["available_at"] = "2026-10-26T08:00:00Z"
        self.fail(self.b)

    def test_future_target_hidden_in_observation_rejected(self):
        self.b["observations"][0]["features"]["sensor"] = {"future_target": 42}
        self.fail(self.b)

    def test_oracle_future_label_even_nested_rejected(self):
        self.b["observations"][0]["features"]["samples"] = [{"oracle_reward": 0.9}]
        self.fail(self.b)

    def test_unpreserved_source_conflicts_rejected(self):
        self.b["observations"][0]["conflicts_count"] = 2
        self.b["observations"][0]["conflicts_preserved"] = False
        self.fail(self.b)

    def test_model_cannot_claim_authorized_execution(self):
        self.b["predictions"][0]["authority"] = "AUTHORIZED"
        self.fail(self.b)

    def test_model_cannot_self_verify(self):
        self.b["verification_status"] = "VERIFIED"
        self.fail(self.b)

    def test_unchanged_content_hash_not_independent_verification(self):
        self.b["source_artifact_sha256"] = "a" * 64
        out = MODULE.check_bundle(self.b)
        self.assertEqual(out["source_cryptographic_authenticity"], "UNVERIFIED")

    def test_unknown_source_checksum_rejected(self):
        self.b["observations"][0]["record_sha256"] = "not sha"
        self.fail(self.b)

    def test_protected_test_not_read_by_tool(self):
        self.b["split"] = "TEST"
        self.fail(self.b)

    def test_duplicate_source_ids_rejected(self):
        second = deepcopy(self.b["observations"][0])
        self.b["observations"].append(second)
        self.fail(self.b)

    def test_source_ref_mismatch_rejected(self):
        self.b["predictions"][0]["observation_ids"] = ["observation:other"]
        self.fail(self.b)

    def test_later_target_available_before_prediction_rejected(self):
        self.b["outcomes"][0]["available_at"] = "2026-10-26T07:30:00+01:00"
        self.fail(self.b)

    def test_outcome_known_later_but_valid_before_prediction_rejected(self):
        self.b["outcomes"][0]["observed_at"] = "2026-10-26T07:20:00+01:00"
        self.fail(self.b)

    def test_forged_signed_verifier_field_rejected(self):
        self.b["verification_signature"] = "self-signed"
        self.fail(self.b)

    def test_nonfinite_probability_rejected(self):
        self.b["predictions"][0]["output"]["probability"] = float("nan")
        self.fail(self.b)

    def test_unqualified_choice_distribution_rejected(self):
        self.b["predictions"][0]["output"] = {
            "kind": "choice", "probabilities": {"inspect": .9, "defer": .9}
        }
        self.fail(self.b)

    def test_normalized_choice_distribution_structurally_valid(self):
        self.b["predictions"][0]["output"] = {
            "kind": "choice", "probabilities": {"inspect": .6, "defer": .4}
        }
        self.assertEqual(MODULE.check_bundle(self.b)["structure"], "PASS")

    def test_structural_defer_without_authority(self):
        self.b["predictions"][0]["output"] = {"kind": "defer", "reason": "insufficient source"}
        self.assertEqual(MODULE.check_bundle(self.b)["woyengi_governing_authority"], "NOT_GRANTED")

    def test_cli_fixture_and_failure_exit_codes(self):
        p = subprocess.run([sys.executable, str(ROOT / "scripts" / "validate_ecosystem_boundary.py"),
                            str(FIXTURE)], text=True, capture_output=True, check=False)
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertEqual(json.loads(p.stdout)["structure"], "PASS")
        with tempfile.TemporaryDirectory() as temp:
            tampered = Path(temp) / "bundle.json"
            self.b["predictions"][0]["authority"] = "AUTHORIZED"
            tampered.write_text(json.dumps(self.b), encoding="utf-8")
            p = subprocess.run([sys.executable, str(ROOT / "scripts" / "validate_ecosystem_boundary.py"),
                                str(tampered)], text=True, capture_output=True, check=False)
            self.assertEqual(p.returncode, 2, p.stderr)
            self.assertEqual(json.loads(p.stdout)["structure"], "FAIL")

    def test_cli_invalid_nan_constant_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            bad = Path(temp) / "bundle.json"
            bad.write_text('{"x":NaN}', encoding="utf-8")
            p = subprocess.run([sys.executable, str(ROOT / "scripts" / "validate_ecosystem_boundary.py"),
                                str(bad)], text=True, capture_output=True, check=False)
            self.assertEqual(p.returncode, 2, p.stderr)


if __name__ == "__main__":
    unittest.main()
