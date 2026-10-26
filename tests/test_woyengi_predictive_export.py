"""Constructed Woyengi v1 model-read export consumer; no real auth/data/scores."""
from __future__ import annotations

from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "validate_woyengi_predictive_export.py"
sys.path.insert(0, str(SCRIPT.parent))
from validate_woyengi_predictive_export import adapt_predictive_read_export, BoundaryError


def example():
    # Exact Woyengi packages/predictive-export/src/index.ts output field contract.
    return {
        "kind": "PREDICTIVE_READ_EXPORT",
        "schemaVersion": 1, "status": "ADVISORY_ONLY",
        "subject": "entity:project-alpha", "predicate": "project:lead",
        "validAt": "2026-02-05T00:00:00Z",
        "recordedAt": "2026-02-15T00:00:00Z",
        "predictionCutoff": "2026-02-15T00:00:00Z",
        "selectedClaimId": "claim:priya",
        "conflictingClaimIds": ["claim:daniel"],
        "observations": [
            {
                "claimId": "claim:daniel",
                "validFrom": "2026-02-01T00:00:00Z",
                "knownFrom": "2026-02-01T00:00:00Z",
                "sourceAvailableAt": "2026-02-01T00:00:00Z",
                "featureHash": "a" * 64,
                "features": {"leader": "entity:daniel"},
            },
            {
                "claimId": "claim:priya",
                "validFrom": "2026-02-01T00:00:00Z",
                "knownFrom": "2026-02-10T00:00:00Z",
                "sourceAvailableAt": "2026-02-10T00:00:00Z",
                "featureHash": "b" * 64,
                "features": {"leader": "entity:priya"},
            },
        ],
        "qualification": "STRUCTURAL_EXPORT_ONLY",
        "publisherAuthenticity": "NOT_ATTESTED_BY_THIS_EXPORTER",
        "independentRewardVerification": "NOT_PERFORMED",
        "actionAuthority": "NOT_GRANTED",
    }


class WoyengiExportConsumerTests(unittest.TestCase):
    def setUp(self):
        self.export = example()

    def deny(self):
        with self.assertRaises(BoundaryError):
            adapt_predictive_read_export(self.export)

    def test_candidate_shape_and_preserved_conflicts(self):
        r = adapt_predictive_read_export(self.export)
        self.assertEqual(r["status"], "STRUCTURAL_ONLY")
        self.assertEqual(r["observation_count"], 2)
        self.assertEqual(r["selected_claim_id"], "claim:priya")
        self.assertEqual(r["conflicting_claim_ids"], ["claim:daniel"])
        self.assertEqual(r["source_rights"], "UNVERIFIED_BY_THIS_CONSUMER")
        self.assertEqual(r["woyengi_action_authority"], "NOT_GRANTED")
        self.assertEqual(r["derived_feature_hashes"], "FORMAT_VALID_ONLY_NOT_RECOMPUTED")
        self.assertNotIn("future_labels", r["observations"][0]["features"])

    def test_selected_missing_rejected(self):
        self.export["selectedClaimId"] = "claim:never-existed"
        self.deny()

    def test_unregistered_conflict_rejected(self):
        self.export["conflictingClaimIds"] = ["claim:unknown"]
        self.deny()

    def test_duplicate_claim_id_rejected(self):
        self.export["observations"][1]["claimId"] = "claim:daniel"
        self.deny()

    def test_claim_known_after_historical_projection_rejected(self):
        self.export["observations"][1]["knownFrom"] = "2026-02-16T00:00:00Z"
        self.deny()

    def test_source_available_after_prediction_rejected(self):
        self.export["observations"][1]["sourceAvailableAt"] = "2026-02-16T00:00:00Z"
        self.deny()

    def test_unverified_state_cannot_claim_authority(self):
        self.export["actionAuthority"] = "AUTHORIZED"
        self.deny()

    def test_untrusted_reward_claim_rejected(self):
        self.export["independentRewardVerification"] = "VERIFIED"
        self.deny()

    def test_source_rights_self_attestation_is_not_consumer_verification(self):
        r = adapt_predictive_read_export(self.export)
        self.assertNotEqual(r["source_rights"], "VERIFIED")

    def test_hidden_future_label_rejected(self):
        self.export["observations"][0]["features"] = {"future_label": True}
        self.deny()

    def test_nonfinite_feature_rejected(self):
        self.export["observations"][0]["features"] = {"temperature": float("inf")}
        self.deny()

    def test_missing_observation_hash_rejected(self):
        self.export["observations"][0]["featureHash"] = "not-a-real-hash"
        self.deny()

    def test_opaque_extra_credential_rejected(self):
        self.export["verifierPrivateKey"] = "spoofed"
        self.deny()

    def test_bitemporal_offsets_normalized_for_cutoff_comparison(self):
        self.export["recordedAt"] = "2026-02-14T19:00:00-05:00"
        r = adapt_predictive_read_export(self.export)
        self.assertEqual(r["observation_count"], 2)
        self.export["observations"][1]["knownFrom"] = "2026-02-15T01:00:00+00:00"
        self.deny()

    def test_duplicate_conflict_denied(self):
        self.export["conflictingClaimIds"].append("claim:daniel")
        self.deny()

    def test_cli_exit_statuses(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "read.json"
            path.write_text(json.dumps(self.export), encoding="utf-8")
            good = subprocess.run([sys.executable, str(SCRIPT), str(path)],
                                  capture_output=True, text=True, timeout=15)
            self.assertEqual(good.returncode, 0, good.stderr)
            self.assertEqual(json.loads(good.stdout)["status"], "STRUCTURAL_ONLY")
            self.export["status"] = "AUTHORIZED"
            path.write_text(json.dumps(self.export), encoding="utf-8")
            bad = subprocess.run([sys.executable, str(SCRIPT), str(path)],
                                 capture_output=True, text=True, timeout=15)
            self.assertEqual(bad.returncode, 2, bad.stderr)
            self.assertEqual(json.loads(bad.stdout)["status"], "REJECTED")


if __name__ == "__main__":
    unittest.main()
