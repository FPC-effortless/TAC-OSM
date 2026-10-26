"""Constructed Woyengi native ClaimLedger projection shape: no model or real source."""
from __future__ import annotations

from copy import deepcopy
import sys
from pathlib import Path
import unittest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
from woyengi_projection_adapter import adapt_native_projection, BoundaryError


def claim(cid: str, value: str, recorded: str, valid: str, level: int) -> dict:
    # Follows the real Woyengi packages/core/src/index.ts ClaimRecord shape.
    return {
        "kind": "claim", "id": cid, "subject": "entity:project-alpha",
        "predicate": "project:lead", "object": {"entity": value},
        "validTime": {"from": valid}, "transactionTime": {"from": recorded},
        "observationIds": ["observation:source:" + cid],
        "evidenceIds": [], "provenance": {"derivedFrom": [], "transformations": []},
        "authority": {"level": level, "basis": "constructed unit input"},
        "confidence": .85, "lifecycle": "verified"
    }


def projection() -> dict:
    # Constructed analogy to the real Woyengi projection.test.ts scenario.
    daniel = {"claim": claim(
        "claim:daniel-leads-alpha", "entity:daniel",
        "2026-02-01T00:00:00Z", "2026-01-10T00:00:00Z", 30),
        "effectiveLifecycle": "verified"}
    priya = {"claim": claim(
        "claim:priya-leads-alpha", "entity:priya",
        "2026-02-10T00:00:00Z", "2026-02-01T00:00:00Z", 80),
        "effectiveLifecycle": "verified"}
    return {
        "subject": "entity:project-alpha", "predicate": "project:lead",
        "validAt": "2026-02-05T00:00:00Z",
        "recordedAt": "2026-02-15T00:00:00Z",
        "selected": deepcopy(priya), "candidates": [deepcopy(priya), deepcopy(daniel)],
        "conflicts": [deepcopy(daniel)], "trace": [
            {"stage": "candidate-discovery", "inputCount": 2, "outputCount": 2, "detail": "fixture"},
            {"stage": "transaction-time", "inputCount": 2, "outputCount": 2, "detail": "fixture"},
            {"stage": "valid-time", "inputCount": 2, "outputCount": 2, "detail": "fixture"},
            {"stage": "lifecycle", "inputCount": 2, "outputCount": 2, "detail": "fixture"},
            {"stage": "selection", "inputCount": 2, "outputCount": 1, "detail": "fixture"},
        ]
    }


class WoyengiShapeAdapterTests(unittest.TestCase):
    def setUp(self):
        self.p = projection()
        self.accessed = []
        def authorize(principal, purpose, subject, predicate, ids, cutoff):
            self.accessed.append((principal, purpose, subject, predicate, ids, cutoff))
            return True
        self.authorize = authorize
        self.features = lambda c: {"entity": c["object"]["entity"]}

    def adapt(self):
        return adapt_native_projection(
            self.p, model_cutoff="2026-02-15T00:00:00Z",
            principal="user:authorized-researcher", purpose="predict:read-only",
            source_artifact_sha256="a" * 64,
            read_authorized=self.authorize,
            feature_allowlist=self.features
        )

    def test_real_woyengi_shape_preserves_all_candidates_and_conflicts(self):
        result = self.adapt()
        self.assertEqual(result["status"], "STRUCTURAL_ONLY")
        self.assertEqual(result["selected_claim_id"], "claim:priya-leads-alpha")
        self.assertEqual(result["conflicting_claim_ids"], ["claim:daniel-leads-alpha"])
        self.assertEqual(len(result["observations"]), 2)
        self.assertEqual(result["governing_authority"], "NOT_GRANTED")
        self.assertEqual(result["trusted_source_authentication"], "NOT_VERIFIED_BY_ADAPTER")
        self.assertEqual(len(self.accessed), 1)
        self.assertEqual(self.accessed[0][0:2], ("user:authorized-researcher", "predict:read-only"))
        self.assertEqual(self.accessed[0][4], ["claim:daniel-leads-alpha", "claim:priya-leads-alpha"])
        self.assertEqual([o["available_at"] for o in result["observations"]],
            ["2026-02-01T00:00:00Z", "2026-02-10T00:00:00Z"])
        self.assertTrue(all(o["conflicts_preserved"] for o in result["observations"]))

    def test_read_permission_denial_fails_closed(self):
        self.authorize = lambda *_: False
        with self.assertRaisesRegex(BoundaryError, "not authorized"):
            self.adapt()

    def test_no_default_or_untrusted_feature_selector(self):
        with self.assertRaises(BoundaryError):
            adapt_native_projection(self.p, model_cutoff="2026-02-15T00:00:00Z",
                principal="user:authorized-researcher", purpose="predict:read-only",
                source_artifact_sha256="a"*64,
                read_authorized=None, feature_allowlist=self.features)

    def test_late_recorded_competing_claim_denied(self):
        self.p["candidates"][0]["claim"]["transactionTime"]["from"] = "2026-03-01T00:00:00Z"
        with self.assertRaises(BoundaryError):
            self.adapt()

    def test_future_valid_at_denied(self):
        self.p["validAt"] = "2026-03-01T00:00:00Z"
        with self.assertRaises(BoundaryError):
            self.adapt()

    def test_late_projection_known_at_denied(self):
        self.p["recordedAt"] = "2026-04-01T00:00:00Z"
        with self.assertRaises(BoundaryError):
            self.adapt()

    def test_future_source_claim_denied(self):
        for c in self.p["candidates"]:
            if c["claim"]["id"] == "claim:daniel-leads-alpha":
                c["claim"]["transactionTime"]["from"] = "2026-02-16T00:00:00Z"
        with self.assertRaises(BoundaryError):
            self.adapt()

    def test_suppressed_conflict_denied(self):
        self.p["conflicts"] = []
        with self.assertRaisesRegex(BoundaryError, "conflict"):
            self.adapt()

    def test_invented_conflict_denied(self):
        self.p["conflicts"] = [deepcopy(self.p["selected"])]
        with self.assertRaises(BoundaryError):
            self.adapt()

    def test_selected_mutated_outside_candidate_set_denied(self):
        self.p["selected"]["claim"]["object"]["entity"] = "entity:other"
        with self.assertRaises(BoundaryError):
            self.adapt()

    def test_source_entity_collision_denied(self):
        self.p["candidates"][0]["claim"]["subject"] = "entity:other"
        with self.assertRaises(BoundaryError):
            self.adapt()

    def test_duplicate_claim_denied(self):
        self.p["candidates"].append(deepcopy(self.p["candidates"][0]))
        with self.assertRaises(BoundaryError):
            self.adapt()

    def test_authority_impersonation_ignored(self):
        self.p["selected"]["claim"]["authority"]["level"] = 999
        self.p["candidates"][0]["claim"]["authority"]["level"] = 999
        result = self.adapt()
        self.assertNotIn("AUTH", str(result["governing_authority"]))
        self.assertEqual(result["governing_authority"], "NOT_GRANTED")

    def test_nested_oracle_feature_rejected(self):
        self.features = lambda c: {"future_labels": {"reward": 1.0}}
        with self.assertRaises(BoundaryError):
            self.adapt()

    def test_source_confidence_out_of_range_denied(self):
        self.p["candidates"][0]["claim"]["confidence"] = 1.3
        with self.assertRaises(BoundaryError):
            self.adapt()

    def test_unknown_lifecycle_denied(self):
        self.p["candidates"][0]["effectiveLifecycle"] = "accepted_by_ai"
        with self.assertRaises(BoundaryError):
            self.adapt()

    def test_missing_source_trace_denied(self):
        self.p["trace"] = []
        with self.assertRaises(BoundaryError):
            self.adapt()

    def test_no_claims_fail_not_fabricate(self):
        self.p["candidates"] = []
        self.p["conflicts"] = []
        self.p["selected"] = None
        with self.assertRaises(BoundaryError):
            self.adapt()

    def test_float_nonfinite_feature_rejected(self):
        self.features = lambda c: {"sensor": float("nan")}
        with self.assertRaises(BoundaryError):
            self.adapt()


if __name__ == "__main__":
    unittest.main()
