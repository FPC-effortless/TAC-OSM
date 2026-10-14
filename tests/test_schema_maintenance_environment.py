"""DEV-017 public temporal process and physical-cost validity gates."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tac_osm.development.schema_maintenance import (
    make_pair, exact_public_targets, execute,
)


class SchemaMaintenanceEnvironmentTests(unittest.TestCase):
    def test_histories_have_exactly_512_public_observations(self):
        for ep in make_pair(43, 512, 128, "pending_cancel", "at"):
            self.assertEqual(len(ep.observations), 512)
            self.assertEqual([o.tick for o in ep.observations], list(range(1, 513)))

    def test_current_value_is_identical_but_mature_future_is_opposite(self):
        left, right = make_pair(59, 512, 128, "pending_cancel", "at")
        e = left.target_entity
        self.assertEqual(left.current_targets[e], right.current_targets[e])
        self.assertNotEqual(left.wait_targets[e], right.wait_targets[e])

    def test_unresolved_intervention_does_not_act_before_due(self):
        left, right = make_pair(59, 512, 128, "pending_cancel", "before")
        self.assertEqual(left.wait_targets[left.target_entity], 1)
        self.assertEqual(right.wait_targets[right.target_entity], 1)

    def test_reissue_requires_ignoring_obsolete_generation(self):
        left, right = make_pair(71, 512, 128, "reissued_cancel", "after")
        e = left.target_entity
        self.assertEqual(len(left.world._pending), 2)
        self.assertEqual(left.world._generation[e], 1)
        self.assertEqual(right.world._generation[e], 2)
        self.assertEqual(left.wait_targets[e], 0)
        self.assertEqual(right.wait_targets[e], 1)

    def test_public_reference_reproduces_all_temporal_conditions(self):
        for h in (32, 128, 512):
            for role in ("retention", "acquisition", "pending_cancel", "reissued_cancel"):
                for deadline in ("before", "at", "after"):
                    for ep in make_pair(43, h, 64, role, deadline):
                        self.assertEqual(exact_public_targets(ep.observations, ep.tick, ep.horizon), ep.wait_targets)

    def test_always_service_succeeds_but_pays_for_unnecessary_service(self):
        ep = make_pair(43, 32, 16, "retention", "at")[0]
        always = execute(ep.world, ep.horizon, tuple(range(8)), 0)
        optimal = execute(ep.world, ep.horizon, tuple(e for e, bit in enumerate(ep.wait_targets) if bit), 0)
        self.assertEqual(always["failures"], 0)
        self.assertGreater(always["unnecessary_services"], 0)
        self.assertGreater(always["total_cost"], optimal["total_cost"])

    def test_inspections_have_explicit_cost(self):
        ep = make_pair(71, 32, 16, "retention", "at")[0]
        none = execute(ep.world, ep.horizon, (), 0)
        sensed = execute(ep.world, ep.horizon, (), 2)
        self.assertAlmostEqual(sensed["total_cost"] - none["total_cost"], .2)

    def test_training_and_evaluation_domains_are_disjoint(self):
        train = make_pair(43, 32, 16, "pending_cancel", "at", domain="train")
        evaluate = make_pair(43, 32, 16, "pending_cancel", "at")
        self.assertFalse({c.fingerprint for c in train} & {c.fingerprint for c in evaluate})


if __name__ == "__main__":
    unittest.main()
