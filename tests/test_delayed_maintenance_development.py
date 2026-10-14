"""DEV-016 validity gates; runnable without third-party packages."""
from dataclasses import fields
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tac_osm.development.delayed_maintenance import (
    Action, MaintenanceWorld, PublicObservation, SchemaReference, make_pair,
)


class DelayedMaintenanceTests(unittest.TestCase):
    def test_selected_observation_depends_on_action(self):
        world = MaintenanceWorld((0, 1))
        self.assertEqual(world.act(Action("inspect", 0)).value, 0)
        self.assertEqual(world.act(Action("inspect", 1)).value, 1)
        self.assertEqual(world.act(Action("wait", 0)).value, -1)

    def test_service_has_no_immediate_effect_and_matures_at_due_tick(self):
        world = MaintenanceWorld((1, 0))
        world.act(Action("service", 0, 0, 2))
        self.assertEqual(world.act(Action("inspect", 0)).value, 1)
        world.advance()
        self.assertEqual(world.act(Action("inspect", 0)).value, 1)
        world.advance()
        self.assertEqual(world.act(Action("inspect", 0)).value, 0)

    def test_reset_invalidates_older_pending_service_even_at_same_tick(self):
        world = MaintenanceWorld((0, 1))
        world.act(Action("service", 0, 1, 2))
        world.advance()
        world.advance(((0, 0),))
        self.assertEqual(world.act(Action("inspect", 0)).value, 0)

    def test_prediction_is_scored_before_receipt_and_receipt_cannot_change_it(self):
        world = MaintenanceWorld((1,))
        prediction = 0
        receipt = world.inspect_receipt(0, prediction)
        self.assertFalse(receipt.accepted)
        self.assertEqual(prediction, 0)
        self.assertEqual(receipt.observation.value, 1)

    def test_public_interface_excludes_target_and_generations(self):
        self.assertEqual({f.name for f in fields(PublicObservation)},
                         {"tick", "entity", "kind", "value", "due"})

    def test_all_roles_have_opposite_answers_and_identical_current_query(self):
        for role in ("retention", "acquisition", "cancelled_service", "pending_cancel"):
            left, right = make_pair(43, 32, 4, role=role)
            self.assertNotEqual(left.expected, right.expected)
            self.assertEqual(left.candidate, right.candidate)
            self.assertEqual(left.query_entity, right.query_entity)
            for case in (left, right):
                reference = SchemaReference(8)
                for obs in case.observations:
                    reference.observe(obs)
                self.assertEqual(reference.predict(case.query_entity, case.candidate), case.expected)
                self.assertEqual(reference.state_scalars, 41)
                self.assertEqual(case.service_expected, 0)

    def test_generation_effect_is_visible_in_process(self):
        left, _ = make_pair(71, 32, 4, role="cancelled_service")
        self.assertTrue(any(o.kind == "scheduled" and o.entity == left.query_entity
                            for o in left.observations))
        self.assertTrue(any(o.kind == "change" and o.entity == left.query_entity
                            for o in left.observations))

    def test_replay_and_training_domain_are_disjoint(self):
        a = make_pair(59, 32, 4)
        self.assertEqual(a, make_pair(59, 32, 4))
        b = make_pair(59, 32, 4, domain="train")
        self.assertFalse({x.fingerprint for x in a} & {x.fingerprint for x in b})

    def test_invalid_inputs_fail(self):
        for callback in (lambda: Action("invalid", 0), lambda: Action("service", 0, 2),
                         lambda: Action("service", 0, 0, 0),
                         lambda: MaintenanceWorld((2,)), lambda: SchemaReference(385),
                         lambda: make_pair(43, 1, 4),
                         lambda: make_pair(43, 32, 4, role="invalid")):
            with self.assertRaises(ValueError):
                callback()


if __name__ == "__main__":
    unittest.main()
