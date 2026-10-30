"""Constructed callback fixtures test engineering checks, never capability."""
import importlib.util
import json
from pathlib import Path
import sys
import unittest

import numpy as np


PATH = Path(__file__).resolve().parents[1] / "research/mathematical-engineering/langmath095/persistence.py"
SPEC = importlib.util.spec_from_file_location("langmath095_persistence", PATH)
persistence = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = persistence
SPEC.loader.exec_module(persistence)
PersistenceAdapter = persistence.PersistenceAdapter
probe_persistence = persistence.probe_persistence


def accounting(**changes):
    costs = {name: 0 for name in persistence.REQUIRED_COSTS}
    costs.update(changes)
    return costs


def fixture_adapter(**changes):
    def initial():
        return {"memory": np.zeros(2), "clock": 0}

    def observe(state, event):
        return {"memory": state["memory"] + event, "clock": state["clock"] + 1}

    def predict(state, query):
        return {"score": state["memory"] * query["current"]}

    def patch(state, name, value):
        return {**state, name: value}

    def verify(state, event):
        if not event["available"]:
            return state
        return {**state, "memory": state["memory"] + event["value"]}

    return PersistenceAdapter(**{
        "name": "constructed-engineering-only", "initial": initial,
        "observe": observe, "predict": predict,
        "state_views": lambda state: {"memory": state["memory"]},
        "patch_state": patch, "query_fields": ("current",),
        "metadata_fields": ("clock",), "cache_components": (),
        "verify": verify, **changes,
    })


class PersistenceEngineeringTests(unittest.TestCase):
    def setUp(self):
        self.events = [np.array([1., 2.]), np.array([3., 4.])]
        self.query = {"current": np.array([1., 1.])}

    def run_probe(self, adapter=None, **kwargs):
        return probe_persistence(adapter or fixture_adapter(), self.events, self.query,
                                 resource_costs=accounting(), **kwargs)

    def test_complete_query_only_interventions_and_input_ownership(self):
        originals = [event.copy() for event in self.events]
        result = self.run_probe(donor_events=[np.array([10., 20.])])
        self.assertEqual(result["status"], "PASS")
        self.assertEqual(result["checks"]["no_source_replay_after_acquisition"]["status"], "PASS")
        self.assertEqual(result["interventions"]["reset_all_components"]["max_abs_output_difference"], 6.)
        self.assertEqual(result["interventions"]["donor_all_components"]["max_abs_output_difference"], 14.)
        self.assertEqual(result["usefulness"]["status"], "INCONCLUSIVE")
        self.assertEqual(result["training_steps"], 0)
        self.assertGreater(result["resources"]["total_state_bytes"], 16)
        json.dumps(result, allow_nan=False)
        for actual, original in zip(self.events, originals):
            np.testing.assert_array_equal(actual, original)

    def test_no_effect_does_not_pass_persistence(self):
        adapter = fixture_adapter(predict=lambda state, query: {"score": query["current"].copy()})
        result = self.run_probe(adapter)
        self.assertEqual(result["checks"]["retained_state_dependence"]["status"], "FAIL")
        self.assertEqual(result["status"], "FAIL")

    def test_missing_costs_are_inconclusive_never_zero(self):
        result = probe_persistence(fixture_adapter(), self.events, self.query)
        self.assertEqual(result["checks"]["complete_resource_accounting"]["status"], "INCONCLUSIVE")
        self.assertIsNone(result["resources"]["total_episode_seconds"])
        self.assertIsNone(result["resources"]["total_state_bytes"])

    def test_explicit_unknown_cost_is_inconclusive(self):
        result = probe_persistence(fixture_adapter(), self.events, self.query,
            resource_costs=accounting(learning_seconds=None))
        self.assertEqual(result["checks"]["complete_resource_accounting"]["status"], "INCONCLUSIVE")
        self.assertNotIn("contract_execution", result["checks"])

    def test_metadata_declaration_is_required(self):
        result = self.run_probe(fixture_adapter(metadata_fields=None))
        self.assertEqual(result["checks"]["state_inventory"]["status"], "INCONCLUSIVE")
        self.assertIsNone(result["resources"]["total_state_bytes"])

    def test_metadata_limit_is_enforced(self):
        result = self.run_probe(fixture_adapter(metadata_limit_bytes=1))
        self.assertEqual(result["status"], "FAIL")
        self.assertIn("metadata", result["checks"]["contract_execution"]["reason"])

    def test_hidden_history_cannot_obtain_complete_accounting(self):
        def observe(state, event):
            return {"memory": state["memory"] + event, "clock": state["clock"] + 1,
                    "unexposed_raw_history": event.copy()}
        result = self.run_probe(fixture_adapter(observe=observe))
        self.assertEqual(result["checks"]["state_inventory"]["unexposed_arrays"], 1)
        self.assertEqual(result["checks"]["state_inventory"]["status"], "INCONCLUSIVE")
        self.assertIsNone(result["resources"]["total_state_bytes"])

    def test_borrowed_raw_input_alias_is_detected(self):
        def observe(state, event):
            return {"memory": event, "clock": state["clock"] + 1}
        result = self.run_probe(fixture_adapter(observe=observe))
        self.assertEqual(result["checks"]["no_source_replay_after_acquisition"]["status"], "FAIL")

    def test_closure_replay_negative_fixture_is_detected(self):
        hidden = []
        def observe(state, event):
            hidden.append(event)
            return {"memory": state["memory"] + event, "clock": state["clock"] + 1}
        def predict(state, query):
            return {"score": state["memory"] + hidden[-1]}
        result = self.run_probe(fixture_adapter(observe=observe, predict=predict))
        self.assertEqual(result["checks"]["no_source_replay_after_acquisition"]["status"], "FAIL")

    def test_mutable_list_closure_reference_is_poisoned_in_place(self):
        hidden = []
        def observe(state, event):
            hidden.append(event)
            return {"memory": state["memory"] + event, "clock": state["clock"] + 1}
        def predict(state, query):
            return {"score": state["memory"] + np.asarray(hidden[-1])}
        result = probe_persistence(fixture_adapter(observe=observe, predict=predict),
            [[1., 2.], [3., 4.]], self.query, resource_costs=accounting())
        self.assertEqual(result["checks"]["no_source_replay_after_acquisition"]["status"], "FAIL")
        self.assertEqual(hidden[-1], [-3., -4.])

    def test_immutable_tuple_payload_is_partial_poison_coverage(self):
        result = probe_persistence(fixture_adapter(), [(1., 2.), (3., 4.)],
                                   self.query, resource_costs=accounting())
        check = result["checks"]["no_source_replay_after_acquisition"]
        self.assertEqual(check["status"], "INCONCLUSIVE")
        self.assertEqual(check["mutated_input_leaves"], 0)
        self.assertEqual(check["unpoisoned_input_leaves"], 4)

    def test_scalar_event_payload_is_partial_poison_coverage(self):
        result = probe_persistence(fixture_adapter(), [1., 2.], self.query,
                                   resource_costs=accounting())
        check = result["checks"]["no_source_replay_after_acquisition"]
        self.assertEqual(check["status"], "INCONCLUSIVE")
        self.assertEqual(check["unpoisoned_input_leaves"], 2)

    def test_shared_acquisition_array_is_not_poisoned_twice(self):
        event = np.array([1., 2.])
        poisoned, coverage = persistence._poison([event, event])
        self.assertIs(poisoned[0], poisoned[1])
        np.testing.assert_array_equal(event, [-1., -2.])
        self.assertEqual(coverage["mutated_input_leaves"], 1)

    def test_thin_view_meters_full_unique_numpy_allocation(self):
        def observe(state, event):
            backing = np.ones(1_000_000, dtype=np.float64)
            return {"memory": backing[:2], "clock": 1}
        result = self.run_probe(fixture_adapter(observe=observe))
        self.assertEqual(result["resources"]["state_array_payload_bytes"], 16)
        self.assertEqual(result["resources"]["resident_array_bytes"], 8_000_000)
        self.assertGreaterEqual(result["resources"]["total_state_bytes"], 8_000_000)
        backing = np.ones(100)
        inventory = persistence._inventory({"a": backing[:2], "b": backing[50:52]})
        self.assertEqual(inventory["resident_array_bytes"], 800)

    def test_opaque_buffer_backing_cannot_pass_complete_accounting(self):
        def observe(state, event):
            return {"memory": np.frombuffer(bytearray(16), dtype=np.float64) + event,
                    "clock": 1}
        # Keep the buffer-backed view itself, rather than an owning result.
        def buffer_observe(state, event):
            value = np.frombuffer(bytearray(16), dtype=np.float64)
            value[:] = event
            return {"memory": value, "clock": 1}
        result = self.run_probe(fixture_adapter(observe=buffer_observe))
        self.assertEqual(result["checks"]["state_inventory"]["status"], "INCONCLUSIVE")
        self.assertEqual(result["checks"]["complete_resource_accounting"]["status"], "INCONCLUSIVE")
        self.assertIsNone(result["resources"]["resident_array_bytes"])
        self.assertIsNone(result["resources"]["total_state_bytes"])

    def test_external_state_view_cannot_pass_inventory(self):
        external = np.zeros(100)
        result = self.run_probe(fixture_adapter(
            state_views=lambda state: {"memory": state["memory"], "external": external}))
        self.assertEqual(result["checks"]["state_inventory"]["status"], "INCONCLUSIVE")
        self.assertEqual(result["checks"]["state_inventory"]["external_state_views"], 1)

    def test_nonfinite_prediction_fails(self):
        result = self.run_probe(fixture_adapter(predict=lambda state, query: {"score": np.array([np.nan])}))
        self.assertEqual(result["status"], "FAIL")
        self.assertIn("nonfinite", result["checks"]["contract_execution"]["reason"])

    def test_nonfinite_state_fails(self):
        result = self.run_probe(fixture_adapter(observe=lambda state, event: {"memory": np.array([np.inf]), "clock": 1}))
        self.assertEqual(result["status"], "FAIL")

    def test_nonfinite_acquisition_is_rejected_before_observer(self):
        result = probe_persistence(fixture_adapter(), [np.array([np.nan, 1.])], self.query,
                                   resource_costs=accounting())
        self.assertEqual(result["status"], "FAIL")

    def test_raw_source_query_fields_are_rejected(self):
        for field in ("history", "raw_source", "truth"):
            with self.subTest(field=field):
                result = probe_persistence(fixture_adapter(query_fields=(field,)), self.events,
                                           {field: np.array([1.])})
                self.assertEqual(result["status"], "FAIL")

    def test_nested_source_query_is_rejected(self):
        result = probe_persistence(fixture_adapter(), self.events,
                                   {"current": {"observations": self.events}})
        self.assertEqual(result["status"], "FAIL")

    def test_state_mutating_query_fails_purity(self):
        def predict(state, query):
            state["memory"] += 1
            return {"score": state["memory"].copy()}
        result = self.run_probe(fixture_adapter(predict=predict))
        self.assertEqual(result["checks"]["query_state_purity"]["status"], "FAIL")

    def test_later_query_metadata_mutation_is_detected(self):
        calls = []
        def predict(state, query):
            calls.append(True)
            if len(calls) == 2:
                state["clock"] += 1
            return {"score": state["memory"].copy()}
        result = self.run_probe(fixture_adapter(predict=predict))
        self.assertEqual(result["checks"]["query_state_purity"]["status"], "FAIL")

    def test_feedback_is_separate_and_missing_nan_does_not_write(self):
        result = self.run_probe(feedback={"available": True, "value": np.array([.5, .5])},
            unavailable_feedback={"available": False, "value": np.array([np.nan, np.nan])})
        self.assertEqual(result["checks"]["unavailable_feedback_does_not_write"]["status"], "PASS")
        self.assertEqual(result["checks"]["feedback_preserves_committed_prediction"]["status"], "PASS")
        self.assertEqual(result["usefulness"]["status"], "INCONCLUSIVE")

    def test_missing_feedback_write_negative_fixture(self):
        def verify(state, event):
            return {**state, "memory": state["memory"] + 1}
        result = self.run_probe(fixture_adapter(verify=verify), unavailable_feedback={"available": False})
        self.assertEqual(result["checks"]["unavailable_feedback_does_not_write"]["status"], "FAIL")

    def test_verification_cannot_add_unaccounted_history(self):
        def verify(state, event):
            return {**state, "unexposed_feedback_cache": event["value"].copy()}
        result = self.run_probe(fixture_adapter(verify=verify),
                                feedback={"available": True, "value": np.ones(100)})
        self.assertEqual(result["checks"]["verified_state_inventory"]["status"], "INCONCLUSIVE")
        self.assertIsNone(result["resources"]["total_state_bytes"])

    def test_truth_only_reaches_external_evaluator(self):
        def evaluator(outputs, truth):
            self.assertEqual(set(outputs), {"score"})
            return {"fixture_squared_error": float(np.square(outputs["score"] - truth).mean())}
        result = self.run_probe(truth=np.array([4., 6.]), evaluate=evaluator)
        self.assertEqual(result["external_metrics"]["fixture_squared_error"], 0.)
        self.assertEqual(result["usefulness"]["status"], "INCONCLUSIVE")
        self.assertEqual(result["learned_structure"]["status"], "INCONCLUSIVE")

    def test_bad_resource_number_fails(self):
        result = probe_persistence(fixture_adapter(), self.events, self.query,
            resource_costs=accounting(model_bytes=float("nan")))
        self.assertEqual(result["status"], "FAIL")


if __name__ == "__main__":
    unittest.main()
