"""Actual slow16 native model diagnostics on constructed software fixtures."""
import unittest
import torch
from tac_osm.research_probe import inspect_variant,probe_dependencies,probe_memory
from tac_osm.research_s1temp006_adapter import persistent_slow16_adapter
from tac_osm.variants.s1temp006_persistent_slow import VelocityAnchoredSlowSystem1

def sample():
    torch.manual_seed(76033)
    return dict(observation_history=torch.randn(2,24,14)*.2,
                prior_logged_action_history=torch.randn(2,24,14)*.1,
                preceding_measured_observation=torch.randn(2,14)*.2)

class SlowActualResearchAdapter(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(76031)
        self.model=VelocityAnchoredSlowSystem1(memory_mode="slow16")
        self.sample=sample()

    def test_genuine_named_persistent_state_lesions_and_gradients(self):
        report=inspect_variant(self.model,self.sample,persistent_slow16_adapter,
                               isolation=True,profile=True)
        self.assertEqual(report["architecture"]["total_trainable"],95907)
        self.assertEqual(report["runtime"]["state_bytes_batch"],2*664)
        self.assertEqual(set(report["memory"]["state_components"]),
          {"additive_fast72","persistent_slow16","associative_memory",
           "preceding_measured_observation"})
        self.assertEqual(report["entity_isolation"]["status"],"PASS")
        self.assertTrue(report["memory"]["query_does_not_write_state"])
        self.assertEqual(report["memory"]["candidate_permutation"]["status"],
                         "INCONCLUSIVE_NOT_DECLARED")
        names=report["dependencies"]["heads"]
        self.assertTrue(names["next_measured_state_forecast"][
             "persistent_slow16"]["connected"])
        self.assertGreater(names["next_measured_state_forecast"][
             "persistent_slow16"]["gradient_l2"],0)
        les=report["memory"]["interventions"]["persistent_slow16"][
            "head_mean_absolute_change"]
        self.assertGreater(les["next_measured_state_forecast"],0)

    def test_no_fake_slow_state_on_unqualified_additive_model(self):
        a=VelocityAnchoredSlowSystem1(memory_mode="additive")
        with self.assertRaises(ValueError):
            probe_memory(a,self.sample,persistent_slow16_adapter)

    def test_illicit_future_source_field_fails_closed(self):
        bad={**self.sample,"future_measured_state":torch.ones(2,14)}
        with self.assertRaises(ValueError):
            inspect_variant(self.model,bad,persistent_slow16_adapter)

if __name__=="__main__":unittest.main()
