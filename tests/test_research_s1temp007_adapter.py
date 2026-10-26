"""S1TEMP007 generic Observatory tests on genuine four-rate System1 state."""
import unittest
import torch
from tac_osm.research_probe import inspect_variant
from tac_osm.research_s1temp007_adapter import multiscale16_observatory_adapter
from tac_osm.variants.s1temp006_persistent_slow import VelocityAnchoredSlowSystem1

class ActualFourRateGenericAdapter(unittest.TestCase):
    def test_true_native_state_types_and_source_interventions(self):
        torch.manual_seed(3877)
        model=VelocityAnchoredSlowSystem1(memory_mode="multiscale16")
        sample=dict(
            observation_history=torch.randn(2,25,14)*.1,
            prior_logged_action_history=torch.randn(2,25,14)*.1,
            preceding_measured_observation=torch.randn(2,14)*.1)
        report=inspect_variant(model,sample,
                    multiscale16_observatory_adapter,isolation=True,profile=True)
        self.assertEqual(report["architecture"]["total_trainable"],95907)
        self.assertEqual(report["runtime"]["state_bytes_batch"],1328)
        self.assertEqual(report["entity_isolation"]["status"],"PASS")
        self.assertTrue(report["memory"]["query_does_not_write_state"])
        self.assertEqual(report["memory"]["candidate_permutation"]["status"],
                         "INCONCLUSIVE_NOT_DECLARED")
        self.assertIn("persistent_slow16",report["memory"]["interventions"])
        self.assertTrue(report["dependencies"]["heads"][
            "next_measured_state_forecast"]["persistent_slow16"]["connected"])
        self.assertGreater(report["dependencies"]["heads"][
            "next_measured_state_forecast"]["persistent_slow16"]["gradient_l2"],0)
        self.assertFalse(report["source_authorized_to_act"] if
              "source_authorized_to_act" in report else False)

    def test_reject_untrusted_extra_future_measurement(self):
        model=VelocityAnchoredSlowSystem1(memory_mode="multiscale16")
        s=dict(observation_history=torch.zeros(2,3,14),
               prior_logged_action_history=torch.zeros(2,3,14),
               preceding_measured_observation=torch.zeros(2,14),
               future_oracle_target=torch.ones(2,14))
        with self.assertRaises(ValueError):
            inspect_variant(model,s,multiscale16_observatory_adapter)

    def test_reject_original_memory_mode_claiming_multi(self):
        model=VelocityAnchoredSlowSystem1(memory_mode="slow16")
        s=dict(observation_history=torch.zeros(2,3,14),
               prior_logged_action_history=torch.zeros(2,3,14),
               preceding_measured_observation=torch.zeros(2,14))
        with self.assertRaises(ValueError):
            inspect_variant(model,s,multiscale16_observatory_adapter)
if __name__=="__main__":unittest.main()
