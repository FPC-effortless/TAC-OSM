"""S1RET008 version-independent Observatory on actual source-shaped neural state."""
import unittest
import torch
from tac_osm.research_probe import inspect_variant,probe_memory
from tac_osm.research_s1ret008_adapter import (
    source_entity_system1_adapter,FIELDS)
from tac_osm.variants.s1ret008_entity_system1 import SourceEntitySystem1

def original_shaped_sample():
    torch.manual_seed(39098)
    B,T=2,25
    x=torch.randn(B,T,14)*.1
    presence=torch.ones(B,T,14)
    slot=torch.randint(0,8,(B,T),dtype=torch.int64)
    priority=torch.ones(B,T)
    return dict(observation_history=x,
        observed_smart_presence_history=presence,
        preceding_measured_observation=torch.zeros(B,14),
        source_entity_slot_history=slot,
        observed_priority_history=priority)

class SourceEntityDiagnosticContract(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(39001)
        self.model=SourceEntitySystem1(
           entity_mode="entity_priority",forecast_anchor="persistence").eval()
        self.sample=original_shaped_sample()

    def test_true_state_lesions_not_fake_labels_or_invented_action(self):
        d=inspect_variant(self.model,self.sample,
                          source_entity_system1_adapter,
                          isolation=True,profile=True)
        self.assertEqual(d["architecture"]["total_trainable"],95907)
        self.assertEqual(d["runtime"]["state_bytes_batch"],1328)
        self.assertTrue(d["memory"]["query_does_not_write_state"])
        self.assertEqual(d["entity_isolation"]["status"],"PASS")
        self.assertEqual(d["memory"]["candidate_permutation"]["status"],
                         "INCONCLUSIVE_NOT_DECLARED")
        self.assertEqual(set(d["memory"]["state_components"]),
             {"fast_running_sum","associative_memory","previous_measured_observation"})
        self.assertTrue(d["dependencies"]["heads"][
             "source_future7day_failure_logit"]["associative_memory"]["connected"])

    def test_future_real_failure_target_cannot_enter_source_graph(self):
        for forbidden in ("next_7day_failure",
                          "future_replacement_action","oracle_future_source",
                          "observed_true_failure_label","actual_maintenance_action"):
            bad={**self.sample,forbidden:torch.ones(2,25)}
            with self.assertRaises(ValueError):
                probe_memory(self.model,bad,source_entity_system1_adapter)

    def test_no_cross_drive_velocity_prior_can_masquerade_as_source(self):
        other=SourceEntitySystem1(entity_mode="entity_priority",
                                  forecast_anchor="velocity")
        with self.assertRaises(ValueError):
            inspect_variant(other,self.sample,source_entity_system1_adapter)

    def test_not_a_hazard_or_executable_action_model(self):
        f=self.model(self.sample["observation_history"],
                     self.sample["observed_smart_presence_history"],
                     source_slot=self.sample["source_entity_slot_history"],
                     observed_priority=self.sample["observed_priority_history"],
                     initial_previous_observation=self.sample["preceding_measured_observation"])
        self.assertFalse(f.is_calibrated)
        self.assertFalse(f.authorized_to_act)
        self.assertEqual(f.scope,"advisory_only")

if __name__=="__main__":unittest.main()
