"""Full native S1TEMP008 actual Observatory source-slot contract.

Synthetic fixtures ONLY for source-program wiring (not trained capability).
"""
import unittest
import torch
from tac_osm.research_probe import inspect_variant
from tac_osm.research_s1temp008_adapter import source_entity_system1_adapter
from tac_osm.variants.s1temp008_source_entity_world import SourceEntityVelocitySystem1

class ActualSourceIndexedModelAdapter(unittest.TestCase):
    def sample(self):
        torch.manual_seed(39111)
        return dict(observation_history=torch.randn(2,20,14)*.08,
            prior_logged_action_history=torch.randn(2,20,14)*.03,
            preceding_measured_observation=torch.randn(2,14)*.08,
            source_entity_slot_history=torch.randint(0,8,(2,20),dtype=torch.int64))
    def test_genuine_memory_lesions_and_source_dependency(self):
        model=SourceEntityVelocitySystem1(address_mode="source_indexed")
        report=inspect_variant(model,self.sample(),source_entity_system1_adapter,
                               isolation=True,profile=True)
        self.assertEqual(report["architecture"]["total_trainable"],95907)
        self.assertEqual(report["runtime"]["state_bytes_batch"],1328)
        self.assertEqual(report["entity_isolation"]["status"],"PASS")
        self.assertTrue(report["memory"]["query_does_not_write_state"])
        self.assertEqual(report["memory"]["candidate_permutation"]["status"],
                         "INCONCLUSIVE_NOT_DECLARED")
        self.assertIn("associative_memory",report["memory"]["interventions"])
        self.assertTrue(report["dependencies"]["heads"][
            "next_observed_state"]["associative_memory"]["connected"])
        self.assertGreater(report["dependencies"]["heads"][
            "next_observed_state"]["associative_memory"]["gradient_l2"],0)
    def test_all_source_oracle_fields_forbidden(self):
        model=SourceEntityVelocitySystem1(address_mode="source_indexed")
        for name in ("future_status","future_action","target_label","OfferID_from_future"):
            sample={**self.sample(),name:torch.ones(2,14)}
            with self.assertRaises(ValueError):
                inspect_variant(model,sample,source_entity_system1_adapter)
    def test_wrong_source_slot_dtype_is_not_quietly_coerced(self):
        model=SourceEntityVelocitySystem1(address_mode="source_indexed")
        sample=self.sample()
        sample["source_entity_slot_history"]=sample["source_entity_slot_history"].float()
        with self.assertRaises(ValueError):
            inspect_variant(model,sample,source_entity_system1_adapter)
    def test_wrong_model_reference_fails_closed(self):
        from tac_osm.variants.s1temp006_persistent_slow import VelocityAnchoredSlowSystem1
        with self.assertRaises(ValueError):
            inspect_variant(VelocityAnchoredSlowSystem1(memory_mode="multiscale16"),
                self.sample(),source_entity_system1_adapter)
if __name__=="__main__":unittest.main()
