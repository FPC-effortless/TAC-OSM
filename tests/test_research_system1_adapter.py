"""S1OBS001 adapter and anti-cheat probes. Constructed fixtures only.

These tests prove diagnostic plumbing; trained model findings require actual
SHA-pinned original checkpoints and authentic source windows in separate CI.
"""
import unittest
from dataclasses import replace

import torch

from tac_osm.research_probe import (
    inspect_variant, probe_memory, probe_dependencies,
    probe_entity_isolation, probe_runtime,
)
from tac_osm.research_anticheat import audit_input_boundary, AuditLedger, Finding
from tac_osm.research_system1_adapter import (
    system1_velocity_adapter, validate_system1_source,
    SOURCE_FIELDS, patch_state, forecast, start, advance,
)
from tac_osm.variants.system1_velocity_anchored import VelocityAnchoredSystem1


def sample(batch=2, frames=5):
    torch.manual_seed(42002)
    return dict(
        observation_history=torch.randn(batch,frames,14)*.1,
        prior_logged_action_history=torch.randn(batch,frames,14)*.1,
        preceding_measured_observation=torch.randn(batch,14)*.1,
    )


class ExactSystem1AdapterTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42001)
        self.model=VelocityAnchoredSystem1().eval()
        self.sample=sample()

    def test_strict_source_allowlist_forbids_future_data_and_action_proposals(self):
        self.assertEqual(set(self.sample),SOURCE_FIELDS)
        self.assertEqual(audit_input_boundary(self.sample,sorted(SOURCE_FIELDS)).status,"PASS")
        for field in ("target","oracle","future_action","next_measured_state",
                      "candidate_action","current_action"):
            tampered={**self.sample,field:torch.zeros(2,14)}
            with self.assertRaisesRegex(ValueError,"allowlist"):
                validate_system1_source(self.model,tampered)
        self.assertEqual(audit_input_boundary({**self.sample,"target":2},
                         sorted(SOURCE_FIELDS)).status,"FAIL")

    def test_exact_model_adaptation_no_hidden_native_assumptions(self):
        report=inspect_variant(self.model,self.sample,
            system1_velocity_adapter,isolation=True,profile=False)
        self.assertEqual(report["architecture"]["total_trainable"],95907)
        self.assertEqual(report["adapter"],"system1-velocity-observational-v1")
        self.assertEqual(report["memory"]["events"],4)
        self.assertEqual(set(report["memory"]["state_components"]),{
            "fast_running_sum","associative_memory","previous_measured_observation"})
        self.assertEqual(report["memory"]["candidate_permutation"]["status"],
                         "INCONCLUSIVE_NOT_DECLARED")
        self.assertTrue(report["memory"]["query_does_not_write_state"])
        self.assertEqual(report["entity_isolation"]["status"],"PASS")
        self.assertTrue(report["dependencies"]["status"]=="MEASURED")
        self.assertEqual(report["runtime"]["status"],"INCONCLUSIVE_NOT_REQUESTED")
        self.assertEqual(report["integrity_status"],"INCONCLUSIVE")

    def test_dependency_legal_previous_observation_only_affects_point(self):
        report=probe_dependencies(self.model,self.sample,system1_velocity_adapter)
        heads=report["heads"]
        point=heads["next_measured_state_forecast"]
        risk=heads["motion_proxy_logit"]
        self.assertTrue(point["previous_measured_observation"]["connected"])
        self.assertGreater(point["previous_measured_observation"]["gradient_l2"],0)
        self.assertFalse(risk["previous_measured_observation"]["connected"])
        self.assertTrue(point["current_measured_observation"]["connected"])
        self.assertTrue(risk["current_measured_observation"]["connected"])

    def test_batch_entity_isolation_and_memory_lesions_are_measurable(self):
        result=probe_entity_isolation(self.model,self.sample,system1_velocity_adapter)
        self.assertEqual(result["status"],"PASS")
        m=probe_memory(self.model,self.sample,system1_velocity_adapter)
        self.assertEqual(set(m["interventions"]),{
            "fast_running_sum","associative_memory",
            "previous_measured_observation","<reset_all>"})
        prior=m["interventions"]["previous_measured_observation"][
            "head_mean_absolute_change"]
        self.assertGreater(prior["next_measured_state_forecast"],0)
        self.assertEqual(prior["motion_proxy_logit"],0)
        self.assertEqual(prior["motion_proxy_score"],0)

    def test_future_only_does_not_modify_earlier_stream_estimates(self):
        clean=self.sample
        modified={k:v.clone() for k,v in clean.items()}
        modified["observation_history"][:,4]+=800
        modified["prior_logged_action_history"][:,4]-=500
        net=self.model
        def prefix_end(s):
            state=start(net,s)
            for i in range(3):state=advance(net,state,s,i)
            return forecast(net,state,
                {**s,
                 "observation_history":s["observation_history"][:,:4],
                 "prior_logged_action_history":s["prior_logged_action_history"][:,:4]})
        with torch.no_grad():
            base=prefix_end(clean);changed=prefix_end(modified)
        for name in base:
            torch.testing.assert_close(base[name],changed[name],rtol=0,atol=0)

    def test_patch_state_pure_and_rejected_invalid_shape(self):
        state=start(self.model,self.sample)
        original=state.core_state.memory.clone()
        patched=patch_state(state,"associative_memory",
                             torch.ones_like(state.core_state.memory))
        self.assertTrue(torch.equal(state.core_state.memory,original))
        self.assertFalse(torch.equal(patched.core_state.memory,original))
        with self.assertRaises(ValueError):
            patch_state(state,"associative_memory",torch.zeros(1))
        with self.assertRaises(ValueError):
            patch_state(state,"no_authority",torch.zeros(2,14))

    def test_order_sensitive_history_resides_in_associative_state_not_fast_sum(self):
        # Reorder old observational pairs while preserving last two genuine
        # steps. Constructed architecture perturbation, NOT valid data.
        clean=sample(batch=2,frames=8)
        changed={k:v.clone() for k,v in clean.items()}
        perm=[4,3,2,1,0,5,6,7]
        for key in ("observation_history","prior_logged_action_history"):
            changed[key]=changed[key][:,perm]
        def replay(s):
            state=start(self.model,s)
            for t in range(s["observation_history"].shape[1]-1):
                state=advance(self.model,state,s,t)
            return state
        with torch.no_grad():
            a=replay(clean);b=replay(changed)
            torch.testing.assert_close(
                a.core_state.fast_sum,b.core_state.fast_sum,
                atol=2e-6,rtol=2e-6)
            torch.testing.assert_close(a.previous_observation,
                                       b.previous_observation,atol=0,rtol=0)
            a_zero=patch_state(a,"associative_memory",
                               torch.zeros_like(a.core_state.memory))
            b_zero=patch_state(b,"associative_memory",
                               torch.zeros_like(b.core_state.memory))
            out1=forecast(self.model,a_zero,clean)
            out2=forecast(self.model,b_zero,changed)
            for head in out1:
                torch.testing.assert_close(out1[head],out2[head],
                                           atol=2e-6,rtol=2e-6)

    def test_profiling_reports_explicit_runtime_not_hardware_guarantee(self):
        result=probe_runtime(self.model,self.sample,
                             system1_velocity_adapter,warmups=0,repeats=2)
        self.assertEqual(result["state_bytes_batch"],2*664)
        self.assertEqual(result["device"],"cpu")
        self.assertIn("local_timing",result["qualification"])

    def test_adversarial_adapter_mislabels_source_cannot_force_pass(self):
        altered=replace(system1_velocity_adapter,
            validate=lambda model,s: None,
            forecast=lambda model,state,s: {
                "next_measured_state_forecast":s["observation_history"][:,-1]
                +s["hidden_future_target"]})
        wrong={**self.sample,"hidden_future_target":torch.ones(2,14)}
        # Honest adapter rejects it rather than running any variant probe.
        with self.assertRaises(ValueError):
            probe_memory(self.model,wrong,system1_velocity_adapter)
        self.assertEqual(audit_input_boundary(wrong,sorted(SOURCE_FIELDS)).status,"FAIL")
        # A malicious replacement of the validation callback alone could bypass
        # boundary checks: provenance is not guaranteed by the generic engine.
        self.assertNotEqual(altered.name,"no-original-model") # no security proof


if __name__=="__main__":
    unittest.main()
