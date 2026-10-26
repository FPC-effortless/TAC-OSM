"""S1RET008 source-only entity identity, admission, scan/step contract.

ALL fixtures are constructed, never claim real Backblaze prediction skill.
"""
import unittest
import torch
from tac_osm.variants.s1ret008_entity_affine import (
    entity_affine_prefix,entity_row_step,source_row_read,check_entity_inputs)
from tac_osm.variants.s1ret008_entity_system1 import SourceEntitySystem1
from tac_osm.variants.system1_velocity_anchored import VelocityAnchoredSystem1

class SourceEntityCausalContracts(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(88001)
        self.x=torch.randn(2,43,14)*.1
        self.u=torch.randn(2,43,14)*.1
        self.prior=torch.randn(2,14)*.1
        self.slot=torch.randint(0,8,(2,43),dtype=torch.int64)
        self.grade=torch.rand(2,43)
        self.grade[:,0]=1.
        base=VelocityAnchoredSystem1().eval()
        self.global_world=SourceEntitySystem1(entity_mode="global").eval()
        self.models={mode:SourceEntitySystem1(entity_mode=mode).eval()
           for mode in ("entity_all","entity_priority","entity_none")}
        for model in (self.global_world,*self.models.values()):
            model.load_state_dict(base.state_dict(),strict=True)
        self.reference=base

    def test_all_registered_weights_and_float_state_equal(self):
        for m in (self.reference,self.global_world,*self.models.values()):
            self.assertEqual(sum(p.numel() for p in m.parameters()),95907)
            self.assertEqual(m.carried_fp32_bytes,664)
        a=self.reference.state_dict()
        for m in (self.global_world,*self.models.values()):
            for k,v in a.items():
                torch.testing.assert_close(v,m.state_dict()[k],atol=0,rtol=0)

    def test_unmodified_global_reference_exact_original(self):
        with torch.no_grad():
            a=self.reference(self.x,self.u,initial_previous_observation=self.prior)
            b=self.global_world(self.x,self.u,source_slot=self.slot,
                observed_priority=self.grade,
                initial_previous_observation=self.prior)
        for name in ("next_observation","adverse_event_logit",
                     "residual_scale","adverse_event_score"):
            torch.testing.assert_close(getattr(a,name),getattr(b,name),
                                       atol=0,rtol=0)

    def test_scan_matches_single_step_across_all_source_modes(self):
        for name,model in (("global",self.global_world),*self.models.items()):
            with self.subTest(mode=name):
                with torch.no_grad():
                    big=model(self.x,self.u,source_slot=self.slot,
                        observed_priority=self.grade,
                        initial_previous_observation=self.prior)
                    state=model.initial_state(2,previous_observation=self.prior)
                    steps=[]
                    for t in range(self.x.shape[1]):
                        f,state=model.step(state,self.x[:,t],self.u[:,t],
                            source_slot=self.slot[:,t],
                            observed_priority=self.grade[:,t])
                        steps.append(f)
                    self.assertEqual(state.core_state.fast_sum.shape,(2,88))
                    self.assertEqual(state.core_state.memory.shape,(2,8,8))
                    for field in ("next_observation","adverse_event_logit",
                                  "adverse_event_score","residual_scale"):
                        got=torch.stack([getattr(z,field) for z in steps],dim=1)
                        torch.testing.assert_close(getattr(big,field),got,
                            atol=5e-5,rtol=5e-5)

    def test_exact_identity_isolation_under_alternating_source_entities(self):
        base=torch.arange(128,dtype=torch.float32).reshape(2,8,8)/100
        vals=torch.full((2,8),.3)
        gate=torch.tensor([.6,.4])
        slot=torch.tensor([3,6],dtype=torch.long)
        relevance=torch.tensor([.9,.95])
        z=entity_row_step(base,vals,gate,slot,relevance,"entity_priority")
        for batch in range(2):
            for row in range(8):
                if row!=int(slot[batch]):
                    torch.testing.assert_close(z[batch,row],base[batch,row],
                                               atol=0,rtol=0)
        read=source_row_read(base,slot)
        torch.testing.assert_close(read,base[torch.arange(2),slot],atol=0,rtol=0)

    def test_low_observed_priority_prevents_all_admission_but_all_control_writes(self):
        vals=torch.randn(2,22,8).tanh()
        gate=torch.ones(2,22)*.5
        slot=torch.arange(22).reshape(1,22).expand(2,-1).remainder(8).long()
        prio=torch.zeros(2,22)
        inactive,_,_=entity_affine_prefix(vals,gate,slot,prio,"entity_priority")
        unconditional,_,_=entity_affine_prefix(vals,gate,slot,prio,"entity_all")
        self.assertTrue(torch.equal(inactive,torch.zeros_like(inactive)))
        self.assertGreater(float(unconditional.abs().sum()),0)

    def test_future_only_state_and_future_entity_slot_cannot_change_prefix(self):
        model=self.models["entity_priority"]
        with torch.no_grad():
            base=model(self.x,self.u,source_slot=self.slot,
                observed_priority=self.grade,initial_previous_observation=self.prior)
            x=self.x.clone();u=self.u.clone();slot=self.slot.clone()
            p=self.grade.clone()
            x[:,-1]+=500;u[:,-1]-=400;slot[:,-1]=(slot[:,-1]+3)%8
            p[:,-1]=1-p[:,-1]
            changed=model(x,u,source_slot=slot,observed_priority=p,
                initial_previous_observation=self.prior)
            for field in ("next_observation","adverse_event_logit","residual_scale"):
                torch.testing.assert_close(getattr(base,field)[:,:-1],
                    getattr(changed,field)[:,:-1],atol=0,rtol=0)

    def test_identity_and_priority_boundaries_fail_closed(self):
        bad=self.slot.clone();bad[0,0]=8
        with self.assertRaises(ValueError):
            self.models["entity_priority"](self.x,self.u,
                  source_slot=bad,observed_priority=self.grade,
                  initial_previous_observation=self.prior)
        badp=self.grade.clone();badp[0,0]=float("nan")
        with self.assertRaises(ValueError):
            self.models["entity_priority"](self.x,self.u,
                  source_slot=self.slot,observed_priority=badp,
                  initial_previous_observation=self.prior)
        with self.assertRaises(ValueError):
            self.models["entity_priority"](self.x,self.u,
                  source_slot=self.slot,observed_priority=self.grade,
                  initial_previous_observation=torch.full_like(
                      self.prior,float("nan")))

    def test_current_observed_source_relevance_carries_gradients_into_native_gates(self):
        model=self.models["entity_priority"].train()
        grade=torch.ones_like(self.grade)
        out=model(self.x,self.u,source_slot=self.slot,
                  observed_priority=grade,
                  initial_previous_observation=self.prior)
        objective=out.next_observation.square().mean()+.1*out.adverse_event_logit.square().mean()
        objective.backward()
        named=dict(model.named_parameters())
        for field in ("core.core.address.weight","core.core.content.weight",
                      "core.core.memory_gates.weight",
                      "core.core.readout.0.weight"):
            self.assertTrue(named[field].grad is not None,field)
            self.assertGreater(float(named[field].grad.abs().sum()),0.,field)

    def test_real_nonpolicy_contract(self):
        y=self.models["entity_priority"](self.x[:,:2],self.u[:,:2],
          source_slot=self.slot[:,:2],observed_priority=self.grade[:,:2],
          initial_previous_observation=self.prior)
        self.assertEqual(y.scope,"advisory_only")
        self.assertFalse(y.authorized_to_act)
        self.assertFalse(y.is_calibrated)

if __name__=="__main__":
    unittest.main()
