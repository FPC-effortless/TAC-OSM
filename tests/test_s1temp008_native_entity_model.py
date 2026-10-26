"""Complete native source-entity System1 graph tests, constructed fixtures.

No source FIT, DEV or TEST case outcome is inspected in these tests.
"""
import unittest
import torch
from tac_osm.variants.s1temp008_source_entity_world import (
    SourceEntityVelocitySystem1)

class NativeEightEntitySystem1(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(39001)
        self.x=torch.randn(2,70,14)*.04
        self.u=torch.randn_like(self.x)*.04
        self.prev=torch.randn(2,14)*.03
        self.slots=torch.randint(0,8,(2,70),dtype=torch.int64)
        self.matched={
            arm:SourceEntityVelocitySystem1(address_mode=arm).eval()
            for arm in ("source_indexed","oblivious")}
        a=self.matched["source_indexed"].state_dict()
        self.matched["oblivious"].load_state_dict(a,strict=True)

    def test_equal_weights_and_state_capacity_not_hidden_extra(self):
        self.assertEqual([sum(p.numel() for p in m.parameters())
            for m in self.matched.values()],[95907]*2)
        self.assertEqual([m.carried_fp32_bytes for m in self.matched.values()],
                          [664]*2)
        for k,v in self.matched["source_indexed"].state_dict().items():
            torch.testing.assert_close(self.matched["oblivious"].state_dict()[k],
                                       v,rtol=0,atol=0)
    def test_both_native_parallel_stream_70_step(self):
        for name,model in self.matched.items():
            with torch.no_grad():
                all_time=model(self.x,self.u,self.slots,
                    initial_previous_observation=self.prev)
                state=model.initial_state(2,previous_observation=self.prev)
                steps=[]
                for t in range(self.x.shape[1]):
                    result,state=model.step_source(
                        state,self.x[:,t],self.u[:,t],self.slots[:,t])
                    steps.append(result)
                for field in ("next_observation","adverse_event_logit",
                              "adverse_event_score","residual_scale"):
                    source=torch.stack([getattr(r,field) for r in steps],1)
                    torch.testing.assert_close(source,getattr(all_time,field),
                                               rtol=4e-5,atol=4e-5)
                self.assertEqual(state.core_state.memory.shape,(2,8,8))
                self.assertEqual(state.core_state.fast_sum.shape,(2,88))
                self.assertFalse(all_time.authorized_to_act)
                self.assertFalse(all_time.is_calibrated)

    def test_legitimate_prior_source_ids_influence_indexed_model(self):
        m=self.matched["source_indexed"]
        a=m(self.x,self.u,self.slots,initial_previous_observation=self.prev)
        modified=self.slots.clone()
        modified[:,5:30]=(modified[:,5:30]+1)%8
        b=m(self.x,self.u,modified,initial_previous_observation=self.prev)
        self.assertGreater((a.next_observation-b.next_observation).abs().max().item(),0)
        self.assertGreater((a.adverse_event_logit-b.adverse_event_logit).abs().max().item(),0)

    def test_oblivious_control_uses_same_source_slots_but_ignores_only_identity(self):
        m=self.matched["oblivious"]
        a=m(self.x,self.u,self.slots)
        b=m(self.x,self.u,torch.zeros_like(self.slots))
        torch.testing.assert_close(a.next_observation,b.next_observation,atol=0,rtol=0)
        torch.testing.assert_close(a.adverse_event_logit,b.adverse_event_logit,atol=0,rtol=0)

    def test_native_point_and_aux_gradients_reach_all_learned_matrices(self):
        for m in self.matched.values():
            out=m(self.x[:,:16],self.u[:,:16],self.slots[:,:16])
            loss=out.next_observation.square().mean()+out.adverse_event_logit.square().mean()
            g=torch.autograd.grad(loss,list(m.parameters()),allow_unused=True)
            names=list(dict(m.named_parameters()).keys())
            self.assertTrue(all(v is not None and torch.isfinite(v).all().item()
                                for v in g),names)
            self.assertGreater(sum(v.norm().item() for v in g),0)

    def test_cannot_infer_id_from_source_future_or_wrong_type(self):
        for m in self.matched.values():
            with self.assertRaises(TypeError):
                m(self.x,self.u)
            for invalid in (self.slots.float(),torch.full_like(self.slots,8),
                            self.slots[:,:-1]):
                with self.assertRaises(ValueError):
                    m(self.x,self.u,invalid)
            with self.assertRaises(ValueError):
                m.step(m.initial_state(2),self.x[:,0],self.u[:,0])

    def test_no_future_target_or_source_action_lookahead(self):
        for m in self.matched.values():
            x=self.x.clone();u=self.u.clone();slot=self.slots.clone()
            a=m(x,u,slot,initial_previous_observation=self.prev)
            x[:,-1]+=17;u[:,-1]-=13;slot[:,-1]=(slot[:,-1]+1)%8
            b=m(x,u,slot,initial_previous_observation=self.prev)
            torch.testing.assert_close(
                a.next_observation[:,:-1],b.next_observation[:,:-1],
                rtol=0,atol=0)
            torch.testing.assert_close(
                a.adverse_event_logit[:,:-1],b.adverse_event_logit[:,:-1],
                rtol=0,atol=0)
if __name__=="__main__":unittest.main()
