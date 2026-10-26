"""S1TEMP006 parameter/capacity/mathematical and real streaming contracts.

Fixtures are constructed SOFTWARE tests; cannot claim real learned retention.
"""
import unittest
import torch
from tac_osm.variants.system1_velocity_anchored import VelocityAnchoredSystem1
from tac_osm.variants.s1temp006_persistent_slow import (
    VelocityAnchoredSlowSystem1,slow_scan,temporal_previous_context,
    FAST_DIM,SLOW_DIM,ALPHA
)

class RealPersistentBankContract(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(76001)
        self.obs=torch.randn(2,75,14)*.15
        self.act=torch.randn(2,75,14)*.14
        self.prev=torch.randn(2,14)*.1
        self.base=VelocityAnchoredSystem1().eval()
        self.add=VelocityAnchoredSlowSystem1(memory_mode="additive").eval()
        self.slow=VelocityAnchoredSlowSystem1(memory_mode="slow16").eval()
        for m in (self.add,self.slow):
            m.load_state_dict(self.base.state_dict(),strict=True)

    def test_equal_weights_and_bounded_unchanged_float_memory(self):
        self.assertEqual([sum(p.numel() for p in m.parameters())
            for m in (self.base,self.add,self.slow)],[95907]*3)
        self.assertEqual([m.carried_fp32_bytes
            for m in (self.base,self.add,self.slow)],[664]*3)
        self.assertEqual(self.slow.temporal_slow_fp32_bytes,64)
        for k,v in self.base.state_dict().items():
            torch.testing.assert_close(v,self.slow.state_dict()[k],rtol=0,atol=0)

    def test_original_additive_exact_same_forward(self):
        with torch.no_grad():
            a=self.base(self.obs,self.act,
                        initial_previous_observation=self.prev)
            b=self.add(self.obs,self.act,
                       initial_previous_observation=self.prev)
        for name in ("next_observation","adverse_event_logit",
                     "residual_scale","adverse_event_score"):
            torch.testing.assert_close(getattr(a,name),getattr(b,name),atol=0,rtol=0)

    def test_finite_chunked_parallel_EMA_matches_true_recurrence(self):
        torch.manual_seed(76002)
        basis=torch.randn(2,1100,SLOW_DIM)*.1
        measured=slow_scan(basis)
        state=torch.zeros_like(basis[:,0])
        for t in range(1100):
            state=(1.-ALPHA)*state+ALPHA*basis[:,t]
            torch.testing.assert_close(measured[:,t],state,atol=3e-6,rtol=3e-6)
        self.assertTrue(torch.isfinite(measured).all().item())

    def test_source_time_safe_batched_vs_single_streaming_at_75_frames(self):
        for model in (self.add,self.slow):
            with torch.no_grad():
                seq=model(self.obs,self.act,
                    initial_previous_observation=self.prev)
                st=model.initial_state(2,previous_observation=self.prev)
                out=[]
                for t in range(self.obs.shape[1]):
                    f,st=model.step(st,self.obs[:,t],self.act[:,t])
                    out.append(f)
                for field in ("next_observation","adverse_event_logit",
                              "adverse_event_score","residual_scale"):
                    stream=torch.stack([getattr(o,field) for o in out],dim=1)
                    torch.testing.assert_close(getattr(seq,field),stream,
                                               atol=3e-5,rtol=3e-5)
                self.assertEqual(st.core_state.fast_sum.shape,(2,88))
                self.assertEqual(st.core_state.memory.shape,(2,8,8))
                self.assertTrue(st.has_previous)

    def test_order_sensitive_persistent_bank_with_same_final_measurement(self):
        # This permutation is an engineered falsifier, not a genuine robot
        # chronology or source for task capability.
        x=self.obs[:,:20].clone();u=self.act[:,:20].clone()
        shuffled=torch.cat((x[:,9:17],x[:,0:9],x[:,17:]),1)
        permu=torch.cat((u[:,9:17],u[:,0:9],u[:,17:]),1)
        with torch.no_grad():
            base,_=self.add.core.core(x,u,return_trace=True)
            perm,_=self.add.core.core(shuffled,permu,return_trace=True)
            # Fast old sum is a bag of features, independent of old order.
            e=self.add.core.core.encoder(torch.cat((x,u),-1)).tanh()
            ep=self.add.core.core.encoder(torch.cat((shuffled,permu),-1)).tanh()
            zf=torch.zeros(e.shape[0],e.shape[1],88)
            zr=torch.zeros(e.shape[0],e.shape[1],8)
            b=self.add.core.core.fast_map(torch.cat((e,zf,zr),-1))[...,:88].tanh()
            bp=self.add.core.core.fast_map(torch.cat((ep,zf,zr),-1))[...,:88].tanh()
            torch.testing.assert_close(b.sum(1),bp.sum(1),atol=1e-5,rtol=1e-5)
            slow_a=slow_scan(b[:,:,FAST_DIM:])[:,-1]
            slow_b=slow_scan(bp[:,:,FAST_DIM:])[:,-1]
            self.assertGreater((slow_a-slow_b).abs().max().item(),1e-6)
    def test_persistent_slow_gradient_reaches_long_past(self):
        basis=torch.randn(1,128,SLOW_DIM,requires_grad=True)
        terminal=slow_scan(basis)[:,-1].square().sum()
        grad=torch.autograd.grad(terminal,basis)[0]
        self.assertGreater(grad[:,0].abs().max().item(),0)
        self.assertTrue(torch.isfinite(grad).all().item())
    def test_persistent_bank_is_not_policy_hazard_calibration(self):
        f=self.slow(self.obs[:,:2],self.act[:,:2],
            initial_previous_observation=self.prev)
        self.assertFalse(f.is_calibrated)
        self.assertFalse(f.authorized_to_act)
        self.assertEqual(f.scope,"advisory_only")

if __name__=="__main__":unittest.main()
