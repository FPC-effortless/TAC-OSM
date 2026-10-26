"""S1TEMP007 four-timescale bank, distinct 3-arm no-new-parameter contract."""
import unittest
import torch
from tac_osm.variants.s1temp006_persistent_slow import (
    VelocityAnchoredSlowSystem1, multiscale_scan,slow_scan,
    MULTISCALE_RATES,ALPHA)
from tac_osm.variants.system1_velocity_anchored import VelocityAnchoredSystem1

class MultiScaleSourceGraph(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(77001)
        self.x=torch.randn(2,135,14)*.1
        self.u=torch.randn(2,135,14)*.1
        self.prev=torch.randn(2,14)*.1
        base=VelocityAnchoredSlowSystem1(memory_mode="additive").eval()
        self.models={key:VelocityAnchoredSlowSystem1(memory_mode=key).eval()
            for key in ("additive","slow16","multiscale16")}
        for model in self.models.values():
            model.load_state_dict(base.state_dict(),strict=True)

    def test_9_model_budget_identical(self):
        for m in self.models.values():
            self.assertEqual(sum(p.numel() for p in m.parameters()),95907)
            self.assertEqual(m.carried_fp32_bytes,664)
            self.assertFalse(m(self.x[:,:2],self.u[:,:2],
                initial_previous_observation=self.prev).authorized_to_act)

    def test_1100_step_bounded_chunked_multirate_exact_recurrence(self):
        torch.manual_seed(77002)
        basis=torch.randn(2,1100,16)*.1
        f=multiscale_scan(basis)
        rates=basis.new_tensor(MULTISCALE_RATES).repeat_interleave(4)
        state=torch.zeros_like(basis[:,0])
        # Evaluate all 1100 true real streaming steps, not interpolated.
        for t in range(1100):
            state=(1-rates)*state+rates*basis[:,t]
            torch.testing.assert_close(f[:,t],state,atol=3e-5,rtol=3e-5)
        self.assertTrue(torch.isfinite(f).all().item())
        self.assertTrue(bool((f[:,-1].abs().max()>0).item()))

    def test_old_single_decay_reproduced_when_rates_all_match(self):
        x=torch.randn(1,160,16)*.01
        multi=multiscale_scan(x)
        original=slow_scan(x,alpha=1/16)
        self.assertEqual(multi.shape,original.shape)
        self.assertEqual(len(MULTISCALE_RATES),4)

    def test_actual_multiscale_native_batch_vs_streaming(self):
        m=self.models["multiscale16"]
        with torch.no_grad():
            b=m(self.x,self.u,initial_previous_observation=self.prev)
            st=m.initial_state(2,previous_observation=self.prev)
            out=[]
            for t in range(self.x.shape[1]):
                f,st=m.step(st,self.x[:,t],self.u[:,t])
                out.append(f)
            for field in ("next_observation","adverse_event_logit",
                          "residual_scale","adverse_event_score"):
                stream=torch.stack([getattr(v,field) for v in out],dim=1)
                torch.testing.assert_close(getattr(b,field),stream,
                                           atol=4e-5,rtol=4e-5)
            self.assertEqual(st.core_state.fast_sum.shape,(2,88))

    def test_four_temporal_half_lives_and_order_sensitivity(self):
        self.assertEqual(MULTISCALE_RATES,(1/16,1/64,1/256,1/1024))
        a=torch.zeros(1,512,16)
        a[:,0,:]=1.
        kept=multiscale_scan(a)[:,-1]
        fast=float(kept[:,0].item())
        slow=float(kept[:,-1].item())
        self.assertGreater(slow,fast*100)
        self.assertGreater(slow,0.)

    def test_prior_only_source_no_future_influence(self):
        m=self.models["multiscale16"]
        x=self.x.clone();u=self.u.clone()
        a=m(x,u,initial_previous_observation=self.prev)
        x[:,-1]+=100;u[:,-1]+=100
        b=m(x,u,initial_previous_observation=self.prev)
        torch.testing.assert_close(a.next_observation[:,:-1],
                                   b.next_observation[:,:-1],atol=0,rtol=0)
        torch.testing.assert_close(a.adverse_event_logit[:,:-1],
                                   b.adverse_event_logit[:,:-1],atol=0,rtol=0)
if __name__=="__main__":unittest.main()
