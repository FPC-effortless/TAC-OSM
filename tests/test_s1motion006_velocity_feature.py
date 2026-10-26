"""Constructed System1 velocity-input intervention: source/AD parity tests."""
import unittest
import torch
from tac_osm.variants.system1_velocity_anchored import VelocityAnchoredSystem1
from tac_osm.variants.s1motion006_velocity_feature import VelocityEventSystem1

class RealClockVelocityHeadTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(36001)
        self.x=torch.randn(2,12,14)*.2
        self.u=torch.randn(2,12,14)*.2
        self.prev=torch.randn(2,14)*.2
        self.legacy=VelocityAnchoredSystem1().eval()
        self.control=VelocityEventSystem1(feature_mode="core_prediction").eval()
        self.treated=VelocityEventSystem1(feature_mode="core_plus_measured_velocity").eval()
        for m in (self.control,self.treated):
            m.load_state_dict(self.legacy.state_dict(),strict=True)
    def test_matched_trainable_params_and_identical_initial_weights(self):
        self.assertEqual([sum(p.numel() for p in m.parameters()) for m in
                         (self.legacy,self.control,self.treated)],[95907]*3)
        self.assertEqual([m.carried_fp32_bytes for m in
                          (self.legacy,self.control,self.treated)],[664]*3)
    def test_full_original_control_regression_at_same_weights(self):
        with torch.no_grad():
            a=self.legacy(self.x,self.u,initial_previous_observation=self.prev)
            b=self.control(self.x,self.u,initial_previous_observation=self.prev)
        for field in ("next_observation","adverse_event_logit",
                      "adverse_event_score","residual_scale"):
            torch.testing.assert_close(getattr(a,field),getattr(b,field),
                                       rtol=0,atol=0)
        self.assertFalse(a.authorized_to_act)
        self.assertFalse(b.is_calibrated)
    def test_factual_point_output_unchanged_by_feature_ablations(self):
        a=self.control(self.x,self.u,initial_previous_observation=self.prev)
        b=self.treated(self.x,self.u,initial_previous_observation=self.prev)
        torch.testing.assert_close(a.next_observation,b.next_observation,rtol=0,atol=0)
        self.assertGreater((a.adverse_event_logit-b.adverse_event_logit).abs().max().item(),0.)
    def test_gradient_source_previous_reaches_treatment_motion_head(self):
        prior=self.prev.clone().detach().requires_grad_(True)
        a=self.control(self.x,self.u,initial_previous_observation=prior)
        b=self.treated(self.x,self.u,initial_previous_observation=prior)
        g0=torch.autograd.grad(a.adverse_event_logit.sum(),prior,
                               retain_graph=True,allow_unused=True)[0]
        g1=torch.autograd.grad(b.adverse_event_logit.sum(),prior,
                               retain_graph=True,allow_unused=True)[0]
        self.assertIsNone(g0)
        self.assertIsNotNone(g1)
        self.assertGreater(g1.norm().item(),0)
    def test_batched_scan_matches_source_primed_real_step(self):
        for m in (self.control,self.treated):
            m.eval()
            with torch.no_grad():
                f=m(self.x,self.u,initial_previous_observation=self.prev)
                s=m.initial_state(2,previous_observation=self.prev)
                nxt=[]
                for t in range(self.x.shape[1]):
                    y,s=m.step(s,self.x[:,t],self.u[:,t])
                    nxt.append(y)
                for attr in ("next_observation","adverse_event_logit",
                             "adverse_event_score","residual_scale"):
                    actual=torch.stack([getattr(y,attr) for y in nxt],dim=1)
                    torch.testing.assert_close(getattr(f,attr),actual,
                                               atol=2e-5,rtol=2e-5)
    def test_future_observation_and_action_do_not_change_prior_prefix(self):
        x=self.x.clone();u=self.u.clone()
        base=self.treated(x,u,initial_previous_observation=self.prev)
        x[:,-1]+=100;u[:,-1]+=100
        changed=self.treated(x,u,initial_previous_observation=self.prev)
        for attr in ("next_observation","adverse_event_logit","residual_scale"):
            torch.testing.assert_close(getattr(base,attr)[:,:-1],
                                       getattr(changed,attr)[:,:-1],atol=0,rtol=0)
    def test_previous_frame_source_validity_fail_closed(self):
        with self.assertRaises(ValueError):
            self.treated(self.x,self.u,
                initial_previous_observation=torch.full_like(self.prev,float("nan")))

if __name__=="__main__":unittest.main()
