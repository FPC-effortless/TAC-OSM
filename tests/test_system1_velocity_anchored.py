"""Source-agnostic software invariants only, NOT real predictive competence."""
import unittest

import torch

from tac_osm.variants.system1_velocity_anchored import VelocityAnchoredSystem1


class VelocityAnchoredEngineeringTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(26002)
        self.model = VelocityAnchoredSystem1().double().eval()

    def inputs(self, batch=2, length=17):
        x = torch.randn(batch, length, 14, dtype=torch.float64)*.2
        a = torch.randn(batch, length, 14, dtype=torch.float64)*.1
        return x, a

    def test_one_shared_core_no_parameter_inflation_count_extra_state(self):
        self.assertEqual(sum(p.numel() for p in self.model.parameters()), 95907)
        self.assertEqual(self.model.carried_fp32_bytes, 664)
        state = self.model.initial_state(2)
        self.assertEqual(state.previous_observation.numel(), 28)
        self.assertEqual(state.core_state.memory.numel(), 128)
        self.assertFalse(state.has_previous)

    def test_zero_gain_is_exact_causal_velocity(self):
        model = VelocityAnchoredSystem1(correction_gain=0).double().eval()
        x, a = self.inputs(length=7)
        prior = torch.randn(2,14,dtype=torch.float64)
        y = model(x,a,initial_previous_observation=prior)
        expected = 2*x-torch.cat((prior[:,None],x[:,:-1]),1)
        torch.testing.assert_close(y.next_observation,expected,atol=0,rtol=0)
        self.assertEqual(y.scope,"advisory_only")
        self.assertFalse(y.authorized_to_act)
        self.assertFalse(y.is_calibrated)

    def test_batch_and_step_match_including_carry_over(self):
        for length in (1,3,17):
            x,a=self.inputs(length=length)
            prior=torch.randn(2,14,dtype=torch.float64)
            with torch.no_grad():
                batch=self.model(x,a,initial_previous_observation=prior)
                state=self.model.initial_state(2,previous_observation=prior)
                outputs=[]
                for t in range(length):
                    pred,state=self.model.step(state,x[:,t],a[:,t])
                    outputs.append(pred)
                for name in ("next_observation","adverse_event_logit",
                             "adverse_event_score","residual_scale"):
                    got=torch.stack([getattr(pred,name) for pred in outputs],1)
                    torch.testing.assert_close(got,getattr(batch,name),
                                               atol=2e-7,rtol=2e-7)
                self.assertTrue(state.has_previous)
                self.assertEqual(state.core_state.count,length)
                torch.testing.assert_close(state.previous_observation,x[:,-1],
                                           atol=0,rtol=0)

    def test_bounded_innovation_and_future_observation_no_leak(self):
        x,a=self.inputs(length=13)
        with torch.no_grad():
            y=self.model(x,a)
            base=2*x-torch.cat((x[:,:1],x[:,:-1]),1)
            bound=self.model.correction_gain*self.model.core.core.delta_scale
            self.assertTrue(((y.next_observation-base).abs() <=
                             bound.view(1,1,-1)+1e-12).all().item())
            change_x=x.clone();change_a=a.clone()
            change_x[:,8:]+=987;change_a[:,8:]-=555
            changed=self.model(change_x,change_a)
            torch.testing.assert_close(y.next_observation[:,:8],
                                       changed.next_observation[:,:8],
                                       atol=2e-8,rtol=2e-8)

    def test_gradients_reach_shared_core_with_physical_prior(self):
        x,a=self.inputs(length=5)
        output=self.model(x,a)
        loss=output.next_observation.square().mean()+.01*output.adverse_event_logit.square().mean()
        loss.backward()
        for param in (self.model.core.core.memory_gates.weight,
                      self.model.core.core.encoder.weight,
                      self.model.core.reactive_head[0].weight):
            self.assertIsNotNone(param.grad)
            self.assertTrue(torch.isfinite(param.grad).all().item())
            self.assertGreater(param.grad.abs().sum().item(),0)

    def test_fail_closed_missing_or_future_priming_and_gain(self):
        x,a=self.inputs()
        for gain in (-.1,.3,float("nan"),True):
            with self.assertRaises(ValueError):
                VelocityAnchoredSystem1(correction_gain=gain)
        with self.assertRaises(ValueError):
            self.model.initial_state(2,previous_observation=torch.randn(2,13,dtype=torch.float64))
        with self.assertRaises(ValueError):
            self.model(x,a,initial_previous_observation=torch.randn(2,13,dtype=torch.float64))


if __name__ == "__main__":
    unittest.main()
