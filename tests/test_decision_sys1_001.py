"""Numerical/software tests only; no fitted model or task scores."""
import hashlib
import importlib.util
from pathlib import Path
import unittest
import numpy as np
import torch
from tac_osm.predictive_state_agent import Belief, PredictiveStateAgent
from tac_osm.decision_sys1_001 import SurpriseGatedPredictiveStateAgent
ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('sys1_reference', ROOT / 'research/decision-sys1-001/reference.py')
reference = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reference)
torch.set_num_threads(1)


class CorrectionTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(601)
        self.old = PredictiveStateAgent(4, 2, 1).double()
        self.new = SurpriseGatedPredictiveStateAgent(4, 2, 1).double()
        self.new.load_state_dict(self.old.state_dict(), strict=True)
        self.x = torch.randn(3, 4, dtype=torch.float64)
        self.state = self.old.observe(self.old.initial(3), self.x)
        self.a = torch.randn(3, 2, dtype=torch.float64)
        self.y = torch.tensor([[.4], [.7], [.2]], dtype=torch.float64)
        self.p = torch.tensor([[.1], [.3], [.9]], dtype=torch.float64)
        self.mask = torch.ones(3, 1, dtype=torch.bool)

    def verify(self, **kw):
        args = dict(state=self.state, executed_action=self.a, outcome=self.y,
                    predicted_outcome=self.p, available=self.mask)
        args.update(kw)
        return self.new.verify(**args)

    def assert_same(self, a, b):
        for x, y in zip(a, b):
            self.assertTrue(torch.equal(x, y))

    def test_01_parent_source_exact(self):
        b = (ROOT / 'src/tac_osm/predictive_state_agent.py').read_bytes()
        self.assertEqual(hashlib.sha1(f'blob {len(b)}\0'.encode() + b).hexdigest(),
                         '00e7c715d280e8195a38c473db5b2ef3a90291ff')

    def test_02_parent_zero_residual_drift_preserved(self):
        result = self.old.verify(self.state, self.a, self.y, self.y, self.mask)
        self.assertTrue(all(not torch.equal(x, y) for x, y in zip(result, self.state)))

    def test_03_zero_residual_noop(self):
        self.assert_same(self.verify(predicted_outcome=self.y), self.state)

    def test_04_missing_nan_noop(self):
        self.assert_same(self.verify(executed_action=torch.full_like(self.a, float('nan')),
                        outcome=torch.full_like(self.y, float('nan')),
                        predicted_outcome=torch.full_like(self.p, float('nan')),
                        available=torch.zeros_like(self.mask)), self.state)

    def test_05_mixed_mask_gradient_finite(self):
        a, y, p, mask = self.a.clone(), self.y.clone(), self.p.clone(), self.mask.clone()
        mask[1] = False
        a[1] = y[1] = p[1] = float('nan')
        r = self.verify(executed_action=a, outcome=y, predicted_outcome=p, available=mask)
        for orig, out in zip(self.state, r):
            self.assertTrue(torch.equal(orig[1], out[1]))
        sum(t.square().sum() for t in r).backward()
        for v in self.new.parameters():
            if v.grad is not None:
                self.assertTrue(bool(torch.isfinite(v.grad).all()))

    def test_06_observe_unchanged(self):
        self.assert_same(self.new.observe(self.state, self.x), self.old.observe(self.state, self.x))

    def test_07_prediction_unchanged_and_read_only(self):
        before = Belief(*(s.clone() for s in self.state))
        candidates = self.a.unsqueeze(1).repeat(1, 3, 1)
        self.assert_same(self.new.predict(self.state, self.x, candidates), self.old.predict(self.state, self.x, candidates))
        self.assert_same(self.state, before)

    def test_08_candidate_permutation(self):
        candidates = torch.randn(3, 5, 2, dtype=torch.float64)
        order = [4, 1, 0, 3, 2]
        a = self.new.predict(self.state, self.x, candidates)
        b = self.new.predict(self.state, self.x, candidates[:, order])
        for x, y in zip(a, b):
            torch.testing.assert_close(x[:, order], y, rtol=1e-12, atol=1e-12)

    def test_09_numpy_reference(self):
        weights = {k: v.detach().numpy() for k, v in self.new.state_dict().items()}
        result = reference.correction([t.detach().numpy() for t in self.state], self.a.numpy(), self.y.numpy(), self.p.numpy(), self.mask.numpy(), weights)
        for a, b in zip(self.verify(), result):
            np.testing.assert_allclose(a.detach().numpy(), b, rtol=1e-12, atol=1e-12)

    def test_10_surprise_scaled_bound(self):
        r = self.verify()
        scale = torch.tanh(self.y - self.p).square().mean(-1, keepdim=True)
        for x, y in zip(r[:2], self.state[:2]):
            self.assertTrue(bool(((x-y).abs() <= .25 * scale + 1e-15).all()))

    def test_11_nonzero_feedback_is_active(self):
        r = self.verify()
        self.assertTrue(all(not torch.equal(x, y) for x, y in zip(r, self.state)))

    def test_12_no_parameter_or_checkpoint_key_change(self):
        self.assertEqual(list(self.old.state_dict()), list(self.new.state_dict()))
        self.assertEqual(sum(p.numel() for p in self.new.parameters()), 25023)
        self.assertEqual(sum(p.numel() for p in self.new.parameters()), sum(p.numel() for p in self.old.parameters()))

    def test_13_entity_batch_isolation(self):
        whole = self.verify()
        for i in range(3):
            r = self.new.verify(Belief(*(x[i:i+1] for x in self.state)), self.a[i:i+1], self.y[i:i+1], self.p[i:i+1], self.mask[i:i+1])
            for a, b in zip(r, whole):
                torch.testing.assert_close(a, b[i:i+1], rtol=1e-12, atol=1e-12)

    def test_14_available_nan_rejected(self):
        with self.assertRaises(ValueError):
            self.verify(outcome=torch.full_like(self.y, float('nan')))
        with self.assertRaises(ValueError):
            self.verify(executed_action=torch.full_like(self.a, float('inf')))

    def test_15_shape_mask_dtype_rejected(self):
        with self.assertRaises(ValueError):
            self.verify(available=self.mask.double())
        with self.assertRaises(ValueError):
            self.verify(executed_action=self.a[:, :1])
        with self.assertRaises(ValueError):
            self.verify(outcome=self.y.float())

    def test_16_finite_difference_feedback_weight(self):
        state = Belief(*(x.detach() for x in self.state))
        out = self.verify(state=state)
        loss = sum(t.square().sum() for t in out)
        loss.backward()
        p = self.new.verify_map.weight
        i, j = 7, 66
        analytic = float(p.grad[i, j])
        eps = 1e-5
        with torch.no_grad():
            orig = p[i, j].item()
            p[i, j] = orig + eps
            up = sum(t.square().sum() for t in self.verify(state=state)).item()
            p[i, j] = orig - eps
            down = sum(t.square().sum() for t in self.verify(state=state)).item()
            p[i, j] = orig
        self.assertAlmostEqual(analytic, (up-down)/(2*eps), delta=1e-8)

    def test_17_prediction_detach_contract(self):
        p = self.p.clone().requires_grad_()
        y = self.y.clone().requires_grad_()
        r = self.verify(predicted_outcome=p, outcome=y)
        sum(t.square().sum() for t in r).backward()
        self.assertIsNone(p.grad)
        self.assertIsNotNone(y.grad)

    def test_18_fp32_and_fp64_zero_noop(self):
        for dtype in [torch.float32, torch.float64]:
            m = SurpriseGatedPredictiveStateAgent(4, 2, 1).to(dtype)
            s = m.observe(m.initial(3), self.x.to(dtype))
            r = m.verify(s, self.a.to(dtype), self.y.to(dtype), self.y.to(dtype), self.mask)
            self.assert_same(s, r)

    def test_19_repeated_confirmations_do_not_drift(self):
        state = self.state
        with torch.no_grad():
            for _ in range(128):
                state = self.new.verify(state, self.a, self.y, self.y, self.mask)
        self.assert_same(state, self.state)

    def test_20_small_residual_has_vanishing_update(self):
        with torch.no_grad():
            big = self.verify(predicted_outcome=self.y + .01)
            small = self.verify(predicted_outcome=self.y + .001)
        big_delta = max((a-b).abs().max().item() for a, b in zip(big, self.state))
        small_delta = max((a-b).abs().max().item() for a, b in zip(small, self.state))
        self.assertLess(small_delta, .02 * big_delta)


if __name__ == '__main__':
    unittest.main()
