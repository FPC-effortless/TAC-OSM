"""Risk-graph mechanical checks; no dataset or optimization."""
import unittest
import torch
from tac_osm.predictive_state_agent import Belief, PredictiveStateAgent
from tac_osm.decision_sys1_001 import NativeRiskStateModel

torch.set_num_threads(1)


class RiskTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(602)
        self.m = NativeRiskStateModel().double()
        self.x = torch.randn(2, 212, dtype=torch.float64)
        self.s = self.m.observe(self.m.initial(2), self.x)

    def test_01_exact_count_and_no_dormant_modules(self):
        self.assertEqual(sum(p.numel() for p in self.m.parameters()), 99963)
        self.assertFalse(any(k.startswith(('verify_', 'action.', 'next_observation.')) for k in self.m.state_dict()))

    def test_02_state_storage(self):
        self.assertEqual(sum(x.numel() for x in self.m.initial(1)), 128)
        self.assertEqual(sum(x.numel() * x.element_size() for x in self.m.float().initial(1)), 512)

    def test_03_probability_normalization(self):
        h, lp = self.m.predict(self.s, self.x)
        self.assertEqual(h.shape, (2, 4))
        self.assertEqual(lp.shape, (2, 5))
        torch.testing.assert_close(lp.exp().sum(-1), torch.ones(2, dtype=torch.float64), rtol=1e-12, atol=1e-12)

    def test_04_read_only(self):
        old = Belief(*(x.clone() for x in self.s))
        self.m.predict(self.s, self.x)
        for a, b in zip(old, self.s):
            self.assertTrue(torch.equal(a, b))

    def test_05_extreme_logits_finite(self):
        for val in [-1000., 0., 1000.]:
            lp = self.m.log_probabilities(torch.full((2, 4), val, dtype=torch.float64))
            self.assertTrue(bool(torch.isfinite(lp).all()))
            torch.testing.assert_close(lp.exp().sum(-1), torch.ones(2, dtype=torch.float64), rtol=1e-12, atol=1e-12)

    def test_06_cumulative_risks_monotone(self):
        _, lp = self.m.predict(self.s, self.x)
        risks = lp[:, 1:].flip(-1).exp().cumsum(-1)
        self.assertTrue(bool((risks[:, 1:] >= risks[:, :-1]).all()))

    def test_07_native_observe_exact(self):
        parent = PredictiveStateAgent(212, 1, 1).double()
        parent.load_state_dict({**parent.state_dict(), **{k: v for k, v in self.m.state_dict().items() if not k.startswith('risk_head.')}}, strict=True)
        expected = parent.observe(self.s, self.x)
        actual = self.m.observe(self.s, self.x)
        for a, b in zip(expected, actual):
            self.assertTrue(torch.equal(a, b))

    def test_08_gradients_all_parameters_reachable(self):
        state = self.m.initial(2)
        for _ in range(4):
            state = self.m.observe(state, torch.randn_like(self.x))
        _, lp = self.m.predict(state, self.x)
        (-lp[:, 2].sum()).backward()
        for name, p in self.m.named_parameters():
            self.assertIsNotNone(p.grad, name)
            self.assertTrue(bool(torch.isfinite(p.grad).all()), name)
            self.assertGreater(p.grad.abs().sum().item(), 0, name)

    def test_09_fp32_normalization(self):
        m = self.m.float()
        x = self.x.float()
        _, lp = m.predict(m.observe(m.initial(2), x), x)
        torch.testing.assert_close(lp.exp().sum(-1), torch.ones(2), rtol=1e-6, atol=1e-6)

    def test_10_batch_isolation(self):
        whole = self.m.predict(self.s, self.x)
        for i in range(2):
            one = self.m.predict(Belief(*(x[i:i+1] for x in self.s)), self.x[i:i+1])
            for a, b in zip(one, whole):
                torch.testing.assert_close(a, b[i:i+1], rtol=1e-12, atol=1e-12)

    def test_11_snapshot_replay(self):
        clone = NativeRiskStateModel().double()
        clone.load_state_dict(self.m.state_dict(), strict=True)
        for a, b in zip(clone.predict(self.s, self.x), self.m.predict(self.s, self.x)):
            self.assertTrue(torch.equal(a, b))

    def test_12_invalid_shapes(self):
        with self.assertRaises(ValueError):
            self.m.predict(self.s, self.x[:, :-1])
        with self.assertRaises(ValueError):
            self.m.log_probabilities(torch.zeros(2, 5))


if __name__ == '__main__':
    unittest.main()
