"""DEV064B enabling-action preflight tests. Never import or open reserved test data."""
import unittest
import numpy as np
import torch
from scripts.run_enabling_casm_dev064b import (
    ARMS, TEST_SEED, VAL_SEED, generate, model,
    oracle, transition, world, features,
)

class TestChainedCASMResearch(unittest.TestCase):
    def test_test_partition_is_sealed(self):
        with self.assertRaises(PermissionError):
            generate(TEST_SEED, 1, purpose="train")
        with self.assertRaises(PermissionError):
            generate(VAL_SEED, 1, purpose="test")

    def test_train_validation_reproducibility(self):
        a, q, manifest = generate(VAL_SEED, 12, purpose="validation")
        b, r, manifest2 = generate(VAL_SEED, 12, purpose="validation")
        self.assertTrue(torch.equal(a, b))
        self.assertTrue(torch.equal(q, r))
        self.assertEqual(manifest["x_sha256"], manifest2["x_sha256"])
        self.assertEqual(tuple(a.shape), (12, 25))
        self.assertEqual(tuple(q.shape), (12, 5))
        self.assertTrue(bool(torch.isfinite(q).all()))

    def test_transition_probability(self):
        rng = np.random.default_rng(12345)
        for _ in range(20):
            w = world(rng)
            for state_index in range(8):
                state = tuple((state_index >> i) & 1 for i in range(3)) + (0,)
                for action in range(1, 5):
                    results = transition(w, state, action)
                    self.assertAlmostEqual(sum(p for p, _ in results), 1, places=10)
                    for prob, nxt in results:
                        self.assertGreaterEqual(prob, 0)
                        self.assertEqual(len(nxt), 4)
                        self.assertTrue(all(b in (0, 1) for b in nxt))

    def test_oracle_bellman(self):
        rng = np.random.default_rng(44)
        w = world(rng)
        for s in range(8):
            state = tuple((s >> i) & 1 for i in range(3)) + (0,)
            for horizon in range(1, 5):
                values = oracle(w, state, horizon)
                self.assertAlmostEqual(values[0], w.penalty * sum(state[:3]), places=8)
                for action in range(1, 5):
                    expected = (w.setup_cost if action == 1 else w.costs[action - 2]) + sum(
                        p * min(oracle(w, nxt, horizon - 1))
                        for p, nxt in transition(w, state, action))
                    self.assertAlmostEqual(values[action], expected, places=8)

    def test_all_arms_receive_same_public_features(self):
        rng = np.random.default_rng(323)
        w = world(rng)
        row = torch.from_numpy(features(w, (1, 0, 1, 0), 4))[None].repeat(3, 1)
        self.assertEqual(tuple(row.shape), (3, 25))
        for arm in ARMS:
            with self.subTest(arm=arm):
                m = model(arm)
                scores = m(row)
                self.assertEqual(tuple(scores.shape), (3, 5))
                self.assertTrue(bool(torch.isfinite(scores).all()))
                self.assertTrue(all(p.requires_grad for p in m.parameters()))

if __name__ == "__main__":
    unittest.main()
