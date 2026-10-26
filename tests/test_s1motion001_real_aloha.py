"""S1MOTION001 engineering fixtures only; NEVER measured research results."""
import unittest

import numpy as np
import torch

from scripts.s1motion001_real_aloha_fit import (
    FIT, DEV, MatchedGRU, System1ParallelWorld, prepare,
    windows, batch, objective, auroc, auprc,
)


def dummy_recordings():
    """Constructed arrays only for verifying causality, shapes and splits."""
    rows = {}
    for ep in FIT + DEV:
        recording = []
        for t in range(257):
            x = (np.arange(14, dtype=float) + ep * .01 +
                 .03 * np.sin(t * (.033 + ep*.0002) + np.arange(14) * .13))
            action = np.arange(14, dtype=float) + .01 * t + ep * .05
            recording.append((t, t / 50, x, action))
        rows[ep] = recording
    return rows


class RealMotionEngineeringTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        src = dummy_recordings()
        cls.train, cls.dev, cls.norm = prepare(
            {e: src[e] for e in FIT}, {e: src[e] for e in DEV})
        cls.src = src

    def test_exact_actor_splits_and_observation_time_alignment(self):
        self.assertEqual(set(self.train), set(FIT))
        self.assertEqual(set(self.dev), set(DEV))
        self.assertFalse(set(FIT) & set(DEV))
        self.assertTrue(set(DEV).isdisjoint(set(range(35, 50))))
        x = np.stack([r[2] for r in self.src[0]])
        u = np.stack([r[3] for r in self.src[0]])
        xm = np.asarray(self.norm["state_mean"])
        xs = np.asarray(self.norm["state_sd"])
        um = np.asarray(self.norm["action_mean"])
        us = np.asarray(self.norm["action_sd"])
        z = self.train[0]
        np.testing.assert_allclose(z["x"][0], (x[1]-xm)/xs, rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(z["y"][0], (x[2]-xm)/xs, rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(z["prior_u"][0], (u[0]-um)/us, rtol=1e-5, atol=1e-5)
        self.assertEqual(len(z["event"]), 255)
        self.assertEqual(z["episode_frames"], 255)

    def test_delta_scale_and_motion_threshold_fit_only(self):
        changed = dummy_recordings()
        for e in DEV:
            for i in range(257):
                old = changed[e][i]
                changed[e][i] = (old[0], old[1], old[2]*1e8, old[3]*1e6)
        _, _, after = prepare(
            {e: changed[e] for e in FIT}, {e: changed[e] for e in DEV})
        self.assertEqual(self.norm, after)

    def test_original_comparator_budget_and_objective(self):
        model = System1ParallelWorld()
        gru = MatchedGRU(torch.ones(14))
        self.assertEqual(sum(p.numel() for p in model.parameters()), 95907)
        self.assertEqual(sum(p.numel() for p in gru.parameters()), 95869)
        self.assertLess(abs(95907 - 95869)/95907, .01)
        sample = batch(self.train, *windows(self.train, 26001)[0])
        for m in (model, gru):
            loss = objective(m, *sample)
            self.assertTrue(torch.isfinite(loss).item())
            loss.backward()
            self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0
                                for p in m.parameters()))

    def test_only_fit_episode_samplers_and_equal_windows(self):
        self.assertEqual(windows(self.train,26001), windows(self.train,26001))
        plans = windows(self.train,26001)
        self.assertEqual(len(plans), 192)
        for es, starts in plans:
            self.assertEqual(len(es), 8)
            self.assertTrue(set(es).issubset(FIT))
            self.assertTrue(all(0 <= j <= 255-32 for j in starts))

    def test_ranking_metrics_have_known_limits(self):
        labels = [0, 0, 1, 1]
        perfect = [0.1, .2, .8, .9]
        inverse = [.8, .9, .1, .2]
        self.assertAlmostEqual(auroc(labels, perfect), 1.0)
        self.assertAlmostEqual(auprc(labels, perfect), 1.0)
        self.assertAlmostEqual(auroc(labels, inverse), 0.0)


if __name__ == "__main__":
    unittest.main()
