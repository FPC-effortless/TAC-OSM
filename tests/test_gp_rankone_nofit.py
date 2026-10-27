"""GP-UNC-002 rank-one covariance update falsifiers (constructed-only)."""
import math
import random
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tac_osm.gp_rankone_nofit import RankOneProjectedResidualGP
from tac_osm.gp_sparse_nofit import ProjectedResidualGP
from tac_osm.gp_uncertainty_nofit import posterior


class GPFixedMemoryRankOne(unittest.TestCase):
    def test_80_constructed_streams_agree_with_precision_reference(self):
        # m=1/2/4/8, 20 independent deterministic stream seeds: 80 cases.
        for m in (1, 2, 4, 8):
            for seed in range(20):
                rng = random.Random(20261027 + seed)
                inducing = [[rng.uniform(-2, 2), rng.uniform(-2, 2)] for _ in range(m)]
                a = RankOneProjectedResidualGP(inducing, length_scale=1.4,
                                                noise_variance=.1)
                b = ProjectedResidualGP(inducing, length_scale=1.4,
                                         noise_variance=.1)
                for i in range(18):
                    x = [rng.uniform(-2, 2), rng.uniform(-2, 2)]
                    r = rng.uniform(-3, 3)
                    a.observe(x, r, feature_at=i, issued_at=i+.1,
                              label_at=i+.5, cutoff=i+1)
                    b.observe(x, r, feature_at=i, label_at=i+.5, cutoff=i+1)
                for _ in range(3):
                    q = [rng.uniform(-2, 2), rng.uniform(-2, 2)]
                    fast, slow = a.predict(q), b.predict(q)
                    self.assertAlmostEqual(fast.mean, slow.mean, delta=1e-8)
                    self.assertAlmostEqual(fast.latent_variance,
                                           slow.latent_variance, delta=1e-8)
                self.assertEqual(a.observations, 18)

    def test_initial_prior_and_noise(self):
        g = RankOneProjectedResidualGP([[0.], [1.]], noise_variance=.2)
        prior = g.predict([0.])
        self.assertAlmostEqual(prior.mean, 0)
        self.assertAlmostEqual(prior.latent_variance, 1, places=7)
        self.assertAlmostEqual(prior.observed_variance-prior.latent_variance, .2)

    def test_on_basis_exact_gp_agrees(self):
        x = [[0.], [1.], [2.]]
        y = [2., -1., 3.]
        g = RankOneProjectedResidualGP(x, noise_variance=.1)
        for i in range(3):
            g.observe(x[i], y[i], feature_at=i, issued_at=i+.1,
                      label_at=i+.5, cutoff=i+1)
        for query in x:
            a, b = g.predict(query), posterior(x, y, query, noise_variance=.1)
            self.assertAlmostEqual(a.mean, b.mean, delta=1e-7)
            self.assertAlmostEqual(a.latent_variance, b.latent_variance, delta=1e-7)

    def test_stored_state_constant_after_long_constructed_stream(self):
        g = RankOneProjectedResidualGP([[0.], [1.], [2.]], noise_variance=.25)
        initial = g.retained_numeric_scalars
        for i in range(512):
            g.observe([float(i%4)], math.sin(i), feature_at=i,
                      issued_at=i+.1, label_at=i+.5, cutoff=i+1)
        self.assertEqual(g.retained_numeric_scalars, initial)
        self.assertEqual(g.observations, 512)
        self.assertTrue(math.isfinite(g.predict([.75]).observed_variance))

    def test_future_label_and_backdated_prediction_do_not_mutate(self):
        g = RankOneProjectedResidualGP([[0.]])
        bad = [(0, 3, 5, 4), (1, 4, 4, 5), (5, 2, 4, 5)]
        for f, issued, label, cutoff in bad:
            with self.assertRaises(ValueError):
                g.observe([0.], 1., feature_at=f, issued_at=issued,
                          label_at=label, cutoff=cutoff)
        self.assertEqual(g.observations, 0)
        self.assertEqual(g.posterior_mean, [0.])
        g.observe([0.], 1., feature_at=0, issued_at=1, label_at=2, cutoff=3)
        with self.assertRaises(ValueError):
            g.observe([0.], 1., feature_at=0, issued_at=1,
                      label_at=2, cutoff=2)
        self.assertEqual(g.observations, 1)

    def test_nonfinite_and_invalid_shapes_fail_closed(self):
        g = RankOneProjectedResidualGP([[0., 0.]])
        for x, r in (([0.], 1.), ([0., float('inf')], 1.),
                     ([0., 0.], float('nan'))):
            with self.assertRaises(ValueError):
                g.observe(x, r, feature_at=0, issued_at=.1,
                          label_at=.5, cutoff=1)
        self.assertEqual(g.observations, 0)

    def test_duplicate_inducing_regularized(self):
        g = RankOneProjectedResidualGP([[0.], [0.]], jitter=1e-8)
        for i in range(15):
            g.observe([0.], i / 10, feature_at=i,
                      issued_at=i+.1, label_at=i+.5, cutoff=i+1)
        p = g.predict([0.])
        self.assertGreaterEqual(p.latent_variance, 0.)
        self.assertTrue(math.isfinite(p.mean))

    def test_independent_streams_do_not_share_state(self):
        a = RankOneProjectedResidualGP([[0.]])
        b = RankOneProjectedResidualGP([[0.]])
        a.observe([0.], 2, feature_at=0, issued_at=.1, label_at=.5, cutoff=1)
        self.assertNotEqual(a.predict([0.]).mean, b.predict([0.]).mean)
        self.assertEqual(b.observations, 0)


if __name__ == "__main__":
    unittest.main()
