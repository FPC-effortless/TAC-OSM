"""Projected-process GP-UNC-001 no-fit mathematical and leakage falsifiers."""
import math
from pathlib import Path
import sys
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tac_osm.gp_sparse_nofit import ProjectedResidualGP
from tac_osm.gp_uncertainty_nofit import posterior


class SparseGPNoFit(unittest.TestCase):
    def fit_constructed(self, points, outcomes, *, inducing=None, noise=.1):
        gp = ProjectedResidualGP(inducing or points, noise_variance=noise)
        for i, (x, y) in enumerate(zip(points, outcomes)):
            gp.observe(x, y, feature_at=i, label_at=i+1, cutoff=i+1)
        return gp

    def test_prior_and_far_field(self):
        gp = ProjectedResidualGP([[0.], [2.]], noise_variance=.1)
        a = gp.predict([0.])
        far = gp.predict([100.])
        self.assertAlmostEqual(a.mean, 0.)
        self.assertAlmostEqual(far.mean, 0.)
        self.assertAlmostEqual(far.latent_variance, 1., places=6)
        self.assertAlmostEqual(a.observed_variance-a.latent_variance, .1)

    def test_full_inducing_at_train_points_matches_exact(self):
        points = [[0.], [1.], [2.]]
        outcomes = [1., -2., 0.5]
        gp = self.fit_constructed(points, outcomes)
        for x in points:
            exact = posterior(points, outcomes, x, noise_variance=.1)
            p = gp.predict(x)
            self.assertAlmostEqual(p.mean, exact.mean, places=6)
            self.assertAlmostEqual(p.latent_variance, exact.latent_variance, places=6)

    def test_approximation_is_not_misrepresented_as_exact_off_basis(self):
        points = [[0.], [1.], [2.]]
        gp = self.fit_constructed(points, [1., -2., 0.5], inducing=[[0.]])
        q = gp.predict([1.])
        exact = posterior(points, [1., -2., 0.5], [1.], noise_variance=.1)
        self.assertGreater(abs(q.mean-exact.mean), .01)

    def test_more_observations_reduce_variance_at_fixed_query(self):
        gp = ProjectedResidualGP([[0.], [1.]], noise_variance=.2)
        prior = gp.predict([0.]).latent_variance
        gp.observe([0.], 1., feature_at=0, label_at=1, cutoff=1)
        mid = gp.predict([0.]).latent_variance
        gp.observe([0.], 1., feature_at=1, label_at=2, cutoff=2)
        final = gp.predict([0.]).latent_variance
        self.assertLess(final, mid)
        self.assertLess(mid, prior)

    def test_memory_retention_constant_as_stream_grows(self):
        gp = ProjectedResidualGP([[0.], [1.], [2.]], noise_variance=.5)
        amount = gp.retained_numeric_scalars
        for i in range(25):
            gp.observe([float(i%3)], float(i%4), feature_at=i, label_at=i+1, cutoff=i+1)
        self.assertEqual(gp.observations, 25)
        self.assertEqual(gp.retained_numeric_scalars, amount)
        self.assertTrue(math.isfinite(gp.predict([0.5]).mean))

    def test_future_label_rejected_without_mutation(self):
        gp = ProjectedResidualGP([[0.]])
        with self.assertRaises(ValueError):
            gp.observe([0.], 1., feature_at=1, label_at=5, cutoff=4)
        self.assertEqual(gp.observations, 0)
        self.assertEqual(gp.last_cutoff, None)
        self.assertEqual(gp.information, [0.])

    def test_out_of_order_cutoff_rejected(self):
        gp = ProjectedResidualGP([[0.]])
        gp.observe([0.], 1., feature_at=1, label_at=2, cutoff=5)
        with self.assertRaises(ValueError):
            gp.observe([0.], 1., feature_at=0, label_at=1, cutoff=4)
        self.assertEqual(gp.observations, 1)

    def test_invalid_features_and_nonfinite_outputs(self):
        gp = ProjectedResidualGP([[0., 0.]])
        with self.assertRaises(ValueError):
            gp.observe([0.], 1., feature_at=1, label_at=2, cutoff=2)
        with self.assertRaises(ValueError):
            gp.observe([0., 0.], float('inf'), feature_at=1, label_at=2, cutoff=2)
        with self.assertRaises(ValueError):
            gp.predict([float('nan'), 0.])
        self.assertEqual(gp.observations, 0)

    def test_duplicates_regularized_and_size_limits(self):
        gp = ProjectedResidualGP([[0.], [0.]], jitter=1e-8)
        gp.observe([0.], 1., feature_at=0, label_at=1, cutoff=1)
        self.assertGreaterEqual(gp.predict([0.]).latent_variance, 0.)
        with self.assertRaises(ValueError):
            ProjectedResidualGP([[float(i)] for i in range(65)])

    def test_positive_noise_avoids_exact_interpolation(self):
        gp = self.fit_constructed([[0.]], [5.], noise=1.)
        p = gp.predict([0.])
        self.assertLess(p.mean, 5)
        self.assertGreater(p.latent_variance, 0)


if __name__ == "__main__":
    unittest.main()
