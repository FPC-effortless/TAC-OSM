"""Constructed/no-fit GP-UNC-001 falsifiers for all six family contracts."""
import math
from pathlib import Path
import sys
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from tac_osm.gp_uncertainty_nofit import (FAMILIES, ARMS, study_matrix, rbf_kernel,
    posterior, gaussian_nll, split_conformal_radius, validate_training_rows)


class GpUncertaintyNoFit(unittest.TestCase):
    def test_every_family_every_arm_explicitly_unqualified(self):
        self.assertEqual(len(FAMILIES), 6)
        self.assertEqual(len(ARMS), 5)
        cells = study_matrix()
        self.assertEqual(len(cells), 30)
        self.assertEqual({(f, a) for f, a, _ in cells}, {(f, a) for f in FAMILIES for a in ARMS})
        self.assertEqual({status for _, _, status in cells}, {'NOT_FITTED_OR_QUALIFIED'})

    def test_rbf_symmetric_and_diagonal(self):
        self.assertAlmostEqual(rbf_kernel((0,), (1,)), math.exp(-0.5))
        self.assertEqual(rbf_kernel((1, 2), (1, 2), signal_variance=2), 2)
        self.assertEqual(rbf_kernel((1, 2), (2, 4)), rbf_kernel((2, 4), (1, 2)))

    def test_transcript_one_observation_example(self):
        p = posterior([(0.,)], [2.], (1.,), noise_variance=1e-9)
        self.assertAlmostEqual(p.mean, 2 * math.exp(-0.5), places=6)
        self.assertAlmostEqual(p.latent_variance, 1-math.exp(-1), places=6)

    def test_near_far_and_noisy_observation(self):
        near = posterior([(0,)], [2.], (0,), noise_variance=.05)
        far = posterior([(0,)], [2.], (100,), noise_variance=.05)
        self.assertGreater(near.latent_variance, 0)
        self.assertLess(near.latent_variance, far.latent_variance)
        self.assertAlmostEqual(far.mean, 0)
        self.assertAlmostEqual(far.latent_variance, 1)
        self.assertAlmostEqual(near.observed_variance - near.latent_variance, .05)

    def test_two_observations_reduce_uncertainty(self):
        a = posterior([(0,)], [2], (1,), noise_variance=.01)
        b = posterior([(0,), (2,)], [2, -1], (1,), noise_variance=.01)
        self.assertLess(b.latent_variance, a.latent_variance)

    def test_duplicate_inputs_stable_with_noise(self):
        p = posterior([(0,), (0,)], [1, 1], (0,), noise_variance=.1)
        self.assertGreaterEqual(p.latent_variance, 0)
        self.assertTrue(math.isfinite(p.mean))

    def test_empty_prior(self):
        self.assertEqual(posterior([], [], (3,), noise_variance=.1).mean, 0)

    def test_nonfinite_and_invalid_covariance_rejected(self):
        for params in ({'length_scale':0}, {'noise_variance':0}, {'signal_variance':-1}):
            with self.subTest(params=params), self.assertRaises(ValueError):
                posterior([(0,)], [1], (0,), **params)
        with self.assertRaises(ValueError): posterior([(0,)], [float('nan')], (0,))
        with self.assertRaises(ValueError): posterior([(0, 1)], [1], (0,))
        with self.assertRaises(ValueError): posterior([(0,)] * 129, [0] * 129, (0,))

    def test_gaussian_nll_penalizes_overconfidence_on_outliers(self):
        self.assertGreater(gaussian_nll(8, 0, 0.01), gaussian_nll(8, 0, 1.))
        with self.assertRaises(ValueError): gaussian_nll(1, 0, 0)

    def test_conformal_order_statistic(self):
        self.assertEqual(split_conformal_radius(range(1, 20), alpha=.1), 18)
        with self.assertRaises(ValueError): split_conformal_radius([1, 2], alpha=.1)
        with self.assertRaises(ValueError): split_conformal_radius([1, -2, 3], alpha=.5)

    def test_causal_cutoff_and_duplicates(self):
        good = [{'id':'a','feature_at':1,'label_at':3},{'id':'b','feature_at':2,'label_at':4}]
        self.assertEqual(validate_training_rows(good, cutoff=4), 2)
        with self.assertRaises(ValueError): validate_training_rows(good, cutoff=3)
        with self.assertRaises(ValueError): validate_training_rows(good + [good[0]], cutoff=4)
        with self.assertRaises(ValueError): validate_training_rows([{'id':'x','feature_at':5,'label_at':4}], cutoff=6)


if __name__ == '__main__': unittest.main()
