"""No-fit 72-case FP64 numerical equivalence against NumPy's independent solver."""
from pathlib import Path
import sys
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import numpy as np
from tac_osm.gp_uncertainty_nofit import posterior


class NumpyReferenceParity(unittest.TestCase):
    def test_seventy_two_posterior_reference_cases(self):
        rng = np.random.default_rng(20261027)
        max_mean, max_var, cases = 0.0, 0.0, 0
        for n in (1, 2, 4, 8, 16, 32):
            for trial in range(12):
                x = rng.normal(size=(n, 3))
                y = rng.normal(size=n)
                query = rng.normal(size=3)
                if trial in (2, 5):
                    x[0] = x[-1]  # nearly singular unregularized kernel
                noise, length, sig = 0.01, 1.7, 1.2
                distance2 = ((x[:, None, :] - x[None, :, :]) ** 2).sum(-1)
                covariance = sig * np.exp(-0.5 * distance2 / length ** 2)
                covariance += (noise + 1e-10) * np.eye(n)
                cross = sig * np.exp(-0.5 * ((x - query) ** 2).sum(-1) / length ** 2)
                mean = float(cross @ np.linalg.solve(covariance, y))
                variance = float(sig - cross @ np.linalg.solve(covariance, cross))
                checked = posterior(x.tolist(), y.tolist(), query.tolist(),
                                    length_scale=length, signal_variance=sig,
                                    noise_variance=noise)
                max_mean = max(max_mean, abs(mean - checked.mean))
                max_var = max(max_var, abs(variance - checked.latent_variance))
                cases += 1
        self.assertEqual(cases, 72)
        self.assertLess(max_mean, 1e-10)
        self.assertLess(max_var, 1e-10)
        print(f'GP_NO_FIT_NUMPY_PARITY cases={cases} max_mean={max_mean:.3e} max_var={max_var:.3e}')


if __name__ == '__main__':
    unittest.main()
