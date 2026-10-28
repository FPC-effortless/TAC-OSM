"""TEMP021 constructed mathematical regressions. NO authentic inference in CI."""
from pathlib import Path
import tempfile
import unittest
import numpy as np
import torch
from replay_original_500k_calibration import fit, moments, replay, score, CAL, SCORE

class Temperature021MathTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): torch.set_num_threads(2)

    def test_exact_source_roles_never_overlap(self):
        self.assertEqual(CAL,("usr_45.txt","usr_50.txt","usr_51.txt"))
        self.assertEqual(SCORE,("usr_52.txt","usr_90.txt","usr_toc.txt"))
        self.assertFalse(set(CAL)&set(SCORE))

    def test_bounded_convex_scalar(self):
        z=torch.tensor([[2.,0.,-1.],[0.5,1.,3.]],dtype=torch.float64)
        y=torch.tensor([0,2])
        for a in (.25,1.,4.):
            loss,gradient,curvature=moments(z,y,a)
            self.assertTrue(np.isfinite([loss,gradient,curvature]).all())
            self.assertGreaterEqual(curvature,0.)
        self.assertLessEqual(moments(z,y,fit(z,y))[0],moments(z,y,1.)[0]+1e-12)

    def test_flat_logits_identity_fallback(self):
        z=torch.zeros(3,128,dtype=torch.float64)
        y=torch.tensor([0,1,127])
        self.assertEqual(fit(z,y),1.)

    def test_positive_scalar_preserves_top1_and_top5(self):
        z=torch.arange(128,dtype=torch.float64)[None,:].expand(5,-1)
        y=torch.tensor([0,100,123,126,127])
        s=score(z,y,.35)
        self.assertEqual(s["identity"]["top1"],s["calibrated"]["top1"])
        self.assertEqual(s["identity"]["top5"],s["calibrated"]["top5"])

    def test_wrong_archive_rejected_before_model_load(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/"original.zip"
            p.write_bytes(b"tampered original")
            with self.assertRaisesRegex(ValueError,"original archive SHA256 mismatch"):
                replay(p,Path(d))

if __name__=="__main__":
    unittest.main()
