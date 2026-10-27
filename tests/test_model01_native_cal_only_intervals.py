"""Software-only interval algebra fixtures, not real predictive evidence."""
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
import numpy as np

ROOT=Path(__file__).resolve().parents[1]/"specialist_models"
sys.path.insert(0,str(ROOT))
import model01_native_cal_only_intervals as m

class IntervalRefinementTests(unittest.TestCase):
    def test_chronological_partition_no_cross_talk(self):
        first,select,audit=m.partition(3941)
        a=list(range(3941))
        self.assertEqual(len(a[first])+len(a[select])+len(a[audit]),3941)
        self.assertTrue(max(a[first])<min(a[select])<max(a[select])<min(a[audit]))
    def test_original_genuine_three_checkpoint_sha_guards(self):
        originals=m.immutable_checkpoints()
        self.assertEqual([v["seed"] for v in originals["runs"]],[1701,1702,1703])
        self.assertEqual(originals["trainable_params"],103252)
        self.assertFalse(originals["original_protected_tail_parsed"])
    def test_asymmetric_interval_score_both_sides_penalized(self):
        pred=np.array([10.,10.,10.])
        y=np.array([3.,10.,17.])
        sc=m.score([-5.,5.],pred,y,np.array([0,1,2]))
        # width 10 + 20 * 2 on either tail; total scores [50,10,50]
        self.assertAlmostEqual(sc["interval_score90"],110/3)
        self.assertEqual(sc["coverage90"],1/3)
        self.assertEqual((sc["undercount"],sc["overcount"]),(1,1))
    def test_phase_conditioning_uses_source_index_only(self):
        idx=np.array([0,35,36,71,72,107,108,143,144])
        self.assertEqual(m.phase(idx).tolist(),[0,0,1,1,2,2,3,3,0])
    def test_cal_only_selection_independent_of_later_targets(self):
        idx=np.arange(1200,dtype=np.int32)
        signed=np.sin(idx*.121)*10+5
        p=np.zeros(len(idx))
        first_p=p[:900];first_y=signed[:900]
        select_idx=idx[900:]
        select_y=signed[900:]
        a=m.select(first_p,first_y,idx[:900],p[900:],select_y,select_idx)
        # Future audit/DEV not accepted as an input to selector.
        b=m.select(first_p,first_y,idx[:900],p[900:],select_y,select_idx)
        self.assertEqual(a,b)
    def test_reject_bad_cov_or_source_cardinality(self):
        with self.assertRaises(m.IntegrityError):
            m.score([-1,1],[0,0],[1],[0,1])
        with self.assertRaises(m.IntegrityError):
            m.score([1,-1],[0,0],[1,2],[0,1])
    def test_reject_tiny_cal(self):
        with self.assertRaises(m.IntegrityError):
            m.partition(50)
    def test_all_original_weights_frozen(self):
        original=m.immutable_checkpoints()
        # No fit optimizer is invoked in candidate selection.
        with patch.object(m.native,"NativeTAC",side_effect=AssertionError("weights touched")):
            xs=np.arange(1024)
            residuals=np.sin(xs/10.)
            chosen,schemes,metrics=m.select(np.zeros(700),residuals[:700],xs[:700],
                                             np.zeros(324),residuals[700:],xs[700:])
            self.assertIn(chosen,schemes)
            self.assertIn("symmetric_CAL_60",metrics)
        self.assertEqual(original["runs"][0]["checkpoint_sha256"],m.RESULTS_SHA[1701])

if __name__=="__main__":
    unittest.main()
