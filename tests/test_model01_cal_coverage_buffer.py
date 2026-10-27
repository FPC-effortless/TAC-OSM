"""Constructed tests only: real model evaluation belongs to source workflow."""
import sys,unittest
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]/"specialist_models"
sys.path.insert(0,str(ROOT))
import model01_cal_coverage_buffer as m

class SourceBufferTests(unittest.TestCase):
    def test_phase_interval_respects_declared_levels(self):
        indices=np.arange(1600,dtype=int)
        signed=np.where((indices%144)<72,-3.+np.sin(indices/11),9.+np.cos(indices/13))
        bands={lvl:m.buffered_intervals(signed,indices,lvl) for lvl in m.LEVELS}
        for level,config in bands.items():
            self.assertEqual(set(config),set(map(str,range(4))))
            self.assertTrue(all(hi>lo for lo,hi in config.values()))
        for ph in range(4):
            widths=[bands[l][str(ph)][1]-bands[l][str(ph)][0] for l in m.LEVELS]
            self.assertLessEqual(widths[0],widths[1])
            self.assertLessEqual(widths[1],widths[2])

    def test_true_nominal_96_covers_no_less_fit_residuals_than_90(self):
        idx=np.arange(1600,dtype=int)
        residual=np.sin(idx/6)*10.+np.cos(idx/17)*4
        widths={}
        for lvl in m.LEVELS:
            cfg=m.buffered_intervals(residual,idx,lvl)
            r=m.previous.score(cfg,np.zeros(len(idx)),residual,idx)
            widths[lvl]=r["coverage90"]
        self.assertTrue(widths[.96]>=widths[.94]>=widths[.90])

    def test_never_accept_unregistered_mass(self):
        idx=np.arange(1600,dtype=int)
        with self.assertRaises(m.ExperimentError):
            m.buffered_intervals(np.ones(len(idx)),idx,.99)

    def test_no_hidden_future_label_in_params(self):
        idx=np.arange(1600,dtype=int)
        vals=np.sin(idx/15)
        expected=m.buffered_intervals(vals,idx,.94)
        other=np.random.default_rng(77).normal(500,100,50)
        self.assertEqual(expected,m.buffered_intervals(vals,idx,.94))
        self.assertEqual(len(other),50) # other held-out observations never input to offsets

    def test_missing_or_insufficient_phase_group_blocked(self):
        idx=np.arange(140,dtype=int)
        with self.assertRaises(m.ExperimentError):
            m.buffered_intervals(np.ones(len(idx)),idx,.94)

    def test_nan_and_input_mismatch_blocked(self):
        idx=np.arange(1600,dtype=int)
        vals=np.ones(len(idx))
        vals[99]=float("nan")
        with self.assertRaises(m.ExperimentError):m.buffered_intervals(vals,idx,.94)
        with self.assertRaises(m.ExperimentError):m.buffered_intervals(vals[:-1],idx,.94)

    def test_live_frozen_original_models_pinned(self):
        r=m.previous.immutable_checkpoints()
        self.assertEqual(r["trainable_params"],103252)
        self.assertEqual([x["seed"] for x in r["runs"]],[1701,1702,1703])
        self.assertFalse(r["original_protected_tail_parsed"])

    def test_manual_status_noncrossing(self):
        self.assertEqual(m.LEVELS,(.90,.94,.96))
        self.assertEqual(m.PUBLISHER_SHA,m.previous.native.SOURCE)
        self.assertEqual(m.run.__doc__,None)

if __name__=="__main__":
    unittest.main()
