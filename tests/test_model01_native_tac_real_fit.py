"""Synthetic fixtures for mechanics only; not neural real-world performance."""
import io,hashlib,json,os,sys,tempfile,unittest
from pathlib import Path
import numpy as np
from unittest.mock import patch
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"specialist_models"))
import model01_native_tac_real_fit as m

class NativeTACSourceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_original_live_learned_parameter_budget(self):
        net=m.NativeTAC()
        self.assertEqual(m.params(net),103252)
        self.assertTrue(95000<=m.params(net)<=105000)

    def test_causal_source_history_has_no_future_target(self):
        src=np.arange(200,dtype="float64")
        x,y,ii=m.windows(src,range(50,54),0,1)
        self.assertEqual(x.shape,(4,48,3))
        self.assertEqual(ii.tolist(),[50,51,52,53])
        self.assertEqual(x[0,-1,0].item(),50)
        self.assertEqual(y[0].item(),56)
        amended=src.copy()
        amended[56]=99999
        xx,_,_=m.windows(amended,range(50,54),0,1)
        self.assertTrue(torch.equal(x[0],xx[0]))

    def test_fast_slow_and_matrix_backpropagate(self):
        torch.manual_seed(3)
        net=m.NativeTAC()
        x=torch.randn(4,48,3)
        y=net(x)
        self.assertEqual(y.shape,(4,))
        self.assertTrue(torch.isfinite(y).all())
        y.square().mean().backward()
        for name in ("fast.weight_ih","slow_gate.weight","key.weight","value.weight","readgate.weight"):
            self.assertIsNotNone(dict(net.named_parameters())[name].grad)

    def test_invalid_history_cannot_silently_truncate(self):
        with self.assertRaises(m.StudyError):
            m.NativeTAC()(torch.zeros(2,49,3))

    def test_no_training_without_explicit_cpu_fit_gate(self):
        # Keep preflight entirely offline: missing source must never be fetched.
        with tempfile.TemporaryDirectory() as directory:
            with patch.dict(os.environ,{},clear=True):
                rc=m.main(["--out",str(Path(directory)/"receipt.json")])
                self.assertEqual(rc,2)

    def test_fixture_checkpoint_restores_exact_scores_without_refitting(self):
        t=np.linspace(20,50,170)
        train=m.windows(t,range(50,100),35,10)
        cal=m.windows(t,range(105,130),35,10)
        dev=m.windows(t,range(133,161),35,10)
        old=m.UPDATES
        try:
            m.UPDATES=2
            saved=m.fit_seed(1701,train,cal,dev,35,10)
            self.assertEqual(saved["steps"],2)
            replay=m.replay_seed(saved,cal,dev,35,10)
            self.assertTrue(replay["CAL_DEV_replayed_within_1e6"])
            self.assertFalse(replay["fitted_during_replay"])
            corrupt=dict(saved)
            corrupt["checkpoint_sha256"]="0"*64
            with self.assertRaises(m.StudyError):m.replay_seed(corrupt,cal,dev,35,10)
        finally:m.UPDATES=old

if __name__=="__main__":unittest.main()
