"""Constructed mechanics/parity tests only; original source CI provides skill evidence."""
import hashlib
import sys
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]/"specialist_models"
sys.path.insert(0,str(ROOT))
import model01_matched_gru_control as m

class FrozenMatchedGRUTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_learned_param_gap_below_one_percent(self):
        n=m.tac.params(m.ControlGRU())
        self.assertEqual(n,102883)
        self.assertTrue(95000<=n<=105000)
        self.assertLess(abs(n-103252)/103252,.01)

    def test_pinned_native_history_model_and_budget(self):
        source,gap=m.preflight()
        self.assertEqual(source["trainable_params"],103252)
        self.assertEqual([r["seed"] for r in source["runs"]],[1701,1702,1703])
        self.assertEqual(source["train_updates_per_seed"],160)
        self.assertEqual(source["batch_size"],64)
        self.assertFalse(source["original_protected_tail_parsed"])
        self.assertLess(gap,.01)

    def test_same_input_shape_and_explicit_residual_target(self):
        model=m.ControlGRU()
        x=torch.randn(4,48,3)
        pred=model(x)
        self.assertEqual(pred.shape,(4,))
        self.assertTrue(torch.isfinite(pred).all())

    def test_sequence_order_changes_prediction(self):
        model=m.ControlGRU().eval()
        x=torch.randn(2,48,3)
        y=x.flip(1)
        with torch.no_grad():
            original=model(x)
            backwards=model(y)
        self.assertFalse(torch.allclose(original,backwards))

    def test_real_gru_parameters_receive_backprop(self):
        model=m.ControlGRU()
        x=torch.randn(4,48,3)
        model(x).square().mean().backward()
        self.assertIsNotNone(model.rnn.weight_ih_l0.grad)
        self.assertIsNotNone(model.rnn.weight_hh_l0.grad)
        self.assertIsNotNone(model.head[0].weight.grad)

    def test_bad_input_window_fails_instead_of_adapting(self):
        with self.assertRaises(m.ControlError):
            m.ControlGRU()(torch.zeros(2,49,3))

    def test_numpy_minibatches_same_as_original_receipt(self):
        original,_=m.preflight()
        for ref in original["runs"]:
            sampler=np.random.default_rng(ref["seed"])
            digest=hashlib.sha256()
            for step in range(m.tac.UPDATES):
                ids=sampler.integers(0,9814,size=m.tac.BATCH)
                digest.update(ids.tobytes())
            # Actually check FIT count from original original-source window definition.
            fit_size=len(range(m.tac.HISTORY-1,m.tac.uci.FIT_END-m.tac.uci.H))
            digest=hashlib.sha256()
            sampler=np.random.default_rng(ref["seed"])
            for step in range(m.tac.UPDATES):
                digest.update(sampler.integers(0,fit_size,size=m.tac.BATCH).tobytes())
            self.assertEqual(digest.hexdigest(),ref["sample_order_sha256"])

    def test_causal_window_construction_does_not_read_future(self):
        a=np.arange(160,dtype="float64")
        x,y,ii=m.tac.windows(a,[50,51],35.,10.)
        self.assertEqual(x.shape,(2,48,3))
        self.assertEqual(ii.tolist(),[50,51])
        a[56]=999999
        xx,_,_=m.tac.windows(a,[50,51],35.,10.)
        self.assertTrue(torch.equal(x,xx))

    def test_fit_requires_explicit_authorization_before_source_fetch(self):
        with patch.dict("os.environ",{},clear=True):
            with patch.object(m.tac.uci,"fetch_official_zip",side_effect=AssertionError("should not fetch")):
                with self.subTest("fail closed"):
                    with __import__("tempfile").TemporaryDirectory() as d:
                        self.assertEqual(m.main(["--out",str(Path(d)/"out.json")]),2)

    def test_original_checkpoint_mechanics_fixture_only(self):
        original,_=m.preflight()
        samples=np.linspace(5,75,180)
        fit=m.tac.windows(samples,range(51,115),30.,10.)
        cal=m.tac.windows(samples,range(115,135),30.,10.)
        dev=m.tac.windows(samples,range(140,167),30.,10.)
        old=m.tac.UPDATES
        try:
            m.tac.UPDATES=2
            # Two artificial update draw sequence must be patched into original
            # reference digest; this verifies exact replay but not a source skill.
            seed=1701
            rng=np.random.default_rng(seed)
            sha=hashlib.sha256()
            for _ in range(2):
                sha.update(rng.integers(0,len(fit[0]),size=m.tac.BATCH).tobytes())
            expected={"sample_order_sha256":sha.hexdigest(),
                      "checkpoint_sha256":original["runs"][0]["checkpoint_sha256"]}
            bundle=m.fit_seed(seed,fit,cal,dev,30.,10.,expected)
            restored=m.replay_seed(bundle,cal,dev,30.,10.,expected)
            self.assertTrue(restored["original_CAL_DEV_replayed_with_declared_numeric_tolerances"])
            self.assertFalse(restored["trained_during_replay"])
        finally:
            m.tac.UPDATES=old

if __name__=="__main__":
    unittest.main()
