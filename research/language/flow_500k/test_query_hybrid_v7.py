"""GRAPH011 conservative source-query inference regression contracts."""
import unittest
import torch
from learned_context_query_v6 import LearnedContextQuery
from source_calibrated_tac_v5 import SourceCalibratedTAC
from source_query_hybrid_v6 import SourceQueryHybridV6
from source_query_hybrid_v7 import CalibratedSourceQueryV7
from tac_wave_ced_500k_v4 import SlotTAC500K

class CalibratedQueryV7Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)
    def setUp(self):
        torch.manual_seed(2011)
        self.base=SourceCalibratedTAC(SlotTAC500K().eval(),torch.ones(128)/128,capacity=64).eval()
        self.query=LearnedContextQuery(torch.ones(128)/128,capacity=64).eval()

    def test_fixed_weight_exact_cache_invariance_and_parameter_budget(self):
        model=CalibratedSourceQueryV7(self.base,self.query).eval()
        self.assertEqual(model.learned.learned_parameters(),2816)
        self.assertEqual(self.base.backbone.parameter_count(),499290)
        x=torch.randint(0,9,(1,70))
        with torch.no_grad():
            exact=self.base(x,source_ids=("source-a",))
            out,diag=model(x,source_ids=("source-a",),return_diagnostics=True)
            old=SourceQueryHybridV6(self.base,self.query,fusion=.1)(x,source_ids=("source-a",))
        self.assertTrue(torch.allclose(out,old,atol=1e-7,rtol=1e-7))
        self.assertTrue(torch.allclose(out.sum(-1),torch.ones((1,70)),atol=1e-5))
        self.assertTrue(bool((out>0).all()))
        hit=diag["exact_hit"]
        self.assertFalse(bool((hit&diag["approximate_hit"]).any()))
        self.assertTrue(torch.allclose(out[hit],exact[hit],atol=1e-7))

    def test_source_owner_chunk_parity(self):
        model=CalibratedSourceQueryV7(self.base,self.query).eval()
        x=torch.randint(0,12,(1,71))
        with torch.no_grad():
            full=model(x,source_ids=("owner",))
            a,st=model(x[:,:30],source_ids=("owner",),return_state=True)
            b=model(x[:,30:],start=30,source_ids=("owner",),initial_state=st)
        self.assertTrue(torch.allclose(torch.cat((a,b),1),full,atol=3e-3,rtol=3e-3))
        with self.assertRaisesRegex(ValueError,"owner"):
            model(x[:,30:],start=30,source_ids=("wrong",),initial_state=st)

    def test_bad_gate_values_fail_closed(self):
        for value in (-.1,1.1,float("nan"),float("inf"),True):
            with self.assertRaises(ValueError):
                CalibratedSourceQueryV7(self.base,self.query,approx_weight=value)

if __name__=="__main__":unittest.main()
