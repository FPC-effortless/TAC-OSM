"""GRAPH010 exact-first fusion contract; constructed operator fixtures only."""
import unittest,torch
from learned_context_query_v6 import LearnedContextQuery
from source_query_hybrid_v6 import SourceQueryHybridV6
from source_calibrated_tac_v5 import SourceCalibratedTAC
from tac_wave_ced_500k_v4 import SlotTAC500K

class Hybrid(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)
    def setUp(self):
        torch.manual_seed(710)
        prior=torch.ones(128)/128
        backbone=SlotTAC500K().eval()
        self.exact=SourceCalibratedTAC(backbone,prior,capacity=64).eval()
        self.q=LearnedContextQuery(prior,capacity=64).eval()
        self.hybrid=SourceQueryHybridV6(self.exact,self.q).eval()

    def test_same_base_exact_hits_unchanged_and_miss_only(self):
        x=torch.randint(0,9,(1,56))
        with torch.no_grad():
            base=self.exact(x,source_ids=("A",))
            fused,diagnostic=self.hybrid(x,source_ids=("A",),return_diagnostics=True)
        self.assertTrue(torch.isfinite(fused).all())
        self.assertTrue(torch.all(fused>0))
        self.assertTrue(torch.allclose(fused.sum(-1),torch.ones((1,56)),atol=1e-5))
        self.assertEqual(diagnostic["added_trainable_parameters"],2816)
        ex=diagnostic["exact_hit"]
        self.assertFalse(bool((ex&diagnostic["approximate_hit"]).any()))
        if bool(ex.any()):
            self.assertTrue(torch.allclose(base[ex],fused[ex],atol=1e-7))
        if bool(diagnostic["approximate_hit"].any()):
            self.assertGreater(float((base-fused)[diagnostic["approximate_hit"]].abs().max()),0)
        self.assertEqual(self.exact.backbone.parameter_count(),499290)

    def test_full_vs_carried_query_state_and_owner_rejection(self):
        x=torch.randint(0,11,(1,49))
        with torch.no_grad():
            full=self.hybrid(x,source_ids=("A",))
            early,st=self.hybrid(x[:,:25],source_ids=("A",),return_state=True)
            late,state=self.hybrid(x[:,25:],start=25,source_ids=("A",),
                                   initial_state=st,return_state=True)
        self.assertTrue(torch.allclose(torch.cat((early,late),1),full,rtol=3e-3,atol=3e-3))
        self.assertEqual(state["count"],49)
        with self.assertRaisesRegex(ValueError,"owner"):
            self.hybrid(x[:,25:],start=25,source_ids=("B",),initial_state=st)

    def test_zero_extra_query_fusion_is_identical_to_graph009(self):
        source=SourceQueryHybridV6(self.exact,self.q,fusion=0.)
        x=torch.randint(0,7,(1,37))
        with torch.no_grad():
            one=self.exact(x,source_ids=("A",))
            two=source(x,source_ids=("A",))
        self.assertTrue(torch.equal(one,two))

if __name__=="__main__":unittest.main()
