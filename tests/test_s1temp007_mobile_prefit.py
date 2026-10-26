"""S1TEMP007 registered real mobile task source, architecture and objective tests.

These are constructed software preflight checks only, not trained evidence.
"""
import unittest
from pathlib import Path
import torch
from scripts.s1temp007_original_mobile_h1024_fit import (
    lock,model_pair,sample_plans,get_episode,FIT,DEV,TEST,WINDOWS,SEEDS,
    ARMS,TERMINAL_FRAMES,TRAIN_BATCH,TRAIN_UPDATES_PER_WINDOW,
)
from tac_osm.variants.s1motion005_gradient_budget import compute_gradients

class MobileSourceEligibility(unittest.TestCase):
    def test_30_authentic_full_FIT_sha_source_and_heldout_denial(self):
        src=lock()
        self.assertEqual(src["revision_sha"],
                         "7249d8a935db04f846f9e669b788f26c3a19b645")
        self.assertEqual(src["source_eligibility_run"],40296589314)
        self.assertEqual(len(src["FIT_receipts"]),30)
        self.assertEqual(src["expected_frames_each"],2000)
        self.assertTrue(all(len(x["sha256"])==64 for x in src["FIT_receipts"]))
        self.assertFalse(set(FIT)&set(DEV))
        self.assertFalse(set(FIT)&set(TEST))
        with self.assertRaises(ValueError):
            get_episode(30,src,Path("/tmp/no-dev"),released_dev=False)
        with self.assertRaises(ValueError):
            get_episode(40,src,Path("/tmp/no-test"),released_dev=True)

    def test_equal_four_real_fit_window_plans(self):
        a=sample_plans(SEEDS[0]); b=sample_plans(SEEDS[0])
        self.assertEqual(a,b)
        self.assertEqual(len(a),192)
        for length in WINDOWS:
            self.assertEqual(sum(p[0]==length for p in a),48)
        self.assertLess(max(TERMINAL_FRAMES),1998)
        self.assertGreaterEqual(min(TERMINAL_FRAMES)-1024+1,0)
        for W,eps,starts in a:
            self.assertEqual(len(eps),TRAIN_BATCH)
            self.assertTrue(all(e in FIT for e in eps))
            self.assertTrue(all(0<=s<=1998-W for s in starts))

    def test_all_three_init_identical_and_AD_source_budget(self):
        paired=model_pair(SEEDS[0],torch.ones(14)*.015)
        self.assertEqual(set(paired),set(ARMS))
        a=paired["additive"]
        for model in paired.values():
            self.assertEqual(sum(x.numel() for x in model.parameters()),95907)
            self.assertEqual(model.carried_fp32_bytes,664)
            for n,v in a.state_dict().items():
                torch.testing.assert_close(model.state_dict()[n],v,
                                           atol=0,rtol=0)
        torch.manual_seed(77)
        x=torch.randn(2,32,14)*.2
        u=torch.randn_like(x)*.2
        y=x+torch.randn_like(x)*.003
        event=torch.randint(0,2,(2,32)).float()
        previous=torch.randn(2,14)*.1
        for model in paired.values():
            point,aux,grad,meta,point_grads=compute_gradients(
                model,(x,u,y,event,previous),arm="tac_projected")
            self.assertTrue(torch.isfinite(point+aux).item())
            self.assertTrue(meta["local_first_order_sgd_nonharm"])
            self.assertTrue(all(torch.isfinite(g).all().item() for g in grad.values()))
if __name__=="__main__":unittest.main()
