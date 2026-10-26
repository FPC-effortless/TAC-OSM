"""S1TEMP006 original authentic source contract and matched budgets (NO FIT)."""
import unittest
from pathlib import Path
import torch
from scripts.s1temp006_real_long_history_fit import (
    lock,make_batch,sample_plans,model_pair,get_episode,TERMINAL_FRAMES,
    WINDOWS,SEEDS,ARMS,FIT,DEV,TEST,TRAIN_UPDATES_PER_WINDOW,
)
class SourceAndEqualBudget(unittest.TestCase):
    def test_authentic_repository_sha_locked_source_honest_paritions(self):
        source=lock()
        self.assertEqual(source["source_eligibility_run"],40290896322)
        self.assertEqual(len(source["FIT_receipts"]),25)
        self.assertEqual(source["expected_frames_each"],1500)
        self.assertEqual(source["revision_sha"],
                         "4f8c712349b83a40e25ef43cbddfbcb9816e3e63")
        self.assertEqual(set(FIT)&set(DEV),set())
        self.assertEqual(set(DEV)&set(TEST),set())
        with self.assertRaises(ValueError):
            get_episode(25,source,Path("/tmp/s1temp006-no-dev"),released_dev=False)
        with self.assertRaises(ValueError):
            get_episode(35,source,Path("/tmp/s1temp006-no-test"),released_dev=True)

    def test_equal_training_source_plan_and_long_history_range(self):
        a=sample_plans(SEEDS[0]);b=sample_plans(SEEDS[0])
        self.assertEqual(a,b)
        self.assertEqual(len(a),len(WINDOWS)*TRAIN_UPDATES_PER_WINDOW)
        self.assertEqual([w for w,_,_ in a].count(512),64)
        self.assertEqual([w for w,_,_ in a].count(128),64)
        self.assertEqual([w for w,_,_ in a].count(32),64)
        self.assertGreater(min(TERMINAL_FRAMES),512)
        self.assertLess(max(TERMINAL_FRAMES),1498)
        for W,ids,starts in a:
            self.assertEqual(len(ids),4)
            self.assertTrue(all(i in FIT for i in ids))
            self.assertTrue(all(0<=s<=1498-W for s in starts))

    def test_matched_neural_weights_state_and_train_gradients(self):
        variants=model_pair(SEEDS[0],torch.ones(14)*.03)
        a,b=(variants[k] for k in ARMS)
        self.assertEqual(a.carried_fp32_bytes,b.carried_fp32_bytes)
        self.assertEqual(a.carried_fp32_bytes,664)
        self.assertEqual(sum(p.numel() for p in a.parameters()),95907)
        for k,v in a.state_dict().items():
            torch.testing.assert_close(v,b.state_dict()[k],rtol=0,atol=0)
        torch.manual_seed(37)
        x=torch.randn(2,32,14)*.2;u=torch.randn_like(x)*.1
        prev=torch.randn(2,14)*.1
        y=x+torch.randn_like(x)*.02
        event=torch.randint(0,2,(2,32)).float()
        from tac_osm.variants.s1motion005_gradient_budget import compute_gradients
        for arm in ARMS:
            p,aux,g,meta,pg=compute_gradients(
               variants[arm],(x,u,y,event,prev),arm="tac_projected")
            self.assertTrue(torch.isfinite(p+aux).item())
            self.assertTrue(meta["local_first_order_sgd_nonharm"])

if __name__=="__main__":unittest.main()
