"""SEQ010 synthetic miniature fixtures test SOFTWARE contracts only, no skill claim."""
from collections import deque
import tempfile
from pathlib import Path
import unittest
import numpy as np
import torch

from scripts.seq010_real_recurrent import (
    N_FEATURES, HISTORY, MODELS, ResidualRecurrent, TRAIN_STEPS,
    digest_partition, sample_window, window, prepare, score, persistence_score,
)

class TestSEQ010CodeContract(unittest.TestCase):
    def test_source_split_deterministic(self):
        ids = [f"trainvehicle-{i}" for i in range(1000)]
        v = [x for x in ids if digest_partition(x)]
        tr = [x for x in ids if not digest_partition(x)]
        self.assertGreater(len(v), 150)
        self.assertGreater(len(tr), 700)
        self.assertFalse(set(v)&set(tr))
        self.assertEqual(v, [x for x in ids if digest_partition(x)])
        self.assertEqual(sample_window("truck-12",17),sample_window("truck-12",17))

    def test_causal_window_excludes_target(self):
        hist=deque(maxlen=HISTORY+1)
        for i in range(11):
            a=np.full((N_FEATURES,), float(i), dtype=np.float32)
            hist.append((float(i),a))
        inputs=window(list(hist)[:-1])
        self.assertEqual(inputs.shape,(HISTORY,N_FEATURES))
        self.assertEqual(float(inputs[-1,0]),9.0)
        self.assertNotIn(10.0,inputs[:,0])
        self.assertTrue(np.array_equal(hist[-1][1],np.full((N_FEATURES,),10,dtype=np.float32)))

    def test_shared_missing_mask_and_initial_persistence(self):
        source=np.ones((130,HISTORY,N_FEATURES),dtype=np.float32)
        target=np.ones((130,N_FEATURES),dtype=np.float32)
        source[0,-1,0]=np.nan
        target[1,1]=np.nan
        prepared=prepare(source,target)
        inp,delta,mask=prepared[:3]
        self.assertFalse(bool(mask[0,0]))
        self.assertFalse(bool(mask[1,1]))
        self.assertEqual(int(mask.sum()),130*N_FEATURES-2)
        for arm in MODELS:
            model=ResidualRecurrent(arm)
            torch.testing.assert_close(model(inp),torch.zeros_like(delta),rtol=0,atol=0)
            self.assertEqual(score(model,prepared)["macro_mae"],0.0)
        self.assertEqual(persistence_score(prepared)["macro_mae"],0.0)

    def test_updates_reach_learned_head(self):
        x=torch.randn(6,HISTORY,2*N_FEATURES)
        for arm in MODELS:
            net=ResidualRecurrent(arm)
            out=net(x)
            loss=(out-1).square().mean()
            loss.backward()
            self.assertIsNotNone(net.head.weight.grad)
            self.assertGreater(net.head.weight.grad.abs().sum().item(),0.0)

if __name__=="__main__":
    unittest.main()
