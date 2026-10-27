"""Diagnostic contract tests for MLAG001; constructed inputs ONLY.

Authentic Vim corpus is not bundled or silently replaced with fabricated
examples. Real FIT/DEV and binary checks require the exact SHA-pinned
original source archive, original checkpoints and manual chapters.
"""
import hashlib
import unittest
import numpy as np
import torch
import torch.nn.functional as F
import train_multilag_real_fit as exp
from parallel_tac_language_gate_original import ParallelTACLanguage

class MultiLagContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): torch.set_num_threads(1)

    def test_original_python_model_immutable(self):
        from pathlib import Path
        source=Path(__file__).with_name("parallel_tac_language_gate_original.py")
        self.assertEqual(hashlib.sha256(source.read_bytes()).hexdigest(),exp.SHA)
        model=ParallelTACLanguage(decoder=75,position_mode="multiscale",gate_mode="silu")
        self.assertEqual(sum(p.numel() for p in model.parameters() if p.requires_grad),100680)

    def test_same_precommitted_optimizer_budget(self):
        self.assertEqual(exp.SEEDS,(1701,1702,1703,1704,1705))
        self.assertEqual(set(exp.MODES),{"tanh","silu"})
        self.assertEqual(exp.V,128)
        self.assertAlmostEqual(.15+.15,.3)
        p=torch.tensor([3.,5.,7.],dtype=torch.float64,requires_grad=True)
        main,h64,h96=p.unbind()
        exact=main+.3*(h64+h96)/2
        paired=main+.15*h64+.15*h96
        self.assertEqual(float((paired-exact).abs()),0)
        grads=torch.autograd.grad(paired,p)[0]
        self.assertTrue(torch.allclose(grads,torch.tensor([1.,.15,.15],dtype=torch.float64)))

    def test_valid_historical_targets_past_only(self):
        torch.manual_seed(1701)
        model=ParallelTACLanguage(decoder=75,post=10,position_mode="multiscale",gate_mode="silu")
        x=torch.randint(128,(2,128),dtype=torch.long)
        y,trace=model(x,return_trace=True)
        for lag in (64,96):
            logits,labels=model.historical_logits(x,trace,lag)
            self.assertEqual(tuple(logits.shape),(2,128-lag,128))
            self.assertTrue(torch.equal(labels,x[:,:128-lag]))
        main=F.cross_entropy(y.reshape(-1,128),torch.roll(x,-1,1).reshape(-1))
        h64,t64=model.historical_logits(x,trace,64)
        h96,t96=model.historical_logits(x,trace,96)
        loss=main+.15*F.cross_entropy(h64.reshape(-1,128),t64.reshape(-1))+.15*F.cross_entropy(h96.reshape(-1,128),t96.reshape(-1))
        self.assertTrue(bool(torch.isfinite(loss)))
        loss.backward()
        grads=[p.grad for p in model.parameters() if p.requires_grad]
        self.assertTrue(all(g is not None and bool(torch.isfinite(g).all()) for g in grads))

    def test_constructed_sample_order_is_seed_deterministic(self):
        docs=[np.arange(512,dtype=np.int64)%128,
              (np.arange(600,dtype=np.int64)*17)%128]
        a,b=exp.sample(docs,np.random.default_rng(1701))
        c,d=exp.sample(docs,np.random.default_rng(1701))
        self.assertTrue(torch.equal(a,c))
        self.assertTrue(torch.equal(b,d))
        self.assertTrue(torch.equal(b[:,0],a[:,1]))
        self.assertEqual(tuple(a.shape),(8,128))

if __name__=="__main__":
    unittest.main()
