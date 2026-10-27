"""No-training, no-test, label-blind selective historical retrieval unit tests."""
import unittest
import numpy as np
import torch
from selective_history import (fit_only_frequency_prior, query_selective_history,
                               TRAINED_HISTORICAL_LAG)

class _HistorySpy:
    def __init__(self):
        self.embedding=torch.nn.Embedding(128,48)
        self.read_calls=0
    def historical_logits(self,tokens,trace,lag):
        self.read_calls+=1
        logits=torch.zeros(tokens.shape[0],tokens.shape[1]-lag,128)
        logits[...,7]=2
        return logits,tokens[:,:-lag]

class SelectiveHistoryTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.model=_HistorySpy()
        self.tokens=torch.randint(128,(3,128),generator=torch.Generator().manual_seed(2026))
        self.fit_prior=fit_only_frequency_prior([np.array([0,0,1,3,127],dtype=np.int64)])

    def test_fit_only_prior_normalized_and_all_tokens_positive(self):
        p=self.fit_prior
        self.assertEqual(tuple(p.shape),(128,))
        self.assertTrue(bool(torch.all(p>0)))
        self.assertAlmostEqual(float(p.sum()),1,places=12)
        self.assertGreater(float(p[0]),float(p[2]))
        self.assertGreater(float(p[127]),float(p[42]))

    def test_in_horizon_uses_trained_model_exactly(self):
        r=query_selective_history(self.model,self.tokens,{},64,self.fit_prior)
        self.assertFalse(r.abstained)
        self.assertTrue(r.used_trained_memory)
        self.assertEqual(self.model.read_calls,1)
        z=torch.zeros(3,64,128);z[...,7]=2
        self.assertTrue(torch.allclose(r.log_probabilities,torch.log_softmax(z,-1)))
        self.assertTrue(torch.allclose(torch.exp(r.log_probabilities).sum(-1),
                                       torch.ones((3,64)),atol=1e-6))

    def test_untrained_horizon_never_calls_model_or_uses_targets(self):
        r=query_selective_history(self.model,self.tokens,{},96,self.fit_prior)
        self.assertEqual(self.model.read_calls,0)
        self.assertTrue(r.abstained)
        self.assertFalse(r.used_trained_memory)
        self.assertEqual(tuple(r.log_probabilities.shape),(3,32,128))
        self.assertTrue(torch.allclose(torch.exp(r.log_probabilities).sum(-1),
                                       torch.ones(3,32),atol=1e-6))
        poison=self.tokens.clone()
        poison[:,0:32]=(poison[:,0:32]+71)%128
        r2=query_selective_history(self.model,poison,{},96,self.fit_prior)
        self.assertTrue(torch.equal(r.log_probabilities,r2.log_probabilities))
        self.assertEqual(self.model.read_calls,0)

    def test_fail_closed_bad_baseline_and_out_of_bounds_history(self):
        for invalid in (torch.zeros(128),torch.ones(128),
                        torch.ones(128,dtype=torch.int64),torch.ones(126)):
            with self.assertRaises(ValueError):
                query_selective_history(self.model,self.tokens,{},96,invalid)
        for lag in (0,128,129,-1,True):
            with self.assertRaises(ValueError):
                query_selective_history(self.model,self.tokens,{},lag,self.fit_prior)
        with self.assertRaisesRegex(ValueError,"positive"):
            fit_only_frequency_prior([np.array([1])],0)
        with self.assertRaisesRegex(ValueError,"out-of-range"):
            fit_only_frequency_prior([np.array([0,128])])

    def test_fallback_does_not_add_trainable_parameters(self):
        self.assertEqual(self.fit_prior.numel(),128)
        self.assertFalse(self.fit_prior.requires_grad)
        self.assertEqual(TRAINED_HISTORICAL_LAG,64)

if __name__=="__main__":
    unittest.main()
