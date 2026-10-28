"""TAC-LANG-MEM-500K-007 numerical contracts. Constructed operator checks only."""
import unittest
import torch
from tac_wave_ced_500k_v3 import WaveCED500KSlow
from historical_500k_v3 import HistoricalWaveCED500KSlow

class DualTimescaleChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
    def setUp(self):
        torch.manual_seed(71)

    def test_active_parameter_budget_and_slow_bounds(self):
        model=WaveCED500KSlow()
        logits,trace=model(torch.randint(0,128,(2,16)),return_trace=True)
        self.assertEqual(model.parameter_count(),499290)
        self.assertTrue(bool(torch.isfinite(logits).all()))
        alpha=trace['gate_alpha'][...,2]
        self.assertTrue(bool(((alpha>=.995)&(alpha<=.999)).all()))
        self.assertTrue(bool((trace['gate_erase'][...,2]<=.002).all()))
        self.assertTrue(bool((trace['gate_beta']<=1-trace['gate_alpha']+1e-7).all()))
        logits.square().mean().backward()
        self.assertEqual(sum(p.numel() for p in model.parameters() if p.grad is not None),499290)
        self.assertGreater(float(model.concept_q.weight.grad.norm()),1e-10)
        self.assertGreater(float(model.concept_k.weight.grad.norm()),1e-10)

    def test_cannot_resume_as_other_owner_or_missing_owner(self):
        model=WaveCED500KSlow().eval()
        x=torch.randint(0,128,(1,8))
        with torch.no_grad():
            _,state=model(x,source_ids=('document-A',),return_state=True)
            with self.assertRaisesRegex(ValueError,'source identities'):
                model(x,start=8,initial_state=state,source_ids=('document-B',))
            with self.assertRaisesRegex(ValueError,'source identities'):
                model(x,start=8,initial_state=state)
            y=model(x,start=8,initial_state=state,source_ids=('document-A',))
        self.assertEqual(y.shape,(1,8,128))

    def test_history_queries_are_owner_bound_and_strictly_past(self):
        model=WaveCED500KSlow().eval()
        x=torch.randint(0,128,(1,15))
        query=(torch.arange(15,dtype=torch.long)-1)[None,:]
        with torch.no_grad():
            self.assertTrue(bool(torch.isfinite(model(x,source_ids=('A',),query_ordinals=query)).all()))
            bad=query.clone()
            bad[0,11]=11
            with self.assertRaisesRegex(ValueError,'earlier source ordinal'):
                model(x,source_ids=('A',),query_ordinals=bad)
            bad0=query.clone()
            bad0[0,0]=-3
            with self.assertRaisesRegex(ValueError,'earlier source ordinal'):
                model(x,source_ids=('A',),query_ordinals=bad0)
            with self.assertRaisesRegex(ValueError,'owner-bound'):
                model(x,query_ordinals=query)

    def test_historical_one_token_chunk_matches_full(self):
        model=HistoricalWaveCED500KSlow().eval()
        x=torch.randint(0,128,(1,201))
        with torch.no_grad():
            full,h=model.forward_with_history(x,lags=(64,192),source_ids=('A',))
            _,_,st=model.forward_with_history(x[:,:200],lags=(64,192),source_ids=('A',),return_state=True)
            partial,hp=model.forward_with_history(x[:,200:],lags=(64,192),start=200,initial_state=st,source_ids=('A',))
        self.assertTrue(torch.allclose(partial,full[:,200:],atol=2e-4,rtol=2e-4))
        self.assertTrue(torch.allclose(hp[192],h[192][:,-1:],atol=2e-4,rtol=2e-4))

    def test_slow_state_affects_192_token_query(self):
        model=HistoricalWaveCED500KSlow().eval()
        x=torch.randint(0,128,(1,193))
        with torch.no_grad():
            _,_,st=model.forward_with_history(x[:,:128],lags=(64,192),source_ids=('A',),return_state=True)
            zero={k:(v.clone() if isinstance(v,torch.Tensor) else v) for k,v in st.items()}
            zero['aligned'][:,1]=0
            _,h=model.forward_with_history(x[:,128:],lags=(192,),start=128,initial_state=st,source_ids=('A',))
            _,h0=model.forward_with_history(x[:,128:],lags=(192,),start=128,initial_state=zero,source_ids=('A',))
        self.assertGreater(float((h[192]-h0[192]).abs().max()),1e-8)

    def test_history_head_parameter_budget(self):
        self.assertEqual(HistoricalWaveCED500KSlow().parameter_count(),503514)

if __name__=='__main__':
    unittest.main()
