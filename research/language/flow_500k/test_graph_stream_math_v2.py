"""TAC-LANG-GRAPH-006 regression gates. CONSTRUCTED operator tests only.
No real language accuracy or original checkpoint performance claims.
"""
import unittest
import torch
import torch.nn.functional as F
from tac_wave_ced_500k_v2 import WaveCED500K
from historical_500k_v2 import HistoricalWaveCED500K

class GraphStreamingRepairTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)
    def setUp(self):torch.manual_seed(1806)

    def test_active_concept_graph_changes_logits_and_gradient(self):
        m=WaveCED500K().double()
        x=torch.randint(0,128,(2,11),dtype=torch.long)
        graph,trace=m(x,return_trace=True,graph_scale=1.0)
        severed=m(x,graph_scale=0.0)
        effect=float((graph-severed).abs().max())
        self.assertGreater(effect,1e-8,"graph still cancels out of logits")
        self.assertTrue(bool(torch.isfinite(graph).all()))
        loss=graph.square().mean()
        g=torch.autograd.grad(loss,(m.concept_q.weight,m.concept_k.weight))
        for grad in g:
            self.assertTrue(bool(torch.isfinite(grad).all()))
            self.assertGreater(float(grad.norm()),1e-12,
              "finite but functionally zero graph gradient cannot pass")
        # Slot specificity, not a disguised arithmetic mean
        weights=trace["concept_read_weights"]
        self.assertEqual(weights.shape,(2,11,6))
        self.assertGreater(float((weights-1/6).abs().max()),.05)
        self.assertTrue(torch.allclose(weights.sum(-1),torch.ones(2,11,dtype=weights.dtype),atol=1e-12))
        self.assertEqual(m.parameter_count(),499290)

    def test_query_and_key_perturbations_both_change_logit(self):
        m=WaveCED500K().double()
        x=torch.randint(0,128,(1,13))
        original=m(x).detach()
        for name in ("concept_q","concept_k"):
            with torch.no_grad():
                p=getattr(m,name).weight
                p[0,0]+=1.0
            altered=m(x).detach()
            self.assertGreater(float((original-altered).abs().max()),1e-10,
                              name+" is functionally inactive")
            with torch.no_grad():getattr(m,name).weight[0,0]-=1.0

    def test_carried_one_token_historical_query_uses_global_ordinals(self):
        m=HistoricalWaveCED500K().double().eval()
        x=torch.randint(0,128,(1,201))
        with torch.no_grad():
            full,history,position=m.forward_with_history(
                x,lags=(64,192),return_positions=True)
            _,_,state=m.forward_with_history(x[:,:200],lags=(64,192),return_state=True)
            resumed,h2,pos2,state2=m.forward_with_history(
                x[:,200:],lags=(64,192),start=200,initial_state=state,
                return_state=True,return_positions=True)
        self.assertEqual(tuple(resumed.shape),(1,1,128))
        for lag in (64,192):
            self.assertEqual(tuple(h2[lag].shape),(1,1,128))
            self.assertEqual(pos2[lag]["current_ordinals"],(200,))
            self.assertEqual(pos2[lag]["target_ordinals"],(200-lag,))
            self.assertEqual(pos2[lag]["local_start"],0)
            self.assertTrue(torch.allclose(h2[lag],history[lag][:,-1:],atol=2e-7,rtol=2e-7))
        self.assertEqual(state2["count"],201)

    def test_chunk_boundary_valid_only_suffix_and_empty_without_error(self):
        m=HistoricalWaveCED500K().eval()
        x=torch.randint(0,128,(1,206))
        with torch.no_grad():
            _,state=m.forward(x[:,:185],return_state=True)
            y,h,p=m.forward_with_history(x[:,185:206],lags=(192,256),
                 start=185,initial_state=state,return_positions=True)
        self.assertEqual(y.shape,(1,21,128))
        self.assertEqual(h[192].shape,(1,14,128))
        self.assertEqual(p[192]["current_ordinals"],tuple(range(192,206)))
        self.assertEqual(p[192]["target_ordinals"],tuple(range(14)))
        self.assertEqual(h[256].shape,(1,0,128))
        self.assertEqual(p[256]["local_start"],21)

    def test_future_queries_and_carried_state_fail_closed(self):
        m=HistoricalWaveCED500K()
        x=torch.randint(0,128,(1,5))
        for bad in ((0,),(-1,),("64",)):
            with self.assertRaises(ValueError):
                m.forward_with_history(x,lags=bad)
        with self.assertRaises(ValueError):
            m.forward_with_history(x,lags=(2,),key_shift=2)
        with self.assertRaises(ValueError):
            m.forward_with_history(x,lags=(2,),start=200)
        with self.assertRaises(ValueError):
            m.forward(x,start=-1)

    def test_write_coupling_preserves_uniform_state_bound_under_chunks(self):
        m=WaveCED500K().eval()
        state=None
        with torch.no_grad():
            for i in range(20):
                x=torch.randint(0,128,(2,16))
                _,trace,state=m.forward(x,start=i*16,initial_state=state,
                                        return_trace=True,return_state=True)
                a=trace["gate_alpha"];w=trace["gate_beta"]
                self.assertTrue(bool(torch.all(w<=1-a+1e-7)))
                self.assertTrue(bool(torch.all(w>=0)))
                self.assertLessEqual(float(state["native"].norm(dim=(-2,-1)).max()),(8.**.5)+1e-3)
                self.assertLessEqual(float(state["aligned"].norm(dim=(-2,-1)).max()),4+1e-3)
        self.assertEqual(state["count"],320)

if __name__=="__main__":unittest.main()
