"""Untrained causal/cache/gradient checks; no timing or superiority experiment."""
import unittest
import torch
from tac_osm.decision_sys1_baselines_002 import GRURiskModel,TransformerRiskModel
from tac_osm.decision_sys1_001 import NativeRiskStateModel

torch.set_num_threads(1)

class BaselineTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(2102)
        self.x=torch.randn(2,9,212,dtype=torch.float64)
    def test_01_counts(self):
        counts=[sum(p.numel() for p in C().parameters()) for C in [NativeRiskStateModel,GRURiskModel,TransformerRiskModel]]
        self.assertEqual(counts,[99963,99977,99816]);self.assertLess(max(counts)/min(counts)-1,.01)
    def test_02_gru_cache_parity(self):
        m=GRURiskModel().double().eval();expected,_=m(self.x);state=m.initial(2);actual=[]
        for x in self.x.unbind(1):y,state=m.step(x,state);actual.append(y)
        torch.testing.assert_close(expected,torch.stack(actual,1),rtol=1e-10,atol=1e-10)
    def test_03_transformer_cache_parity(self):
        m=TransformerRiskModel().double().eval();expected=m(self.x);cache=None;actual=[]
        for x in self.x.unbind(1):y,cache=m.step(x,cache);actual.append(y)
        torch.testing.assert_close(expected,torch.stack(actual,1),rtol=1e-10,atol=1e-10)
        self.assertEqual(cache.length,9)
    def test_04_transformer_future_poison(self):
        m=TransformerRiskModel().double().eval();x=self.x.clone();x[:,5:]=1e8
        torch.testing.assert_close(m(self.x)[:,:5],m(x)[:,:5],rtol=1e-10,atol=1e-10)
    def test_05_gru_future_poison(self):
        m=GRURiskModel().double().eval();x=self.x.clone();x[:,5:]=1e8
        torch.testing.assert_close(m(self.x)[0][:,:5],m(x)[0][:,:5],rtol=1e-10,atol=1e-10)
    def test_06_transformer_gradient_parity(self):
        a=TransformerRiskModel().double();b=TransformerRiskModel().double();b.load_state_dict(a.state_dict())
        a(self.x).square().sum().backward();cache=None;ys=[]
        for x in self.x.unbind(1):y,cache=b.step(x,cache);ys.append(y)
        torch.stack(ys,1).square().sum().backward()
        for (n,p),(m,q) in zip(a.named_parameters(),b.named_parameters()):
            self.assertEqual(n,m);torch.testing.assert_close(p.grad,q.grad,rtol=1e-9,atol=1e-9)
    def test_07_all_baseline_parameters_reachable(self):
        for C in [GRURiskModel,TransformerRiskModel]:
            m=C().double();y=m(self.x);y=y[0] if isinstance(y,tuple) else y;y.square().sum().backward()
            for n,p in m.named_parameters():
                self.assertIsNotNone(p.grad,n);self.assertTrue(torch.isfinite(p.grad).all(),n);self.assertGreater(p.grad.abs().sum(),0.,n)
    def test_08_cache_storage_accounted(self):
        m=TransformerRiskModel().float().eval();cache=None
        for x in self.x.float().unbind(1):_,cache=m.step(x,cache)
        count=sum(t.numel()*t.element_size() for pair in cache.layers for t in pair)
        self.assertEqual(count,2*2*2*9*64*4)
        g=GRURiskModel();self.assertEqual(g.initial(1).numel()*4,512)
    def test_09_checkpoint_replay(self):
        for C in [GRURiskModel,TransformerRiskModel]:
            a=C().double().eval();b=C().double().eval();b.load_state_dict(a.state_dict())
            ya,yb=a(self.x),b(self.x)
            if isinstance(ya,tuple):ya,yb=ya[0],yb[0]
            self.assertTrue(torch.equal(ya,yb))
    def test_10_cached_fp32_parity(self):
        m=TransformerRiskModel().eval();x=self.x.float();cache=None;out=[]
        for t in x.unbind(1):y,cache=m.step(t,cache);out.append(y)
        torch.testing.assert_close(m(x),torch.stack(out,1),rtol=2e-5,atol=2e-6)
    def test_11_batch_isolation(self):
        m=TransformerRiskModel().double().eval()
        torch.testing.assert_close(m(self.x)[0],m(self.x[:1])[0],rtol=1e-10,atol=1e-10)
    def test_12_input_gradient_future_is_zero(self):
        m=TransformerRiskModel().double();x=self.x.clone().requires_grad_()
        m(x)[:,:4].sum().backward();self.assertEqual(x.grad[:,4:].abs().sum(),0.)
if __name__=='__main__':unittest.main()
