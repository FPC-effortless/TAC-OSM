import copy,unittest
import torch
from .common import build
from .model import run_chunks
from collect016.reference import whole_reference
from collect011.model import named_state_tensors
class Prefit(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)
    def test_count_and_all_parameter_credit(self):
        m=build('attention',16000);self.assertEqual(sum(p.numel() for p in m.parameters()),99875)
        x=torch.arange(2*41).reshape(2,41)%256;z,_,bal,counts=run_chunks(m,x,chunk=17)
        self.assertEqual(float(bal),0.);self.assertEqual(float(counts.sum()),0.)
        z.square().mean().backward();self.assertTrue(all(p.grad is not None and p.grad.abs().sum()>0 and torch.isfinite(p.grad).all() for p in m.parameters()))
    def test_full_graph_reference_and_gradients(self):
        for dtype in (torch.float32,torch.float64):
            m=build('attention',16000).to(dtype);n=copy.deepcopy(m);x=torch.arange(18).reshape(2,9)%256
            a,s=m(x);b,t=whole_reference(n,x);tol=3e-5 if dtype==torch.float32 else 2e-9
            torch.testing.assert_close(a,b,atol=tol,rtol=tol)
            for k,v in named_state_tensors(s).items():torch.testing.assert_close(v,named_state_tensors(t)[k],atol=tol,rtol=tol)
            a.square().mean().backward();b.square().mean().backward()
            for (k,v),(kk,w) in zip(m.named_parameters(),n.named_parameters()):
                self.assertEqual(k,kk);torch.testing.assert_close(v.grad,w.grad,atol=tol,rtol=tol)
    def test_connected_chunk_and_credit(self):
        m=build('attention',16000);x=torch.arange(2*136).reshape(2,136)%256
        captures=[]
        def hook(module,args,out):out.retain_grad();captures.append(out)
        h=m.core.embedding.register_forward_hook(hook);z,s,_,_=run_chunks(m,x);z[:,-1].square().sum().backward();h.remove()
        self.assertGreater(float(captures[0].grad.abs().sum()),1e-14)
        with torch.no_grad():a,ss=m(x);torch.testing.assert_close(a,z,atol=4e-5,rtol=4e-5)
        self.assertEqual(tuple(s.native.layers[0].recent_key.shape),(2,4,136,16))
        self.assertEqual(len(s.phase),0)
    def test_future_and_owner(self):
        m=build('attention',16000);x=torch.arange(62).reshape(2,31)%256;p=x.clone();p[:,19:]=(p[:,19:]+97)%256;q=x.clone();q[1]=(q[1]+49)%256
        with torch.no_grad():a,_=m(x);b,_=m(p);c,_=m(q)
        torch.testing.assert_close(a[:,:19],b[:,:19],atol=0,rtol=0);torch.testing.assert_close(a[0],c[0],atol=0,rtol=0)
if __name__=='__main__':unittest.main()
