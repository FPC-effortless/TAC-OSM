"""Actual-source CPU boundary, full-graph/reference, credit and provenance tests."""
import copy,random,unittest
import torch
from torch.nn import functional as F
from .common import build,EXPECTED,verify_data
from .data import pools,history,training_batch
from .model import run_chunks
from .reference import whole_reference
from collect011.model import named_state_tensors
class Prefit(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)
    def test_counts_and_all_parameter_credit(self):
        for arm in EXPECTED:
            m=build(arm,16000);self.assertEqual(sum(p.numel() for p in m.parameters()),EXPECTED[arm])
            x=torch.arange(2*41).reshape(2,41)%256;z,_,bal,_=run_chunks(m,x,chunk=17,detach=arm=='detach_128')
            loss=F.cross_entropy(z.flatten(0,1),(x+1).flatten()%256)+.01*bal;loss.backward()
            self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum()>0 for p in m.parameters()))
    def test_source_graph_forward_state_and_surrogate_gradients(self):
        for arm in EXPECTED:
          for dtype in (torch.float32,torch.float64):
            a=build(arm,16000).to(dtype);b=copy.deepcopy(a);x=(torch.arange(2*9).reshape(2,9)*7+13)%256
            mask=torch.ones_like(x,dtype=torch.bool);mask[:,3]=False
            y,s=a(x,write_mask=mask);z,t=whole_reference(b,x,write_mask=mask)
            tol=3e-5 if dtype==torch.float32 else 2e-9
            torch.testing.assert_close(y,z,atol=tol,rtol=tol)
            for k,v in named_state_tensors(s).items():torch.testing.assert_close(v,named_state_tensors(t)[k],atol=tol,rtol=tol)
            y.square().mean().backward();z.square().mean().backward()
            for (n,p),(nn,q) in zip(a.named_parameters(),b.named_parameters()):
                self.assertEqual(n,nn);torch.testing.assert_close(p.grad,q.grad,atol=tol,rtol=tol)
    def test_chunk_values_and_state(self):
        for arm in EXPECTED:
            m=build(arm,16000);x=torch.arange(2*35).reshape(2,35)%256
            with torch.no_grad():
                a,s,_,_=run_chunks(m,x,chunk=35);b,t,_,_=run_chunks(m,x,chunk=11)
            torch.testing.assert_close(a,b,atol=4e-5,rtol=4e-5)
            for k,v in named_state_tensors(s).items():torch.testing.assert_close(v,named_state_tensors(t)[k],atol=4e-5,rtol=4e-5)
    def test_cross_boundary_gradient_is_not_silently_detached(self):
        norms={}
        for detach in (False,True):
            m=build('full_bptt',16000);activations=[]
            def capture(mod,args,out):out.retain_grad();activations.append(out)
            h=m.core.embedding.register_forward_hook(capture)
            x=(torch.arange(2*136).reshape(2,136)*7+13)%256
            z,_,_,_=run_chunks(m,x,chunk=128,detach=detach);z[:,-1].square().sum().backward();h.remove()
            g=activations[0].grad;norms[detach]=0. if g is None else float(g.abs().sum())
        self.assertGreater(norms[False],1e-14);self.assertEqual(norms[True],0.)
    def test_future_poison_and_owner_isolation(self):
        for arm in EXPECTED:
            m=build(arm,16000);x=torch.arange(2*31).reshape(2,31)%256;poison=x.clone();poison[:,19:]=(poison[:,19:]+97)%256
            other=x.clone();other[1]=(other[1]+51)%256
            with torch.no_grad():a,_=m(x);b,_=m(poison);c,_=m(other)
            torch.testing.assert_close(a[:,:19],b[:,:19],atol=0,rtol=0)
            torch.testing.assert_close(a[0],c[0],atol=0,rtol=0)
    def test_training_source_trace_and_task_separation(self):
        data,_=verify_data();pool=pools(data['train']);dpool=pools(data['dev'])
        for step in range(1,17):
            x,y,mask,meta=training_batch(data['train'],pool,seed=16000,step=step)
            self.assertEqual(tuple(x.shape),(2,384));self.assertEqual(mask.sum(),4)
            for f in meta['facts']:
                i=f['source_offset'];self.assertEqual(data['train'][i:i+8].decode('ascii'),f['quote']);self.assertIn(f['key'][0],'abcdefghijklm')
            self.assertEqual(bytes(y[1][mask[1]].tolist()).hex(),meta['answer_hex'])
        h=history(data['dev'],dpool,random.Random(160002),n=4,partition='dev',gap=256)
        self.assertTrue(all(k[0] in 'nopqrstuvwxyz' for k in h['keys']))
    def test_initial_shared_parameters(self):
        a=build('full_bptt',16000);b=build('detach_128',16000);c=build('current_only',16000)
        bd=dict(b.named_parameters());cd=dict(c.named_parameters())
        for n,p in a.named_parameters():
            self.assertTrue(torch.equal(p,bd[n]))
            if n in cd and p.shape==cd[n].shape:self.assertTrue(torch.equal(p,cd[n]))
if __name__=='__main__':unittest.main()
