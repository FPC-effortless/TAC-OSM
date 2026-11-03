import copy,json,unittest
import torch
from .common import build,EXPECTED,verify_data,require,ROOT
from .model import run_chunks,tensors
from .reference import whole_reference
from .data import pools,training_batch

class Prefit(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)
    def test_actual_whole_source_reference_and_backward(self):
        for family in ('phase','context','gru'):
          for dtype in (torch.float32,torch.float64):
            with self.subTest(family=family,dtype=dtype):
                a=build(family+'_first',18000).to(dtype);b=copy.deepcopy(a)
                x=torch.arange(18).reshape(2,9)%256;mask=torch.ones_like(x,dtype=torch.bool)
                if family!='gru':mask[:,3]=False
                y,s=a(x,write_mask=mask);z,t=whole_reference(b,x,write_mask=mask)
                tol=4e-5 if dtype==torch.float32 else 3e-9
                torch.testing.assert_close(y,z,atol=tol,rtol=tol)
                for n,v in tensors(s).items():torch.testing.assert_close(v,tensors(t)[n],atol=tol,rtol=tol)
                y.square().sum().backward();z.square().sum().backward()
                for (n,p),(nn,q) in zip(a.named_parameters(),b.named_parameters()):
                    self.assertEqual(n,nn);self.assertIsNotNone(p.grad,n);torch.testing.assert_close(p.grad,q.grad,atol=tol*10,rtol=tol*10)
                self.assertEqual(sum(p.numel() for p in a.parameters()),EXPECTED[family])
    def test_causal_owner_chunk_and_activation_credit(self):
        for family in ('phase','context','gru'):
            m=build(family+'_joint',18000);x=torch.arange(74).reshape(2,37)%256;future=x.clone();future[:,19:]=(future[:,19:]+61)%256;other=x.clone();other[1]=(other[1]+13)%256
            with torch.no_grad():
                y,s=m(x);z,_=m(future);q,_=m(other);torch.testing.assert_close(y[:,:19],z[:,:19],atol=0,rtol=0);torch.testing.assert_close(y[0],q[0],atol=0,rtol=0)
                zz,ss,_,_=run_chunks(m,x,chunk=8);torch.testing.assert_close(y,zz,atol=4e-5,rtol=4e-5)
                for n,v in tensors(s).items():torch.testing.assert_close(v,tensors(ss)[n],atol=4e-5,rtol=4e-5)
                with self.assertRaises(ValueError):m(x[:,:2],state=s,owners=tuple(reversed(s.owners)))
            captures=[]
            def hook(module,args,out):out.retain_grad();captures.append(out)
            h=m.core.embedding.register_forward_hook(hook);z,_,_,_=run_chunks(m,x,chunk=8);z[1,-1].square().sum().backward();h.remove()
            self.assertGreater(float(captures[0].grad[1].abs().sum()),0.)
            self.assertEqual(float(captures[0].grad[0].abs().sum()),0.)
    def test_actual_episode_answer_poison(self):
        data,_=verify_data();x,y,mask,first,meta=training_batch(data['train'],pools(data['train']),seed=18000,step=1)
        changed=x.clone()
        for row,r in enumerate(meta['records'],1):changed[row,r['first_position']+1:]=(changed[row,r['first_position']+1:]+73)%256
        for family in ('phase','context','gru'):
            m=build(family+'_first',18000)
            with torch.no_grad():a,_=m(x);b,_=m(changed)
            for row,r in enumerate(meta['records'],1):torch.testing.assert_close(a[row,:r['first_position']+1],b[row,:r['first_position']+1],atol=0,rtol=0)
    def test_paired_initialization_and_target_boundaries(self):
        a=build('phase_joint',18000);b=build('context_first',18000)
        for (n,p),(nn,q) in zip(a.named_parameters(),b.named_parameters()):self.assertEqual(n,nn);self.assertTrue(torch.equal(p,q))
        data,_=verify_data();pool=pools(data['train']);seen=set()
        for step in (1,255,256,257,511,512,513,1024):
            x,y,mask,first,meta=training_batch(data['train'],pool,seed=18000,step=step)
            self.assertEqual(tuple(x.shape),(4,128));self.assertEqual(int(mask.sum()),12);self.assertEqual(int(first.sum()),3)
            for row,r in enumerate(meta['records'],1):
                p=r['first_position'];seen.add(p);self.assertTrue(96<=p<=123);self.assertEqual(bytes(x[row,p-8:p+1].tolist()),b'Copy '+r['key'].encode()+b': ')
                self.assertEqual(y[row,p].item(),bytes.fromhex(r['answer_hex'])[0]);self.assertEqual(x[row,p].item(),32)
                self.assertEqual(tuple(y[row,p:p+4].tolist()),tuple(bytes.fromhex(r['answer_hex'])))
        self.assertGreater(len(seen),1)
if __name__=='__main__':unittest.main()
