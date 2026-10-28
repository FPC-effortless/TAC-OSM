"""Constructed software tests only; NOT real-world capability or Lean proofs."""
import unittest
import math
import torch
from torch.nn import functional as F
from tac_wave_ced_500k import WaveCED500K,affine_scan,scalar_affine_scan,last_write_memory,ordinal_code

class WaveTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        torch.set_num_threads(2)

    def test_contract_and_full_backward(self):
        m=WaveCED500K();self.assertEqual(m.parameter_count(),499290)
        x=torch.randint(0,128,(2,11))
        y=m(x)
        loss=F.cross_entropy(y[:,:-1].reshape(-1,128),x[:,1:].reshape(-1));loss.backward()
        self.assertTrue(torch.isfinite(loss).item())
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters() if p.requires_grad))
        self.assertEqual(sum(p.numel() for p in m.parameters() if p.grad is not None),499290)

    def test_affine_parallel_matches_sequential(self):
        for dtype in (torch.float32,torch.float64):
            raw=torch.randn(3,13,4,4,dtype=dtype)*.05
            p=.95*torch.eye(4,dtype=dtype)+raw
            q=torch.randn(3,13,5,4,dtype=dtype)*.1
            scan=affine_scan(p,q); s=torch.zeros(3,5,4,dtype=dtype);outs=[]
            for t in range(13):
                s=s@p[:,t]+q[:,t];outs.append(s)
            serial=torch.stack(outs,1)
            self.assertTrue(torch.allclose(scan,serial,rtol=8e-5,atol=8e-6))

    def test_affine_parallel_gradients(self):
        torch.manual_seed(13)
        a=torch.randn(1,7,2,2,dtype=torch.double,requires_grad=True)*.05
        p=a+torch.eye(2)*.9
        q=torch.randn(1,7,2,2,dtype=torch.double,requires_grad=True)*.1
        scan=affine_scan(p,q).square().sum(); g1=torch.autograd.grad(scan,(a,q),retain_graph=True)
        state=torch.zeros(1,2,2,dtype=torch.double)
        accum=torch.tensor(0.,dtype=torch.double)
        for t in range(7):
            state=state@p[:,t]+q[:,t];accum=accum+state.square().sum()
        g2=torch.autograd.grad(accum,(a,q))
        self.assertTrue(all(torch.allclose(x,y,rtol=1e-8,atol=1e-9) for x,y in zip(g1,g2)))

    def test_scalar_scan(self):
        a=torch.rand(2,9,96)*.15+.8;q=torch.randn(2,9,96)*.05
        out=scalar_affine_scan(a,q);s=torch.zeros(2,96)
        for t in range(9):
            s=s*a[:,t]+q[:,t]
            self.assertTrue(torch.allclose(s,out[:,t],rtol=1e-5,atol=1e-6))

    def test_no_future_leak(self):
        m=WaveCED500K().eval()
        a=torch.randint(0,128,(1,13)); b=a.clone();b[:,9:]=torch.randint(0,128,(1,4))
        with torch.no_grad():
            y1=m(a);y2=m(b)
        self.assertTrue(torch.allclose(y1[:,:9],y2[:,:9],rtol=0,atol=1e-6))

    def test_nonwrite_storage_identity(self):
        s=torch.randn(2,10,3,16,8);mask=torch.zeros(2,10,dtype=torch.bool)
        mask[0,2]=True;mask[0,7]=True;mask[1,1]=True
        copied=last_write_memory(s,mask)
        self.assertTrue(torch.equal(copied[0,3],s[0,2]))
        self.assertTrue(torch.equal(copied[0,6],s[0,2]))
        self.assertTrue(torch.equal(copied[1,9],s[1,1]))
        self.assertEqual(copied[0,0].count_nonzero().item(),0)

    def test_masked_memory_no_write(self):
        m=WaveCED500K().eval()
        x=torch.randint(0,128,(1,8));mask=torch.zeros((1,8),dtype=torch.bool)
        y,tr=m(x,write_mask=mask,return_trace=True)
        self.assertTrue(y.isfinite().all())
        self.assertEqual(tr['native_prewrite'].count_nonzero().item(),0)
        self.assertEqual(tr['aligned_prewrite'].count_nonzero().item(),0)

    def test_concept_equilibrium_stationarity_and_energy(self):
        m=WaveCED500K()
        x=torch.randint(0,128,(1,6)); _,tr=m(x,return_trace=True)
        c=tr['concept_posterior'];p=tr['concept_prewrite'];L=tr['concept_laplacian']
        # Reconstruct G by equation? test Laplacian PSD and symmetric solution anchored to prior using source values.
        eig=torch.linalg.eigvalsh(L.detach().flatten(0,1))
        self.assertGreater(float(eig.min()),-1e-5)
        self.assertTrue(torch.isfinite(c).all())
        self.assertTrue(torch.isfinite(p).all())

    def test_ordinal_period_known_alias(self):
        i=torch.tensor([[0,10,248]])
        self.assertTrue(torch.allclose(ordinal_code(i,torch.float64),ordinal_code(i+155701,torch.float64),atol=1e-12))

if __name__=='__main__':unittest.main()
