"""GRAPH015 constructed numerical regression tests only; no learned skill evidence."""
import unittest
import torch
from torch.nn import functional as F
from graph015_delta_tac_500k import Graph015DeltaTAC500K,affine_scan

class DeltaContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_active_count(self):
        m=Graph015DeltaTAC500K()
        self.assertEqual(m.parameter_count(),499290)

    def test_every_learned_coordinate_attached(self):
        torch.manual_seed(15);m=Graph015DeltaTAC500K();x=torch.randint(0,128,(2,9))
        p=m(x);F.cross_entropy(p.reshape(-1,128),x.reshape(-1)).backward()
        missing=[name for name,param in m.named_parameters() if param.grad is None]
        self.assertFalse(missing,missing)
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in m.parameters()))

    def test_registered_gate_bounds(self):
        m=Graph015DeltaTAC500K();x=torch.randint(0,128,(2,12))
        _,tr=m(x,return_trace=True)
        rho,beta=tr['delta_rho'],tr['delta_beta']
        self.assertTrue(bool(((rho>=.994999)&(rho<=.999001)).all()))
        self.assertTrue(bool(((beta>0)&(beta<=rho)).all()))

    def test_prewrite_no_future_access(self):
        torch.manual_seed(15);m=Graph015DeltaTAC500K().eval();x=torch.randint(0,128,(1,16))
        with torch.no_grad():
            a=m(x);poison=x.clone();poison[:,8:]=(poison[:,8:]+25)%128;b=m(poison)
        self.assertTrue(torch.allclose(a[:,:8],b[:,:8],atol=5e-5,rtol=5e-5))

    def test_conditional_spectral_bound(self):
        torch.manual_seed(7)
        k=F.normalize(torch.randn(4,8,dtype=torch.float64),dim=-1)
        rho=torch.tensor([.995,.996,.998,.999],dtype=k.dtype)
        beta=rho*torch.tensor([.05,.3,.6,.95],dtype=k.dtype)
        P=rho[:,None,None]*torch.eye(8,dtype=k.dtype)-beta[:,None,None]*k[:,:,None]*k[:,None,:]
        self.assertTrue(bool((torch.linalg.matrix_norm(P,ord=2)<=rho+1e-12).all()))
        self.assertTrue(bool((torch.linalg.eigvalsh(P).min(-1).values>=-1e-12).all()))

    def test_corrective_update_reduces_keyed_residual_no_decay(self):
        torch.manual_seed(18)
        M=torch.randn(8,8,dtype=torch.float64)
        key=F.normalize(torch.randn(8,dtype=torch.float64),dim=0)
        value=torch.randn(8,dtype=torch.float64)
        beta=.7
        before=(value-M@key).norm()
        updated=M+beta*(value-M@key)[:,None]*key[None,:]
        after=(value-updated@key).norm()
        self.assertTrue(torch.allclose(after/before,torch.tensor(1-beta,dtype=M.dtype),atol=1e-12))

    def test_parallel_scan_gradient_matches_serial_for_delta(self):
        torch.manual_seed(7)
        B,T,D,R=2,8,8,16
        k=F.normalize(torch.randn(B,T,D,dtype=torch.float64),dim=-1)
        v=torch.randn(B,T,R,dtype=torch.float64,requires_grad=True)
        rho=torch.full((B,T),.997,dtype=torch.float64)
        beta=torch.full((B,T),.35,dtype=torch.float64)
        eye=torch.eye(D,dtype=torch.float64)
        P=rho[...,None,None]*eye-beta[...,None,None]*k.unsqueeze(-1)*k.unsqueeze(-2)
        Q=beta[...,None,None]*v.unsqueeze(-1)*k.unsqueeze(-2)
        par=affine_scan(P,Q)
        M=torch.zeros(B,R,D,dtype=torch.float64)
        out=[]
        for t in range(T):
            M=M@P[:,t]+Q[:,t];out.append(M)
        serial=torch.stack(out,dim=1)
        self.assertTrue(torch.allclose(par,serial,atol=3e-12,rtol=3e-12))
        gp=torch.autograd.grad(par.square().sum(),v,retain_graph=True)[0]
        gs=torch.autograd.grad(serial.square().sum(),v)[0]
        self.assertTrue(torch.allclose(gp,gs,atol=2e-10,rtol=2e-10))

if __name__=="__main__": unittest.main()
