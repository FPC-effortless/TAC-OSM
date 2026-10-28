"""GRAPH014 numerical software tests. Constructed inputs are NOT benchmark data."""
import unittest
import torch
from torch.nn import functional as F
from graph014_selective_ssm_500k import SelectiveDiagonalSSM500K

class NewSSMContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_active_parameters_and_original_tac_band(self):
        m=SelectiveDiagonalSSM500K()
        self.assertEqual(m.parameter_count(),498288)
        self.assertLess(abs(498288-499290)/499290,.01)

    def test_parallel_vs_chronological_state_and_finite_gradient(self):
        torch.manual_seed(144)
        for length in (1,3,17,64,129):
            a=torch.rand(2,length,5,dtype=torch.float64,requires_grad=True)
            b=torch.randn(2,length,5,dtype=torch.float64,requires_grad=True)
            par=SelectiveDiagonalSSM500K.parallel_scan(a,b)
            ser=SelectiveDiagonalSSM500K.serial_scan(a,b)
            self.assertTrue(torch.allclose(par,ser,atol=2e-12,rtol=2e-12))
            gp=torch.autograd.grad(par.square().sum(),(a,b),retain_graph=True)
            gs=torch.autograd.grad(ser.square().sum(),(a,b))
            for p,s in zip(gp,gs):
                self.assertTrue(torch.allclose(p,s,atol=2e-10,rtol=2e-10))

    def test_source_causal_future_poison(self):
        torch.manual_seed(145)
        m=SelectiveDiagonalSSM500K().eval()
        x=torch.randint(0,128,(2,67),dtype=torch.long)
        with torch.no_grad():
            first=m(x)
            changed=x.clone()
            changed[:,37:]=(changed[:,37:]+7)%128
            second=m(changed)
        self.assertTrue(torch.allclose(first[:,:37],second[:,:37],atol=1e-6,rtol=1e-6))

    def test_forward_backprop_and_serial_prefix_equivalence(self):
        torch.manual_seed(146)
        m=SelectiveDiagonalSSM500K()
        x=torch.randint(0,128,(2,17),dtype=torch.long)
        out=m(x);old=m(x,use_serial=True)
        self.assertEqual(tuple(out.shape),(2,17,128))
        self.assertTrue(torch.allclose(out,old,atol=2e-5,rtol=2e-5))
        loss=F.cross_entropy(out.reshape(-1,128),x.reshape(-1))
        loss.backward()
        self.assertEqual(sum(p.numel() for p in m.parameters() if p.grad is not None),498288)
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in m.parameters()))

    def test_malformed_inputs_rejected(self):
        m=SelectiveDiagonalSSM500K()
        with self.assertRaises(ValueError):m(torch.zeros(1,2,dtype=torch.float))
        with self.assertRaises(ValueError):m(torch.full((1,2),128,dtype=torch.long))
        with self.assertRaises(ValueError):m(torch.zeros(1,0,dtype=torch.long))
        with self.assertRaises(ValueError):SelectiveDiagonalSSM500K.parallel_scan(torch.ones(1,2,3),torch.ones(1,2,4))

if __name__=="__main__":
    unittest.main()
