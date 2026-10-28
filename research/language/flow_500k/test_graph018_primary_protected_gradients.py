"""GRAPH018 constructed gradient-algebra tests. NOT trained predictive evidence."""
import unittest
import torch
from graph018_primary_protected_gradients import primary_guard,protected_backward

class ProtectedMathTests(unittest.TestCase):
    def test_negative_aux_becomes_orthogonal(self):
        main=torch.tensor([2.,1.],dtype=torch.float64)
        aux=torch.tensor([-4.,2.],dtype=torch.float64)
        out,changed=primary_guard(main,aux)
        self.assertTrue(changed)
        self.assertAlmostEqual(float(torch.dot(main,out)),0.,places=12)

    def test_nonconflicting_aux_unchanged(self):
        main=torch.tensor([2.,1.],dtype=torch.float64)
        aux=torch.tensor([3.,-1.],dtype=torch.float64)
        out,changed=primary_guard(main,aux)
        self.assertFalse(changed)
        self.assertTrue(torch.equal(out,aux))

    def test_zero_primary_unchanged(self):
        main=torch.zeros(2,dtype=torch.float64)
        aux=torch.tensor([-1.,3.],dtype=torch.float64)
        out,changed=primary_guard(main,aux)
        self.assertFalse(changed)
        self.assertTrue(torch.equal(out,aux))

    def test_bad_gradients_rejected(self):
        with self.assertRaises(ValueError):
            primary_guard(torch.ones(2),torch.ones(3))
        with self.assertRaises(ValueError):
            primary_guard(torch.ones(2),torch.tensor([float("nan"),1.]))

    def test_adversarial_multi_loss_does_not_oppose_main(self):
        w=torch.nn.Parameter(torch.tensor([1.,2.],dtype=torch.float64))
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__();self.w=w
        m=M()
        l1=(w*w).sum();l4=-(3*w[0]+w[1]);l16=-(w[0]+3*w[1])
        gmain=torch.autograd.grad(l1,w,retain_graph=True)[0]
        receipt=protected_backward(m,{1:l1,4:l4,16:l16},weights={4:.15,16:.15})
        self.assertGreaterEqual(float(torch.dot(gmain,w.grad)),float(gmain.square().sum())-1e-10)
        self.assertEqual(receipt["opposing_component_removed"],{4:1,16:1})

    def test_shared_and_separate_heads_have_finite_gradients(self):
        torch.manual_seed(18)
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.shared=torch.nn.Linear(3,3)
                self.main=torch.nn.Linear(3,2)
                self.h4=torch.nn.Linear(3,2)
                self.h16=torch.nn.Linear(3,2)
            def forward(self,x):
                z=self.shared(x)
                return {1:self.main(z),4:self.h4(z),16:self.h16(z)}
        m=M();outs=m(torch.randn(2,3))
        report=protected_backward(m,{h:v.square().mean() for h,v in outs.items()})
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters()))
        self.assertEqual(report["training"],"NO UPDATE APPLIED")

if __name__=="__main__":
    unittest.main()
