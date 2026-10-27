"""Synthetic unit diagnostics; never primary performance evidence."""
import unittest
import torch
import torch.nn.functional as F
from gradient_guard import gradient_guard_backward, source_shared
from parallel_tac_language_gate_original import ParallelTACLanguage

class GradientGuardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_applied_projection_is_first_order_nonantagonistic(self):
        torch.manual_seed(192)
        net=ParallelTACLanguage(decoder=75,position_mode="multiscale",gate_mode="silu")
        opt=torch.optim.SGD(net.parameters(),lr=.001)
        x=torch.randint(0,128,(2,128));y=torch.roll(x,-1,1)
        out,trace=net(x,return_trace=True)
        a,t=net.historical_logits(x,trace,64)
        b,v=net.historical_logits(x,trace,96)
        main=F.cross_entropy(out.reshape(-1,128),y.reshape(-1))
        hist=.15*F.cross_entropy(a.reshape(-1,128),t.reshape(-1))+.15*F.cross_entropy(b.reshape(-1,128),v.reshape(-1))
        selected=[p for _,p in source_shared(net)]
        main_grads=torch.autograd.grad(main,selected,retain_graph=True)
        diagnostic=gradient_guard_backward(net,main,hist,opt)
        local_dot=sum((g.double()*p.grad.double()).sum().item()
                      for g,p in zip(main_grads,selected))
        self.assertGreaterEqual(local_dot,-1e-10)
        if diagnostic["raw_dot"]<0:
            self.assertTrue(diagnostic["projected"])
        self.assertEqual(sum(p.numel() for p in net.parameters()),100680)
        torch.nn.utils.clip_grad_norm_(net.parameters(),1.)
        opt.step()
        self.assertTrue(all(torch.isfinite(p).all() for p in net.parameters()))

    def test_source_encoders_are_only_modified_subset(self):
        net=ParallelTACLanguage(decoder=75,position_mode="multiscale",gate_mode="silu")
        names=[n for n,p in source_shared(net)]
        self.assertIn("rotation",names)
        self.assertNotIn("hist_project.weight",names)
        self.assertNotIn("native_gates.weight",names)
        self.assertNotIn("embedding.weight",names)

if __name__=="__main__":
    unittest.main()
