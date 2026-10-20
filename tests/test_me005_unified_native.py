"""ME005 unit contracts only; NO synthetic scientific accuracy claims."""
import unittest
import torch
from torch.nn import functional as F
from scripts.me005_unified_native_tac import (
    D, UnifiedNativeTAC, EvidenceLedger, sparsemax, load_me004_backbone)

class NativeUnifiedContracts(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(505)
        self.net=UnifiedNativeTAC()
        x=torch.randn(2,10,2*D)
        x[:,:,D:]=(torch.rand(2,10,D)>.25).float()
        self.x=x
    def test_output_preserved_by_zero_residual(self):
        x=self.x.clone()
        x[:,:,:D]*=x[:,:,D:]
        torch.testing.assert_close(self.net(self.x),self.net.backbone(x),atol=2e-6,rtol=2e-6)
    def test_missing_values_do_not_leak(self):
        x=self.x.clone();y=self.net(x)
        x[:,:,:D]=torch.where(x[:,:,D:]>0,x[:,:,:D],torch.full_like(x[:,:,:D],999.))
        self.assertTrue(torch.equal(self.net(x),y))
    def test_bounds_norm_and_stream(self):
        s=self.net.initial(2,self.x.device,self.x.dtype)
        for t in range(self.x.shape[1]):
            s,op,diag=self.net.step(self.x[:,t],s)
            for val in (s.fast,s.slow,s.slots):
                self.assertLessEqual(float(val.detach().abs().max()),1.00001)
            self.assertLessEqual(float(s.assoc.detach().flatten(1).norm(dim=1).max()),2.00001)
            self.assertTrue(torch.allclose(diag['edges'].sum(-1),torch.ones_like(diag['edges'].sum(-1)),atol=1e-5))
        d=self.net(self.x,details=True)
        torch.testing.assert_close(d['state'].fast,s.fast,atol=1e-6,rtol=1e-6)
    def test_gradient_to_internal_after_residual_opens(self):
        with torch.no_grad():self.net.correction.weight.normal_(0,.001)
        loss=F.mse_loss(self.net(self.x),torch.randn(2,2,D))
        loss.backward()
        self.assertGreater(float(self.net.graph.query.weight.grad.abs().max()),0)
        for p in self.net.parameters():
            if p.grad is not None:self.assertTrue(torch.isfinite(p.grad).all())
    def test_role_binding_exact_unbinding(self):
        role=self.net.operator.roles[0]
        x=torch.randn(5,16)
        self.assertTrue(torch.equal(x*role*role,x))
    def test_critic_cannot_authorize_action(self):
        d=self.net(self.x,details=True)
        self.assertIsNone(d['outcome_logit'])
        with self.assertRaises(ValueError):self.net(self.x,recorded_actions=torch.zeros(2,3))
    def test_persistent_memory_receipts(self):
        v=EvidenceLedger(2,receipt_validator=lambda key,context,success,receipt: receipt in {'receipt-one','receipt-two','receipt-three'})
        v.propose('rule',{'proof':'requested'})
        with self.assertRaises(ValueError):v.verify('rule','one',True,None)
        self.assertFalse(v.verify('rule','one',True,'receipt-one'))
        self.assertFalse(v.verify('rule','one',True,'receipt-two'))
        self.assertTrue(v.verify('rule','two',True,'receipt-three'))
        self.assertIn('rule',v.persistent)
        locked=EvidenceLedger(1)
        locked.propose('unverified',{'x':1})
        with self.assertRaises(PermissionError):locked.verify('unverified','context',True,'fake-receipt')
        self.assertNotIn('unverified',locked.persistent)
        replay=EvidenceLedger(2,receipt_validator=lambda *args:True)
        replay.propose('r',{})
        self.assertFalse(replay.verify('r','a',True,'same-receipt'))
        with self.assertRaises(ValueError):replay.verify('r','b',True,'same-receipt')
    def test_energy_refinement_is_contracting_to_fixed_attractor(self):
        h=torch.randn(2,64)
        unit=self.net.reason
        p=F.softmax(unit.q(h)@unit.prototypes.T/8.,dim=-1)
        target=p@unit.prototypes
        refined,steps,_=unit(h,torch.ones(2,1))
        self.assertLessEqual(float(((refined-target)**2).sum().detach()),
                             float(((h-target)**2).sum().detach())+1e-5)
        self.assertTrue(1<=float(steps.detach())<=4)
    def test_sparsemax_simplex(self):
        x=torch.randn(2,5,4)
        p=sparsemax(x)
        torch.testing.assert_close(p.sum(-1),torch.ones(2,5))
        self.assertTrue((p>=0).all())

if __name__=="__main__":
    unittest.main()
