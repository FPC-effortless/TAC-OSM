"""ME006 math/software only — no synthetic scientific claim."""
import unittest
import torch
from torch import nn
from scripts.me006_causal_relevance import CausalRelevanceAdapter,future_gain

class FrozenDummy(nn.Module):
    def forward(self,x):
        return x[:,-1,:12].unsqueeze(1).expand(-1,2,-1)*.04

class CausalRelevanceContracts(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1);torch.manual_seed(20261020)
        self.x=torch.randn(3,8,24)
        self.x[:,:,12:]=(torch.rand(3,8,12)>.3).float()

    def test_zero_residual_preserves_frozen_pretrained_output(self):
        for mode in ("direct","associative"):
            with self.subTest(mode=mode):
                model=CausalRelevanceAdapter(FrozenDummy(),mode)
                out=model(self.x,True)
                self.assertTrue(torch.equal(out["forecast"],out["baseline"]))

    def test_mask_does_not_leak_unobserved_values(self):
        model=CausalRelevanceAdapter(FrozenDummy())
        base=model(self.x)
        x=self.x.clone()
        x[:,:,:12]=torch.where(x[:,:,12:]>0,x[:,:,:12],1e10)
        self.assertTrue(torch.equal(base,model(x)))

    def test_memory_norm_and_causal_history_sensitivity(self):
        model=CausalRelevanceAdapter(FrozenDummy())
        output=model(self.x,True)
        self.assertLessEqual(float(output["fast_state"].detach().flatten(1).norm(dim=-1).max()),2.001)
        newer=self.x.clone();newer[:,-1,:12]+=1.3
        self.assertFalse(torch.equal(output["fast_state"],model(newer,True)["fast_state"]))

    def test_gradients_after_residual_unfreezing(self):
        model=CausalRelevanceAdapter(FrozenDummy());target=torch.randn(3,2,12)
        (model(self.x)-target).square().mean().backward()
        with torch.no_grad():model.decode.weight.add_(-.3*model.decode.weight.grad)
        model.zero_grad(set_to_none=True)
        (model(self.x)-target).square().mean().backward()
        self.assertGreater(float(model.encode.weight.grad.abs().max()),0)

    def test_future_labels_never_part_of_inference_arguments(self):
        model=CausalRelevanceAdapter(FrozenDummy())
        d=model(self.x,True);target=torch.randn_like(d["forecast"])
        self.assertAlmostEqual(float(future_gain(d["forecast"],target,d["baseline"])),0,places=6)
        with self.assertRaises(ValueError):
            future_gain(d["forecast"],target[:,:1],d["baseline"])

if __name__=="__main__":unittest.main(verbosity=2)
