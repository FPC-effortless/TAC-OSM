"""Independent directional finite differences for the original TAC language model.

Unlike the existing Torch-vs-Torch autograd checks, these compare full-model
autograd to a two-sided *function-value* difference. This is a numerical
diagnostic over constructed tokens, not a proof of universal AD correctness,
and it is not evidence of real-data language quality.
"""
from __future__ import annotations
import math
import unittest
import torch
import torch.nn.functional as F
from parallel_tac_language_gate_original import ParallelTACLanguage

class LanguageNumericalDerivativeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def build(self):
        torch.manual_seed(20261027)
        m=ParallelTACLanguage(decoder=75,post=10,position_mode="multiscale",
                              gate_mode="silu", optimized_scan=True)
        return m.double().eval()

    @staticmethod
    def objective(model, tokens, labels, lag):
        logits,trace=model(tokens,return_trace=True)
        history,targets=model.historical_logits(tokens,trace,lag)
        return F.cross_entropy(logits.reshape(-1,128),labels.reshape(-1)) + (
            0.3*F.cross_entropy(history.reshape(-1,128),targets.reshape(-1)))

    def test_full_graph_finite_difference_for_each_of_36_tensor_groups(self):
        m=self.build()
        rng=torch.Generator().manual_seed(88)
        tokens=torch.randint(128,(1,9),generator=rng,dtype=torch.long)
        labels=torch.randint(128,(1,9),generator=rng,dtype=torch.long)
        lag=2
        variables=list(m.named_parameters())
        self.assertEqual(len(variables),36)
        loss=self.objective(m,tokens,labels,lag)
        gradients=torch.autograd.grad(loss,[p for _,p in variables],allow_unused=True)
        step=2e-5
        with torch.no_grad():
            for (name,p),g in zip(variables,gradients):
                if g is None:
                    continue
                # Exact one-tensor direction isolates this named learned group.
                direction=torch.randn(p.shape,generator=rng,dtype=p.dtype)
                direction=direction/direction.norm().clamp_min(1e-12)
                p.add_(step*direction)
                plus=float(self.objective(m,tokens,labels,lag))
                p.add_(-2*step*direction)
                minus=float(self.objective(m,tokens,labels,lag))
                p.add_(step*direction)
                estimated=(plus-minus)/(2*step)
                analytic=float((g*direction).sum())
                self.assertTrue(math.isfinite(estimated),name)
                self.assertLess(abs(estimated-analytic),
                                2e-5+0.003*abs(analytic),
                                f"{name}: finite difference={estimated}, AD={analytic}")

    def test_real_lag64_objective_finite_difference(self):
        # Training used a 64-step old-character objective. This tests its
        # full 65-step differentiable path without retraining.
        m=self.build()
        rng=torch.Generator().manual_seed(3991)
        tokens=torch.randint(128,(1,65),generator=rng,dtype=torch.long)
        labels=torch.randint(128,(1,65),generator=rng,dtype=torch.long)
        loss=self.objective(m,tokens,labels,64)
        p=m.source_silu_mod.weight
        g=torch.autograd.grad(loss,p)[0]
        direction=torch.randn(p.shape,generator=rng,dtype=p.dtype)
        direction=direction/direction.norm()
        step=1e-5
        with torch.no_grad():
            p.add_(step*direction)
            plus=float(self.objective(m,tokens,labels,64))
            p.add_(-2*step*direction)
            minus=float(self.objective(m,tokens,labels,64))
            p.add_(step*direction)
        estimated=(plus-minus)/(2*step)
        analytic=float((g*direction).sum())
        self.assertLess(abs(estimated-analytic),2e-5+0.003*abs(analytic))

if __name__=="__main__":
    unittest.main()
