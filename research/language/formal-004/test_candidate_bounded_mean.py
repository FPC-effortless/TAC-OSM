"""Constructed FP32 numerical diagnostics for competing 88-state accumulators.

Not language quality evidence. Crucially, test proves proposed online mean
is NOT a universal numerical improvement; it improves selected long-prefix
rounding patterns and can lose to source FP32 sum/count on others.
"""
import unittest
import torch
from candidate_bounded_mean import chronological_mean

class BoundedMeanDiagnostics(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_exact_real_formula_and_same_88_channel_count(self):
        for n in (1,17,513):
            g=torch.Generator().manual_seed(n)
            basis=(torch.rand(n,88,generator=g,dtype=torch.float64)*2-1)
            s=chronological_mean(basis,"source_sum_divide")
            m=chronological_mean(basis,"online_mean")
            self.assertEqual(m.shape,(88,))
            self.assertLess(float((s-m).abs().max()),5e-13)
            self.assertLessEqual(float(m.abs().max()),1+1e-12)

    def test_fp32_mean_can_improve_but_not_universal(self):
        n=4096
        late=torch.ones(n,dtype=torch.float32)
        late[n//2:]=1e-5
        alternating=torch.tensor([1.0,-.99999]*(n//2),dtype=torch.float32)
        testcases=[("late-small",late),("alternating",alternating)]
        for name,values in testcases:
            data=values[:,None].expand(-1,88)
            exact=values.double().mean()
            source=chronological_mean(data,"source_sum_divide")[0].double()
            proposal=chronological_mean(data,"online_mean")[0].double()
            err0=float(abs(source-exact))
            err1=float(abs(proposal-exact))
            if name=="late-small":
                self.assertLess(err1,err0)
            else:
                self.assertLess(err0,err1)
            self.assertLess(max(err0,err1),2e-5)

    def test_fail_closed(self):
        with self.assertRaisesRegex(ValueError,"FP32/FP64"):
            chronological_mean(torch.ones((3,88),dtype=torch.float16),"online_mean")
        with self.assertRaisesRegex(ValueError,"finite"):
            chronological_mean(torch.tensor([[float("nan")]]),"online_mean")
        with self.assertRaisesRegex(ValueError,"mode"):
            chronological_mean(torch.ones((3,88)),"wrong")

if __name__=="__main__":
    unittest.main()
