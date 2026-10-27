"""CPU exact-rational tests for the actual source 8-key and 248-head dot shapes.

These are arithmetic unit diagnostics only. Nothing here is pretrained
language performance or a proof of every finite-precision kernel/device.
"""
import unittest
import torch
import torch.nn.functional as F
from audit_tensor_rounding import audit_float32_dot,exact_rational_dot
from parallel_tac_language_gate_original import ParallelTACLanguage
from source_independent_reference import phi

class LanguageTensorRoundingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_exact_rational_dot_and_adversarial_cancellation(self):
        # 16,777,216 + 1 - 16,777,216 is exactly one in rationals;
        # different FP32 reduction trees may not produce that value.
        a=torch.tensor([16777216.,1.,-16777216.])
        b=torch.ones_like(a)
        exact,_=exact_rational_dot(a,b)
        self.assertEqual(exact,1)
        for operation in ("matmul","einsum","rounded_product_sum"):
            receipt=audit_float32_dot(a,b,operation)
            self.assertTrue(receipt["within_candidate_budget"],receipt)

    def test_kernel_shape_8_16_48_88_128_248(self):
        rng=torch.Generator().manual_seed(202614)
        for n in (8,16,48,88,128,248):
            for scale in (1e-3,1.,1e3):
                x=torch.randn((n,),generator=rng,dtype=torch.float32)*scale
                y=torch.randn((n,),generator=rng,dtype=torch.float32)/scale
                for operation in ("matmul","einsum","rounded_product_sum"):
                    receipt=audit_float32_dot(x,y,operation)
                    self.assertTrue(receipt["within_candidate_budget"],
                                    f"n={n}, scale={scale}, {receipt}")

    def test_actual_source_address_and_decoder_weights(self):
        torch.manual_seed(99)
        model=ParallelTACLanguage(decoder=75,post=10,
            gate_mode="silu",position_mode="multiscale").float().eval()
        with torch.no_grad():
            address=phi(17,torch.float32,torch.device("cpu"),1)[0]
            learned_key=F.normalize(address@model.rotation[0],dim=0,eps=1e-6)
            memory=torch.randn((16,8),generator=torch.Generator().manual_seed(32))
            for row in memory:
                receipt=audit_float32_dot(row,learned_key,"einsum")
                self.assertTrue(receipt["within_candidate_budget"],receipt)
            # Real 248-dimensional language decoder weight rows.
            features=torch.randn(248,generator=torch.Generator().manual_seed(33))
            for i in (0,1,37,74):
                receipt=audit_float32_dot(
                    model.read_in.weight[i],features,"matmul")
                self.assertTrue(receipt["within_candidate_budget"],receipt)

    def test_normal_and_subnormal_diagnostic_not_theorem(self):
        x=torch.full((8,),1e-20,dtype=torch.float32)
        y=x.clone()
        receipts=[audit_float32_dot(x,y,mode)
            for mode in ("matmul","einsum","rounded_product_sum")]
        for receipt in receipts:
            self.assertGreaterEqual(receipt["actual_absolute_error"],0)
            # The observation is kept even if the backend flushes
            # subnormals and DOES NOT satisfy the candidate budget.

    def test_reject_finite_precision_misuse(self):
        with self.assertRaisesRegex(ValueError,"float32"):
            exact_rational_dot(torch.ones(8,dtype=torch.float64),
                               torch.ones(8,dtype=torch.float64))
        with self.assertRaisesRegex(ValueError,"nonfinite"):
            audit_float32_dot(torch.tensor([float("nan"),1.]),
                              torch.ones(2))
        with self.assertRaisesRegex(ValueError,"equal nonempty"):
            audit_float32_dot(torch.ones(8),torch.ones(7))

if __name__=="__main__":
    unittest.main()
