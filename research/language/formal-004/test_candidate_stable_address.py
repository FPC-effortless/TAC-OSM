"""Untrained, controlled mathematics diagnostics for TAC-LANG-REPAIR-016.

THESE ARE SYNTHETIC OPERATOR UNIT TESTS, NEVER real-language results.
No checkpoint is altered and no headline performance improvement is claimed.
"""
import unittest
import torch
import torch.nn.functional as F
from candidate_stable_address import (
    POSITION_CYCLE,PERIODS,phase_code,epoch_code,smooth_key,
    conditioned_rotation,repaired_source_key)
from parallel_tac_language_gate_original import ParallelTACLanguage

class CandidateStableAddressTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_first_era_phase_byte_equal_original_source(self):
        model=ParallelTACLanguage(decoder=75,post=10,gate_mode="silu",
            position_mode="multiscale")
        for dtype in (torch.float32,torch.float64):
            for t in (0,1,7,128,4096,POSITION_CYCLE-1):
                source=model.keys_raw_phi(1,"cpu",dtype,t)[0]
                self.assertTrue(torch.equal(phase_code(t,dtype),source))
                self.assertTrue(torch.equal(epoch_code(t,dtype),source))

    def test_forced_old_collision_and_epoch_separation(self):
        for t in (0,1,91,4096,POSITION_CYCLE-1):
            self.assertTrue(torch.allclose(phase_code(t),phase_code(t+POSITION_CYCLE),
                atol=1e-13,rtol=0))
            a=epoch_code(t)
            b=epoch_code(t+POSITION_CYCLE)
            self.assertTrue(torch.any(a!=b))
            self.assertAlmostEqual(float((b-a)[0]),0.125,delta=1e-14)
            self.assertTrue(torch.equal(a[1:],b[1:]))
        self.assertEqual(POSITION_CYCLE,155701)
        # This does NOT make a finite 8D associative memory lossless.
        # It only removes the exact forced period at the address-input level.

    def test_smooth_key_bounds_and_nondiscontinuous_gradient(self):
        for tau in (.01,.05):
            for scale in (0.,1e-9,1e-6,.005,.01,1.,100.):
                g=torch.Generator().manual_seed(200+int(scale*1e6))
                x=(torch.randn(8,dtype=torch.float64,generator=g)*scale)
                y=smooth_key(x,tau)
                self.assertTrue(torch.isfinite(y).all())
                self.assertLessEqual(float(y.norm()),1.+1e-12)
                jac=torch.autograd.functional.jacobian(
                    lambda z:smooth_key(z,tau),x)
                op=float(torch.linalg.svdvals(jac).max())
                self.assertLessEqual(op,1/tau+1e-10)
                self.assertTrue(torch.isfinite(jac).all())
            zero=torch.zeros(8,dtype=torch.float64,requires_grad=True)
            y=smooth_key(zero,tau)
            grad=torch.autograd.grad(y[0],zero)[0]
            self.assertAlmostEqual(float(grad[0]),1/tau,delta=1e-10)

    def test_conditioned_rotation_min_singular_value(self):
        for scale in (0.,1.,10.,1e6):
            torch.manual_seed(124)
            raw=torch.randn(2,8,8,dtype=torch.float64)*scale
            repaired=conditioned_rotation(raw,0.05)
            self.assertGreaterEqual(float(torch.linalg.svdvals(repaired).min()),
                                    1-8*.05-1e-11)
            self.assertTrue(torch.isfinite(repaired).all())
        eye=torch.eye(8,dtype=torch.float64)
        original=torch.stack([eye,torch.roll(eye,1,1)])
        self.assertTrue(torch.equal(conditioned_rotation(original),original))

    def test_parameter_free_combination_reduces_same_residue_address_collision(self):
        eye=torch.eye(8,dtype=torch.float64)
        rotation=torch.stack([eye,torch.roll(eye,1,1)])
        for t in (0,22,2048):
            a=repaired_source_key(rotation,t)
            b=repaired_source_key(rotation,t+POSITION_CYCLE)
            self.assertEqual(tuple(a.shape),(2,8))
            self.assertGreater(float((a-b).norm()),1e-4)
            self.assertLessEqual(float(a.norm(dim=-1).max()),1.0+1e-12)
        self.assertEqual(rotation.numel(),128)

    def test_no_claim_of_unbounded_fp32_epoch_precision(self):
        # The monotone era/(era+1) saturates in finite FP32.
        # At sufficiently large eras the tag stops disambiguating.
        base=13
        q=10**10
        a=epoch_code(base+POSITION_CYCLE*q,torch.float32)
        b=epoch_code(base+POSITION_CYCLE*(q+1),torch.float32)
        self.assertTrue(torch.equal(a,b))

    def test_fail_closed_arguments(self):
        with self.assertRaises(ValueError): smooth_key(torch.ones(8),0)
        with self.assertRaises(ValueError): smooth_key(torch.ones(7),.01)
        with self.assertRaises(ValueError): epoch_code(-1)
        with self.assertRaises(ValueError): epoch_code(0,era_scale=0)
        with self.assertRaises(ValueError): conditioned_rotation(torch.ones(8,8))
        with self.assertRaises(ValueError): conditioned_rotation(torch.ones(2,8,8),.125)

if __name__=="__main__":
    unittest.main()
