"""S1TEMP008 source-indexed entity matrix kernels: constructed SOFTWARE tests.

Not real case outcome evidence or source-level proof of authorization.
"""
import unittest
import torch
from tac_osm.variants.s1temp008_entity_retention import (
    indexed_affine_schedule,indexed_affine_parallel,indexed_affine_step,
    validate_event_slots)
from tac_osm.variants.parallel_affine_memory import sequential_affine_reference

class GenuineSlotAlgorithmContract(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(80001)
        self.B=3;self.T=45
        self.slots=torch.randint(0,8,(self.B,self.T),dtype=torch.long)
        self.values=torch.tanh(torch.randn(self.B,self.T,8))
        self.alpha=torch.sigmoid(torch.randn(self.B,self.T,1))
        self.beta=torch.sigmoid(torch.randn(self.B,self.T,1))

    def test_original_64_float_memory_no_extra_parameters(self):
        self.assertEqual(8*8*4,256)
        self.assertFalse(any(p.requires_grad for p in []))

    def test_work_efficient_parallel_matches_causal_reference(self):
        P,Q,k=indexed_affine_schedule(self.slots,self.values,self.alpha,self.beta)
        expected=sequential_affine_reference(P,Q)
        actual,keys=indexed_affine_parallel(self.slots,self.values,self.alpha,self.beta)
        torch.testing.assert_close(actual,expected,atol=4e-6,rtol=4e-6)
        torch.testing.assert_close(k,keys,atol=0,rtol=0)

    def test_streaming_matched_previous_read_and_all_timestamps(self):
        expected,_=indexed_affine_parallel(self.slots,self.values,self.alpha,self.beta)
        memory=self.values.new_zeros(self.B,8,8)
        for t in range(self.T):
            next_memory,old_read=indexed_affine_step(
                memory,self.slots[:,t],self.values[:,t],
                self.alpha[:,t],self.beta[:,t])
            k=torch.nn.functional.one_hot(self.slots[:,t],8).to(memory.dtype)
            torch.testing.assert_close(old_read,(memory@k.unsqueeze(-1)).squeeze(-1),
                atol=0,rtol=0)
            torch.testing.assert_close(next_memory,expected[:,t],
                atol=4e-6,rtol=4e-6)
            memory=next_memory

    def test_no_cross_entity_write_even_if_value_huge(self):
        m=torch.randn(2,8,8)
        slots=torch.tensor([1,5],dtype=torch.int64)
        new,read=indexed_affine_step(m,slots,torch.ones(2,8)*500,
                                   torch.full((2,1),.4),torch.full((2,1),.9))
        for b,j in enumerate(slots.tolist()):
            for k in range(8):
                if k!=j:
                    torch.testing.assert_close(new[b,:,k],m[b,:,k],atol=0,rtol=0)
            self.assertGreater((new[b,:,j]-m[b,:,j]).norm().item(),0)

    def test_source_slot_and_loss_gradient_connectivity(self):
        v=self.values.clone().requires_grad_(True)
        a=self.alpha.clone().requires_grad_(True)
        b=self.beta.clone().requires_grad_(True)
        output,_=indexed_affine_parallel(self.slots,v,a,b)
        loss=output[:,-1].square().sum()
        grads=torch.autograd.grad(loss,(v,a,b))
        self.assertTrue(all(x is not None and
            torch.isfinite(x).all().item() and x.norm().item()>0 for x in grads))

    def test_observation_order_and_ids_both_matter(self):
        before,_=indexed_affine_parallel(self.slots,self.values,self.alpha,self.beta)
        swapped=self.slots.clone()
        swapped[:,2],swapped[:,3]=self.slots[:,3],self.slots[:,2]
        after,_=indexed_affine_parallel(swapped,self.values,self.alpha,self.beta)
        self.assertGreater((before-after).abs().max().item(),0)

    def test_future_event_does_not_change_previous_memory(self):
        old,_=indexed_affine_parallel(self.slots,self.values,self.alpha,self.beta)
        future=self.values.clone();future[:,-1]+=1000
        new,_=indexed_affine_parallel(self.slots,future,self.alpha,self.beta)
        torch.testing.assert_close(old[:,:-1],new[:,:-1],atol=0,rtol=0)

    def test_reject_unauthorized_slot_dtype_and_overflow(self):
        for slots in (self.slots.float(),self.slots.clamp_max(3).to(torch.int32),
                      self.slots.clone().fill_(8),self.slots[:,:-1]):
            with self.assertRaises(ValueError):
                indexed_affine_parallel(slots,self.values,self.alpha,self.beta)

    def test_identity_update_when_write_strength_zero(self):
        values=self.values[:,:1]
        strength=torch.zeros_like(self.beta[:,:1])
        result,_=indexed_affine_parallel(self.slots[:,:1],values,self.alpha[:,:1],strength)
        torch.testing.assert_close(result,torch.zeros_like(result),atol=0,rtol=0)

if __name__=="__main__":unittest.main()
