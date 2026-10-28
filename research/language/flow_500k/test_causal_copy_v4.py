import unittest,torch
from causal_copy_mixture_v4 import suffix_source_queries,copy_mixture_probs
class Tests(unittest.TestCase):
 def test_strict_source_only_and_expected_suffix(self):
  x=torch.tensor([[5,9,3,8,5,9,3,8,1,2,5,9,3,8,4]],dtype=torch.long)
  q,h=suffix_source_queries(x,4)
  self.assertTrue(bool((q[h]<torch.arange(x.size(1)).expand_as(q)[h]).all()))
  self.assertEqual(q[0,7].item(),4)
 def test_normalized_nonnegative_no_hit_is_model(self):
  x=torch.tensor([[1,2,3,4,1,2,3,4,5,6,7,8]],dtype=torch.long)
  l=torch.randn(1,x.size(1),128)
  prior=torch.ones(128)/128
  p,q,h=copy_mixture_probs(l,x,prior,.5,k=3)
  self.assertTrue(torch.allclose(p.sum(-1),torch.ones_like(p.sum(-1)),atol=1e-6))
  self.assertTrue(bool((p>=0).all()))
  self.assertTrue(torch.allclose(p[:,0],l[:,0].softmax(-1),atol=1e-6))
 def test_future_poison_does_not_change_earlier_distribution(self):
  torch.manual_seed(7);x=torch.randint(0,128,(1,55));l=torch.randn(1,55,128);prior=torch.ones(128)/128
  a,_,_=copy_mixture_probs(l,x,prior,.5)
  poisoned=x.clone();poisoned[:,35:]=(poisoned[:,35:]+19)%128
  b,_,_=copy_mixture_probs(l,poisoned,prior,.5)
  self.assertTrue(torch.allclose(a[:,:35],b[:,:35],atol=1e-6))
if __name__=='__main__':unittest.main()
