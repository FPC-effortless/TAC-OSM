import unittest,torch
from source_slot_memory_v4 import SourceSlotMemory
from tac_wave_ced_500k_v4 import SlotTAC500K
from historical_500k_v4 import HistoricalSlotTAC500K
class Checks(unittest.TestCase):
 def setUp(self):torch.manual_seed(709);torch.set_num_threads(2)
 def test_bound_exact_and_admission(self):
  mem=SourceSlotMemory(256,16);v=torch.rand(1,230,16)*2-1
  q=torch.arange(230)[None,:]-192;q[:,:192]=-1
  read,state,valid=mem.step_chunk(v,start=0,owner_ids=('A',),queries=q)
  self.assertTrue(torch.equal(read[:,192:],v[:,:38]))
  self.assertTrue(bool(valid[:,192:].all()))
  self.assertLessEqual(float(state['values'].abs().max()),1.)
  self.assertEqual(state['ordinals'].numel(),256)
  self.assertEqual(state['count'],230)
 def test_cross_chunk_correct_source_and_gradient(self):
  mem=SourceSlotMemory(256,16)
  x=torch.randn(1,200,16).tanh().requires_grad_()
  q=torch.arange(200)[None,:]-192;q[:,:192]=-1
  full,state,valid=mem.step_chunk(x,start=0,owner_ids=('doc',),queries=q)
  _,first,_=mem.step_chunk(x[:,:128],start=0,owner_ids=('doc',))
  tail,second,ok=mem.step_chunk(x[:,128:],start=128,owner_ids=('doc',),initial_state=first,queries=q[:,128:])
  self.assertTrue(torch.allclose(full[:,128:],tail,atol=1e-6,rtol=1e-6))
  self.assertTrue(torch.equal(full[:,192],x[:,0]))
  full[:,192].sum().backward()
  self.assertGreater(float(x.grad[:,0].abs().sum()),0)
  self.assertEqual(float(x.grad[:,191].abs().sum()),0)
  with self.assertRaisesRegex(ValueError,'owner'):
   mem.step_chunk(x[:,128:],start=128,owner_ids=('other',),initial_state=first,queries=q[:,128:])
 def test_future_expired_and_state_corruption(self):
  m=SourceSlotMemory(8,16)
  x=torch.ones(1,8,16)
  with self.assertRaisesRegex(ValueError,'strictly past'):
   m.step_chunk(x,start=0,owner_ids=('A',),queries=torch.arange(8)[None,:])
  _,state,_=m.step_chunk(x,start=0,owner_ids=('A',))
  z=state.copy();z['ordinals']=state['ordinals'].clone();z['ordinals'][0,0]=18
  with self.assertRaisesRegex(ValueError,'expired or future'):
   m.step_chunk(x[:,:1],start=8,owner_ids=('A',),initial_state=z)
 def test_model_parameter_and_historical_real_gradient(self):
  a=HistoricalSlotTAC500K(256)
  self.assertEqual(a.parameter_count(),503514)
  x=torch.randint(0,128,(1,197))
  logits,h=a.forward_with_history(x,source_ids=('doc',))
  self.assertEqual(h[192].shape,(1,5,128))
  loss=torch.nn.functional.cross_entropy(h[192].reshape(-1,128),x[:,:5].reshape(-1))
  loss.backward()
  self.assertGreater(float(a.aligned_value.weight.grad.norm()),0)
 def test_model_chunk_streaming(self):
  a=SlotTAC500K(256).eval();x=torch.randint(0,128,(1,200))
  with torch.no_grad():
   full=a(x,source_ids=('A',))
   _,state=a(x[:,:128],source_ids=('A',),return_state=True)
   late=a(x[:,128:],start=128,source_ids=('A',),initial_state=state)
  self.assertTrue(torch.allclose(full[:,128:],late,rtol=2e-3,atol=2e-3))
if __name__=='__main__':unittest.main()
