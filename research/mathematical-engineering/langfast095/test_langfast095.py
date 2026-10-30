"""LANGFAST095 original model source-level identity and gradient error-path checks."""
import sys
from pathlib import Path
import pytest
import torch
ROOT=Path(__file__).resolve().parents[3]
sys.path[:0]=[str(ROOT/'research/mathematical-engineering/langmem086'),str(Path(__file__).resolve().parent)]
from compact_tac import CompactDualClippedTAC086
from optimized095 import OptimizedCompactTAC095

def pair():
 torch.manual_seed(42)
 original=CompactDualClippedTAC086()
 optimized=OptimizedCompactTAC095()
 optimized.load_state_dict(original.state_dict(),strict=True)
 assert list(original.state_dict())==list(optimized.state_dict())
 return original,optimized
@pytest.mark.parametrize('steps',[1,3,9])
def test_identical_forward_native_state(steps):
 torch.set_num_threads(1)
 a,b=pair();x=torch.randint(0,256,(2,steps))
 y1,s1=a(x,owners=('a','b'));y2,s2=b(x,owners=('a','b'))
 assert torch.equal(y1,y2)
 assert s1.clock==s2.clock==steps
 assert all(torch.equal(s1.tensors[k],s2.tensors[k]) for k in s1.tensors)
 assert s1.tensor_bytes()==s2.tensor_bytes()

def test_same_parameter_gradients_training_unchanged():
 torch.set_num_threads(1)
 a,b=pair()
 inputs=torch.randint(0,256,(1,3))
 x,_=a(inputs,owners=('train',));x.float().square().mean().backward()
 y,_=b(inputs,owners=('train',));y.float().square().mean().backward()
 for (name,pa),(name2,pb) in zip(a.named_parameters(),b.named_parameters()):
  assert name==name2
  assert pa.grad is not None and pb.grad is not None,name
  assert torch.allclose(pa.grad,pb.grad,atol=1e-7,rtol=1e-6),name

@pytest.mark.parametrize('bad_owners',[(),('x','x'),('x',''),('x',8)])
def test_original_owner_error_preserved(bad_owners):
 a,b=pair()
 for m in (a,b):
  with pytest.raises(ValueError):
   m.start(bad_owners)

def test_original_nonfinite_error_preserved():
 from dataclasses import replace
 a,b=pair()
 for model in (a,b):
  s=model.start(('owner',))
  corrupt=replace(s,tensors={**s.tensors,'fast_memory':torch.full_like(s.tensors['fast_memory'],float('nan'))})
  with pytest.raises(ValueError):
   model(torch.tensor([[1]]),state=corrupt,owners=('owner',))

def test_identity_only_nonpersistent_buffers():
 a,b=pair()
 assert set(a.state_dict())==set(b.state_dict())
 assert 'eye4_langfast095' not in b.state_dict()
 assert 'eye8_langfast095' not in b.state_dict()
 assert sum(p.numel() for p in a.parameters())==sum(p.numel() for p in b.parameters())==100521
