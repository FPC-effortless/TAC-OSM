"""LANGHEAD094 model-graph scoped invariants; constructed data is only test input."""
from pathlib import Path
import sys
import pytest
import torch
sys.path.insert(0,str(Path(__file__).resolve().parent))
from model094 import ARMS,PARAMS,make_model

@pytest.mark.parametrize('arm',ARMS)
def test_actual_model_live_size(arm):
 m=make_model(arm)
 n=sum(p.numel() for p in m.parameters())
 assert n==PARAMS[arm] and 95000<=n<=105000
 assert max(PARAMS.values())/min(PARAMS.values())<1.01

@pytest.mark.parametrize('arm',ARMS)
def test_true_original_model_causality_prefix_and_streaming(arm):
 torch.set_num_threads(2);torch.manual_seed(31)
 m=make_model(arm).eval()
 x=torch.randint(0,256,(1,12),dtype=torch.long);future=x.clone();future[:,8:]=(future[:,8:]+101)%256
 with torch.no_grad():
  complete,_=m(x,owners=('a',))
  poisoned,_=m(future,owners=('a',))
  assert torch.equal(complete[:,:8],poisoned[:,:8])
  state=m.start(('a',));logits=[]
  for j in range(x.shape[1]):
   row,state=m(x[:,j:j+1],state=state,owners=('a',));logits.append(row)
  assert state.clock==x.shape[1]
  assert torch.allclose(complete,torch.cat(logits,1),atol=3e-5,rtol=0)

@pytest.mark.parametrize('arm',ARMS)
def test_gradients_connected_and_finite(arm):
 torch.set_num_threads(2);torch.manual_seed(43)
 m=make_model(arm)
 x=torch.randint(0,256,(2,10),dtype=torch.long)
 output,_=m(x,owners=('aa','bb'))
 loss=torch.nn.functional.cross_entropy(output.reshape(-1,256),torch.randint(0,256,(20,)))
 loss.backward()
 for name,param in m.named_parameters():
  assert param.grad is not None,name
  assert torch.isfinite(param.grad).all(),name
 if arm=='tied_dot':assert float(m.raw_strength.grad.abs().max())>0

@pytest.mark.parametrize('arm',ARMS)
def test_batch_owner_isolation(arm):
 torch.set_num_threads(2);torch.manual_seed(45)
 m=make_model(arm).eval()
 a=torch.randint(0,256,(1,7),dtype=torch.long)
 b=torch.randint(0,256,(1,7),dtype=torch.long)
 with torch.no_grad():
  full,_=m(torch.cat([a,b]),owners=('a','b'))
  single,_=m(a,owners=('a',))
  assert torch.allclose(full[0],single[0],atol=3e-5,rtol=0)

@pytest.mark.parametrize('arm',ARMS)
def test_owner_mismatch_rejected(arm):
 m=make_model(arm)
 with pytest.raises(ValueError):
  m(torch.tensor([[7]],dtype=torch.long),state=m.start(('old',)),owners=('new',))

def test_diagonal_contractivity_for_equal_observations():
 m=make_model('selective_ssm')
 e=m.embedding(torch.tensor([14]))[0]
 a=torch.exp(-torch.nn.functional.softplus(m.raw_decay)*torch.sigmoid(m.gate(e)))
 assert bool(((a>0)&(a<1)).all())
 for t in range(5):
  p=torch.full((128,),float(t));q=p+2
  assert torch.linalg.vector_norm(a*p-a*q)<=torch.linalg.vector_norm(p-q)+1e-6
