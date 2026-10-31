"""Constructed implementation checks; no trained capability evidence."""
import importlib.util
from pathlib import Path
import sys
import pytest
import torch

HERE=Path(__file__).resolve().parent
s=importlib.util.spec_from_file_location('lm100_runner',HERE/'run.py');R=importlib.util.module_from_spec(s);sys.modules[s.name]=R;s.loader.exec_module(R)
ROOT=HERE.parents[2]
SOURCE=ROOT/'research/mathematical-engineering/langtrain099/model.py'
if not SOURCE.exists():SOURCE=HERE/'src/model099.py'
M=R.load_source(SOURCE)

@pytest.fixture(scope='module')
def model():
 torch.set_num_threads(2);torch.manual_seed(100001)
 return M.ScaledContextualAssociativeTAC099().eval()

@pytest.mark.parametrize('mask',range(32))
def test_complete_factorial_is_selective_and_owner_preserving(model,mask):
 with torch.inference_mode():
  _,s=model(torch.arange(12)[None],owners=('test',));before={k:v.clone() for k,v in s.tensors.items()}
  v=R.intervene(M,s,f'reset_{mask:02d}')
  assert v.clock==s.clock and v.owners==s.owners
  for j,key in enumerate(R.FACTORS):
   if key in ('context','lexical','previous_key'):
    assert torch.equal(v.tensors[key],torch.zeros_like(before[key]) if mask&(1<<j) else before[key])
   else:
    bank=0 if key=='fast' else 1
    for n in ('memory','mass'):assert torch.equal(v.tensors[n][:,bank],torch.zeros_like(before[n][:,bank]) if mask&(1<<j) else before[n][:,bank])
  assert all(torch.equal(before[k],s.tensors[k]) for k in before)

@pytest.mark.parametrize('disabled',([0],[1],[0,1]))
def test_disabled_bank_is_zero_on_every_timestep_and_others_preserved(model,disabled):
 with torch.inference_mode():
  _,s=model(torch.arange(12)[None],owners=('test',));x=torch.tensor([[10,12,14,16,18]])
  base,st,tr=model(x,state=s,owners=s.owners,return_trace=True)
  with R.suppress_banks(M,disabled):out,end,trace=model(x,state=s,owners=s.owners,return_trace=True)
  for j in disabled:
   assert torch.count_nonzero(trace['memory'][:,:,j])==0
   assert torch.count_nonzero(trace['mass'][:,:,j])==0
   assert torch.count_nonzero(end.tensors['memory'][:,j])==0
  for j in set([0,1])-set(disabled):
   assert torch.equal(trace['memory'][:,:,j],tr['memory'][:,:,j])
  for name in ('context','lexical','previous_key'):assert torch.equal(st.tensors[name],end.tensors[name])
  restored,_=model(x,state=s,owners=s.owners);assert torch.equal(base,restored)

def test_shuffle_preserves_coupled_mass_invariant(model):
 with torch.inference_mode():
  _,s=model(torch.arange(32)[None],owners=('test',))
  for name in ('shuffle_fast','shuffle_slow','shuffle_banks'):
   t=R.intervene(M,s,name).tensors
   assert bool((t['memory'].abs()<=t['mass'][:,:,None,:]+2e-6).all())

def test_zero_state_every_byte_has_no_historical_input(model):
 with torch.inference_mode():
  a,_=model(torch.tensor([[63]]),owners=('a',));b,_=model(torch.tensor([[63]]),owners=('a',))
  assert torch.equal(a,b)

def test_exact_original_grid_decoder_grammar():
 assert R.decode(b'221234;ignored')==b'221234'
 for v in (b'221234',b'22ab34;',b'771234;',b'22123;'):
  with pytest.raises(ValueError):R.decode(v)
