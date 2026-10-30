"""LANGPARALLEL096 software-only causal/scan invariants, not new model training."""
import sys
from pathlib import Path
import pytest
import torch
ROOT=Path(__file__).resolve().parents[3]
sys.path[:0]=[str(ROOT/'research/mathematical-engineering/context026'),str(ROOT/'research/mathematical-engineering/langlayer038'),str(Path(__file__).resolve().parent)]
from context_transition import ContextTransition026,scalar_affine_scan
from recurrence_only import RecurrenceOnlyDepth038,RecurrenceOnlyUpperBlock
from frozen_compare096 import STUDIES,LENGTHS,ANCHORS,DOCS,make
def test_two_original_research_arms_and_no_new_100k_fit():
 assert set(STUDIES)=={'flat_parallel','layered_parallel'}
 assert STUDIES['flat_parallel'][2]==500122
 assert STUDIES['layered_parallel'][2]==499901
 assert LENGTHS==(64,128) and len(ANCHORS)==8 and len(DOCS)==3
 for arm in STUDIES:
  m=make(arm)
  assert sum(p.numel() for p in m.parameters())==STUDIES[arm][2]
def test_parallel_scan_matches_serial_original_vector():
 torch.manual_seed(41)
 p=torch.rand(2,12,48,dtype=torch.float64)*.97
 q=torch.randn(2,12,48,dtype=torch.float64)
 out=scalar_affine_scan(p,q)
 prev=torch.zeros(2,48,dtype=torch.float64);series=[]
 for i in range(12):
  prev=p[:,i]*prev+q[:,i];series.append(prev)
 assert torch.allclose(out,torch.stack(series,1),atol=1e-12,rtol=0)
@pytest.mark.parametrize('arm',list(STUDIES))
def test_model_real_causal_token_prefix(arm):
 torch.set_num_threads(2)
 torch.manual_seed(102)
 m=make(arm);m.eval()
 x=torch.randint(0,256,(1,20));q=x.clone();q[:,13:]=(q[:,13:]+61)%256
 with torch.no_grad():
  a=m(x)
  b=m(q)
  assert a.shape==(1,20,256)
  assert torch.allclose(a[:,:13],b[:,:13],rtol=1e-5,atol=3e-5)
@pytest.mark.parametrize('arm',list(STUDIES))
def test_no_unapproved_carried_streaming_interface(arm):
 m=make(arm)
 with pytest.raises(ValueError,match='reset'):
  m(torch.tensor([[1,2,3]],dtype=torch.long),start=2)
@pytest.mark.parametrize('arm',list(STUDIES))
def test_reject_bad_byte_source(arm):
 m=make(arm)
 with pytest.raises(ValueError):
  m(torch.tensor([[3,300]],dtype=torch.long))
def test_layer_uses_concrete_parallel_upper_scan():
 torch.manual_seed(5)
 m=RecurrenceOnlyUpperBlock(280,59).eval()
 z=torch.randn(2,13,280)
 with torch.no_grad():
  y,t=m(z,return_trace=True)
  assert y.shape==z.shape
  assert t['state'].shape==(2,13,59)
  assert torch.allclose(t['prior_state'][:,0],torch.zeros_like(t['prior_state'][:,0]))
  assert torch.all((t['alpha']>0)&(t['alpha']<.995))
def test_no_concept_graph_reintroduced_into_upper_recurrence():
 m=RecurrenceOnlyDepth038()
 assert not hasattr(m.layernorm[1],'concept_q')
 assert not hasattr(m.layernorm[1],'concept_k')
 assert hasattr(m,'concept_q') and hasattr(m,'concept_k')
 assert hasattr(m,'native_key') and hasattr(m,'rotation')
