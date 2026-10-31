"""No-fit checks for conventional controls; not evidence of learned capability."""
import importlib.util
from pathlib import Path
import sys
import pytest
import torch
import numpy as np

s=importlib.util.spec_from_file_location('lm101_conventional',Path(__file__).with_name('conventional.py'))
C=importlib.util.module_from_spec(s);sys.modules[s.name]=C;s.loader.exec_module(C)
torch.set_num_threads(1)

@pytest.mark.parametrize('name',C.CONTROL_NAMES)
def test_live_parameters_all_tensors_active_and_finite(name):
 torch.manual_seed(10101);m=C.make_control(name).double()
 assert 99500<=m.counts()['trainable']<=100600
 x=torch.arange(32).reshape(2,16)
 out,state=m(x,owners=('a','b'))
 assert out.shape==(2,16,256) and state.clock==16
 weights=torch.linspace(-.7,.9,out.numel(),dtype=torch.float64).reshape_as(out)
 (out*weights).mean().backward()
 for key,p in m.named_parameters():
  assert p.grad is not None and bool(torch.isfinite(p.grad).all()),key
  assert float(p.grad.abs().sum())>0,key

@pytest.mark.parametrize('name',C.CONTROL_NAMES)
def test_streaming_prefix_and_causal_future_poison(name):
 torch.manual_seed(10102);m=C.make_control(name).double().eval()
 x=torch.tensor([[1,3,5,7,9,11,13],[2,4,6,8,10,12,14]])
 with torch.inference_mode():
  full,last=m(x,owners=('a','b'));one,state=m(x[:,:3],owners=('a','b'))
  two,end=m(x[:,3:],state=state,owners=('a','b'))
  assert torch.allclose(torch.cat((one,two),1),full,atol=1e-9,rtol=1e-9)
  assert all(torch.allclose(v,end.tensors[k],atol=1e-9,rtol=1e-9) for k,v in last.tensors.items())
  future=x.clone();future[:,4:]=200
  alt,_=m(future,owners=('a','b'));assert torch.allclose(alt[:,:4],full[:,:4],atol=1e-9,rtol=1e-9)

def test_selective_scan_against_literal_serial_equations():
 torch.manual_seed(10103);a=torch.rand(2,17,8,dtype=torch.float64);b=torch.randn_like(a);s=torch.randn(2,8,dtype=torch.float64)
 assert torch.allclose(C.scan(a,b,s,True),C.scan(a,b,s,False),atol=1e-12,rtol=1e-12)

@pytest.mark.parametrize('name',C.CONTROL_NAMES)
def test_reject_owner_foreign_state_and_future_keyword(name):
 m=C.make_control(name);x=torch.tensor([[3,5,7]])
 _,state=m(x,owners=('a',))
 with pytest.raises(ValueError):m(x,state=state,owners=('b',))
 with pytest.raises(ValueError):m(x,owners=('a',),future_labels=torch.zeros_like(x))


def test_relative_parameter_gap_below_one_percent():
 counts=[100370]+[C.make_control(n).counts()['trainable'] for n in C.CONTROL_NAMES]
 assert (max(counts)-min(counts))/max(counts)<.01

@pytest.mark.parametrize('name',C.CONTROL_NAMES)
def test_independent_numpy_full_forward_and_final_state(name):
 s=importlib.util.spec_from_file_location('lm101_control_reference',Path(__file__).with_name('conventional_reference.py'))
 N=importlib.util.module_from_spec(s);s.loader.exec_module(N)
 torch.manual_seed(10104);m=C.make_control(name).double().eval();x=torch.tensor([[1,7,13,19,21,31,8]])
 got,state=m(x,owners=('a',));expected,last=N.forward(name,N.parameters(m),x.numpy())
 assert np.allclose(got.detach().numpy(),expected,atol=1e-10,rtol=1e-9)
 for k,v in state.tensors.items():assert np.allclose(v.detach().numpy(),last[k],atol=1e-10,rtol=1e-9)

@pytest.mark.parametrize('name',C.CONTROL_NAMES)
def test_each_parameter_tensor_one_independent_finite_difference_coordinate(name):
 s=importlib.util.spec_from_file_location('lm101_control_fd_reference',Path(__file__).with_name('conventional_reference.py'))
 N=importlib.util.module_from_spec(s);s.loader.exec_module(N)
 torch.manual_seed(10105);m=C.make_control(name).double().eval();x=torch.tensor([[1,3,7,9,13]])
 out,_=m(x,owners=('a',));w=torch.linspace(-.7,.9,out.numel(),dtype=torch.float64).reshape_as(out)
 (out*w).mean().backward();p=N.parameters(m);epsilon=1e-5
 for key,value in m.named_parameters():
  index=np.unravel_index(int(value.grad.abs().argmax()),tuple(value.shape));expected=float(value.grad[index]);original=p[key][index];values=[]
  for sign in (-1,1):
   p[key][index]=original+sign*epsilon;prediction,_=N.forward(name,p,x.numpy());values.append(float((prediction*w.numpy()).mean()))
  p[key][index]=original;numerical=(values[1]-values[0])/(2*epsilon)
  assert np.isclose(expected,numerical,atol=1e-7,rtol=1e-4),(key,expected,numerical)
