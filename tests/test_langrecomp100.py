"""Constructed execution/gradient gates; actual original FIT work is Actions-only."""
import importlib.util,json,os,sys
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest import mock
import numpy as np
import pytest
import torch
ROOT=Path(__file__).resolve().parents[1];HERE=ROOT/'research/mathematical-engineering/langrecomp100'
def load(name,path=None):
    key='test_langrecomp100_'+name;spec=importlib.util.spec_from_file_location(key,path or HERE/(name+'.py'));m=importlib.util.module_from_spec(spec);sys.modules[key]=m;spec.loader.exec_module(m);return m
R=load('recompute');M=load('model',ROOT/'research/mathematical-engineering/langtrain099/model.py');C=load('common');G=load('gate',ROOT/'scripts/validate_research_parity.py')
def small():
    torch.manual_seed(100);return M.ScaledContextualAssociativeTAC099(M.Geometry099(32,48,8,16,138)).double()

def exercise(m,policy,state=None,mask=None):
    x=torch.arange(22).reshape(2,11);owners=('a','b');counter={}
    y,s=R.forward(m,x,owners=owners,state=state,write_mask=mask,policy=policy,counter=counter)
    loss=y.square().mean()+sum(t.square().mean() for t in s.tensors.values());loss.backward()
    return y,s,{n:p.grad.clone() for n,p in m.named_parameters()},counter

@pytest.mark.parametrize('policy',R.POLICIES)
def test_full_source_outputs_state_parameter_gradients_match_native(policy):
    m=small();n=small();a,sa,ga,ca=exercise(m,'native');b,sb,gb,cb=exercise(n,policy)
    torch.testing.assert_close(a,b,atol=0,rtol=0)
    for name in sa.tensors:torch.testing.assert_close(sa.tensors[name],sb.tensors[name],atol=0,rtol=0)
    for name in ga:torch.testing.assert_close(ga[name],gb[name],atol=0,rtol=0)
    assert ca['calls']==1 and cb['calls']==(2 if policy=='recompute' else 1)

def test_carried_state_gradients_and_owners_not_detached():
    results=[]
    for policy in R.POLICIES:
        m=small();_,state=m(torch.arange(10).reshape(2,5),owners=('a','b'))
        tensors={n:v.detach().clone().requires_grad_() for n,v in state.tensors.items()};state=M.State099(state.owners,state.clock,tensors)
        y,s,g,c=exercise(m,policy,state);results.append((y,{n:p.grad.clone() for n,p in tensors.items()}))
        assert s.clock==16 and s.owners==('a','b')
    for n in results[0][1]:torch.testing.assert_close(results[0][1][n],results[1][1][n],atol=0,rtol=0)

def test_explicit_write_mask_and_source_causal_suffix():
    m=small();mask=torch.ones(2,11,dtype=torch.bool);mask[:,3:7]=False
    a,sa,ga,_=exercise(m,'native',mask=mask);n=small();b,sb,gb,_=exercise(n,'recompute',mask=mask)
    torch.testing.assert_close(a,b,atol=0,rtol=0)
    for k in ga:torch.testing.assert_close(ga[k],gb[k],atol=0,rtol=0)
    n.zero_grad(set_to_none=True);x=torch.arange(22).reshape(2,11);poison=x.clone();poison[:,6:]=255
    p,_=R.forward(n,poison,owners=('a','b'),write_mask=mask,policy='recompute')
    torch.testing.assert_close(b[:,:6],p[:,:6],atol=0,rtol=0)

def test_exact10m_source_forward_backward_checkpoint_path():
    torch.manual_seed(100);m=M.ScaledContextualAssociativeTAC099();x=torch.arange(5)[None];counter={}
    y,s=R.forward(m,x,owners=('a',),policy='recompute',counter=counter);y.square().mean().backward()
    assert sum(p.numel() for p in m.parameters())==10000000 and s.tensor_bytes()==11104 and counter['calls']==2
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters())

def test_only_registered_owner_matrix_and_default_scale_guard():
    f=json.loads((HERE/'FAIR_COMPARISON.json').read_text());assert G.check_manifest(f,'preflight')['eligible_for_architecture_superiority'] is False
    for change in ('contract','seed','updates','parent','count'):
        bad=deepcopy(f)
        if change=='contract':bad['parameter_budget_contract']='unrecognized'
        if change=='seed':bad['models'][0]['seed']=0
        if change=='updates':bad['budget']['training_updates']=17
        if change=='parent':bad['parent_checkpoint_run']=1
        if change=='count':bad['models'][0]['deployed_parameters']=10000001
        with pytest.raises(G.ParityError):G.check_manifest(bad,'preflight')
    with pytest.raises(G.ParityError):G.check_model_parameter_target(10000000,10000000)

def test_disabled_optimizer_guard_precedes_data_or_constructor():
    w=load('worker')
    with mock.patch.object(w.C,'read',return_value={'continuation_enabled':False}),mock.patch.object(torch.optim,'AdamW') as opt:
        with pytest.raises(ValueError,match='disabled'):w.run(SimpleNamespace())
        opt.assert_not_called()

def test_scalar_and_multidimensional_tensor_byte_digest():
    a={'step':torch.tensor(2048.),'matrix':torch.arange(24.).reshape(2,3,4)}
    assert C.tensor_digest(a)==C.tensor_digest({k:v.clone() for k,v in a.items()})
    b={**a,'step':torch.tensor(2049.)};assert C.tensor_digest(a)!=C.tensor_digest(b)

def test_saved_hook_counter_keeps_only_metadata_not_tensor_references():
    w=load('worker');m=small();counter=w.Saves(m);a=torch.ones(3,4)
    assert counter.pack(a) is a;assert counter.unpack(a) is a
    assert counter.report()['logical_requested_bytes']==a.numel()*a.element_size()

def test_comparison_rejects_changed_gradient_even_if_output_equal():
    A=load('assemble');row={'seed':9901,'steps':16,'absolute_optimizer_steps':2064,'trainable_parameters':10000000,'deployed_parameters':10000000,'parent_checkpoint_sha256':'p','initial_model_sha256':'i','initial_optimizer_sha256':'o','initial_torch_rng_sha256':'r','sample_sha256':'s','source_commit':'c','source_hashes':{},'history':[]}
    for i in range(16):row['history'].append({'step':i+1,'absolute_step':2049+i,'mode':'language','loss':1.,'outputs_state_sha256':'x','raw_all_parameter_gradient_sha256':'g','parameter_gradient_norms':{},'gradient_norm_before_clip':1.,'model_sha256':'m','optimizer_sha256':'o','torch_rng_sha256':'r','exposure':{},'forward_calls_including_recompute':1})
    other=deepcopy(row)
    for x in other['history']:x['forward_calls_including_recompute']=2
    assert len(A.compare_pair(row,other))==16
    other['history'][5]['raw_all_parameter_gradient_sha256']='changed'
    with pytest.raises(ValueError,match='gradient'):A.compare_pair(row,other)

def test_invalid_policy_rejected():
    with pytest.raises(ValueError):R.forward(small(),torch.ones(1,2,dtype=torch.long),owners=('a',),policy='anything')

@pytest.mark.skipif(os.environ.get('GITHUB_ACTIONS')!='true',reason='optimizer execution only in Actions')
def test_constructed_two_step_AdamW_parameter_and_moment_parity():
    a=small();b=small();oa=torch.optim.AdamW(a.parameters(),lr=3e-5,foreach=False);ob=torch.optim.AdamW(b.parameters(),lr=3e-5,foreach=False)
    for step in range(2):
        for m,o,policy in ((a,oa,'native'),(b,ob,'recompute')):
            o.zero_grad(set_to_none=True);exercise(m,policy);torch.nn.utils.clip_grad_norm_(m.parameters(),1.,foreach=False);o.step()
        assert C.tensor_digest(a.state_dict())==C.tensor_digest(b.state_dict())
        assert C.optimizer_digest(oa,a)==C.optimizer_digest(ob,b)

def test_actual_policy_probe_exposes_all_native_float_state_gradients():
    a=load('probe_adapter').adapter100();m=small();sample={'tokens':torch.arange(6).reshape(2,3),'owners':('a','b')}
    a.validate(m,sample);s=a.start(m,sample);s=a.advance(m,s,sample,0);outputs,leaves=a.gradient_graph(m,s,sample)
    gradients=torch.autograd.grad(outputs['nextbyte_logits'].square().mean(),tuple(leaves.values()),allow_unused=False)
    assert set(leaves)==set(s.tensors) and all(torch.isfinite(g).all() for g in gradients)

def test_independent_full_graph_replay_reference_matches_actual_checkpoint():
    n=load('reference');m=small();x=torch.arange(7)[None]
    p={k:t.detach().numpy().copy() for k,t in m.state_dict().items()}
    y,s=R.forward(m,x,owners=('a',),policy='recompute');ny,ns=n.replay_forward(p,x.numpy())
    np.testing.assert_allclose(y.detach().numpy(),ny,atol=5e-10,rtol=5e-10)
    for key in ns:np.testing.assert_allclose(s.tensors[key].detach().numpy(),ns[key],atol=5e-10,rtol=5e-10)
