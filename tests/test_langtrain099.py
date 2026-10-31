"""Constructed readiness/authorization checks; no optimizer step or DEV scoring."""
from copy import deepcopy
from pathlib import Path
import importlib.util
import sys
from types import SimpleNamespace
from unittest import mock
import numpy as np
import pytest
import torch

ROOT=Path(__file__).resolve().parents[1];HERE=ROOT/'research/mathematical-engineering/langtrain099'
def load(name,path=None):
    spec=importlib.util.spec_from_file_location('test099_'+name,path or HERE/(name+'.py'))
    m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m);return m
M=load('model');N=load('reference');C=load('common');G=load('gate',ROOT/'scripts/validate_research_parity.py')

def test_live_exact10million_count_state_and_finite_connected_gradients():
    torch.manual_seed(9901);m=M.ScaledContextualAssociativeTAC099();x=torch.arange(6)[None]
    assert m.counts()['trainable']==m.counts()['deployed']==m.geometry.parameter_count()==10000000
    assert m.start(('a',)).tensor_bytes()==11104
    y,s=m(x,owners=('a',));y.square().mean().backward()
    for name,p in m.named_parameters():assert p.grad is not None and torch.isfinite(p.grad).all() and p.grad.norm()>0,name
    assert s.clock==6

def test_actual10m_full_numpy_state_and_serial_chunk_equivalence():
    torch.manual_seed(9902);m=M.ScaledContextualAssociativeTAC099().double();x=torch.arange(9)[None]
    with torch.no_grad():
        y,s=m(x,owners=('a',));seq,ss=m(x,owners=('a',),parallel=False)
        first,part=m(x[:,:4],owners=('a',));second,last=m(x[:,4:],state=part,owners=('a',))
    p={n:v.detach().numpy().copy() for n,v in m.state_dict().items()};yn,sn=N.forward(p,x.numpy())
    np.testing.assert_allclose(y.numpy(),yn,atol=5e-10,rtol=5e-10)
    torch.testing.assert_close(y,seq,atol=5e-10,rtol=5e-10);torch.testing.assert_close(y,torch.cat((first,second),1),atol=5e-10,rtol=5e-10)
    for n,v in s.tensors.items():
        np.testing.assert_allclose(v.numpy(),sn[n],atol=5e-10,rtol=5e-10)
        torch.testing.assert_close(v,last.tensors[n],atol=5e-10,rtol=5e-10)

def test_generic_small_geometry_matches_exact098_ancestor():
    old=load('ancestor',ROOT/'research/mathematical-engineering/langmath098/model.py').ContextualAssociativeTAC098().double()
    m=M.ScaledContextualAssociativeTAC099(M.Geometry099(embed=32,context=48,key=8,value=16,ffn=138)).double();m.load_state_dict(old.state_dict(),strict=True)
    x=torch.arange(17)[None];a,sa=old(x,owners=('a',));b,sb=m(x,owners=('a',))
    torch.testing.assert_close(a,b,atol=0,rtol=0)
    for n in sa.tensors:torch.testing.assert_close(sa.tensors[n],sb.tensors[n],atol=0,rtol=0)

def test_causal_suffix_and_state_boundary_with_actual10m():
    m=M.ScaledContextualAssociativeTAC099().eval();x=torch.arange(8)[None];poison=x.clone();poison[:,4:]=255
    with torch.no_grad():a,s=m(x,owners=('a',));b,_=m(poison,owners=('a',))
    torch.testing.assert_close(a[:,:4],b[:,:4],atol=0,rtol=0)
    with pytest.raises(ValueError):m(x,state=s,owners=('other',))
    reset=m.intervene(s,'reset');assert reset.clock==s.clock and all(v.count_nonzero()==0 for v in reset.tensors.values())
    corrupt=m.intervene(s,'corrupt');assert corrupt.clock==s.clock

def test_default100k_guard_is_not_relaxed():
    with pytest.raises(G.ParityError):G.check_model_parameter_target(10000000,10000000)
    G.check_model_parameter_target(100000,100000)

def test_only_exact_owner_scale_matrix_passes():
    manifest=C.read(HERE/'FAIR_COMPARISON.json');assert G.check_manifest(manifest,'preflight')['eligible_for_architecture_superiority'] is False
    for change in ('contract','count','geometry','seed'):
        bad=deepcopy(manifest)
        if change=='contract':bad['parameter_budget_contract']='unrecognized'
        if change=='count':bad['models'][0]['trainable_parameters']=10000001
        if change=='geometry':bad['geometry']['embed']=257
        if change=='seed':bad['models'][0]['seed']=9999
        with pytest.raises(G.ParityError):G.check_manifest(bad,'preflight')

def test_incomplete_training_cannot_pass_final_scale_gate():
    m=C.read(HERE/'FAIR_COMPARISON.json')
    for r in m['models']:r['measured']={'steps':m['budget']['training_updates']-1}
    with pytest.raises(G.ParityError,match='incomplete'):G.check_manifest(m,'final')

def test_training_disabled_blocks_before_optimizer_or_data():
    T=load('train');a=SimpleNamespace(seed=9901)
    with mock.patch.object(T.C,'read',return_value={'fit_enabled':False}),mock.patch.object(torch.optim,'AdamW') as optimizer:
        with pytest.raises(ValueError,match='disabled'):T.run(a)
        optimizer.assert_not_called()

def test_frozen_schedule_and_true_episode_alignment():
    episodes=[{'domain':d,'acquisition_bytes':[1]*64,'query_bytes':[2]*32,'target_bytes':[3]*4} for d in ('lexical',)*4+('arc',)*4]
    windows=[(0,0,'source')]*20;plan=C.schedule(9901,2048,windows,episodes)
    assert plan==C.schedule(9901,2048,windows,episodes)
    assert sum(r['mode']=='episodic' for r in plan)==512
    x,y,future,valid,domains,cost=C.batch(plan[3],[bytes(range(200))],windows,episodes)
    assert x.shape==(4,272) and future.sum()==16 and valid.sum()==396
    assert (x[:,95]==2).all() and (y[:,95]==3).all() and future[:,95:99].all()
    assert domains==('code','code','arc','arc')

def test_fixed_source_data_contract_and_split_counts():
    d=C.read(HERE/'FROZEN_INPUTS.json');assert d['FIT_files']==197 and d['language_DEV_files']==22 and d['language_DEV_windows']==176
    reg=C.read(HERE/'REGISTER.json');assert reg['parameters']==10000000 and reg['training']['precision']=='FP32'

def test_original097_guard_prevents_shared_validator_from_retraining():
    text=(ROOT/'.github/workflows/langacquire097.yml').read_text();assert 'separate099' in text and '(direct097 or not separate099)' in text

def test_actor_only_schema_rejects_future_target():
    A=load('actor')
    with mock.patch.object(A.C,'read_json',return_value=[{'id':'case','domain':'lexical','query_bytes':[1],'generation_limit':4,'target_bytes':[2]}]):
        with pytest.raises(ValueError,match='forbidden'):A.load_actor(Path('fixture'),'query')


def test_source_registered_scaled_probe_boundary():
    m=load('model').ScaledContextualAssociativeTAC099();a=load('probe_adapter').adapter099()
    sample={'tokens':torch.tensor([[1,2,3],[4,5,6]]),'owners':('a','b')}
    a.validate(m,sample);state=a.start(m,sample);state=a.advance(m,state,sample,0)
    forecasts,leaves=a.gradient_graph(m,state,sample)
    assert set(leaves)=={'context','lexical','previous_key','memory','mass'}
    assert forecasts['nextbyte_logits'].shape==(2,256)
    with pytest.raises(ValueError):a.validate(m,{**sample,'target':1})


def test_full10m_scan_serial_parameter_gradients_and_numpy_finite_difference():
    torch.manual_seed(9903);m=M.ScaledContextualAssociativeTAC099().double();x=torch.arange(7)[None]
    y,_=m(x,owners=('a',));y.square().mean().backward();parallel={n:p.grad.clone() for n,p in m.named_parameters()}
    m.zero_grad(set_to_none=True);z,_=m(x,owners=('a',),parallel=False);z.square().mean().backward()
    for n,p in m.named_parameters():torch.testing.assert_close(parallel[n],p.grad,atol=5e-10,rtol=5e-8)
    params={n:v.detach().numpy().copy() for n,v in m.state_dict().items()};eps=1e-5
    for name in ('context_lag','address_value.weight','ffn_out.weight'):
        index=(0,)*params[name].ndim;original=params[name][index]
        params[name][index]=original+eps;plus,_=N.forward(params,x.numpy())
        params[name][index]=original-eps;minus,_=N.forward(params,x.numpy());params[name][index]=original
        numeric=(np.square(plus).mean()-np.square(minus).mean())/(2*eps)
        np.testing.assert_allclose(numeric,parallel[name][index].item(),atol=2e-9,rtol=2e-5)
