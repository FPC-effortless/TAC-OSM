"""Fail-closed mathematical accounting, never model-skill evidence."""
import copy
import json
import pytest
from tac_osm.research_operator_budget import signature_work,prove_pair,digest,verify_profile


def op(name,inputs,outputs,calls=1):
    sig=json.dumps([name,[["torch.float32",s] for s in inputs],[["torch.float32",s] for s in outputs]])
    return dict(stage='forward',signature=sig,calls=calls)


def pair(extra=None):
    base=[op('aten::mm.',[[16,100],[100,100]],[[16,100]],100)]
    return {'operators':base}, {'operators':base+([extra] if extra else [])}


def test_unknown_differing_math_blocks():
    a,b=pair(op('aten::mysterious.',[[16]],[[16]]))
    with pytest.raises(ValueError,match='unbounded'):prove_pair(a,b)


def test_common_unknown_cancels_but_complete_difference_is_reported():
    a,b=pair(op('aten::add.Tensor',[[16]],[[16]]))
    proof=prove_pair(a,b)
    assert proof['relative_full_math_gap_upper_bound']>0
    assert len(proof['complete_unequal_operators'])==1


def test_excess_compute_and_duplicate_rows_block():
    a,b=pair(op('aten::mul.Tensor',[[1000000]],[[1000000]],100))
    with pytest.raises(ValueError,match='exceeds'):prove_pair(a,b)
    a['operators']*=2
    with pytest.raises(ValueError,match='duplicate'):prove_pair(a,b)


def test_matrix_and_weight_invariance_required():
    ops=pair()[0]['operators']
    profile=dict(schema='tac-osm.complete-operator-profile.v1',profiles=[dict(seed=seed,arm=arm,kind=kind,
        weights_before='x',weights_after='x',parameters=99390,finite_gradient_tensors=43,
        operators=ops,operator_sha256=digest(ops),steps=1 if kind=='first' else 63,step_operator_sha256=[digest(ops)]*(1 if kind=='first' else 63)) for seed in (1,2) for arm in ('assoc_historic_query','fast_slow_historic_query') for kind in ('first','warm')])
    assert len(verify_profile(profile))==4
    bad=copy.deepcopy(profile);bad['profiles'][0]['weights_after']='changed'
    with pytest.raises(ValueError,match='weights'):verify_profile(bad)
    profile['profiles'][0]['operator_sha256']='fabricated'
    with pytest.raises(ValueError,match='fingerprint'):verify_profile(profile)


def test_reduction_bound_and_matmul():
    s=op('aten::sum.',[[16,8]], [[]])['signature']
    assert signature_work(s)==128
    s=op('aten::bmm.',[[2,3,4],[2,4,5]], [[2,3,5]])['signature']
    assert signature_work(s)==240



def test_actual_original_graph_loss_mask_and_gradients():
    import os
    from pathlib import Path
    import subprocess
    import sys
    root=os.environ.get('TAC_OSM_CONTENT_QUERY_SOURCE')
    if not root:pytest.skip('dedicated CPU job supplies pinned original implementation')
    code=r'''
from pathlib import Path
import sys
sys.path[:0]=[str(Path.cwd()),str(Path.cwd()/'src')]
import torch
import importlib.util,os
from scripts.original_history_trace import verify_implementation,load_original_modules
from scripts.train_history_query_diverse import full_loss
root=verify_implementation(Path(sys.argv[1]),'fd201207cb76b30470f64e1fc1baecf9de44560d')
source,original=load_original_modules(root)
torch.set_num_threads(1);torch.manual_seed(19)
d=dict(values=torch.randn(2,200,13),valid=torch.ones(2,200,13,dtype=torch.bool),age=torch.zeros(2,200,13),last=torch.zeros(2,200,13))
d['valid'][0,48,0]=False
d['valid'][1,118,2]=False
d['values'][~d['valid']]=float('nan')
for arm in ('assoc_historic_query','fast_slow_historic_query'):
 m=original.create_model(26016,arm)
 # Actual missing values passed to original step are sanitized as source loader does.
 d['values']=torch.nan_to_num(d['values'])
 captured={}
 oldpast,oldfuture=m.historic_event_logits,m.forecast
 def past(*args):
  out=oldpast(*args);out.retain_grad();captured['past']=out;return out
 def future(*args):
  mu,sd=oldfuture(*args);mu.retain_grad();sd.retain_grad();captured['mu']=mu;captured['sd']=sd;return mu,sd
 m.historic_event_logits=past;m.forecast=future
 loss=full_loss(m,d,[[0,40],[1,48]],source)
 loss.backward()
 assert captured['past'].grad[0,1]==0
 assert captured['mu'].grad[1,0,2]==0 and captured['sd'].grad[1,0,2]==0
 peer=os.environ.get('TAC_OSM_HISTORY_MATH_SOURCE')
 if peer:
  spec=importlib.util.spec_from_file_location('pnds_loss',Path(peer)/'foundation/v1/history_query016.py')
  ref=importlib.util.module_from_spec(spec);spec.loader.exec_module(ref)
  owners=torch.tensor([0,1]);anchors=torch.tensor([40,48]);times=anchors[:,None]+torch.arange(0,64,8)[None,:]
  hist=ref.historical_loss(captured['past'].detach().flatten().tolist(),d['values'][owners[:,None],times,0].flatten().tolist(),d['valid'][owners[:,None],times,0].flatten().tolist())
  futureval=ref.forecast_loss(captured['mu'].detach().flatten().tolist(),captured['sd'].detach().flatten().tolist(),d['values'][owners,anchors+70].flatten().tolist(),d['valid'][owners,anchors+70].flatten().tolist())
  assert abs(float(loss.detach())-hist-futureval)<1e-6

 assert torch.isfinite(loss)
 assert sum(p.numel() for p in m.parameters())==99390
 assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters())
 print('PASS_CONSTRUCTED_ORIGINAL_GRAPH',arm)
'''
    p=subprocess.run([sys.executable,'-c',code,root],cwd=Path.cwd(),capture_output=True,text=True)
    assert p.returncode==0,p.stdout+p.stderr
    assert p.stdout.count('PASS_CONSTRUCTED_ORIGINAL_GRAPH')==2
