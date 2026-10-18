"""Information provenance, belief/reference mathematics and learned deployment."""
from pathlib import Path
import sys
import copy
import pytest
import torch
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from data_belief_dev062 import BeliefWorld,PriceBelief,corpus,batch,snapshot
from run_belief_dev062 import gate_fit,make,objective,rollout
from tac_osm.development.belief_agent import configurations


@pytest.fixture(scope='module')
def data():
    torch.set_num_threads(1);specs=configurations();gate=gate_fit();m=make('tac',specs['tac'],44,gate);rows=corpus(9962001,8)
    return specs,gate,rows,m


def test_explicit_action_provenance_and_private_independent_schema():
    w=BeliefWorld(9962002,0);start=len(w.events);w.act(3);trace=w.events[start:]
    assert len(trace)==3 and trace[0][2] and trace[0][41]
    assert trace[1][3] and trace[2][3]
    w2=BeliefWorld(9962002,0);w2.bits=1-w2.bits;start=len(w2.events);w2.act(3)
    assert len(w2.events[start:])==3
    assert [int(e[:6].argmax()) for e in trace]==[int(e[:6].argmax()) for e in w2.events[start:]]


def test_exact_inspection_correction_and_action_prediction_match_enumeration():
    b=PriceBelief([-1,0]);b.condition(0,0,True);assert b.p[2:].sum()==0
    b.update(2,[0]);assert b.p[0]==1
    for matrix in b.transitions:
        np.testing.assert_allclose(matrix.sum(-1),1)
    assert b.policy()[0]==1


@pytest.mark.parametrize('arm',['tac','no_casm','gru','mlp'])
def test_all_matched_controls_train_on_same_public_data(data,arm):
    specs,gate,rows,_=data;m=make(arm,specs[arm],44,gate);x=batch(rows,range(8),m.gate);loss,_=objective(m,x);loss.backward()
    assert torch.isfinite(loss) and m.policy[-1].weight.grad.abs().sum()>0
    assert m.belief[-1].weight.grad.abs().sum()>0
    assert not any(p.requires_grad for p in m.gate.parameters())
    assert abs(specs[arm]['parameters']/specs['tac']['parameters']-1)<=.05


def test_identity_rotation_and_no_privileged_target_inference(data):
    _,_,rows,m=data;m=copy.deepcopy(m).double().eval();x=batch(rows,range(8),copy.deepcopy(m.gate).float());x={k:v.double() if v.is_floating_point() else v for k,v in x.items()};y=copy.deepcopy(x);rot,_=torch.linalg.qr(torch.randn(16,16,dtype=torch.float64))
    y['events'][:,:,6:22]=y['events'][:,:,6:22]@rot;y['events'][:,:,22:38]=y['events'][:,:,22:38]@rot;y['keys']=y['keys']@rot;y['candidates'][:,:,3:]=y['candidates'][:,:,3:]@rot
    def f(z):return m(z['events'],z['present'],z['keys'],z['candidates'],z['prices'])
    with torch.no_grad():
        a=f(x);b=f(y)
        torch.testing.assert_close(a['posterior'],b['posterior'],atol=1e-10,rtol=1e-10)
        y=dict(x,posterior=1-x['posterior'],future=1-x['future'],policy=x['policy'].roll(1,-1));b=f(y)
        torch.testing.assert_close(a['logits'],b['logits'],atol=0,rtol=0)


def test_corresponding_public_reference_has_no_access_to_modified_hidden_bits():
    w=BeliefWorld(9962003,0);a=w.reference().p.copy();w.bits=1-w.bits
    np.testing.assert_array_equal(w.reference().p,a)


def test_empty_initial_evidence_and_visited_state_relabelling(data):
    _,_,_,m=data;w=BeliefWorld(9962004,0);w.events=[w.record(5,None) for _ in range(32)];x=batch([snapshot(w,w.reference())],[0],m.gate)
    assert x['events'].shape[1]>=1
    assert not x['present'].any()
    with torch.no_grad():assert torch.isfinite(objective(m,x)[0])
    c,visited=rollout(copy.deepcopy(m).double(),9962005,32,2,2,True)
    assert len(visited)==sum(len(r['actions']) for r in c['rows'])
    assert all(abs(float(r['posterior'].sum())-1)<1e-6 for r in visited)
