"""Public binding, deployment independence and numeric decoding invariants."""
from pathlib import Path
import sys
import copy
import pytest
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from data_identity_dev0615 import corpus,canonical,batch,IdentityWorld
from run_identity_dev0615 import setup_gate,make,objective,rollout
from tac_osm.development.robust_identity import configurations,choose


@pytest.fixture(scope='module')
def inputs():
    torch.set_num_threads(1)
    specs=configurations();gate=setup_gate();m=make('split_robust',specs['split_robust'],11,gate)
    rows=canonical(corpus(9916150,8),m.admission)
    return specs,gate,rows


@pytest.mark.parametrize('arm',['fixed_bc','split_robust','no_casm_robust','raw_gru_robust','raw_mlp_robust'])
def test_all_controls_have_finite_trainable_predictions(inputs,arm):
    specs,gate,rows=inputs;m=make(arm,specs[arm],11,gate)
    loss,_=objective(m,batch(rows,range(8),m,torch.Generator().manual_seed(4),not m.fixed))
    loss.backward()
    assert torch.isfinite(loss)
    assert m.policy[-1].weight.grad.abs().sum()>0
    assert not any(p.requires_grad for p in m.admission.parameters())


def predict(m,x):
    return m(x['events'],x['present'],x['keys'],x['query_valid'],x['candidates'],x['valid'])


@pytest.mark.parametrize('arm',['split_robust','no_casm_robust'])
def test_identity_rotation_and_manifest_permutation_preserve_actions(inputs,arm):
    specs,gate,rows=inputs;m=make(arm,specs[arm],11,gate).double().eval()
    x=batch(rows,range(8),m,torch.Generator().manual_seed(4),True)
    x={k:v.double() if v.is_floating_point() else v for k,v in x.items()}
    with torch.no_grad():
        expected=predict(m,x);rot,_=torch.linalg.qr(torch.randn(16,16,dtype=torch.float64));y=copy.deepcopy(x)
        for start,end in [(4,20),(20,36)]:y['events'][:,:,start:end]=y['events'][:,:,start:end]@rot
        y['keys']=y['keys']@rot;y['candidates'][:,:,3:]=y['candidates'][:,:,3:]@rot
        actual=predict(m,y)
        torch.testing.assert_close(actual['logits'],expected['logits'],atol=1e-10,rtol=1e-10)
        perm=torch.tensor([5,3,1,0,2,4]);y=copy.deepcopy(x);y['keys']=y['keys'][:,perm];y['query_valid']=y['query_valid'][:,perm]
        actual=predict(m,y)
        torch.testing.assert_close(actual['logits'],expected['logits'],atol=1e-10,rtol=1e-10)


def test_cached_public_history_equals_incremental_runtime_and_no_private_labels(inputs):
    specs,gate,rows=inputs;m=make('split_robust',specs['split_robust'],11,gate).eval()
    x=batch(rows,range(8),m);s=m.initial(x['keys'],x['query_valid'])
    with torch.no_grad():
        for t in range(x['events'].shape[1]):s,_=m.observe(s,x['events'][:,t],x['present'][:,t])
        a=m.decide(s,x['candidates'],x['valid']);b=predict(m,x)
    torch.testing.assert_close(a['logits'],b['logits'],atol=1e-6,rtol=1e-6)
    y=dict(x,current=1-x['current'],policy=x['policy'].roll(1,-1),one=1-x['one'])
    torch.testing.assert_close(predict(m,y)['logits'],b['logits'],atol=0,rtol=0)
    assert set(s)=={'keys','entity_valid','events','present'}


def test_epsilon_decoder_uses_only_valid_choices():
    z=torch.tensor([[5.,5.0000005,100.],[1.,1.000002,100.]],dtype=torch.float64)
    assert choose(z,torch.tensor([[1,1,0],[1,1,0]],dtype=torch.bool)).tolist()==[0,1]


def test_seen_and_unseen_share_physics_but_have_disjoint_label_names():
    a=IdentityWorld(9916151,0,6,128,'seen');b=IdentityWorld(9916151,0,6,128,'unseen')
    assert a.bits.tolist()==b.bits.tolist() and a.parents.tolist()==b.parents.tolist()
    assert max(a.labels)<80<=min(b.labels)
    assert not torch.equal(torch.tensor(a.public()),torch.tensor(b.public()))


def test_teacher_relabels_each_visited_student_state(inputs):
    specs,gate,_=inputs;m=make('split_robust',specs['split_robust'],11,gate).double().eval()
    cells,visited=rollout(m,9916152,2,32,2,'fresh',2,True)
    assert len(visited)==sum(len(r['actions']) for r in cells['rows'])
    assert all(abs(float(r['policy'].sum())-1)<1e-6 for r in visited)
    assert all(r['depth']<2 for r in visited)
