import copy
from pathlib import Path
import sys
import numpy as np
import pytest
import torch
from torch import nn
from tac_osm.development.learned_agent import AgentConfig
from tac_osm.development.temporal_agent import SparseTemporalAgent,proper_and_decision_losses
from tac_osm.development.uncertain_world import PublicBelief,UncertainWorld,STATES,COMMANDS,prior
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from evaluate_uncertainty_dev060 import calibration
from run_memory_foundations_dev060 import RetrievalProbe,data


def model(sparse=True):
    torch.set_num_threads(1);m=SparseTemporalAgent(AgentConfig(dimension=12),seed=600,sparse=sparse).eval()
    m.admission=nn.Linear(22,1)
    with torch.no_grad():m.admission.weight.zero_();m.admission.weight[0,0]=8;m.admission.bias.fill_(-4)
    return m


def test_sparse_closed_gate_skips_expensive_operators_and_preserves_state():
    m=model();state=m.initial(1);event=torch.zeros(1,22)
    called=[]
    hooks=[v.register_forward_hook(lambda *args:called.append(1)) for v in (m.address,m.encoder)]
    with torch.inference_mode():result,trace=m.observe(state,event)
    assert result is state and not called and trace['encoder_rows']==trace['address_rows']==0
    for hook in hooks:hook.remove()


def test_dense_sparse_inference_and_copy_on_write_are_exact():
    sparse=model();dense=model(False);dense.load_state_dict(sparse.state_dict())
    a=sparse.initial(1);b=dense.initial(1)
    with torch.inference_mode():
        for tick in range(12):
            e=torch.randn(1,22);e[0,0]=tick%3==0;e[0,21]=tick/256
            old=tuple(t.clone() for t in a);original=a
            a,ta=sparse.observe(a,e);b,tb=dense.observe(b,e)
            assert all(torch.equal(x,y) for x,y in zip(a,b))
            assert all(torch.equal(x,y) for x,y in zip(original,old))
            assert torch.equal(ta['address'],tb['address'])
        args=(torch.randn(1,13,11),torch.ones(1,13,dtype=torch.bool),torch.randn(1,6,8),torch.ones(1,6,dtype=torch.bool))
        assert torch.equal(sparse(a,*args)['logits'],dense(b,*args)['logits'])


def test_temporal_bias_prefers_newer_identical_content_without_entity_rules():
    m=model();state=m.initial(1);state[0][0,0,:12]=1;state[0][0,1,:12]=1
    state[0][0,0,12]=.1;state[0][0,1,12]=.9;state[1][0,:2]=1
    _,scores=m.read(state,torch.randn(1,1,12))
    assert (scores[:,:,:,1]>scores[:,:,:,0]).all()
    scores[:,:,:,:2].sum().backward();assert m.raw_decay.grad.abs().sum()>0


def test_asymmetric_risk_does_not_reweight_proper_probability_targets():
    completion=torch.randn(4,requires_grad=True);outcomes=torch.randn(4,5,2,requires_grad=True)
    output={'completion':completion,'outcomes':outcomes,'logits':torch.randn(4,5,requires_grad=True)}
    target=torch.rand(4);y=torch.rand(4,5,2);risk=torch.rand(4,5);valid=torch.ones(4,5,dtype=torch.bool)
    a=proper_and_decision_losses(output,target,y,risk,valid);b=proper_and_decision_losses(output,target,y,risk*100,valid)
    assert torch.equal(a['completion_proper'],b['completion_proper']) and torch.equal(a['outcome_proper'],b['outcome_proper'])
    torch.testing.assert_close(b['decision_risk'],100*a['decision_risk'])


def test_bayes_filter_matches_independent_joint_likelihood_and_inspection():
    b=PublicBelief([-1,0]);expected=prior()*np.where(STATES[:,0]==1,.9,.1);expected/=expected.sum()
    np.testing.assert_allclose(b.condition(0,1),expected)
    prediction=expected@b.transitions[4]
    likelihood=np.prod(np.where(STATES==[0,1],.9,.1),axis=1)
    expected=prediction*likelihood;expected/=expected.sum()
    np.testing.assert_allclose(b.update(4,[0,1]),expected)
    result=b.update(1,[0]);assert result[STATES[:,0]==1].sum()==0
    assert np.isclose(b.transitions[3,0,1],.8)  # unnecessary parent repair faults child
    assert np.isclose(b.transitions[4,1,0],.95) # repair child with healthy parent
    assert b.transitions[4,3,3]==1              # parent blocks child repair
    for action in range(5):
        assert np.isclose(sum(m for m,_ in b.branches(b.p,action)),1)
        assert np.allclose(b.transitions[action].sum(-1),1)


def test_bounded_planner_acquires_information_and_depth_one_matches_enumeration():
    b=PublicBelief([-1,-1]);cost=b.action_costs(depth=1)
    for action,(kind,_) in enumerate(COMMANDS):
        expected=20*np.dot(b.p@b.transitions[action],STATES.sum(-1))+(0 if kind=='stop' else .11 if kind=='inspect' else .62)
        assert np.isclose(cost[action],expected)
    # Low fault probability makes a paid exact inspection cheaper than a blind repair.
    b.p=np.array([.9,0.,.1,0.])
    assert COMMANDS[b.choose(2)][0]=='inspect'
    with pytest.raises(ValueError):b.choose(0)


def test_public_payload_and_posterior_cannot_read_hidden_bits_or_parents():
    w=UncertainWorld(6001,0);payload,_=w.payload();reference=w.reference().p.copy()
    w.bits=1-w.bits;w.parents[:]=-1
    other,_=w.payload()
    for key in payload:np.testing.assert_array_equal(payload[key],other[key])
    np.testing.assert_array_equal(w.reference().p,reference)
    assert set(payload)=={'events','candidates','queries'}


def test_repair_receipt_schema_independent_of_hidden_change_and_exact_inspect_tagged():
    for index in range(10):
        w=UncertainWorld(6002,index);w.act(3)
        assert [r['subject'] for r in w.events[-2:]]==w.identities.tolist()
        assert all(r['source']=='repair' for r in w.events[-2:])
        w.act(1);payload,_=w.payload()
        assert payload['events'][-1,22]==1 and payload['events'][-1,23]==0
        assert int(payload['events'][-1,20])==w.bits[0]


def test_identical_public_history_can_require_different_private_actions():
    a=UncertainWorld(6003,0);b=copy.deepcopy(a)
    a.bits=np.array([1,0]);b.bits=np.array([0,1]);a.parents[:]=-1;b.parents[:]=-1
    for key,value in a.payload()[0].items():np.testing.assert_array_equal(value,b.payload()[0][key])
    np.testing.assert_array_equal(a.reference().p,b.reference().p)
    assert np.flatnonzero(a.bits)[0]!=np.flatnonzero(b.bits)[0]


def test_both_probe_arms_receive_time_and_new_continuous_identities():
    ds=data(6004,8,16);a=RetrievalProbe(False,600);b=RetrievalProbe(True,600)
    assert all(torch.equal(x,y) for x,y in zip(a.content.parameters(),b.content.parameters()))
    assert ds['records'].shape==(8,16,10) and ds['query'].shape==(8,8)
    result=calibration([0,.5,1],[0,1,1]);assert result['count']==3
    assert sum(v['count'] for v in result['bins'])==3
    boundary=calibration(np.arange(11)/10,np.zeros(11))
    assert sum(v['count'] for v in boundary['bins'])==11
    with pytest.raises(ValueError):calibration([1.1],[0])
    with pytest.raises(ValueError):calibration([.5],[.5])
