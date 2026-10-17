import inspect
import numpy as np
import pytest
import torch
from tac_osm.development.learned_stack import (
    ARMS, LearnedStack, PublicEvent, PhysicalWorld, collect, FittedFilter, independent_transitions)


@pytest.fixture(autouse=True)
def deterministic():
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)


def test_public_boundary_and_seed_namespaces():
    assert list(inspect.signature(LearnedStack.update).parameters) == ['self','state','public']
    assert list(inspect.signature(LearnedStack.predict).parameters) == ['self','state','action','tick']
    assert set(PublicEvent.__dataclass_fields__) == {'action','sensors','exact','tick'}
    a, _ = collect(8551001, 3)
    b, _ = collect(8551002, 3)
    assert not torch.equal(a[0], b[0])
    with pytest.raises(ValueError): LearnedStack().update(None, torch.full((1,10),float('nan')))


@pytest.mark.parametrize('arm', ARMS)
def test_incremental_matches_batch_and_restoration(arm):
    ds,_ = collect(8551003, 2, length=4)
    model = LearnedStack(13,arm).eval()
    with torch.no_grad():
        batch = model(ds[0],ds[1])
        state = model.initial(ds[0][:,0]); outputs=[]
        for t in range(4):
            state = model.update(state,ds[0][:,t])
            outputs.append(model.predict(state,ds[1][:,t],ds[0][:,t,-1]*20))
        torch.testing.assert_close(batch,torch.stack(outputs,1),rtol=1e-5,atol=1e-6)
        restored = LearnedStack(99,arm).eval()
        restored.load_state_dict(model.state_dict())
        torch.testing.assert_close(batch,restored(ds[0],ds[1]),rtol=0,atol=0)


def test_parameters_matched_and_component_gradients():
    full = LearnedStack(4)
    gru = LearnedStack(4,'gru_matched')
    assert abs(full.trainable_count-gru.trainable_count)/full.trainable_count <= .02
    ds,_ = collect(8551004,4,length=4)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(full(ds[0],ds[1]),ds[2])
    loss.backward()
    for group in ('encoder','memory','router','workspace','compute','head','query'):
        assert sum(float(p.grad.abs().sum()) for p in getattr(full,group).parameters() if p.grad is not None)>0,group
    assert full.last_work['operator_calls'] > 0
    assert full.last_work['operator_calls'] <= 4*4*2


def test_physics_replay_and_costs():
    a,b = PhysicalWorld(8551005,0,'drift'),PhysicalWorld(8551005,0,'drift')
    for action in [1,2,0,3,4]*4:
        ea,ca,la=a.step(action); eb,cb,lb=b.step(action)
        assert (ea,ca,la)==(eb,cb,lb)
        assert ca >= 4*sum(la['after'])
        if action in (3,4): assert ea.sensors[action-3] == la['after'][action-3]
    with pytest.raises(ValueError): a.step(-1)


def test_fitted_filter_uses_public_receipts_only():
    ds,labels=collect(8551006,8)
    for table in (FittedFilter.fit(labels,ds[1]),independent_transitions(labels,ds[1])):
        np.testing.assert_allclose(table.sum(-1),1)
        filt=FittedFilter(table)
        filt.update(PublicEvent(0,(0,1),(1,1),0),initial=True)
        np.testing.assert_allclose(filt.belief.sum(0),[0,1,0,0])
        assert filt.predict().shape == (5,2)


def test_memory_lesion_changes_forecast():
    ds,_=collect(8551007,3,length=4)
    model=LearnedStack(7).eval()
    with torch.no_grad():
        state=model.initial(ds[0][:,0])
        for t in range(4): state=model.update(state,ds[0][:,t])
        a=model.predict(state,ds[1][:,-1],torch.full((3,),4))
        b=model.predict(model.initial(ds[0][:,0]),ds[1][:,-1],torch.full((3,),4))
    assert not torch.allclose(a,b)
