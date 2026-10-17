"""DEV-046 vertical-slice regression: public events, local write, forecasts, safe actions."""
import pytest
torch=pytest.importorskip("torch")

from tac_osm.development.unified_state_prototype import (
    EntityLocalState,UnifiedPredictiveStatePrototype
)
from tac_osm.development.decision_cost_policy import Decision


def test_public_event_to_action_forecast_tensor_contract():
    torch.manual_seed(46)
    model=UnifiedPredictiveStatePrototype(entities=4,dim=12,budget=2)
    events=torch.randn(2,5,16)
    ids=torch.tensor([[0,1,2,3,0],[2,1,0,3,2]],dtype=torch.long)
    ticks=torch.arange(1,6).float().expand(2,-1)
    result=model(events,ids,ticks,torch.tensor([4.,12.]))
    assert result.p_wait.shape==result.p_service.shape==(2,4)
    assert result.selected_entities.shape==(2,2) and result.retrievals==2
    assert (result.p_wait>=0).all() and (result.p_wait<=1).all()


def test_local_update_has_exact_untouched_nonaddressed_entity():
    torch.manual_seed(46)
    m=EntityLocalState(entities=4,dim=12)
    a=m.initial(2,torch.device("cpu"),torch.float32)
    b=m.step(a,torch.randn(2,12),torch.tensor([0,2]),torch.tensor([5.,5.]))
    assert torch.equal(a.fast[0,1:],b.fast[0,1:])
    assert torch.equal(a.slow[1,0:2],b.slow[1,0:2])
    assert not torch.equal(a.fast[0,0],b.fast[0,0])


def test_rejected_verified_receipt_never_changes_state():
    m=EntityLocalState(entities=4,dim=12)
    a=m.initial(1,torch.device("cpu"),torch.float32)
    nextstate=m.step(a,torch.randn(1,12),torch.tensor([0]),torch.tensor([5.]),
                     admit=torch.tensor([False]))
    assert nextstate is a


def test_causal_event_order_can_change_state_with_same_final_event():
    torch.manual_seed(46)
    model=UnifiedPredictiveStatePrototype(entities=2,dim=8,budget=1)
    x=torch.randn(1,3,16)
    y=x[:,[1,0,2]]
    ids=torch.zeros(1,3,dtype=torch.long)
    ticks=torch.tensor([[1.,2.,3.]])
    a=model.encode(x,ids,ticks)
    b=model.encode(y,ids,ticks)
    assert torch.equal(x[:,-1],y[:,-1])
    assert not torch.allclose(a.event[:,0],b.event[:,0])


def test_absent_action_feasibility_requires_human_review():
    model=UnifiedPredictiveStatePrototype(entities=2,dim=8,budget=1)
    admissible=torch.zeros((1,2,3),dtype=torch.bool)
    out=model(torch.zeros(1,1,16),torch.zeros(1,1,dtype=torch.long),
              torch.ones(1,1),torch.ones(1),admissible=admissible)
    assert (out.decisions.action==Decision.HUMAN_REVIEW).all()
    assert out.decisions.requires_human_review.all()


def test_relevance_budget_validation_is_explicit():
    with pytest.raises(ValueError):
        UnifiedPredictiveStatePrototype(entities=2,dim=8,budget=3)(
            torch.zeros(1,1,16),torch.zeros(1,1,dtype=torch.long),
            torch.ones(1,1),torch.ones(1))


def test_action_conditioned_prediction_gradients_reach_public_encoder():
    torch.manual_seed(46)
    model=UnifiedPredictiveStatePrototype(entities=2,dim=8,budget=1)
    inp=torch.randn(1,3,16)
    ids=torch.zeros(1,3,dtype=torch.long)
    t=torch.tensor([[1.,2.,3.]])
    result=model(inp,ids,t,torch.tensor([12.]))
    loss=-(torch.log(result.p_wait[:,0]+1e-6)+torch.log(1-result.p_service[:,0]+1e-6)).mean()
    loss.backward()
    assert model.observation[0].weight.grad is not None
    assert model.observation[0].weight.grad.abs().sum()>0
    assert model.world_predictor.action.weight.grad is not None
    assert model.world_predictor.action.weight.grad.abs().sum()>0
