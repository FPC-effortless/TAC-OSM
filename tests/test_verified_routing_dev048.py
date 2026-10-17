import pytest
torch = pytest.importorskip("torch")
from tac_osm.development.unified_state_prototype import UnifiedPredictiveStatePrototype
from tac_osm.development.verified_routing import sparse_plan, admissible_services, commit_public_receipts
from tac_osm.development.decision_cost_policy import Decision


def test_sparse_predictor_receives_only_the_retrieved_entities():
    model = UnifiedPredictiveStatePrototype(entities=8,dim=8)
    state = model.memory.initial(2,torch.device("cpu"),torch.float32)
    seen = []
    hook = model.world_predictor.register_forward_pre_hook(lambda m,args: seen.append(args[0].shape))
    plan = sparse_plan(model,state,torch.ones(2),2)
    hook.remove()
    assert seen==[torch.Size([2,2,24])]
    assert plan.predicted_entities==4 and plan.routing_candidates==16
    assert (plan.actions==Decision.HUMAN_REVIEW).sum()==12
    with pytest.raises(ValueError):
        admissible_services(plan.actions)


def test_execution_capacity_constraint_is_checked_before_transaction():
    with pytest.raises(ValueError):
        admissible_services(torch.ones(1,8,dtype=torch.long),max_services=2)
    assert admissible_services(torch.tensor([[0,1,0,1]]),max_services=2)==[(1,3)]


def test_valid_corrective_receipt_is_not_rejected_for_disagreeing_with_forecast():
    model = UnifiedPredictiveStatePrototype(dim=8)
    state = model.memory.initial(1,torch.device("cpu"),torch.float32)
    x = torch.randn(1,16)
    rejected = commit_public_receipts(model,state,x,torch.tensor([2]),torch.tensor([5.]),torch.tensor([False]))
    assert rejected is state
    corrected = commit_public_receipts(model,state,x,torch.tensor([2]),torch.tensor([5.]),torch.tensor([True]))
    assert not torch.equal(corrected.fast,state.fast)
    with pytest.raises(ValueError):
        commit_public_receipts(model,corrected,x,torch.tensor([2]),torch.tensor([4.]),torch.tensor([True]))
