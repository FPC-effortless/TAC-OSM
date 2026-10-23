"""Generated fixtures test mathematics/data boundaries only, never capability."""
import copy
import numpy as np
import torch
from scripts.me069_workflow_source import fit_preprocessing,encode_cases,split_cases
from scripts.me069_train_workflow import build,pack,predictions,objective
from tac_osm.workflow_me069 import WorkflowForecast,waiting_quantile


def cases():
    return [{"id":str(i),"events":[["OPEN|COMPLETE","r1",i*1000.+t*2]
        if t%2==0 else ["REVIEW|START","r2",i*1000.+t*2] for t in range(40)]}
        for i in range(10)]


def test_case_temporal_boundary_excludes_unavailable_labels():
    rows=cases();rows[0]["events"][-1][2]=6500
    groups,cuts=split_cases(rows)
    assert "0" not in {c["id"] for c in groups["FIT"]}
    assert cuts["FIT_crossing_excluded"]==1
    assert not ({c["id"] for c in groups["FIT"]}&{c["id"] for c in groups["DEV"]})


def test_train_vocabulary_and_features_do_not_use_case_id_or_future_attributes():
    original=cases();pre=fit_preprocessing(original[:6])
    changed=copy.deepcopy(original);changed[9]["events"][-1][0]="UNSEEN_TARGET"
    changed[0]["id"]="a-completely-different-identifier"
    a=encode_cases(original[:1],pre)[0];b=encode_cases(changed[:1],pre)[0]
    np.testing.assert_array_equal(a["features"],b["features"])
    assert "UNSEEN_TARGET" not in pre["activities"]


def test_native_states_are_bounded_and_predictions_are_causal():
    torch.manual_seed(69001)
    raw=cases();pre=fit_preprocessing(raw);data=pack(encode_cases(raw,pre),[(0,35),(1,35)])
    for arm in ("native_tac","gru_world"):
        model=build(arm,pre)
        with torch.no_grad():model.output.weight.normal_(std=.01)
        original=predictions(model,data)
        changed=list(data);changed[0]=data[0].clone();changed[0][:,20:]=1
        later=predictions(model,tuple(changed))
        torch.testing.assert_close(original.activity_logits.reshape(2,32,-1)[:,:20],
            later.activity_logits.reshape(2,32,-1)[:,:20],rtol=0,atol=0)
        state=model.initial(2);encoded=model.event_encoding(*data[:3])
        for t in range(32):state=model.observe(state,encoded[:,t],data[3][:,t],data[6][:,t])
        if arm=="native_tac":
            assert state.fast.abs().max()<=1 and state.slow.abs().max()<=1
            assert state.memory.square().sum((-2,-1)).sqrt().max()<=1.00001


def test_complete_workflow_objective_trains_finite_parameters():
    torch.manual_seed(69001);raw=cases();pre=fit_preprocessing(raw)
    data=pack(encode_cases(raw,pre),[(0,35),(1,35)])
    for arm in ("native_tac","gru_world"):
        m=build(arm,pre);o=torch.optim.AdamW(m.parameters(),lr=.001)
        for _ in range(2):
            o.zero_grad(set_to_none=True);loss=objective(m,data);loss.backward()
            assert torch.isfinite(loss)
            assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters())
            o.step()


def test_wait_distribution_handles_real_zero_mass():
    p=WorkflowForecast(torch.zeros(2,3),torch.tensor([-10.,10.]),
        torch.tensor([2.,2.]),torch.tensor([0.,0.]))
    median=waiting_quantile(p,.5)
    assert median[0]==0 and median[1]>0
    assert (waiting_quantile(p,.95)>=waiting_quantile(p,.05)).all()
