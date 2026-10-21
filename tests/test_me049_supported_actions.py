"""ME049 checks implement numerical/data-integrity safeguards only.
Constructed fixture rows here are NOT real-world performance evidence.
"""
import gzip
import pytest
import torch
from tac_osm.me049_action_state import ActionStateTAC,make_model,ARMS,authentic_logged_action_loss
from scripts import me049_train_supported_actions as study


def test_me049_action_changes_recurrent_native_dynamics_with_real_typed_contract():
    torch.manual_seed(49049)
    net=ActionStateTAC()
    x=torch.randn(4,12)
    g=torch.zeros(4,dtype=torch.long)
    z=torch.zeros(4,dtype=torch.long)
    o=torch.ones(4,dtype=torch.long)
    all_outputs=net.forward(x,g)
    assert all_outputs.shape==(4,2) and torch.isfinite(all_outputs).all()
    assert torch.allclose(all_outputs[:,0],net.logits_for_logged_action(x,g,z))
    assert torch.allclose(all_outputs[:,1],net.logits_for_logged_action(x,g,o))
    with pytest.raises(ValueError):
        net.logits_for_logged_action(x,g,torch.ones(4,dtype=torch.long)*2)


def test_me049_unobserved_action_has_no_supervised_embedding_gradient():
    torch.manual_seed(49050)
    net=ActionStateTAC()
    x=torch.randn(5,12)
    action=torch.ones(5,dtype=torch.long)
    loss=authentic_logged_action_loss(net,x,action,torch.ones(5),torch.zeros(5))
    loss.backward()
    assert net.action_token.weight.grad is not None
    assert torch.all(net.action_token.weight.grad[0]==0)
    assert torch.isfinite(net.action_token.weight.grad).all()
    assert net.state.transition.weight.grad is not None
    for name in ARMS:
        assert make_model(name).num_active()>0


def _write_small_numeric_fixture(path,n,all_treated=False):
    with gzip.open(path,"wt") as f:
        f.write(",".join([f"f{i}" for i in range(12)]+[
             "treatment","visit","conversion","exposure"])+"\n")
        for idx in range(n):
            a=1 if all_treated else idx%2
            row=[str(float(idx%7+i)) for i in range(12)]
            row+= [str(a),str(int((idx//2)%3!=0)),
                    str(int((idx//2)%5==0)),str(a)]
            f.write(",".join(row)+"\n")


def test_me049_preflight_enforces_authentic_cohort_action_support_without_outcome_peeking(tmp_path,monkeypatch):
    path=tmp_path/"original-numeric-fixture.csv.gz"
    _write_small_numeric_fixture(path,2000)
    monkeypatch.setattr(study,"N_ALL",2000)
    monkeypatch.setattr(study,"CANDIDATES",1500)
    monkeypatch.setattr(study,"PER_ACTION",{"FIT":200,"DEV":100,"TEST":100})
    splits,manifest=study.plan_supported_indices(path)
    assert not manifest["split_selection_saw_features_or_outcomes"]
    assert len(set(splits["FIT"])&set(splits["TEST"]))==0
    assert set(splits)=={"FIT","DEV","TEST"}
    cohort=study.load_partition(path,splits["FIT"],400)
    summary=study.cohort_summary(cohort)
    assert summary["0"]["n"]==summary["1"]["n"]==200
    assert summary["0"]["visit_positive"]>0
    assert summary["1"]["conversion_positive"]>0


def test_me049_fails_closed_when_action_support_missing(tmp_path,monkeypatch):
    path=tmp_path/"only-one-treatment.csv.gz"
    _write_small_numeric_fixture(path,200,all_treated=True)
    monkeypatch.setattr(study,"N_ALL",200)
    monkeypatch.setattr(study,"CANDIDATES",150)
    monkeypatch.setattr(study,"PER_ACTION",{"FIT":5,"DEV":5,"TEST":5})
    with pytest.raises(RuntimeError,match="ACTION_SUPPORT_FAILURE"):
        study.plan_supported_indices(path)
