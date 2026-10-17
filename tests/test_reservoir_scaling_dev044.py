"""DEV-044A predefined scale/retention controls and disjoint public outcome gates."""
import importlib.util
from pathlib import Path

import pytest

torch=pytest.importorskip("torch")
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location("dev044",ROOT/"scripts/run_reservoir_scaling_dev044.py")
m=importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def test_symmetric_writer_profile_initializations_and_frozen_parameters():
    m.paired_profiles(4401)
    assert len(m.ARM_NAMES)==9
    for name in m.PROFILES:
        a=m.build("trainable_"+name,4401)
        b=m.build("frozen_"+name,4401)
        assert a.state_scalars==b.state_scalars==768
        assert all(torch.equal(v,b.state_dict()[key]) for key,v in a.state_dict().items())
        assert not any(p.requires_grad for p in b.memory.parameters())


def test_predefined_scale_changes_are_not_overwritten_by_freezing():
    base=m.build("frozen_base",4401)
    sticky=m.build("frozen_sticky",4401)
    scaled=m.build("frozen_scaled",4401)
    assert not torch.equal(base.memory.decay_logits,sticky.memory.decay_logits)
    assert torch.equal(sticky.memory.decay_logits,scaled.memory.decay_logits)
    assert not torch.equal(sticky.memory.value.weight,scaled.memory.value.weight)
    assert not torch.equal(sticky.memory.key.weight,scaled.memory.key.weight)


def test_known_state_budget_and_baseline_control():
    assert m.build("gru768",4401).state_scalars==768
    assert m.build("delta768",4401).state_scalars==768
    assert m.build("schema41",4401).state_scalars==41


def test_calibration_vs_evaluation_seed_namespaces_disjoint():
    a={c["fingerprint"] for _,conds in m.cases_for("calibration",1,True) for c in conds}
    b={c["fingerprint"] for _,conds in m.cases_for("evaluation",1,True) for c in conds}
    assert len(a)==len(b)==2 and a.isdisjoint(b)


def test_counterfactual_policy_cost_from_all_entities():
    row=dict(probabilities=[[.9,.1,0.]]+[[0.,0.,0.]]*7,
             labels=[[1,0,1]]+[[0,0,0]]*7,
             deadline="at",role="reissued_cancel",target_entity=0)
    service=m.physical([row],.25)
    wait=m.physical([row],1.)
    assert service["total_cost"]==1
    assert service["failures"]==0 and service["unnecessary_services"]==0
    assert wait["total_cost"]==4 and wait["failures"]==1


def test_dev_selection_is_fixed_cost_minimization():
    yes=dict(probabilities=[[.9,.1,0.]]+[[0.,0.,0.]]*7,
             labels=[[1,0,1]]+[[0,0,0]]*7)
    assert m.select_threshold([yes])==.25
    assert m.PROFILES=={"base":(0.,1.,1.),"sticky":(2.,1.,1.),"scaled":(2.,.5,2.)}
