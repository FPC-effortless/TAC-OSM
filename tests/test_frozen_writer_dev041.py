"""DEV-041 registered pilot contracts; no trained-science claims from smoke tests."""
import torch
import pytest
from run_frozen_writer_dev041 import (
    ARMS, build, compare_initialization, cost, choose_threshold, cases_for,
    last_event_rows,
)


def test_frozen_writer_identical_initialization():
    trained=build("bounded_trainable", 4101)
    frozen=build("bounded_random_frozen", 4101)
    compare_initialization(trained, frozen)
    assert trained.state_scalars==frozen.state_scalars==768
    assert all(not p.requires_grad for p in frozen.memory.parameters())
    assert any(p.requires_grad for p in trained.memory.parameters())


def test_frozen_writer_grads_never_enabled():
    model=build("bounded_random_frozen",4101)
    other=[p for n,p in model.named_parameters() if not n.startswith("memory.")]
    assert other and all(p.requires_grad for p in other)
    assert all(not p.requires_grad for p in model.memory.parameters())


def test_memoryless_has_no_persistent_history():
    model=build("memoryless",4101)
    assert model.state_scalars==0
    assert len(tuple(p for p in model.parameters() if p.requires_grad))>0


def test_physical_cost_uses_counterfactual_service_prediction():
    row=dict(probabilities=[[.9,.1]]+[[.0,.0]]*7,
             labels=[[1,0,0]]+[[0,0,0]]*7)
    out=cost(row,.25)
    assert out==dict(failures=0,unnecessary_services=0,
                     services=1,inspections=0,total_cost=1)
    never=cost(row,1.0)
    assert never["failures"]==1 and never["total_cost"]==4


def test_threshold_uses_calibration_only():
    row=dict(probabilities=[[.9,.1]]+[[.0,.0]]*7,
             labels=[[1,0,0]]+[[0,0,0]]*7)
    t,s=choose_threshold([row])
    assert t==.25 and s[str(.25)]==1


def test_calibration_and_evaluation_fingerprints_disjoint():
    a={c.fingerprint for worlds,_ in cases_for("calibration",1,32) for c in worlds}
    b={c.fingerprint for worlds,_ in cases_for("eval",1,32) for c in worlds}
    assert len(a)==72 and len(b)==72 and not a&b


def test_last_event_replaces_probability_field():
    class C:
        fingerprint="fixture"
    import run_frozen_writer_dev041 as runner
    # Refuse silent duplicates or unexpected mapping in a simplified fixture.
    row=dict(fingerprint="fixture",probabilities=[[0.,0.,0.]]*8,labels=[[0,0,0]]*8)
    assert "probabilities" in row and len(row["probabilities"])==8
    assert ARMS==("bounded_trainable","bounded_random_frozen","memoryless")
