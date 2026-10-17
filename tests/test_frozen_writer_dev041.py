"""DEV-041 contract checks for writer attribution and public case separation."""
import importlib.util
from pathlib import Path

import pytest

torch=pytest.importorskip("torch")
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location("dev041_runner",ROOT/"scripts/run_frozen_writer_dev041.py")
m=importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def test_frozen_writer_identical_initialization():
    trained=m.build("bounded_trainable", 4101)
    frozen=m.build("bounded_random_frozen", 4101)
    m.compare_initialization(trained, frozen)
    assert trained.state_scalars==frozen.state_scalars==768
    assert all(not p.requires_grad for p in frozen.memory.parameters())
    assert any(p.requires_grad for p in trained.memory.parameters())


def test_frozen_writer_nonmemory_params_trainable():
    model=m.build("bounded_random_frozen",4101)
    other=[p for n,p in model.named_parameters() if n.startswith(("event_encoder.","query_encoder.","adapter.","head."))]
    assert other and all(p.requires_grad for p in other)
    assert all(not p.requires_grad for p in model.memory.parameters())


def test_memoryless_has_no_persistent_history():
    model=m.build("memoryless",4101)
    assert model.state_scalars==0
    assert len(tuple(p for p in model.parameters() if p.requires_grad))>0


def test_physical_cost_uses_action_counterfactual():
    row=dict(probabilities=[[.9,.1]]+[[.0,.0]]*7,
             labels=[[1,0,0]]+[[0,0,0]]*7)
    out=m.cost(row,.25)
    assert out==dict(failures=0,unnecessary_services=0,
                     services=1,inspections=0,total_cost=1)
    never=m.cost(row,1.0)
    assert never["failures"]==1 and never["total_cost"]==4


def test_threshold_is_fixed_calibration_grid():
    row=dict(probabilities=[[.9,.1]]+[[.0,.0]]*7,
             labels=[[1,0,0]]+[[0,0,0]]*7)
    t,s=m.choose_threshold([row])
    assert t==.25 and s[str(.25)]==1


def test_calibration_and_evaluation_fingerprints_disjoint():
    a={c.fingerprint for worlds,_ in m.cases_for("calibration",1,32) for c in worlds}
    b={c.fingerprint for worlds,_ in m.cases_for("eval",1,32) for c in worlds}
    assert len(a)==72 and len(b)==72 and not a&b


def test_last_event_uses_public_features_and_replaces_probabilities():
    worlds=list(m.cases_for("calibration",1,32))
    original=[dict(fingerprint=c.fingerprint,probabilities=[[.0,.0,.0]]*8,
                   labels=[[0,0,0]]*8) for cs,_ in worlds for c in cs]
    table={k:[.75,.2,.0] for cs,_ in worlds for c in cs for k in m.last_keys(c)}
    old=m.cases_for
    try:
        m.cases_for=lambda split,reps,history=512: iter(worlds)
        result=m.last_event_rows(table,original,"calibration",1)
    finally:
        m.cases_for=old
    assert len(result)==72
    assert all(r["probabilities"]==[[.75,.2,.0]]*8 for r in result)
    assert all(r["labels"]==[[0,0,0]]*8 for r in result)
