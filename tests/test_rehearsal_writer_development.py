"""DEV-012 mechanism tests: bounded recurrent writes, rehearsal, and leak controls."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
from tac_osm.development.rehearsal_writer import (  # noqa: E402
    RehearsalWriterModel, make_matching_models,
)
from tac_osm.development.protection_benchmark import make_pair, tensors  # noqa: E402
from run_rehearsal_writer_dev import (  # noqa: E402
    ARMS, train_case, sample_rehearsal_inputs, summarize, evaluate, conditions,
)


def test_all_arms_match_bytewise_initialization_and_protect_modes():
    arms=make_matching_models(43)
    assert set(arms)==set(ARMS)
    first=arms[ARMS[0]].state_dict()
    for arm in ARMS[1:]:
        assert all(torch.equal(first[k],v) for k,v in arms[arm].state_dict().items())
    assert arms["learned_rehearsal"].mode=="learned_protection"
    assert arms["fixed_rehearsal"].mode=="fixed_protection"


def test_state_size_and_original_casm_decision_path_preserved():
    pair=make_pair(43,32,4,1.,.08,"retention")
    a,v,q,c=tensors([pair.left,pair.right])
    model=RehearsalWriterModel(mode="always_update")
    state=model.encode(a,v,trace=False)
    assert state.shape==(2,4,8,24)
    assert state[0].numel()==768
    assert model.predict(state,q,c).shape==(2,2)
    assert model.rehearsal_logits(state,q).shape==(2,2)


def test_auxiliary_supervision_reaches_real_writer_and_protection_gate():
    pair=make_pair(47,32,4,1.,.08,"acquisition")
    a,v,q,c=tensors([pair.left,pair.right])
    model=RehearsalWriterModel(mode="learned_protection")
    state=model.encode(a,v,trace=False)
    aux=F.cross_entropy(model.rehearsal_logits(state,q),torch.tensor(
        [pair.left.target^pair.left.current_bit,pair.right.target^pair.right.current_bit]))
    aux.backward()
    grad=model.memory.write.weight.grad
    pgrad=model.memory.protection[-1].weight.grad
    assert grad is not None and torch.isfinite(grad).all() and grad.abs().sum()>0
    assert pgrad is not None and torch.isfinite(pgrad).all() and pgrad.abs().sum()>0


def test_rehearsal_sampling_only_observed_visible_events_and_no_query_target_identity():
    a,v,q,c,y,_,h,d,noise=train_case(43,7)
    old,fresh=sample_rehearsal_inputs(a,v,h,d,noise,43,7)
    assert old[0].shape==fresh[0].shape==(4,12)
    assert old[1].shape==fresh[1].shape==(4,)
    assert all(t.item() in (0,1) for t in old[1])
    assert torch.isfinite(old[0]).all() and torch.isfinite(fresh[0]).all()
    # Offline memory rehearsal timestamps do not carry the evaluator's
    # queried history slot or target into model memory update/query forward.
    for j in range(4):
        i=(7*11+j//2*7)%h
        k=h+(7*7+j//2*3)%d
        assert old[1][j]==v[j,i] and fresh[1][j]==v[j,k]


def test_no_memory_predictor_cannot_discriminate_opposite_past():
    model=RehearsalWriterModel()
    pair=make_pair(51,128,16,1.,.25,"retention")
    a,v,q,c=tensors([pair.left,pair.right])
    assert torch.equal(a[0],a[1]) and torch.equal(q[0],q[1]) and torch.equal(c[0],c[1])
    state=model.encode(a,v,trace=False)
    reset=torch.zeros_like(state)
    assert torch.equal(model.predict(reset,q,c)[0],model.predict(reset,q,c)[1])
    assert torch.equal(model.rehearsal_logits(reset,q)[0],model.rehearsal_logits(reset,q)[1])


def test_update_gate_never_hides_full_incoming_observation_from_writer():
    torch.manual_seed(71)
    for mode in ("always_update","fixed_protection","learned_protection"):
        model=RehearsalWriterModel(mode=mode)
        address=torch.randn(2,12)
        state=model.memory.initial(2,address)
        new=model.memory.update(state,address,torch.tensor([0,1]))
        assert new.shape==state.shape and torch.isfinite(new).all()
        gates=model.memory.last_gate
        assert gates is not None and gates.shape==(2,4)
        assert ((gates>=0)&(gates<=1)).all()
        assert new.abs().sum()>0


def test_eval_conditions_pair_roles_and_long_delay():
    c=conditions(False)
    assert len(c)==80
    assert len({tuple(x) for x in c})==80
    assert any(x[0]==512 and x[1]==32 for x in c)
    assert {x[4] for x in c}=={"retention","acquisition"}


def test_invalid_auxiliary_input_shapes_fail_closed():
    model=RehearsalWriterModel()
    state=torch.zeros(2,4,8,24)
    with pytest.raises(ValueError):
        model.rehearsal_logits(state,torch.zeros(2,11))
    with pytest.raises(ValueError):
        model.rehearsal_logits(torch.zeros(2,4,8,23),torch.zeros(2,12))
    with pytest.raises(ValueError):
        model.rehearsal_logits(state,torch.full((2,12),float("nan")))
