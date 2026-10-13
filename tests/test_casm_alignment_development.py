"""DEV-013 CASM/readout alignment mechanism and counterfactual tests."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
from tac_osm.development.casm_readout_alignment import (  # noqa: E402
    ARMS,CASMAlignedModel,matching_models,
)
from tac_osm.development.rehearsal_writer import RehearsalWriterModel  # noqa: E402
from tac_osm.development.protection_benchmark import make_pair,tensors,seed_for  # noqa: E402
from run_casm_alignment_dev import conditions,evaluate  # noqa: E402


def _batch(seed=43):
    p=make_pair(seed,32,4,1.0,.08,"retention")
    return tensors([p.left,p.right]),p


def test_native_casm_decision_matches_existing_tac_osm_before_training():
    torch.manual_seed(3)
    native=RehearsalWriterModel()
    torch.manual_seed(3)
    aligned=CASMAlignedModel()
    x,p=_batch()
    native.eval()
    aligned.eval()
    with torch.no_grad():
        state=native.encode(x[0],x[1],trace=False)
        expected=native.predict(state,x[2],x[3])
        result=aligned.predict(state,x[2],x[3])
    torch.testing.assert_close(result,expected,rtol=2e-6,atol=2e-6)
    assert aligned.last_alignment_dispatch_calls>0


def test_same_initialization_and_768_scalar_state_all_five_arms():
    models=matching_models(43)
    assert set(models)==set(ARMS)
    first=models[ARMS[0]].state_dict()
    for model in models.values():
        assert all(torch.equal(first[k],v) for k,v in model.state_dict().items())
        x,_=_batch()
        state=model.encode(x[0],x[1],trace=False)
        assert state.shape==(2,4,8,24)
        assert state[0].numel()==768


def test_auxiliary_aligned_gradient_reaches_memory_and_actual_casm_selector():
    torch.manual_seed(11)
    model=CASMAlignedModel()
    (a,v,q,c),p=_batch(seed=47)
    state=model.encode(a,v,trace=False)
    bits=torch.tensor([(ep.target^ep.current_bit) for ep in (p.left,p.right)])
    loss=F.cross_entropy(model.aligned_historical_logits(state,q),bits)
    loss.backward()
    for grad in (model.memory.write.weight.grad,
                 model.read_selector[-1].weight.grad,
                 model.compute.selector[-1].weight.grad):
        assert grad is not None and torch.isfinite(grad).all() and grad.abs().sum()>0
    assert model.shared_head.weight.grad is not None
    assert model.shared_head.weight.grad.abs().sum()>0


def test_detached_aux_does_not_reach_writer_or_casm():
    model=CASMAlignedModel()
    (a,v,q,c),p=_batch(seed=49)
    state=model.encode(a,v,trace=False)
    bits=torch.tensor([(ep.target^ep.current_bit) for ep in (p.left,p.right)])
    loss=F.cross_entropy(
        model.aligned_historical_logits(state,q,detach_features=True),bits)
    loss.backward()
    assert model.shared_head.weight.grad is not None
    assert model.shared_head.weight.grad.abs().sum()>0
    assert model.memory.write.weight.grad is None
    assert model.read_selector[-1].weight.grad is None
    assert model.compute.selector[-1].weight.grad is None


def test_no_oracle_target_or_query_role_in_predict_signature():
    model=CASMAlignedModel()
    (a,v,q,c),p=_batch(seed=53)
    assert p.left.target!=p.right.target
    assert torch.equal(q[0],q[1]) and torch.equal(c[0],c[1])
    assert torch.equal(a[0],a[1])
    with torch.no_grad():
        state=model.encode(a,v,trace=False)
        logits=model.predict(state,q,c)
        assert logits.shape==(2,2)
        reset=model.predict(torch.zeros_like(state),q,c)
    torch.testing.assert_close(reset[0],reset[1],atol=1e-6,rtol=1e-6)


def test_diagnostic_xor_is_not_actual_casm_answer():
    model=CASMAlignedModel()
    rows=evaluate(model,241,"aligned_aux",smoke=True)
    assert len(rows)==1
    row=rows[0]
    pair=make_pair(seed_for("dev013-eval",241,32,4,0.0,.08,
                            "retention",0),32,4,0.0,.08,"retention")
    current=tensors([pair.left,pair.right])[3].tolist()
    assert row["diagnostic_xor_pred"]==[int(b)^int(c)
                                         for b,c in zip(row["bit_pred"],current)]
    # The original task decision is separately computed via predict and
    # not replaced with the diagnostic XOR shortcut.
    assert "task_pred" in row
    assert row["state_scalars_per_episode"]==768


def test_eval_registered_conditions_and_roles():
    full=conditions(False)
    assert len(full)==80
    assert {x[4] for x in full}=={"retention","acquisition"}
    assert any(x[0]==512 and x[1]==32 for x in full)


def test_invalid_read_inputs_rejected():
    m=CASMAlignedModel()
    state=torch.zeros(2,4,8,24)
    with pytest.raises(ValueError):
        m.predict(state,torch.zeros(2,11),torch.zeros(2,dtype=torch.long))
    with pytest.raises(ValueError):
        m.predict(state,torch.zeros(2,12),torch.full((2,),2))
    with pytest.raises(ValueError):
        m.aligned_historical_logits(state,torch.full((2,12),float("nan")))
