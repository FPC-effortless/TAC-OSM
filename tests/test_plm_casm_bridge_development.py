"""DEV-015: actual recurrent PLM -> CASM action gradients and no-oracle controls."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
from tac_osm.development.casm_readout_alignment import CASMAlignedModel  # noqa: E402
from tac_osm.development.plm_casm_bridge import PLMCASMBridge,ARMS  # noqa: E402
from tac_osm.development.protection_benchmark import make_pair,tensors  # noqa: E402
from run_plm_casm_bridge_dev import make_models,change_basis  # noqa: E402


def _pair(seed=43):
    p=make_pair(seed,32,4,1.0,.08,"retention")
    return tensors([p.left,p.right]),p


def test_all_five_models_have_identical_starting_tensors_and_768_scalar_state():
    reference=CASMAlignedModel()
    arms,hashes=make_models(reference,43)
    assert set(arms)==set(ARMS)
    assert len(set(hashes.values()))==1
    assert len({sum(p.numel() for p in m.parameters()) for m in arms.values()})==1
    (a,v,q,c),_= _pair()
    for m in arms.values():
        with torch.no_grad():
            state=m.reader.encode(a,v,trace=False)
            result,bit,prob=m(a,v,q,c)
        assert state.shape==(2,4,8,24) and state[0].numel()==768
        assert result.shape==bit.shape==prob.shape==(2,2)


def test_frozen_writer_and_reader_only_modes_keep_writer_gradients_disabled():
    reference=CASMAlignedModel()
    models,_=make_models(reference,43)
    (a,v,q,c),p=_pair()
    target=torch.tensor([p.left.target,p.right.target])
    for mode in ("frozen_task","reader_aux"):
        model=models[mode]
        logits,bit,_=model(a,v,q,c)
        loss=F.cross_entropy(logits,target)
        if mode=="reader_aux":
            loss=loss+F.cross_entropy(bit,target^c)
        loss.backward()
        assert model.reader.memory.write.weight.grad is None
        assert not model.reader.memory.write.weight.requires_grad
        assert model.actor.compute.selector[-1].weight.grad is not None


def test_joint_task_gradient_reaches_true_recurrent_writer_and_casm():
    model=PLMCASMBridge(CASMAlignedModel(),mode="joint_task")
    (a,v,q,c),p=_pair(47)
    logits,_,_=model(a,v,q,c)
    F.cross_entropy(logits,torch.tensor([p.left.target,p.right.target])).backward()
    grad=model.reader.memory.write.weight.grad
    assert grad is not None and torch.isfinite(grad).all() and grad.abs().sum()>0
    assert model.reader.shared_head.weight.grad is not None
    assert model.reader.shared_head.weight.grad.abs().sum()>0
    assert model.actor.compute.selector[-1].weight.grad is not None


def test_joint_auxiliary_supervises_historical_bit_without_teacher_forward():
    model=PLMCASMBridge(CASMAlignedModel(),mode="joint_aux")
    (a,v,q,c),p=_pair(49)
    task,bit,posterior=model(a,v,q,c)
    label=torch.tensor([p.left.target,p.right.target])
    loss=F.cross_entropy(task,label)+F.cross_entropy(bit,label^c)
    loss.backward()
    assert torch.isfinite(posterior).all()
    assert model.reader.memory.write.weight.grad.abs().sum()>0
    assert model.actor.compute.selector[-1].weight.grad.abs().sum()>0


def test_no_memory_signal_must_be_indistinguishable_for_opposite_past():
    model=PLMCASMBridge(CASMAlignedModel(),mode="frozen_task")
    model.eval()
    (a,v,q,c),p=_pair(59)
    assert p.left.target != p.right.target
    assert torch.equal(q[0],q[1]) and torch.equal(c[0],c[1])
    with torch.no_grad():
        logits=model.with_neutral_bit(a,v,q,c)
    torch.testing.assert_close(logits[0],logits[1],atol=1e-6,rtol=1e-6)


def test_signed_coordinate_permutation_preserves_exact_euclidean_target():
    (a,v,q,c),p=_pair(71)
    moved,query=change_basis(a,q,seed_for_case:=91)
    original=((a-q[:,None,:])**2).sum(-1)
    altered=((moved-query[:,None,:])**2).sum(-1)
    torch.testing.assert_close(original,altered,atol=1e-4,rtol=1e-5)
    assert torch.equal(original.argmin(1),altered.argmin(1))
    assert seed_for_case==91
    assert p.left.target != p.right.target


def test_eval_forward_has_no_target_index_or_answer_input():
    model=PLMCASMBridge(CASMAlignedModel(),mode="joint_aux")
    (a,v,q,c),p=_pair(73)
    assert p.left.target_stream_slot==p.right.target_stream_slot
    assert torch.equal(a[0],a[1])
    assert torch.equal(q[0],q[1])
    # Only the visible payload differs at one historical location. No
    # evaluator target, slot or role is accepted by bridge.forward.
    logits,_,_=model(a,v,q,c)
    assert logits.shape==(2,2)


def test_invalid_input_dimensions_or_nonfinite_fail_closed():
    model=PLMCASMBridge(CASMAlignedModel(),mode="joint_aux")
    (a,v,q,c),_=_pair(79)
    with pytest.raises(ValueError):
        model(a,v,q[:,:-1],c)
    with pytest.raises(ValueError):
        model(a,v,q,torch.tensor([0]))
    with pytest.raises(ValueError):
        model(a,v,torch.full_like(q,float("nan")),c)
    with pytest.raises(ValueError):
        PLMCASMBridge(CASMAlignedModel(),mode="not_registered")
