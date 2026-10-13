"""DEV-014: XOR conditional computation vs oracle leakage mechanism gates."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
from tac_osm.development.casm_readout_alignment import CASMAlignedModel  # noqa: E402
from tac_osm.development.casm_action_transfer import (  # noqa: E402
    CASMActionAdapter,teacher_bit_distribution,frozen_memory_posterior,
)
from tac_osm.development.protection_benchmark import make_pair,tensors  # noqa: E402
from run_casm_action_transfer_dev import build_adapters,make_features  # noqa: E402


def test_sparse_dense_identical_starting_tensors_and_parameter_count():
    parent=CASMAlignedModel()
    arms=build_adapters(parent,43)
    first=arms["oracle_sparse"].state_dict()
    for name,m in arms.items():
        assert all(torch.equal(v,first[k]) for k,v in m.state_dict().items()),name
    assert len({sum(p.numel() for p in m.parameters()) for m in arms.values()})==1
    assert arms["oracle_sparse"].top_k==1 and arms["oracle_dense"].top_k==4


def test_original_casm_operator_banks_and_answer_are_reused_at_initialization():
    parent=CASMAlignedModel()
    child=CASMActionAdapter(parent)
    for key,val in parent.compute.state_dict().items():
        torch.testing.assert_close(val,child.compute.state_dict()[key],atol=0,rtol=0)
    for key,val in parent.answer.state_dict().items():
        torch.testing.assert_close(val,child.answer.state_dict()[key],atol=0,rtol=0)


def test_bit_teacher_label_is_never_in_nonoracle_model_call():
    parent=CASMAlignedModel()
    pair=make_pair(43,32,4,0.,.08,"retention")
    visible=make_features(parent,[pair.left,pair.right])
    assert visible["target"][0]!=visible["target"][1]
    assert torch.equal(visible["current"][0],visible["current"][1])
    assert torch.equal(visible["oracle"],F.one_hot(visible["bit"],2).float())
    child=CASMActionAdapter(parent,top_k=1)
    result=child(visible["predicted"],visible["current"])
    assert result.shape==(2,2)
    assert torch.isfinite(result).all()


def test_no_casm_is_affine_in_two_binary_inputs_and_cannot_represent_xor():
    parent=CASMAlignedModel()
    child=CASMActionAdapter(parent,top_k=1)
    hist=torch.tensor([0,0,1,1])
    current=torch.tensor([0,1,0,1])
    with torch.no_grad():
        logit=child(teacher_bit_distribution(hist),current,ablate_compute=True)
    # The difference between the two class logits must be additive across
    # historical bit and current: 00+11 == 01+10. XOR violates this identity.
    margin=logit[:,1]-logit[:,0]
    torch.testing.assert_close(margin[0]+margin[3],margin[1]+margin[2],
                               atol=1e-5,rtol=1e-5)
    assert child.last_operator_sample_calls==0


def test_casm_gradients_and_actual_operator_dispatch():
    parent=CASMAlignedModel()
    hist=torch.tensor([0,0,1,1])
    current=torch.tensor([0,1,0,1])
    target=hist^current
    models=build_adapters(parent,43)
    for model in models.values():
        model.train()
        logits=model(teacher_bit_distribution(hist),current)
        loss=F.cross_entropy(logits,target)
        loss.backward()
        grad=model.compute.selector[-1].weight.grad
        assert grad is not None and torch.isfinite(grad).all() and grad.abs().sum()>0
        model.eval()
        with torch.no_grad():
            model(teacher_bit_distribution(hist),current)
        assert model.last_operator_sample_calls>0


def test_frozen_parent_memory_remains_768_and_has_no_backprop():
    parent=CASMAlignedModel()
    parent.eval()
    for p in parent.parameters():p.requires_grad_(False)
    pair=make_pair(71,128,16,1.,.25,"retention")
    a,v,q,c=tensors([pair.left,pair.right])
    posterior=frozen_memory_posterior(parent,a,v,q)
    assert posterior.shape==(2,2)
    torch.testing.assert_close(posterior.sum(-1),torch.ones(2),atol=1e-6,rtol=1e-6)
    assert not posterior.requires_grad
    assert all(p.grad is None for p in parent.parameters())


def test_uninformative_bit_distribution_cannot_distinguish_paired_targets():
    parent=CASMAlignedModel()
    child=CASMActionAdapter(parent)
    child.eval()
    pair=make_pair(47,128,16,1.,.25,"acquisition")
    _,_,_,current=tensors([pair.left,pair.right])
    neutral=torch.full((2,2),.5)
    with torch.no_grad():
        out=child(neutral,current)
    torch.testing.assert_close(out[0],out[1],atol=1e-6,rtol=1e-6)
    assert pair.left.target!=pair.right.target


def test_probability_simplex_and_current_errors_fail_closed():
    child=CASMActionAdapter(CASMAlignedModel())
    with pytest.raises(ValueError):
        child(torch.randn(2,3),torch.zeros(2,dtype=torch.long))
    with pytest.raises(ValueError):
        child(torch.full((2,2),.6),torch.zeros(2,dtype=torch.long))
    with pytest.raises(ValueError):
        child(torch.full((2,2),.5),torch.tensor([0,2]))
    with pytest.raises(ValueError):
        child(torch.full((2,2),float("nan")),torch.tensor([0,1]))
