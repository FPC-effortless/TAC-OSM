"""Constructed engineering falsifiers; no model capability measurements."""

import pytest
import torch
from torch import nn

from scripts.langrecomp053_training_parity import compare_outputs, one_pass
from tac_osm.research_gradient_coverage import GradientContract


@pytest.fixture
def case():
    torch.manual_seed(55)
    model = nn.Sequential(nn.Embedding(256, 8), nn.Linear(8, 256)).train()
    contract = GradientContract.capture(model)
    x = torch.tensor([[1, 2, 3, 4]])
    result = one_pass(model, x, torch.tensor([[2, 3, 4, 5]]), "native")
    return model, contract, result


def test_common_missing_gradient_is_rejected_despite_legacy_parity(case):
    model, contract, result = case
    name = next(iter(result["grads"]))
    del result["grads"][name]
    result["missing"] = [name]
    assert compare_outputs(result, result)["matching_missing_gradient_parameters"] == [name]
    with pytest.raises(ValueError, match="missing gradient"):
        contract.validate(model, result, logits_shape=(1, 4, 256))


def test_empty_or_unexpected_gradients_cannot_pass(case):
    model, contract, result = case
    for grads in ({}, {**result["grads"], "invented": torch.zeros(1)}):
        changed = {**result, "grads": grads}
        with pytest.raises(ValueError, match="gradient coverage"):
            contract.validate(model, changed, logits_shape=(1, 4, 256))


@pytest.mark.parametrize("fault", ["shape", "dtype", "nan", "inf"])
def test_gradient_schema_and_finiteness_are_independent_requirements(case, fault):
    model, contract, result = case
    name = next(iter(result["grads"]))
    grad = result["grads"][name]
    if fault == "shape":
        result["grads"][name] = grad[:1]  # broadcasts under allclose; must fail schema.
    elif fault == "dtype":
        result["grads"][name] = grad.double()
    else:
        grad.flatten()[0] = float(fault)
    with pytest.raises(ValueError, match="invalid gradient"):
        contract.validate(model, result, logits_shape=(1, 4, 256))


def test_finite_zero_gradient_is_present_not_disconnected(case):
    model, contract, result = case
    for grad in result["grads"].values():
        grad.zero_()
    report = contract.validate(model, result, logits_shape=(1, 4, 256))
    assert report["element_count"] == contract.elements
    assert all(row["nonzero_elements"] == 0 for row in report["parameters"])


@pytest.mark.parametrize("mutation", ["weight", "identity", "requires_grad"])
def test_contract_frozen_before_forward_detects_model_drift(case, mutation):
    model, contract, result = case
    if mutation == "weight":
        with torch.no_grad():
            model[0].weight[0, 0] += 1
    elif mutation == "identity":
        model[0].weight = nn.Parameter(model[0].weight.detach().clone())
    else:
        model[0].weight.requires_grad_(False)
    with pytest.raises(ValueError, match="changed"):
        contract.validate(model, result, logits_shape=(1, 4, 256))


def test_shared_parameter_is_counted_once():
    model = nn.Module()
    model.a = nn.Linear(3, 4, bias=False)
    model.b = nn.Linear(3, 4, bias=False)
    model.b.weight = model.a.weight
    contract = GradientContract.capture(model)
    assert len(contract.trainable) == 1 and contract.elements == 12


def test_full_native_and_checkpoint_coverage_and_hashes_match(case):
    model, contract, native = case
    alternate = one_pass(model, torch.tensor([[1, 2, 3, 4]]),
                         torch.tensor([[2, 3, 4, 5]]), "checkpoint_entire_graph")
    a = contract.validate(model, native, logits_shape=(1, 4, 256))
    b = contract.validate(model, alternate, logits_shape=(1, 4, 256))
    assert a == b
    assert a["tensor_count"] == len(list(model.parameters()))
    assert a["element_count"] == sum(p.numel() for p in model.parameters())


def test_output_shapes_do_not_broadcast(case):
    model, contract, result = case
    result["logits"] = result["logits"][:, :1]
    with pytest.raises(ValueError, match="exact-shape logits"):
        contract.validate(model, result, logits_shape=(1, 4, 256))


def test_frozen_only_model_is_not_complete_training_coverage():
    model = nn.Linear(2, 2).requires_grad_(False)
    with pytest.raises(ValueError, match="no declared trainable"):
        GradientContract.capture(model)
