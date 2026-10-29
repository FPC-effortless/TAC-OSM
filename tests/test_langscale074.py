"""LANGSCALE074: exact live 500K counts, upper gradients, h48 source correspondence."""
import copy
import sys
from pathlib import Path
import pytest
import torch

ROOT=Path(__file__).resolve().parents[1]
for name in ("langscale074","langscale070"):
    p=str(ROOT/"research/mathematical-engineering"/name)
    if p not in sys.path:sys.path.insert(0,p)
from upper_reallocation import ReallocatedUpperTAC074,WIDTHS,COUNTS,expected_parameters
from scaled_recurrence import ScaledRecurrenceTAC070
from scripts.validate_research_parity import (
    _check_manifest_parity_only,check_model_parameter_target,ParityError
)


@pytest.mark.parametrize("tier",["h24","h48","h96","h128"])
def test_exact_live_model_counts_and_active_recurrence(tier):
    torch.manual_seed(74001)
    model=ReallocatedUpperTAC074(tier)
    h,w=WIDTHS[tier]
    assert model.upper_state_width==h and model.ffn_width==w
    assert model.layernorm[1].state_width==h
    assert model.parameter_count()==COUNTS[tier]
    assert sum(p.numel() for p in model.parameters())==COUNTS[tier]
    assert expected_parameters(h,w)==COUNTS[tier]
    assert all(p.requires_grad for p in model.parameters())
    x=torch.tensor([[1,3,5,7,9,11,13,15],[20,19,18,17,16,15,14,13]])
    y=x.clone()
    y[:,6:]=torch.tensor([[24,27],[31,35]])
    with torch.no_grad():
        z=model(x)
        changed=model(y)
    assert z.shape==(2,8,256) and torch.isfinite(z).all()
    assert torch.allclose(z[:,:6],changed[:,:6],atol=1e-6,rtol=1e-6)
    (model(x).square().mean()).backward()
    for name in ("input_proj.weight","forget.weight","offer.weight",
                 "read_gate.weight","output.weight","residual_gain"):
        p=dict(model.layernorm[1].named_parameters())[name]
        assert p.grad is not None and torch.isfinite(p.grad).all() and torch.any(p.grad!=0)
    for p in (model.ffn_up.weight,model.ffn_gate.weight,model.ffn_out.weight):
        assert p.grad is not None and torch.isfinite(p.grad).all() and torch.any(p.grad!=0)


def test_original_070_h48_source_weights_and_logits_identical_unfitted():
    torch.manual_seed(1284)
    old=ScaledRecurrenceTAC070("p500k").eval()
    torch.manual_seed(1284)
    new=ReallocatedUpperTAC074("h48").eval()
    assert tuple(old.state_dict())==tuple(new.state_dict())
    for key in old.state_dict():
        assert torch.equal(old.state_dict()[key],new.state_dict()[key]),key
    x=torch.tensor([[17,11,25,13,8,1,0],[23,7,9,10,20,14,2]])
    with torch.no_grad():
        assert torch.equal(old(x),new(x))


def test_narrow_074_exception_not_general_budget_bypass():
    source={"observed_real_outcomes":True,"sha256":"a"*64}
    seed=1701
    manifest={
        "schema_version":"tac-osm.parity.v1","study_id":"LANGSCALE074",
        "study_type":"single_model_training","execution_level":"internal_research",
        "registered_before_first_fit":True,
        "parameter_budget_contract":"owner-2026-10-29-upper-state-500k-reallocation",
        "scale_axis":"UPPER_RECURRENT_STATE_VS_FFN_REALLOCATION",
        "no_architecture_superiority_claim":True,"real_world_source":source,
        "models":[{"name":"h24","seed":seed,
                  "trainable_parameters":499625,"deployed_parameters":499625}]
    }
    assert _check_manifest_parity_only(manifest)["status"]=="PASS_PREFIT_SINGLE_MODEL"
    with pytest.raises(ParityError):
        check_model_parameter_target(499625,499625)
    for change in (
        lambda m:m.update(study_id="LANGSCALE075"),
        lambda m:m.update(scale_axis="FFN_WIDTH_ONLY"),
        lambda m:m.update(parameter_budget_contract="not-owner"),
        lambda m:m["models"][0].update(trainable_parameters=499624),
        lambda m:m["models"][0].update(name="h160"),
    ):
        bad=copy.deepcopy(manifest)
        change(bad)
        with pytest.raises(ParityError):
            _check_manifest_parity_only(bad)


def test_unregistered_recurrence_width_and_ffn_rejected():
    for tier in ("h32","h160","p500k",""):
        with pytest.raises(ValueError):
            ReallocatedUpperTAC074(tier)
    with pytest.raises(ValueError):
        ReallocatedUpperTAC074("h24",vocab=128)
    for dims in ((0,500),(-1,500),(20,0),(20,0.2),(True,300)):
        with pytest.raises(ValueError):
            expected_parameters(*dims)
