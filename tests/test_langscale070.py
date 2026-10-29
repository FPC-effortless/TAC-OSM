"""LANGSCALE070 no-fit constructor, causal, trained-path gradient and hard size gate."""
import copy
import sys
from pathlib import Path
import pytest
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"research/mathematical-engineering/langscale070"))
from scaled_recurrence import ScaledRecurrenceTAC070,COUNTS,WIDTHS
from scripts.validate_research_parity import (
    _check_manifest_parity_only, check_model_parameter_target, ParityError
)


@pytest.mark.parametrize("tier",["p250k","p500k","p1000k"])
def test_count_forward_causality_gradient_and_state_geometry(tier):
    torch.manual_seed(70070)
    model=ScaledRecurrenceTAC070(scale=tier)
    count=COUNTS[tier]
    assert sum(p.numel() for p in model.parameters() if p.requires_grad)==count
    assert sum(p.numel() for p in model.parameters())==count
    assert model.ffn_width==WIDTHS[tier]
    assert model.upper_state_width==48
    assert model.upper_concept_graph is False
    assert model.scale==tier
    assert model.layernorm[1].state_width==48
    x=torch.tensor([[1,2,3,4,5,6,7,8],[11,10,9,8,7,6,5,4]])
    y=x.clone()
    y[:,6:]=torch.tensor([[27,31],[37,41]])
    model.eval()
    with torch.no_grad():
        logits=model(x)
        changed=model(y)
    assert logits.shape==(2,8,256)
    assert torch.isfinite(logits).all()
    assert torch.allclose(logits[:,:6],changed[:,:6],atol=1e-6,rtol=1e-6)
    model.train()
    (model(x).square().mean()).backward()
    for name in ("input_proj.weight","forget.weight","offer.weight",
                 "read_gate.weight","output.weight","residual_gain"):
        p=dict(model.layernorm[1].named_parameters())[name]
        assert p.grad is not None and torch.isfinite(p.grad).all()
        assert torch.any(p.grad!=0)
    for p in (model.ffn_up.weight,model.ffn_gate.weight,model.ffn_out.weight):
        assert p.grad is not None and torch.isfinite(p.grad).all()
        assert torch.any(p.grad!=0)


def test_unregistered_widths_and_vocab_rejected():
    for tier,vocab in (("p200k",256),("p1500k",256),("p500k",128)):
        with pytest.raises(ValueError):
            ScaledRecurrenceTAC070(tier,vocab)


def test_hard_100k_budget_still_applies_to_all_other_training():
    with pytest.raises(ParityError):
        check_model_parameter_target(500000,500000,label="other-studies")
    manifest={
        "schema_version":"tac-osm.parity.v1",
        "study_id":"LANGSCALE071",
        "study_type":"single_model_training",
        "registered_before_first_fit":True,
        "parameter_budget_contract":"owner-2026-10-29-ffn-only-language-scaling",
        "real_world_source":{"observed_real_outcomes":True,"sha256":"a"*64},
        "models":[{"name":"p500k","seed":1701,
                   "trainable_parameters":499623,"deployed_parameters":499623}],
    }
    with pytest.raises(ParityError):
        _check_manifest_parity_only(manifest)
    manifest["study_id"]="LANGSCALE070"
    manifest["scale_axis"]="FFN_WIDTH_ONLY"
    manifest["execution_level"]="internal_research"
    manifest["no_architecture_superiority_claim"]=True
    assert _check_manifest_parity_only(manifest)["status"]=="PASS_PREFIT_SINGLE_MODEL"
    for name,bad in (("p250k",250132),("p500k",499624),("p1000k",1000238)):
        wrong=copy.deepcopy(manifest)
        wrong["models"][0]["name"]=name
        wrong["models"][0]["trainable_parameters"]=bad
        wrong["models"][0]["deployed_parameters"]=bad
        with pytest.raises(ParityError):
            _check_manifest_parity_only(wrong)
