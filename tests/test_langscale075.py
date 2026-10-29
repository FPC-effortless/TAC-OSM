"""LANGSCALE075 native 8x8 /12x12/16x16 source and causality falsifiers.

Only constructed source tests. Trained genuine publisher source and carried
streaming need separate audited checkpoint/runtime evidence.
"""
import copy
import sys
from pathlib import Path
import pytest
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/"research/mathematical-engineering/langscale075"),
             str(ROOT/"research/mathematical-engineering/langscale070")]
from native_matrix import NativeMatrixTAC075,WIDTHS,COUNTS,model_parameters
from scaled_recurrence import ScaledRecurrenceTAC070
from scripts.validate_research_parity import _check_manifest_parity_only,check_model_parameter_target,ParityError


def test_m8_exact_original_source_initialization_logits_traces_and_gradients():
    torch.manual_seed(7512)
    old=ScaledRecurrenceTAC070("p500k")
    torch.manual_seed(7512)
    newer=NativeMatrixTAC075("m8")
    assert tuple(old.state_dict())==tuple(newer.state_dict())
    for name in old.state_dict():
        assert torch.equal(old.state_dict()[name],newer.state_dict()[name]),name
    x=torch.tensor([[1,3,5,7,9,11,13,15],[19,17,15,13,11,9,7,5]])
    mask=torch.tensor([[1,1,0,1,0,1,1,1],[1,0,1,1,1,0,1,1]],dtype=torch.bool)
    o,to=old(x,write_mask=mask,return_trace=True)
    n,tn=newer(x,write_mask=mask,return_trace=True)
    assert torch.equal(o,n)
    assert tuple(to)==tuple(tn)
    for name in to:
        assert torch.equal(to[name],tn[name]),name
    old.zero_grad();newer.zero_grad()
    (old(x).square().mean()).backward()
    (newer(x).square().mean()).backward()
    for (ko,po),(kn,pn) in zip(old.named_parameters(),newer.named_parameters()):
        assert ko==kn
        assert torch.equal(po.grad,pn.grad),ko


@pytest.mark.parametrize("tier",["m8","m12","m16"])
def test_native_full_matrix_source_geometry_causality_gradients(tier):
    torch.manual_seed(7513)
    model=NativeMatrixTAC075(tier)
    k=WIDTHS[tier]
    assert model.parameter_count()==COUNTS[tier]==model_parameters(k)
    assert model.ffn_width==455 and model.upper_state_width==48
    assert all(p.requires_grad for p in model.parameters())
    assert model.rotation.shape==(2,8,k)
    assert model.native_key.out_features==k and model.native_value.out_features==k
    a=torch.randint(0,256,(2,9),dtype=torch.long)
    d=a.clone()
    d[:,7:]=torch.tensor([[101,103],[107,109]])
    mask=torch.ones((2,9),dtype=torch.bool);mask[:,2]=False
    y,t=model(a,write_mask=mask,return_trace=True)
    x=model(d,write_mask=mask)
    assert y.shape==(2,9,256) and torch.isfinite(y).all()
    assert torch.allclose(y[:,:7],x[:,:7],atol=1e-6,rtol=1e-6)
    assert t["native_memory"].shape==(2,9,k,k)
    assert t["native_prewrite"].shape==(2,9,k,k)
    assert t["aligned_prewrite"].shape==(2,9,2,16,k)
    assert t["native_read"].shape==(2,9,8)
    assert t["native_value"].shape==(2,9,k)
    (model(a).square().mean()).backward()
    for name,p in model.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(),name
        assert p.grad.abs().sum()>0,name
    if k>8:
        assert model.native_key.weight.grad[8:].abs().sum()>0
        assert model.native_value.weight.grad[8:].abs().sum()>0
        assert model.native_read_proj.weight.grad[:,8:].abs().sum()>0
        assert model.rotation.grad[:,:,8:].abs().sum()>0


def test_model_tier_rejects_unregistered_and_policy_gates_other_studies():
    for bad in ("m4","m10","m24","p500k",""):
        with pytest.raises(ValueError):
            NativeMatrixTAC075(bad)
    for k in (9,12.0,True,None):
        with pytest.raises(ValueError):
            model_parameters(k)
    with pytest.raises(ParityError):
        check_model_parameter_target(500175,500175)
    m={"schema_version":"tac-osm.parity.v1",
       "study_id":"LANGSCALE075","study_type":"single_model_training",
       "execution_level":"internal_research",
       "registered_before_first_fit":True,
       "parameter_budget_contract":"owner-2026-10-29-native-associative-capacity",
       "scale_axis":"NATIVE_ASSOCIATIVE_KEY_VALUE_AND_ALIGNED_ADDRESS",
       "no_architecture_superiority_claim":True,
       "real_world_source":{"observed_real_outcomes":True,"sha256":"a"*64},
       "models":[{"name":"m12","seed":1701,
                  "trainable_parameters":500175,"deployed_parameters":500175}]}
    assert _check_manifest_parity_only(m)["status"]=="PASS_PREFIT_SINGLE_MODEL"
    for alter in (lambda m:m.update(study_id="LANGSCALE076"),
                  lambda m:m["models"][0].update(trainable_parameters=500176),
                  lambda m:m["models"][0].update(name="m8"),
                  lambda m:m.update(scale_axis="FFN_WIDTH_ONLY")):
        n=copy.deepcopy(m);alter(n)
        with pytest.raises(ParityError):
            _check_manifest_parity_only(n)
