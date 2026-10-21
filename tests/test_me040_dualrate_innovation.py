"""ME040 genuine architecture numerical properties; NO synthetic success claims."""
import pytest
import torch
from tac_osm.me040_dualrate_innovation import (
    pair_observed_patches, build_model, preflight
)
from tac_osm.me038_innovation_forecast import InnovationMemory


def test_me040_original_parameter_and_core_equivalence():
    evidence=preflight()
    assert evidence["status"]=="ME040_SOFTWARE_ONLY_MATH_PASS"
    assert evidence["recurrent_steps_original"]==12
    assert evidence["recurrent_steps_paired"]==6
    assert evidence["original_weights_untouched"]
    assert not evidence["new_real_fits_executed"]


def test_me040_aggregation_is_causal_at_complete_pairs():
    torch.manual_seed(40040)
    original=torch.randn(3,12,16)
    baseline=pair_observed_patches(original)
    for n in range(1,7):
        changed=original.clone()
        changed[:,2*n:]+=torch.randn_like(changed[:,2*n:])*200
        assert torch.equal(baseline[:,:n],pair_observed_patches(changed)[:,:n])
    assert torch.allclose(baseline,(original[:,0::2]+3*original[:,1::2])*.25)


def test_me040_native_state_size_and_parent_coefficient_invariants():
    a=build_model("innovation_tac",96,40001)
    b=build_model("paired_innovation_tac",96,40001)
    assert sum(p.numel() for p in a.parameters())==sum(p.numel() for p in b.parameters())
    assert len(list(a.state_dict()))==len(list(b.state_dict()))
    assert all(torch.equal(v,b.state_dict()[k]) for k,v in a.state_dict().items())
    toks=torch.randn(3,12,16)
    # Constant-shape latent output with finite numerical state.
    assert a.memory(toks).shape == b.memory(toks).shape == (3,64)
    assert torch.isfinite(b.memory(toks)).all()


def test_me040_invalid_or_future_incomplete_window_rejected():
    with pytest.raises((ValueError,torch.jit.Error)):
        pair_observed_patches(torch.randn(2,11,16))
    with pytest.raises((ValueError,torch.jit.Error)):
        pair_observed_patches(torch.randn(2,12,15))
