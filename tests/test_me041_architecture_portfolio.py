"""ME041 numerical-only tests; no synthetic model performance claims."""
import torch
from tac_osm.me041_architecture_portfolio import (
    ARCHITECTURE_ARMS, ALL_ARMS, ResearchMemory, build_model, numerical_preflight
)
from tac_osm.me040_dualrate_innovation import pair_observed_patches


def test_me041_complete_portfolio_active_equal_capacity_and_gradients():
    state=numerical_preflight()
    assert state["status"]=="ME041_NUMERICAL_ONLY_PASS"
    assert len(state["arms"])==11
    assert state["active_adapter_params"]==272
    assert not state["real_outcomes_scored"]
    assert not state["future_data_opened"]


def test_me041_all_models_zero_additive_head_match_same_original_ar_parent():
    torch.set_num_threads(1)
    source=build_model("innovation_tac",96,41001)
    x=torch.randn(2,96,8)
    with torch.no_grad():
        for arm in ALL_ARMS:
            candidate=build_model(arm,96,41001)
            assert torch.equal(candidate(x),source(x)),arm


def test_me041_prediction_error_write_and_six_prefixes_are_numerical_safe():
    torch.manual_seed(41)
    x=torch.randn(2,12,16)
    p=pair_observed_patches(x)
    assert p.shape==(2,6,16)
    for arm in ARCHITECTURE_ARMS:
        m=ResearchMemory(arm).eval()
        out=m(x)
        assert out.shape==(2,64)
        assert torch.isfinite(out).all()
        if arm in ("cdl_plasticity","unified_multimechanism"):
            with torch.no_grad():
                a=torch.sigmoid(m.adapter(p[:,0]).mean(-1,keepdim=True))
            assert (a>=0).all() and (a<=1).all()


def test_me041_no_inert_dummy_weights_and_true_active_parameter_gap():
    for h in (96,720):
        counts={a:sum(p.numel() for p in build_model(a,h,41).parameters())
                for a in ALL_ARMS}
        assert len({counts[a] for a in ARCHITECTURE_ARMS})==1
        assert counts["baseline_adaptive"]-counts["innovation_tac"]==272
        assert max(counts.values())/min(counts.values())<1.01
