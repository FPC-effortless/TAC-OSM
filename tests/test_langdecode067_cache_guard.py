"""Constructed LANGDECODE067 integrity hardening; no trained success claim."""
import pytest
import torch

from scripts.langdecode067_cache_guard import VersionGuardedCapTAC067
from scripts.langdecode064_cached_cap import CachedCapPositionTAC064
from scripts.langdecode057_validation import BatchedValidationTAC057
from incremental import SHAPES
from factorial_depth import FactorialDepthControl040


@pytest.mark.parametrize("mode",["cap_all","native"])
@pytest.mark.parametrize("dtype",[torch.float32,torch.float64])
def test_guard_does_not_change_original_bitwise_state_and_logits(mode,dtype):
    torch.manual_seed(67067)
    model=FactorialDepthControl040(state_width=48,upper_graph=False).to(dtype).eval()
    original=BatchedValidationTAC057(model,mode)
    cache=CachedCapPositionTAC064(model,mode)
    guarded=VersionGuardedCapTAC067(model,mode)
    owners=("a","b")
    tokens=torch.randint(0,256,(2,132))
    masks=(torch.arange(132)[None,:]+torch.arange(2)[:,None])%4!=0
    with torch.inference_mode():
        outputs=[x.prefill(tokens[:,:128],owners=owners,write_mask=masks[:,:128])
                 for x in (original,cache,guarded)]
        for y,st in outputs[1:]:
            assert torch.equal(y,outputs[0][0])
            assert all(torch.equal(st.tensors[k],outputs[0][1].tensors[k]) for k in SHAPES)
        for t in range(128,132):
            outputs=[engine.step(tokens[:,t],st,owners=owners,write_mask=masks[:,t])
                     for engine,(_,st) in zip((original,cache,guarded),outputs)]
            for y,st in outputs[1:]:
                assert torch.equal(y,outputs[0][0])
                assert all(torch.equal(st.tensors[k],outputs[0][1].tensors[k]) for k in SHAPES)


@pytest.mark.parametrize("dtype",[torch.float32,torch.float64])
def test_version_guard_rejects_in_place_cache_edit(dtype):
    model=FactorialDepthControl040(state_width=48,upper_graph=False).to(dtype).eval()
    engine=VersionGuardedCapTAC067(model,"cap_all")
    state=engine.start(("a",))
    with torch.no_grad():
        engine._cap_all_ref[0,0].add_(0.125)
    with pytest.raises(ValueError,match="cached positional features replaced or modified"):
        engine.step(torch.tensor([97]),state,owners=("a",))
    with pytest.raises(ValueError,match="cached positional features replaced or modified"):
        engine.save_state(state,"/tmp/not-created-state-langdecode067.pt")


def test_version_guard_rejects_reassignment_and_native_mode_cache_injection():
    model=FactorialDepthControl040(state_width=48,upper_graph=False).eval()
    guarded=VersionGuardedCapTAC067(model,"cap_all")
    guarded._cap_all_ref=guarded._cap_all_ref.clone()
    with pytest.raises(ValueError,match="cached positional features replaced or modified"):
        guarded.validate(guarded.start(("a",)),("a",))
    native=VersionGuardedCapTAC067(model,"native")
    native._cap_all_ref=torch.zeros(1,7)
    with pytest.raises(ValueError,match="native-mode cache unexpectedly present"):
        native.validate(native.start(("a",)),("a",))


def test_source_uses_existing_exact_position_kernel_and_no_new_parameters():
    import inspect
    from incremental import IncrementalTAC043
    assert VersionGuardedCapTAC067._advance is CachedCapPositionTAC064._advance
    assert VersionGuardedCapTAC067.step is IncrementalTAC043.step
    assert "Tensor._version" in inspect.getsource(__import__("scripts.langdecode067_cache_guard",fromlist=["x"]))
