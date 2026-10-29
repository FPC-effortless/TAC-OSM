"""Constructed registered PyTorch positional source controls (no neural fit)."""
import sys
from pathlib import Path
import torch

ROOT=Path(__file__).resolve().parents[1]
for name in ("langmemory042","context026","langscale070"):
    p=str(ROOT/"research/mathematical-engineering"/name)
    if p not in sys.path:sys.path.insert(0,p)
from candidate import controlled_position,transform
from context_transition import phi_position
from scaled_recurrence import ScaledRecurrenceTAC070


def test_source_position_cap_is_noop_at_128_and_differs_after():
    for dtype in (torch.float32,torch.float64):
        t=torch.arange(128,dtype=torch.long)
        base=phi_position(t,dtype)
        assert torch.equal(base,transform(base,"cap_all"))
        late=phi_position(torch.tensor([128,129,200,255]),dtype)
        capped=transform(late,"cap_all")
        assert not torch.equal(late,capped)
        assert torch.equal(capped[0],capped[1])
        assert torch.equal(capped[1],capped[-1])


def test_temporary_hook_preserves_full_original_model_logits_through_128():
    torch.manual_seed(72072)
    torch.set_num_threads(2)
    m=ScaledRecurrenceTAC070("p250k").eval()
    x=torch.arange(128,dtype=torch.long).view(1,128).remainder(256)
    with torch.inference_mode():
        original=m(x)
        with controlled_position(m,"cap_all"):
            capped=m(x)
        after=m(x)
    assert torch.equal(original,capped)
    assert torch.equal(original,after)
    assert not m.pos._forward_pre_hooks


def test_hook_cleanup_on_exception():
    m=ScaledRecurrenceTAC070("p250k").eval()
    try:
        with controlled_position(m,"cap_all"):
            assert m.pos._forward_pre_hooks
            raise RuntimeError("constructed early exit")
    except RuntimeError:
        pass
    assert not m.pos._forward_pre_hooks
