"""Synthetic-only callsite parser accounting and unmodified transition checks."""
from types import SimpleNamespace
import pytest
import torch
from scripts.langprofile059_lines import (
    collapse, resolve_advance_callsite, original_line_range,
    FOCUS_OPS, CALLS, profile_raw
)
from scripts.langdecode057_validation import BatchedValidationTAC057
from factorial_depth import FactorialDepthControl040
from incremental import IncrementalTAC043


def event(name, time_us, stack, count=1):
    return SimpleNamespace(name=name, self_cpu_time_total=time_us,
                           stack=stack, count=count)


def test_allow_only_genuine_original_advance_line():
    lo, hi = original_line_range()
    assert lo < hi
    assert resolve_advance_callsite([f"/repo/langstream043/incremental.py:{lo+2} in _advance"],(lo,hi))==lo+2
    assert resolve_advance_callsite([f"/repo/langstream043/incremental.py({lo+2}): _advance"],(lo,hi))==lo+2
    assert resolve_advance_callsite(["/tmp/fake_incremental.py:44"],(lo,hi)) is None
    assert resolve_advance_callsite([f"/tmp/incremental.py:{hi+1}"],(lo,hi)) is None
    assert resolve_advance_callsite([], (lo,hi)) is None


def test_accounting_preserves_unknown_stacks_no_guess():
    lo,hi=original_line_range()
    es=[event("aten::_to_copy",40,[f"/src/incremental.py:{lo+2}"]),
        event("aten::mul",30,[]),
        event("aten::addmm",20,["/tmp/unrelated.py:33"]),
        event("ProfilerStep*",100,[])]
    r=collapse(es,(lo,hi))
    assert r["total_aten_self_cpu_us"]==90
    assert r["attributed_self_cpu_us"]==40
    assert r["unattributed_self_cpu_us"]==50
    assert r["attributed_fraction"]==pytest.approx(40/90)
    assert r["top_advance_lines"][0]["source_line"]==lo+2
    assert r["focus_operator_self_cpu_us"]["aten::_to_copy"]==40
    assert r["focus_operator_self_cpu_us"]["aten::mul"]==30
    assert r["top_advance_lines"][0]["dominant_operators"][0]["name"]=="aten::_to_copy"


@pytest.mark.parametrize("bad",[[], [event("aten::mul",float("nan"),[])],
                                [event("aten::mul",-1,[])],
                                [event("aten::mul",0,[])]])
def test_fail_closed_when_unusable_cpu_profile(bad):
    with pytest.raises(ValueError):
        collapse(bad,(4,200))


def test_empty_or_other_source_callsites_indeed_inconclusive():
    report=collapse([event("aten::mul",10,[])],(4,200))
    assert report["status"]=="NO_ADVANCE_PYTHON_STACKS_AVAILABLE"
    assert report["attributed_fraction"]==0
    assert not report["top_advance_lines"]


def test_profiler_does_not_change_reference_model_and_state():
    torch.manual_seed(59059)
    model=FactorialDepthControl040(state_width=48,upper_graph=False).eval()
    engine=BatchedValidationTAC057(model,"cap_all")
    assert BatchedValidationTAC057._advance is IncrementalTAC043._advance
    state=engine.start(("synthetic-owner",))
    token=torch.tensor([4],dtype=torch.long)
    mask=torch.ones_like(token,dtype=torch.bool)
    with torch.inference_mode():
        a,sa=engine._advance(token,state,mask)
        result=profile_raw(engine,token,state,mask)
        b,sb=engine._advance(token,state,mask)
    assert torch.equal(a,b)
    assert all(torch.equal(sa.tensors[k],sb.tensors[k]) for k in sa.tensors)
    assert state.tensor_bytes()==3904
    assert result["calls"]==CALLS
    assert result["total_aten_self_cpu_us"]>0
    assert 0<=result["attributed_fraction"]<=1
    assert result["status"] in ("SOURCE_CALLSITES_RECORDED",
                                "NO_ADVANCE_PYTHON_STACKS_AVAILABLE")
