"""LANGMEM081 constructed selector causality and negative controls.

These tests use constructed prefix byte strings, NOT real model capability.
The future target is supplied ONLY to add_outcome; selection cannot read it.
"""
from pathlib import Path
import sys
import pytest
import torch
from torch.nn import functional as F
ROOT=Path(__file__).resolve().parents[1]
for n in ("langmem081","langmem079","langmem078","langscale075",
          "langscale070","langmemory042","context026","langbase024"):
    p=str(ROOT/"research/mathematical-engineering"/n)
    if p not in sys.path:sys.path.insert(0,p)
from scripts.langmem081_context_address import select_past,add_outcome,LENGTHS,AGES,CELLS


def _writers(length):
    return {a:F.normalize(torch.tensor([float((a*j)%17+1) for j in range(1,9)]),dim=0)
            for a in range(4,length)}


@pytest.mark.parametrize("L",[2,3,4])
@pytest.mark.parametrize("H",[32,128,256])
def test_source_context_matching_rank_controls_no_future_byte_access(L,H):
    prefix=(b"abcde" * 26)[:129]
    assert len(prefix)==129
    q=F.normalize(torch.arange(1,9,dtype=torch.float32),dim=0)
    source=_writers(129)
    selected=select_past(prefix,"fixture-1",source,q,context_len=L,age_cap=H)
    assert selected["query_clock"]==128
    assert selected["candidate_count"]<=H
    assert all(a<128 for a in selected["source_clock_indices"])
    assert len(selected["matching_clocks"])==selected["matching_count"]
    assert selected["has_historical_match"]
    target_a=30
    target_b=211
    # Both possible future bytes are excluded from the ONLY selector inputs.
    # Thus no target may affect the historical selection or its rankings.
    assert target_a!=target_b
    also=select_past(prefix,"fixture-1",source,q,context_len=L,age_cap=H)
    assert selected==also
    metric={"nll":2.0,"top1":0}
    a=add_outcome(selected,prefix,target_a,metric)
    b=add_outcome(selected,prefix,target_b,metric)
    assert a["model_top_writer_clock"]==b["model_top_writer_clock"]
    assert a["source_positive_recent_clock"]==b["source_positive_recent_clock"]
    assert a["positive_match_count"]==b["positive_match_count"]
    assert a["original_trained_TAC_nextbyte_nll"]==b["original_trained_TAC_nextbyte_nll"]
    assert a["actual_original_next_byte_target"]!=b["actual_original_next_byte_target"]


def test_no_positive_source_context_does_not_count_as_retrieval():
    prefix=bytes(range(129))
    q=F.normalize(torch.arange(1,9,dtype=torch.float32),dim=0)
    res=select_past(prefix,"owner",_writers(129),q,context_len=4,age_cap=128)
    assert res["matching_count"]==0
    assert res["has_historical_match"] is False
    assert res["best_matching_rank"] is None
    assert res["historical_last_match_clock"] is None
    out=add_outcome(res,prefix,79,{"nll":3.1,"top1":0})
    assert out["exact_recent_match_observed_continuation_correct"] is None
    assert out["positive_exists"] is False
    assert out["model_writer_context_hit"] is False
    assert abs(out["exact_history_smoothed_source_nll"]-5.545177444479562)<1e-9


def test_wrong_history_hard_decoy_is_never_correct_context():
    prefix=(b"abcdaabcdb"*13)[:129]
    q=F.normalize(torch.arange(1,9,dtype=torch.float32),dim=0)
    selected=select_past(prefix,"owner",_writers(129),q,context_len=2,age_cap=128)
    assert selected["hardest_wrong_clock"] is not None
    t=selected["query_clock"];a=selected["hardest_wrong_clock"]
    assert prefix[a-2:a]!=prefix[t-1:t+1]
    assert selected["random_clock"]<t


def test_fail_closed_future_inputs_and_bad_source_writer_vectors():
    q=F.normalize(torch.arange(1,9,dtype=torch.float32),dim=0)
    prefix=b"x"*129
    with pytest.raises(ValueError):
        select_past(prefix,"owner",{},q,context_len=2,age_cap=32)
    with pytest.raises(ValueError):
        select_past(prefix+b"x"*500,"owner",_writers(129),q,context_len=2,age_cap=32)
    with pytest.raises(ValueError):
        select_past(prefix,"owner",_writers(129),q,context_len=5,age_cap=32)
    with pytest.raises(ValueError):
        add_outcome({},prefix,256,{"nll":1,"top1":0})


def test_registered_source_evaluation_count():
    assert LENGTHS==(2,3,4)
    assert AGES==(32,128,256)
    assert CELLS==4*192*9==6912
