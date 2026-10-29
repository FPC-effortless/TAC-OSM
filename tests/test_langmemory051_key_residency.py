"""LANGMEMORY051 constructed causality and pure source-analysis checks."""
import pytest
from scripts.langmemory051_key_residency import NoEvictionMG,one_owner,run,SLOTS
from scripts.langmemory050_multivalue import BoundedMultiContinuation

def test_unbounded_mg_is_really_unbounded_and_not_a_deployable_cache():
    s=NoEvictionMG(2)
    for n in range(600):s.update(n.to_bytes(3,"little"),42)
    assert len(s.values)==600
    assert s.lookup((599).to_bytes(3,"little"))==42

def test_oracle_mg_keeps_heavy_source_continuation_without_key_eviction():
    s=NoEvictionMG(2)
    for x in b"XXYXXZXXYX":s.update(b"abc",x)
    assert s.lookup(b"abc")==ord("X")
    assert s.decrements>=0

def test_source_value_diversity_prior_only_and_no_owner_crossing():
    a=one_owner(b"abcXabcY"*210,"ownerA",1536,4)
    b=one_owner(b"abcXabcZ"*210,"ownerB",1536,4)
    assert a["source_sha256"]!=b["source_sha256"]
    assert a["source_analysis"]["source_positions"]==352
    assert a["source_analysis"]["repeat_with_more_than_one_value"]>=0
    assert a["arms"]["mg_no_eviction_2"]["total"]==352
    assert a["arms"]["exact_full_history"]["total"]==352

def test_different_future_suffix_cannot_change_stored_prefix_state():
    a=NoEvictionMG(3);b=NoEvictionMG(3)
    prefix=b"abcXabcY"*80
    for t in range(3,len(prefix)):
        a.update(prefix[t-3:t],prefix[t])
        b.update(prefix[t-3:t],prefix[t])
    assert a.values==b.values
    suffix_a=prefix+b"abcQ"*80
    suffix_b=prefix+b"abcZ"*80
    for t in range(len(prefix),len(suffix_a)):
        a.update(suffix_a[t-3:t],suffix_a[t])
        b.update(suffix_b[t-3:t],suffix_b[t])
    assert a.values!=b.values

def test_mg_key_exists_with_empty_candidate_counter():
    s=NoEvictionMG(2)
    for x in b"ABC":s.update(b"abc",x)
    assert s.contains_key(b"abc")
    assert s.lookup(b"abc") is None

def test_actual_bounded_representations_have_frozen_key_capacity():
    assert SLOTS=={"k1_1344":160,"k2_1344":80,"k3_1344":64,
                   "k1_4096":504,"k2_4096":252,"k3_4096":200}
    for b,k in ((1344,2),(1344,3),(4096,2),(4096,3)):
        z=BoundedMultiContinuation(b,k,2)
        assert z.slots==SLOTS[f"k{k}_{b}"]
        assert z.effective_state_payload_bytes()==b

def test_zero_byte_value_and_exact_key_stored_before_query():
    s=NoEvictionMG(3)
    s.update(b"\x00ab",0)
    assert s.contains_key(b"\x00ab") and s.lookup(b"\x00ab")==0
    assert not s.contains_key(b"abc")

def test_source_decomposition_identity_and_replay():
    raw=b"aaaaabbbbbbccdddd"*100
    x=one_owner(raw,"owner",1536,4)
    y=one_owner(raw,"owner",1536,4)
    d=x["source_analysis"]
    assert d["source_positions"]==352
    assert d["unique_context_queries"]+d["repeat_context_queries"]==352
    assert d["repeat_with_one_value"]+d["repeat_with_more_than_one_value"]==d["repeat_context_queries"]
    assert x["case_identity_sha256"]==y["case_identity_sha256"]
    assert x["arms"]["old_k1_1344"]["correct"]==y["arms"]["old_k1_1344"]["correct"]
