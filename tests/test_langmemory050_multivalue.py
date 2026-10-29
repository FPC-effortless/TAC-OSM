"""Only constructed integrity checks; original source outcomes are in GH CI."""
import pytest
from scripts.langmemory050_multivalue import (
    BoundedMultiContinuation,owner_probe,aggregate,arm_name)

@pytest.mark.parametrize("b,k,ways",[(b,k,w) for b in (1344,4096)
                             for k in (2,3) for w in (1,2)])
def test_true_bytearray_payload_never_exceeds_budget(b,k,ways):
    x=BoundedMultiContinuation(b,k,ways)
    assert len(x.payload)+64==b
    assert x.slots*x.width<=len(x.payload)
    assert x.sets*x.ways==x.slots
    assert x.effective_state_payload_bytes()==b

def test_two_candidate_votes_track_true_heavy_counts_with_no_collision():
    x=BoundedMultiContinuation(1344,2,2)
    key=b"abc"
    for t,value in enumerate(b"XXXXYYYXXY"):
        x.observe(key,value,t)
    assert x.lookup(key)==ord("X")
    assert x.lossy_updates==0
    assert x.key_hits==1
    assert x.occupied()==1

def test_three_values_retain_all_three_when_key_survives():
    x=BoundedMultiContinuation(1344,3,1)
    key=b"def"
    for t,value in enumerate(b"ABAACCCBBBCA"):
        x.observe(key,value,t)
    assert x.lookup(key) in (ord("A"),ord("B"),ord("C"))
    assert x.lossy_updates==0
    assert x.occupied()==1

def test_fourth_continuation_triggers_mg_decrement_without_false_exact_claim():
    x=BoundedMultiContinuation(1344,3,1)
    key=b"def"
    for t,value in enumerate(b"ABCD"):
        x.observe(key,value,t)
    assert x.lossy_updates==1
    assert x.lookup(key) is None

def test_hash_collision_does_not_return_wrong_exact_source_key():
    x=BoundedMultiContinuation(1344,2,1)
    group={}
    for n in range(500):
        key=n.to_bytes(3,"little")
        group.setdefault(fnv_reference(key,x.sets),[]).append(key)
    a,b=next(pair[:2] for pair in group.values() if len(pair)>=2)
    x.observe(a,ord("X"),1)
    x.observe(b,ord("Y"),2)
    assert x.lookup(a) is None
    assert x.lookup(b)==ord("Y")
    assert x.evictions==1

def fnv_reference(key,sets):
    from scripts.langmemory047_bounded import fnv
    return fnv(key)%sets

def test_source_poison_never_changes_past_observation_state():
    raw=b"abcXabcY"*130
    alter=raw[:520]+b"?"*(len(raw)-520)
    a=BoundedMultiContinuation(1344,3,2)
    b=BoundedMultiContinuation(1344,3,2)
    for t in range(3,520):
        a.observe(raw[t-3:t],raw[t],t)
        b.observe(alter[t-3:t],alter[t],t)
    assert a.payload==b.payload
    for t in range(520,len(raw)):
        a.observe(raw[t-3:t],raw[t],t)
        b.observe(alter[t-3:t],alter[t],t)
    assert a.payload!=b.payload

def test_owner_isolation_full_causal_cohort():
    a=owner_probe(b"abcX"*420,"srcA",1536,4)
    b=owner_probe(b"abcY"*420,"srcB",1536,4)
    assert a["source_sha256"]!=b["source_sha256"]
    z=aggregate([a,b])
    assert all(x["queries"]==z["cases"] for x in z["arms"].values())
    assert all(0<=x["success_all_positions"]<=1 for x in z["arms"].values())

def test_correct_refusal_of_bad_byte_and_clock():
    m=BoundedMultiContinuation(4096,3,2)
    with pytest.raises(ValueError):m.observe(b"abc",256,2)
    with pytest.raises(ValueError):m.observe(b"abc",23,65535)
    with pytest.raises(ValueError):m.lookup(b"ab")
