"""LANGMEMORY047 constructed correctness and future-blindness tests, no real scores."""
import pytest
from scripts.langmemory047_bounded import RawRing,FixedKeyTable,fnv,study_owner,aggregate

def test_actual_allocated_bytes_are_hard_budgets():
    for budget in (1344,4096):
        ring=RawRing(budget)
        assert len(ring.buffer)==budget
        for ways in (1,2):
            table=FixedKeyTable(budget,ways)
            assert len(table.buffer)+64==budget
            assert table.actual_payload_bytes()==budget

def test_ring_can_only_read_retained_previous_context():
    ring=RawRing(8)
    with pytest.raises(IndexError):ring.at(0)
    for byte in b"abcXabcYabcX":ring.observe(byte)
    assert ring.lookup(b"abc")[0]==ord("Y") or ring.lookup(b"abc")[0]==ord("X")
    with pytest.raises(IndexError):ring.at(0)
    assert ring.clock==12

def test_table_exact_key_and_candidate_updates():
    m=FixedKeyTable(1344,1)
    assert m.read(b"abc") is None
    m.update(b"abc",88,3)
    assert m.read(b"abc")==88
    m.update(b"abc",89,4)
    assert m.read(b"abc") is None
    m.update(b"abc",89,5)
    assert m.read(b"abc")==89
    assert m.exact_key_hits==2 and len(m.buffer)==1280

def test_set_associative_collision_evicts_oldest_no_wrong_key():
    m=FixedKeyTable(1344,2)
    groups={}
    for i in range(256):
        key=bytes((i,(i+5)%256,(i+37)%256))
        bucket=fnv(key)%m.sets
        groups.setdefault(bucket,[]).append(key)
    keys=next(g[:3] for g in groups.values() if len(g)>=3)
    m.update(keys[0],31,10)
    m.update(keys[1],32,11)
    m.update(keys[2],33,12)
    assert m.evictions==1
    assert m.read(keys[0]) is None
    assert m.read(keys[1])==32
    assert m.read(keys[2])==33

def test_streaming_source_poison_does_not_affect_past_decisions():
    raw=b"abcX-z" * 220
    tampered=raw[:660]+b"Q"*len(raw[660:])
    a,ap=study_owner(raw,"owner",1536,True)
    b,bp=study_owner(tampered,"owner",1536,True)
    assert [row for row in ap if row[0]<660]==[row for row in bp if row[0]<660]
    assert a["case_identity_sha256"]!=b["case_identity_sha256"]
    assert a["allocated_state_bytes"]["table2_1344"]==1344

def test_owner_isolation_and_same_source_replay():
    a=b"abcX"*400
    b=b"abcY"*400
    aa=study_owner(a,"owner_a",1536)
    ab=study_owner(a,"owner_a",1536)
    other=study_owner(b,"owner_b",1536)
    assert aa["source_sha256"]==ab["source_sha256"]
    assert aa["arms"]["table1_1344"]["correct"]==ab["arms"]["table1_1344"]["correct"]
    assert aa["source_sha256"]!=other["source_sha256"]

def test_all_arms_evaluate_identical_positions():
    a=study_owner(b"abcX"*450,"a",1536)
    b=study_owner(b"abcY"*450,"b",1536)
    total=aggregate([a,b])
    assert total["cases"]==a["cases"]+b["cases"]
    assert all(x["calls"]==total["cases"] for x in total["arms"].values())
    assert all(0<=x["accuracy_all_queries"]<=1 for x in total["arms"].values())
