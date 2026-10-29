"""Constructed causal source-retrieval checks, not real-world language gains."""
import random
from scripts.langmemory045_source_probe import BoundedSketch,bucket,exact_history,recent,sample,aggregate

def test_long_occurrence_is_exact_and_not_recent():
    data=b"abcX"+b"z"*100+b"abcY"
    t=len(data)-1
    v,h,far,cost=exact_history(data,t,b"abc")
    a,hits=recent(data,t,b"abc")
    assert (v,h,far)==(ord("X"),1,1)
    assert a is None and hits==0 and cost==t-3

def test_poisoning_future_does_not_change_past_lookup():
    data=b"abcX"+b"z"*100+b"abcYdef"
    t=107
    changed=data[:t]+b"?"*(len(data)-t)
    key=data[t-3:t]
    assert recent(data,t,key)==recent(changed,t,key)
    assert exact_history(data,t,key)==exact_history(changed,t,key)

def test_sketch_bounded_bucket_and_deterministic_owner():
    s=BoundedSketch()
    assert bucket(b"abc")==bucket(b"abc")
    for item in (65,65,66,65):
        s.update(b"abc",item)
    assert s.read(b"abc")==65 and s.total[bucket(b"abc")]==4
    assert len(s.candidate)==256

def test_source_probe_replay_and_owner_scope():
    rng=random.Random(45)
    raw=(b"alpha-beta-gamma\ndef alpha(): pass\n"*20)+bytes(rng.randrange(256) for _ in range(150))
    a=sample(raw,"publisher_A")
    b=sample(raw,"publisher_A")
    c=sample(raw,"publisher_B")
    assert a==b
    assert a["case_identity_sha256"]!=c["case_identity_sha256"]
    assert a["positions_tested"]>0
    assert a["verifier_fraction"]<=.20

def test_full_aggregate_totals_and_no_artificial_metric():
    raw=b"abcdefgabcX"*140
    row=sample(raw,"only-fit")
    agg=aggregate([row,row])
    assert agg["positions"]==2*row["positions_tested"]
    assert agg["verified_fraction"]<=.20
    assert all(0<=x["coverage"]<=1 and 0<=x["accuracy_given_coverage"]<=1 for x in agg["arms"].values())
