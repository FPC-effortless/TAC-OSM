"""No-fit constructed causality, state-bounds and source identity tests."""
import pytest
from scripts.langmemory048_verified_ring_index import VerifiedRingLocator,process_owner,aggregate

@pytest.mark.parametrize("budget,ring,slots",[(1344,1024,40),(4096,3072,128)])
def test_single_buffer_exact_total_bytes(budget,ring,slots):
    for ways in (1,2):
        x=VerifiedRingLocator(budget,ways)
        assert len(x.payload)==budget and x.capacity==ring and x.slots==slots
        assert x.sets*ways*8==budget-ring

def test_latest_continuation_is_verified_from_source_not_majority():
    x=VerifiedRingLocator(1344,2)
    for v in b"abcXabcY":x.observe(v)
    for v in b"abc":x.observe(v)
    assert x.current_key()==b"abc"
    assert x.lookup(b"abc")[0]==ord("Y")
    with pytest.raises(ValueError):x.lookup(b"xyz")

def test_stale_source_reference_fails_safe():
    x=VerifiedRingLocator(1344,2)
    for v in b"abcX":x.observe(v)
    for v in b"z"*1030:x.observe(v)
    assert x.lookup(b"zzz")[0]==ord("z")
    assert not any(x.entry(off)[1]==3 and x.valid_record(off,b"abc")
                   for off in x.slots_for(b"abc"))

def test_future_poison_cannot_change_past_state():
    prefix=b"abcX-xyz\n"*100
    a=VerifiedRingLocator(1344,2);b=VerifiedRingLocator(1344,2)
    for v in prefix:a.observe(v);b.observe(v)
    assert a.payload==b.payload and a.clock==b.clock
    assert a.lookup(a.current_key())==b.lookup(b.current_key())
    for v in b"abcY"*100:a.observe(v)
    for v in b"QQQQ"*100:b.observe(v)
    assert a.payload!=b.payload

def test_hash_fingerprint_is_not_accepted_without_source_content():
    x=VerifiedRingLocator(1344,1)
    for v in b"abcXabc":x.observe(v)
    off=x.slots_for(b"abc")[0]
    assert x.lookup(b"abc")[0]==ord("X")
    x.payload[off:off+2]=((x.fingerprint(b"abc")+1)&0xffff).to_bytes(2,"little")
    assert x.lookup(b"abc")[0] is None

def test_original_source_owner_separation_and_replay():
    a=process_owner(b"abcX"*400,"a",1536)
    b=process_owner(b"abcY"*400,"b",1536)
    c=process_owner(b"abcX"*400,"a",1536)
    assert a["source_sha256"]!=b["source_sha256"]
    assert a["stats"]["raw_1344"]["correct"]==c["stats"]["raw_1344"]["correct"]
    assert a["locators"]["locator2_1344"]["bytes"]==1344

def test_all_arms_and_owner_scopes_have_same_query_positions():
    a=process_owner(b"abcdefXYZ"*200,"a",1536)
    b=process_owner(b"aaaaabbcc"*200,"b",1536)
    z=aggregate([a,b])
    assert z["positions"]==a["positions"]+b["positions"]
    assert all(v["queries"]==z["positions"] for v in z["arms"].values())
    assert all(0<=v["success_all_positions"]<=1 for v in z["arms"].values())
