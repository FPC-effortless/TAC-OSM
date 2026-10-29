"""Constructed controls of chronological and source-owner boundaries, not FIT results."""
from scripts.langmemory046_verification_controls import ExactContextIndex,collect,selections,evaluate,random_rank,RANDOM_SEEDS

def test_index_only_reads_previous_continuations():
    x=ExactContextIndex()
    assert x.read(b"abc")==(None,0)
    x.observe(b"abc",88)
    assert x.read(b"abc")==(88,1)
    x.observe(b"abc",89);x.observe(b"abc",89)
    assert x.read(b"abc")==(89,3) and x.transitions==3

def test_exact_index_tie_break_deterministic():
    x=ExactContextIndex();x.observe(b"abc",99);x.observe(b"abc",51)
    assert x.read(b"abc")==(51,2)
    assert x.packed_lower_bound()>0 and x.python_heap_bytes()>=x.packed_lower_bound()

def test_future_poison_preserves_prefix_predictions():
    data=b"alpha-betatest-foo;bar\n"*65
    changed=data[:900]+b"\xff"*(len(data)-900)
    a,_=collect(data,"owner_A")
    b,_=collect(changed,"owner_A")
    ax=[v for v in a if v["position"]<900]
    bx=[v for v in b if v["position"]<900]
    assert ax==bx

def test_matched_random_call_budget_target_independent():
    data=b"abbaabbaacbbcbbc"*82
    cases,_=collect(data,"owner")
    masks=selections(cases,"owner",RANDOM_SEEDS[0])
    assert len(masks["select_scan"])==len(masks["random_all"])==len(masks["random_uncertain"])
    assert masks["select_scan"]==masks["select_index"]
    altered=[dict(r,truth=(r["truth"]+1)%256) for r in cases]
    assert selections(altered,"owner",RANDOM_SEEDS[0])==masks

def test_index_and_naive_scan_equal():
    data=b"abcX"+b"z"*110+b"abcY"+b"abcX"*95
    cases,_=collect(data,"p")
    masks=selections(cases,"p",29)
    rows=evaluate(cases,masks)
    assert rows["select_scan"]["correct"]==rows["select_index"]["correct"]
    assert rows["select_index"]["scan_positions"]==0
    assert rows["select_scan"]["index_lookups"]==0

def test_source_owner_separation():
    a=b"abcX"+b"x"*600;b=b"abcY"+b"y"*600
    ar,am=collect(a,"a");br,bm=collect(b,"b")
    assert am["owner"]=="a" and bm["owner"]=="b"
    assert am["full_original_source_sha256"]!=bm["full_original_source_sha256"]
    assert random_rank(11,"a",256)!=random_rank(11,"b",256)
    assert len(ar)>0 and len(br)>0

def test_fixed_random_seed_no_target_leakage():
    vals=[random_rank(s,"owner",256) for s in RANDOM_SEEDS]
    assert len(set(vals))==len(RANDOM_SEEDS)
    assert random_rank(11,"owner",256)==vals[0]
