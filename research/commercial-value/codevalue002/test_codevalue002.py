"""Constructed-only CODEVALUE002 checks. No publisher bytes needed."""
import hashlib
import math
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train import REG, ARMS, SEEDS, PARAMS, make, preflight, gitblob, batches, schedule_hash
from evaluate import P3, aggregate, verify_all

def test_registration_complete_and_models_are_real_500k():
    assert preflight()
    assert set(ARMS)=={"tac","gru","ssm"}
    assert SEEDS==(1701,1702,1703)
    assert len(REG["train"]["files"])==14
    assert len(REG["dev"]["files"])==6
    assert REG["budget"]["targets_per_model"] if "targets_per_model" in REG["budget"] else REG["budget"]["target_bytes"]==327680
    for arm in ARMS:
        model=make(arm)
        assert sum(p.numel() for p in model.parameters())==PARAMS[arm]
        assert abs(PARAMS[arm]/499623-1)<.01

def test_git_blob_and_no_preregistered_overlap():
    b=b"real constructed only"
    expected=hashlib.sha1(b"blob "+str(len(b)).encode()+b"\0"+b).hexdigest()
    assert gitblob(b)==expected
    fitpaths={("python/cpython",x["path"]) for x in REG["train"]["files"]}
    devpaths={(x["repo"],x["path"]) for x in REG["dev"]["files"]}
    assert not (fitpaths & devpaths)
    for x in REG["train"]["files"] + REG["dev"]["files"]:
        assert len(x["git_blob_sha1"])==40 and x["size"]>1000

def test_equal_sampler_with_constructed_source_only():
    docs={"test_a":b"abcdef012345"*130,"test_b":b"foobar123456"*130}
    first=[(s, [(o,p,b) for o,p,b in rows]) for s,rows in batches(docs,1701)]
    second=[(s, [(o,p,b) for o,p,b in rows]) for s,rows in batches(docs,1701)]
    assert len(first)==320 and first==second
    assert schedule_hash(docs,1701)==schedule_hash(docs,1701)

def test_p3_causal_and_normalized():
    docs={"own_a":b"abcabcabcabcdefdef"*200,"own_b":b"defdefaaabbbccc"*200}
    p=P3(docs)
    for ctx in (b"abc",b"def",b"xyz",b"x"):
        prob=p.probs(ctx)
        assert len(prob)==256 and all(q>0 and math.isfinite(q) for q in prob)
        assert math.isclose(sum(prob),1.,abs_tol=1e-9)
        assert p.next(ctx) in range(256)

def test_missing_source_counts_fail_closed():
    item={"nll_sum":2., "top1_count":1,"exact":{2:0,4:0,6:0},"group":"requests"}
    with pytest.raises(ValueError,match="denominator"):
        aggregate([item,item])

def test_aggregate_fixed_owner_group_and_offer_denominators():
    groups=REG["evaluation"]["groups"]
    rows=[]
    for group in groups:
        for i in range(8):
            rows.append({"group":group,"nll_sum":64.,"top1_count":32,
                         "exact":{2:1,4:int(i==0),6:0},"forward_ms":1.})
    result=aggregate(rows)
    assert result["targets"]==3072
    assert result["offers"]==48
    assert result["exact"]["2"]==24
    assert result["exact"]["4"]==3
    assert result["oracle_net_keys"]["4"]==9
    assert all(v["offers"]==16 for v in result["by_publisher"].values())

def test_seal_requires_every_checkpoint_before_evaluation(tmp_path):
    with pytest.raises(FileNotFoundError):
        verify_all(tmp_path)

def test_tac_graph_has_real_first_stage_and_graph_free_upper():
    m=make("tac")
    assert m.depth==2 and m.upper_state_width==48 and not m.upper_concept_graph
    assert tuple(m.native_key.weight.shape)==(8,48)
    assert tuple(m.context_lag_weights.shape)==(3,48)
    assert m.ffn_width==455
    assert m.parameter_count()==499623
