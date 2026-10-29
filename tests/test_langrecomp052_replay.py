"""Constructed-only implementation checks, not model training or competence scores."""
import pytest,torch
from scripts.langrecomp052_replay import (
    original_incremental,replay_parity,time_trial,assert_equal)
from scripts.langrecomp052_replay import IncrementalTAC043
from scripts.langlayer041_real_context_probe import model_for

@pytest.fixture(scope="module")
def fresh_engine():
    # Fresh randomly initialized graph is ONLY an engineering fixture.
    # Actual experiment loads sha-pinned original trained model.
    torch.set_num_threads(2)
    return IncrementalTAC043(model_for("h48_plain").eval(),"cap_all")

def test_constructed_causal_stream_and_full_prefix_replay(fresh_engine):
    raw=b"abcabcabcZ"*2
    with torch.inference_mode():
        x=original_incremental(fresh_engine,raw,"fake_fixture",steps=3)
        y=replay_parity(fresh_engine,raw,"fake_fixture",x)
    assert y["max_abs_logit"]<=3e-5
    assert y["incremental_tensor_state_bytes"]>0
    assert y["source_bytes_after_generation"]==len(raw)+3

def test_source_bytes_and_state_accounted_without_false_whole_ram_claim(fresh_engine):
    raw=b"historical-sequence"
    with torch.inference_mode():
        x=original_incremental(fresh_engine,raw,"ownerfixture",steps=2)
        a=time_trial(fresh_engine,raw,"ownerfixture",x["generated"],"incremental")
        b=time_trial(fresh_engine,raw,"ownerfixture",x["generated"],"recompute_every_step")
    assert a["retained_model_state_tensor_bytes"]==x["state_bytes"]
    assert a["retained_raw_prefix_bytes"]==0
    assert b["retained_model_state_tensor_bytes"]==0
    assert b["retained_raw_prefix_bytes"]==len(raw)+2
    assert len(a["decode_step_ms"])==len(b["decode_step_ms"])==2

def test_detect_invalid_logit_and_new_argmax():
    a=torch.tensor([[2.,1.,0.]])
    with pytest.raises(ValueError):assert_equal(a,torch.tensor([[1.,2.,0.]]))
    with pytest.raises(ValueError):assert_equal(a,torch.tensor([[float("nan"),1.,0.]]))

def test_source_stream_owners_do_not_cross_and_evidence_blind_to_label(fresh_engine):
    raw=b"abcdefghijklmnopqrst"
    with torch.inference_mode():
        a=original_incremental(fresh_engine,raw,"owner1",steps=2)
        b=original_incremental(fresh_engine,raw,"owner2",steps=2)
        assert a["generated"]==b["generated"]
        replay_parity(fresh_engine,raw,"owner1",a)

def test_original_checkpoint_hash_is_hardcoded():
    from scripts.langrecomp052_replay import CHECKPOINT_SHA,SOURCE_ORDER_SHA
    assert len(CHECKPOINT_SHA)==64 and len(SOURCE_ORDER_SHA)==64
    assert CHECKPOINT_SHA=="0cbc1b89f53ea2a0fe1371a179a9ff14cc35ae84d09f76bdab1dffeac4ac86cb"
