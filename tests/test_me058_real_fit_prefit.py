"""ME058 *synthetic software fixtures*, never claims of trained model ability."""
import json

import numpy as np
import pytest
import torch

from scripts.me058_profile_original_fit import (
    original_batch, identify_fit_only, schedule_fit_only, hashed, SEEDS)


def small_source():
    rng=np.random.default_rng(42)
    return {
        10:{"state":rng.normal(loc=10.,size=(350,14)).astype(np.float32),
            "action":rng.normal(loc=11.,size=(350,14)).astype(np.float32),
            "human":np.zeros(350,dtype=np.bool_)},
        20:{"state":rng.normal(loc=20.,size=(350,14)).astype(np.float32),
            "action":rng.normal(loc=21.,size=(350,14)).astype(np.float32),
            "human":np.zeros(350,dtype=np.bool_)},
    }


def test_source_schedule_is_reproducible_and_paired_without_label_selection():
    source=small_source()
    source[10]["human"][256]=True
    a,stat=schedule_fit_only(source,SEEDS[0],steps=12)
    b,stat2=schedule_fit_only(source,SEEDS[0],steps=12)
    assert a==b and stat==stat2
    assert stat["eligible_source_windows"]>=100
    assert all(not source[ep]["human"][s+256] for pair in a for ep,s in pair)
    c,_=schedule_fit_only(source,SEEDS[1],steps=12)
    assert a!=c


def test_fit_only_physical_stats_never_join_episode_end_to_start():
    source={}
    for k,val in [(1,0.),(2,1000.)]:
        s=np.full((280,14),val,dtype=np.float32)
        a=s.copy()
        source[k]={"state":s,"action":a,
                   "human":np.zeros(280,dtype=np.bool_)}
    fit=identify_fit_only(source)
    assert max(fit["delta_scale"])==pytest.approx(.0005)
    assert max(fit["servo_gain"])==pytest.approx(0.)


def test_genuine_batch_uses_exact_window_and_both_arms_share_same_tensors():
    source=small_source()
    values=identify_fit_only(source)
    batch=original_batch(source,[[10,0],[20,10]],values)
    states,actions,authority,ids=batch
    assert states.shape==(2,273,14)
    assert actions.shape==(2,272,14)
    assert authority.shape==(2,273,1)
    assert ids==((10,0),(20,10))
    source[10]["human"][256]=True
    with pytest.raises(RuntimeError,match="human control"):
        original_batch(source,[[10,0],[20,10]],values)


def test_order_digest_stable_and_sensitive_to_schedule_order():
    assert hashed([[10,0],[20,1]])==hashed([[10,0],[20,1]])
    assert hashed([[10,0],[20,1]])!=hashed([[20,1],[10,0]])
