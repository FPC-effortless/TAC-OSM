"""Constructed no-fit checks of source cohort selection and reset-context geometry."""
import math
import sys
from pathlib import Path
import pytest
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"research/mathematical-engineering/langscale070"))
from scaled_recurrence import ScaledRecurrenceTAC070
from scripts.langscale071_context_eval import target_cohort,score_length,LENGTHS


def fake_dev():
    return {f"owner{i}.rst":bytes(((x+i*17)%256 for x in range(700)))
            for i in range(4)}


def test_same_target_cohort_has_four_separate_owners_and_hash():
    dev=fake_dev()
    cohort,sha=target_cohort(dev)
    assert len(cohort)==256 and len(sha)==64
    assert target_cohort(dev)==(cohort,sha)
    assert {o for o,_,_ in cohort}==set(dev)
    for o,t,y in cohort:
        assert 256<=t<len(dev[o])
        assert dev[o][t]==y
    bad={**dev,"owner2.rst":b"abc"}
    with pytest.raises(ValueError):
        target_cohort(bad)


def test_original_model_reset_window_source_geometry_and_finite_metric():
    torch.manual_seed(711)
    dev=fake_dev()
    cohort,_=target_cohort(dev)
    model=ScaledRecurrenceTAC070(scale="p250k").eval()
    torch.set_num_threads(2)
    row=score_length(model,dev,cohort,16)
    assert row["context"]==16 and row["n"]==256
    assert len(row["per_document"])==4
    assert math.isfinite(row["nll"]) and 0<=row["top1"]<=1
    # Deliberately wrong authentic target should be rejected.
    poisoned=list(cohort)
    owner,t,y=poisoned[0]
    poisoned[0]=(owner,t,(y+1)%256)
    with pytest.raises(ValueError):
        score_length(model,dev,poisoned,16)


def test_context_set_includes_untrained_256_length_as_extrapolation():
    assert LENGTHS==(16,32,64,128,256)
    assert max(LENGTHS)>128
