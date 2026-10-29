"""No-fit ordinal code mathematical identity tests: synthetic controls only."""
import math
import pytest
import torch
from context_transition import ordinal_code, PERIODS
from scripts.langkernel061_vector_ordinal import (
    vectorized_ordinal, original_pair, vectorized_pair, vectorized_two_calls,
    ARMS, PROMPTS, CALLS, TRIALS
)


@pytest.mark.parametrize("dtype",[torch.float32,torch.float64])
def test_vectorized_reference_is_bitwise_equal_at_period_boundaries(dtype):
    indexes=torch.tensor(sorted({0,1,2,6,7,8,12,13,28,29,58,59,127,128,511,512,8191,8192,65535}),dtype=torch.long)
    ref=ordinal_code(indexes,dtype)
    got=vectorized_ordinal(indexes,dtype)
    assert torch.equal(got,ref), "kernel math changed finite precision"
    assert got.shape==(len(indexes),8)


@pytest.mark.parametrize("clock",[0,1,128,512,8192])
def test_vectorized_pair_exact_full_reference(clock):
    t=torch.tensor([clock],dtype=torch.long)
    expected=original_pair(t,torch.float32)
    assert torch.equal(vectorized_pair(t,torch.float32),expected)
    assert torch.equal(vectorized_two_calls(t,torch.float32),expected)


def test_input_contract_does_not_mutate_or_create_cache():
    p=torch.tensor([0,128,8192])
    before=p.clone()
    out=vectorized_ordinal(p,torch.float32)
    assert torch.equal(p,before)
    assert out.shape==(3,8)
    for bad in (torch.tensor([1.0]),torch.tensor([],dtype=torch.long)):
        with pytest.raises(ValueError):vectorized_ordinal(bad,torch.float32)


def test_original_code_and_periods_never_mutated():
    assert PERIODS==(7,13,29,59)
    assert set(ARMS)=={"original_two_calls","vectorized_two_calls","vectorized_pair"}
    assert PROMPTS==(128,512,8192) and CALLS==64 and TRIALS==7


def test_frozen_math_reference_values():
    p=torch.tensor([0],dtype=torch.long)
    x=vectorized_ordinal(p,torch.float64)
    assert torch.equal(x[0,::2],torch.ones(4,dtype=torch.float64))
    assert torch.equal(x[0,1::2],torch.zeros(4,dtype=torch.float64))
