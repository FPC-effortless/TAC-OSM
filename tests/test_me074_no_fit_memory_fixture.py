"""Deterministic synthetic *fixture integrity* tests, NO model training.

These are zero research-success evidence. Synthetic keyed recall remains
purely unit-test machinery until authentic long-history evidence exists.
"""
import numpy as np
import pytest

from scripts.me074_exposed_real_baselines import iterated_bigram_matrix, exact_markov, sample_ngram

@pytest.mark.parametrize("distractor_length",[16,64,256])
def test_answer_free_future_query_and_exact_fixture(distractor_length):
    rng=np.random.default_rng(74074+distractor_length)
    key=7
    value=29
    events=[("write",key,value)]
    for _ in range(distractor_length):
        another=int(rng.integers(8,40))
        events.append(("distractor_write",another,int(rng.integers(256))))
    events.append(("query",key,None))
    assert events[0][2]==value
    assert events[-1][2] is None
    assert all(e[1]!=key for e in events[1:-1])
    truth={e[1]:e[2] for e in events if e[0].endswith("write")}
    assert truth[key]==value
    assert len(events)==distractor_length+2

def test_absorbing_end_bigram_exact_distribution():
    probabilities=np.array([[.7,.3],[.2,.8]],dtype=float)
    survival=np.array([.8,.5],dtype=float)
    matrix=iterated_bigram_matrix(probabilities,survival)
    assert matrix.shape==(3,3)
    assert np.allclose(matrix.sum(1),1.)
    out=exact_markov(0,matrix)
    assert np.isclose(out[1][-1],.2)
    assert out[8][-1]>=out[4][-1]>=out[2][-1]>=out[1][-1]
    assert all(np.isclose(x.sum(),1.) for x in out.values())

def test_sampled_ngram_one_step_is_analytic_and_absorption_stable():
    state={"vocabulary":2,"first_order_transition_probability":[[.7,.3],[.2,.8]],
        "base_probability":[.5,.5], "context_transition_counts":{str(k):{} for k in (2,3,5)}}
    s=np.array([.8,.5])
    a=sample_ngram([0],state,s,5,"fixture")
    b=sample_ngram([0],state,s,5,"fixture")
    assert all(np.array_equal(a[k],b[k]) for k in (1,2,4,8))
    assert np.allclose(a[1],(np.array([.56,.24,.2])+1e-6)/(1+3e-6))
    assert a[8][-1]>=a[4][-1]>=a[2][-1]>=a[1][-1] - 1e-8
