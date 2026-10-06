import numpy as np
from tac_osm.adaptive_timescale import ARMS,PAIR_TYPES_TEST,PAIR_TYPES_TRAIN,TRAIN_FILTER_ALPHAS,TEST_FILTER_ALPHAS,PARENT_ALPHAS,make_pairs,make_policy,TemporalState,parameter_count
def test_filter_sets_are_disjoint():
    assert set(np.round(TRAIN_FILTER_ALPHAS,12)).isdisjoint(set(np.round(TEST_FILTER_ALPHAS,12)))
def test_new_test_filters_do_not_reuse_prior_transfer_filters():
    prior={0.40,0.58,0.74,0.86,0.93,0.989,0.992}
    assert set(np.round(TEST_FILTER_ALPHAS,12)).isdisjoint(prior)
def test_pair_families_are_complete():
    assert len(PAIR_TYPES_TRAIN)==21 and len(PAIR_TYPES_TEST)==21
def test_pairs_are_exact_negations():
    ex=make_pairs(7,32,10,alphas=TRAIN_FILTER_ALPHAS,pair_types=PAIR_TYPES_TRAIN)
    for a,b in zip(ex[::2],ex[1::2]):
        assert a.label!=b.label and a.current_observation==b.current_observation==0.0 and np.array_equal(b.history,-a.history)
def test_policy_parameter_count_is_fixed():
    assert parameter_count()==86 and make_policy(1,"three_adaptive").parameter_count==86
def test_parent_alpha_initialization_is_exact():
    p=make_policy(1,"three_adaptive")
    assert np.allclose(p.alphas,PARENT_ALPHAS)
def test_state_derivative_is_finite():
    s=TemporalState("three_adaptive",PARENT_ALPHAS)
    values,deriv=s.process_with_derivatives(np.array([1.0,-.5,.25],dtype=float))
    assert np.all(np.isfinite(values)) and np.all(np.isfinite(deriv))


def test_zero_padded_fixed_state_slots_are_actually_zero():
    history=np.array([1.0,-2.0,3.0])
    no_state=TemporalState("no_state",PARENT_ALPHAS).process(history)
    one=TemporalState("one_fixed",PARENT_ALPHAS).process(history)
    assert np.array_equal(no_state,np.zeros(3))
    assert one[1] == 0.0 and one[2] == 0.0
