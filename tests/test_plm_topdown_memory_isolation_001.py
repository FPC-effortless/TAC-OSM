import numpy as np
from tac_osm.memory_isolation import ARMS,GROUND_TRUTH_ALPHAS,STATE_ALPHAS,featurize,make_pairs,representability_sign_accuracy,target_action
def test_fixed_ground_truth_relation():
    assert GROUND_TRUTH_ALPHAS==(0.45,0.985)
def test_state_arm_structure():
    assert ARMS==("no_state","single_timescale","two_timescale","mtsk")
    assert STATE_ALPHAS["no_state"]==(None,None,None)
    assert STATE_ALPHAS["single_timescale"]==(0.90,None,None)
    assert STATE_ALPHAS["two_timescale"]==(0.50,0.98,None)
def test_negation_pairs_are_balanced():
    ex=make_pairs(11,32,10)
    for a,b in zip(ex[::2],ex[1::2]):
        assert a.current_observation==b.current_observation==0.0
        assert a.label!=b.label and np.array_equal(b.history,-a.history)
def test_pair_family_is_not_an_input():
    ex=make_pairs(12,16,4)
    for arm in ARMS:
        X=featurize(ex,arm)
        assert X.shape==(8,4)
def test_no_state_features_are_zero():
    ex=make_pairs(13,16,4)
    assert np.allclose(featurize(ex,"no_state")[:,1:],0.0)
def test_single_and_two_state_padding_is_zero():
    ex=make_pairs(14,16,4)
    one=featurize(ex,"single_timescale")
    two=featurize(ex,"two_timescale")
    assert np.allclose(one[:,2:],0.0)
    assert np.allclose(two[:,3:],0.0)
def test_mtsk_representation_sign_is_above_gate():
    assert representability_sign_accuracy()>=0.90
