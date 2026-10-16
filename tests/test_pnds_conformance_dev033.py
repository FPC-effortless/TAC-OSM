import math
import pytest
from tac_osm.development.pnds_conformance import routing_metrics,admission_rates,four_gate_decision,path_calibration,inspect_engineering_report


def test_softmax_mass_matches_two_level_formula_with_large_logits():
    r=routing_metrics([1002.,1002.,1000.,1000.,1000.],{0,1},[0,1],.5)
    assert r['distractor_mass']==pytest.approx(3/(2*math.exp(4)+3))
    assert r['recall']==1 and r['minimum_relevant_margin']==2
    assert r['total_indexing_cost_measured'] is False


def test_low_distractor_mass_does_not_imply_complete_recall():
    r=routing_metrics([20.,19.,0.],{0,1},[0])
    assert r['distractor_mass']<1e-8 and r['recall']==.5


def test_false_admit_and_contamination_have_different_denominators():
    r=admission_rates([True,False,False,False,True],[True,True,True,True,False])
    assert r['false_admit_rate']==.25 and r['contamination_rate']==.5


def test_missing_truth_and_zero_denominators_are_not_success():
    assert admission_rates([True],[None])['contamination_rate'] is None
    assert admission_rates([False],[True])['contamination_rate'] is None
    assert admission_rates([True],[False])['false_admit_rate'] is None
    assert admission_rates([True,True],[False,None])['contamination_rate'] is None


def test_four_gate_shadow_fails_closed_without_claiming_online_enforcement():
    assert four_gate_decision(True,True,True,True)['would_admit']
    for missing in (None,False):assert not four_gate_decision(True,missing,True,True)['would_admit']
    with pytest.raises(TypeError):four_gate_decision(1,True,True,True)


def test_calibration_keeps_path_lengths_and_acceptance_denominators_separate():
    r=path_calibration([1.,0.,.96,.97],[True,False,False,True],[1,1,10,10])
    assert r['by_length'][0]['brier']==0
    assert r['by_length'][0]['false_commit_fraction']==0
    assert r['by_length'][1]['false_commit_fraction']==.5
    assert r['calibration_guaranteed'] is False


def test_receipt_acceptance_cannot_become_validity_ground_truth():
    r=inspect_engineering_report({'source':'fixture','rows':[{'trace':[{'receipt_acceptance':True,'commit_accepted':True}]}]})
    assert r['commit_decisions']['labeled']==0
    assert r['four_gate_shadow']['would_admit'] is False
    assert r['formal_correspondence_proved'] is False


def test_invalid_routing_and_calibration_records_are_rejected():
    with pytest.raises(ValueError):routing_metrics([0.,1.],{0},[0,0])
    with pytest.raises(ValueError):routing_metrics([float('nan')],{0},[0])
    with pytest.raises(TypeError):path_calibration([.9],[1],[1])
