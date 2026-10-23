"""ME077A numerical scoring unit checks, not simulated model evidence."""
import numpy as np
import pytest
from scripts.me077a_original_motion_audit import auc,ap,brier,ranking,thresholds,bins

def test_perfect_ranking():
    y=np.array([False,True,False,True]);p=np.array([.1,.9,.2,.8])
    assert auc(y,p)==pytest.approx(1.)
    assert ap(y,p)==pytest.approx(1.)

def test_constant_scores_have_chance_ranking_with_ties():
    y=np.array([False,True,False,True]);p=np.ones(4)*.25
    assert auc(y,p)==pytest.approx(.5)
    assert ap(y,p)==pytest.approx(.5)
    assert brier(y,p)==pytest.approx(.3125)

def test_reversed_ranking():
    y=np.array([True,True,False,False]);p=np.array([.1,.2,.9,.8])
    assert auc(y,p)==pytest.approx(0.)
    assert ap(y,p)<.5

def test_fixed_threshold_confusion_not_chosen_after_observations():
    y=np.array([True,False,True,False]);p=np.array([.7,.2,.08,.03])
    row=thresholds(y,p)["0.1"]
    assert (row["TP"],row["FP"],row["FN"],row["TN"])==(1,1,1,1)
    assert row["recall"]==pytest.approx(.5)

def test_probability_bins_count_each_example_once():
    y=np.array([False,True,True,False]);p=np.array([0,.05,.5,1.])
    z=bins(y,p)
    assert sum(r["n"] for r in z)==4
    assert len(z)==7

def test_ap_undefined_if_no_positive_events():
    assert auc(np.array([False,False]),np.array([.1,.9])) is None
    assert ap(np.array([False,False]),np.array([.1,.9])) is None
