"""Constructed mathematical guards only, NOT synthetic capability evidence."""
import pytest
import torch
from scripts.latfuse006_event_decoder import (
    FEATURES,DELAYS,TRAIN_STOP,EVAL_START,
    ridge_fit_and_score,identity_source_and_parent_check
)

def test_learned_query_is_no_oracle_16d_readout_mode():
    assert set(FEATURES)=={
        "current","nonassoc","learned_associative_read","entire_memory_matrix"}
    assert DELAYS==(16,64,128,512)

def test_chronological_source_cohort_horizon_separation():
    a=[(0,TRAIN_STOP-12,1),(0,TRAIN_STOP-36,0)]
    b=[(0,EVAL_START,1),(0,EVAL_START-48,0)]
    assert identity_source_and_parent_check(a,b)

def test_invalid_overlapping_original_historical_cohorts_rejected():
    with pytest.raises(RuntimeError):
        identity_source_and_parent_check([(0,EVAL_START-500,1)],
            [(0,EVAL_START,0)])

def test_external_dual_ridge_reports_real_source_metric_contract():
    # Synthetic points are algebra-only tests. NEVER scientific performance.
    X=torch.tensor([[1.,0.],[0.,1.],[.9,.1],[.1,.9]])
    Y=torch.tensor([1.,0.,1.,0.])
    test=torch.tensor([[.8,.2],[.2,.8]])
    got=ridge_fit_and_score(X,Y,test,torch.tensor([1.,0.]))
    assert got["training_n"]==4 and got["evaluation_n"]==2
    assert got["feature_dim"]==2
    assert got["ridge_lambda"]==1.
    assert 0<=got["balanced_accuracy"]<=1
    assert 0<=got["real_event_Brier"]<=1
    assert 0<=got["physical_source_pair_ordering_accuracy"]<=1

def test_mismatched_paired_source_labels_cannot_enter_retrieval_probe():
    X=torch.randn(4,3)
    with pytest.raises(RuntimeError):
        ridge_fit_and_score(X,torch.tensor([1.,0.,0.,0.]),
            torch.randn(2,3),torch.tensor([1.,0.]))
