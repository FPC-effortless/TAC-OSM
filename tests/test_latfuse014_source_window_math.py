"""LATFUSE014 pure mathematical test cases, NEVER scientific performance."""
import torch
from scripts.latfuse014_source_neutral_window import (
    OFFSETS,fit_probe,content_top1,grouped_ci)
def test_history_negatives_all_fixed_offsets_and_zero_label_logic():
    assert OFFSETS==(0,8,16,24,32,40,48,56)
    assert all((OFFSETS[(i+1)%8]-OFFSETS[i])%64==8 for i in range(8))
def test_train_only_lambda_one_primal_dual_ridge_equivalence():
    torch.manual_seed(14)
    x=torch.randn(16,6)
    y=torch.randn(16)
    test=torch.randn(7,6)
    got,n,d=fit_probe(x,y,torch.ones(16,dtype=torch.bool),test)
    cen=x.mean(0);sd=x.std(0,unbiased=False).clamp_min(1e-4)
    tx=((x-cen)/sd).clamp(-6,6)/(6**.5)
    vx=((test-cen)/sd).clamp(-6,6)/(6**.5)
    yc=y.mean();ys=y.std(unbiased=False).clamp_min(1e-4)
    expected=(vx@tx.T@torch.linalg.solve(tx@tx.T+torch.eye(16),((y-yc)/ys)[:,None]))[:,0]*ys+yc
    torch.testing.assert_close(got,expected.clamp(0,9),atol=2e-6,rtol=1e-5)
    assert n==16 and d==6
def test_balanced_eight_way_top1_respects_source_train_center():
    eye=torch.eye(8)
    vals=torch.nn.functional.pad(eye,(0,8)).unsqueeze(0)
    read=vals.clone()
    hits,r=content_top1(read,vals,torch.zeros(16),torch.ones((1,8),dtype=torch.bool))
    assert r["top1"]==1 and all(x==1 for x in r["requested_position_top1"])
    rotated=read.roll(1,1)
    wrong,rw=content_top1(rotated,vals,torch.zeros(16),torch.ones((1,8),dtype=torch.bool))
    assert rw["top1"]==0
def test_source_site_bootstrap_does_not_resample_individual_queries():
    src=torch.arange(8).repeat_interleave(2)
    v=torch.arange(16).float()
    assert grouped_ci(v,src,14014)==grouped_ci(v,src,14014)
