"""Development-only training, leakage and finite-cache interference tests."""
from __future__ import annotations

from dataclasses import replace

import torch

from tac_osm.integration_002_benchmark import make_pair
from tac_osm.integration_002_learned_dev import (
    CACHE_BUCKETS, LearnedNoisyMetric, bounded_cache, bucket,
    dense_l2, evaluate_seed, select_model_visible, train_seed, training_batch,
)


def test_supervised_training_tensors_have_only_visible_shapes():
    q, x, labels = training_batch(43, 0)
    assert q.shape == (12,12)
    assert x.shape == (12,48,12)
    assert labels.shape == (12,)
    assert ((0 <= labels)&(labels < 32)).all()


def test_initial_model_equals_negative_euclidean_distance():
    torch.manual_seed(1)
    q = torch.randn(3,12)
    x = torch.randn(3,15,12)
    metric=LearnedNoisyMetric()
    ref=-(x-q[:,None,:]).square().sum(-1)
    assert torch.allclose(metric(q,x),ref,atol=1e-5)


def test_training_performs_nonzero_gradient_and_parameter_updates():
    model,record=train_seed(43,steps=3)
    assert record["steps"]==3
    assert record["nonzero_finite_gradient"]
    assert record["parameters_changed"]
    assert record["initial_parameters_sha256"] != record["final_parameters_sha256"]
    assert torch.isfinite(model.projection.weight).all()


def test_inference_selection_cannot_read_oracle_slot_or_answer():
    ep=make_pair(92,32,16,1.0,0.25).left
    poisoned=replace(ep,target_slot=(ep.target_slot+3)%32,target=1-ep.target)
    model=LearnedNoisyMetric()
    assert select_model_visible(model,ep)==select_model_visible(model,poisoned)
    assert dense_l2(ep)==dense_l2(poisoned)


def test_capacity_fixed_and_real_updates_overwrite_existing_slots():
    ep=make_pair(442,512,32,1.0,0.25).left
    yes=bounded_cache(ep,include_writes=True)
    no=bounded_cache(ep,include_writes=False)
    assert yes.build_writes==no.build_writes==512
    assert yes.intervening_writes==32 and no.intervening_writes==0
    assert len(yes.entries)<=CACHE_BUCKETS
    assert yes.overwrites>0 and yes.overwrites>=no.overwrites
    assert 0<=bucket(ep.query_address)<64


def test_bounded_memory_carries_no_privileged_target_metadata():
    ep=make_pair(9,32,4,1.0,0.25).left
    cache=bounded_cache(ep,include_writes=True)
    assert all(set(row)=={"address","value"} for row in cache.entries)
    assert len(cache.entries)<=64
    assert cache.build_writes==32 and cache.intervening_writes==4


def test_paired_bounded_swap_ablation_changes_causal_content():
    p=make_pair(399,32,4,1.0,0.08)
    # Same-present paired episodes differ only at the target value; the
    # non-oracle cache is content-driven and does not inspect target_slot.
    a=bounded_cache(p.left,include_writes=True)
    b=bounded_cache(p.right,include_writes=True)
    assert a.build_writes==b.build_writes==32
    assert a.intervening_writes==b.intervening_writes==4
    assert a.overwrites==b.overwrites


def test_evaluation_uses_a_distinct_namespace_and_reports_work():
    model,_=train_seed(43,steps=2)
    cells=evaluate_seed(model,131,pairs_per_cell=2,H=(32,),D=(4,),
                        fractions=(0.0,1.0),noises=(0.08,0.25))
    assert len(cells)==4
    for c in cells:
        assert c["evaluation_seed"]==131
        assert c["episode_count"]==4
        assert c["dense_query_candidates"]==36
        assert c["cache_build_plus_refresh"]==36
        assert c["mean_bounded_query_candidates"]<=64
        assert c["reset_correct"]==0.5
        for key in ("learned_correct","euclidean_correct","learned_recall_at_1",
                    "euclidean_recall_at_1","bounded_correct",
                    "bounded_nowrite_correct","bounded_retained",
                    "bounded_swapped_correct"):
            assert 0<=c[key]<=1


def test_invalid_training_budget_fails_closed():
    for steps in (-1,0,181):
        try: train_seed(43,steps=steps)
        except ValueError: pass
        else: raise AssertionError("unregistered development budget accepted")
