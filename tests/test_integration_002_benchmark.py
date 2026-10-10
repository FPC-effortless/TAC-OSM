"""Pre-preregistration validity checks for noisy-key INTEGRATION-002 pilot."""
from __future__ import annotations

from tac_osm.integration_002_benchmark import (
    HARD_FRACTIONS, NOISE_LEVELS, diagnostic_rows, make_pair,
    similarity_prediction, validate_pair, visible_inputs,
)


def test_no_model_visible_exact_key_or_oracle_field():
    for h in (32,128,512):
        for d in (1,4,16,32):
            for f in HARD_FRACTIONS:
                for noise in NOISE_LEVELS:
                    p = make_pair(71 + h*100_003 + d*97 + round(f*100)
                                  + round(noise*100),h,d,f,noise)
                    validate_pair(p,h,d)
                    for ep in (p.left,p.right):
                        vis = visible_inputs(ep)
                        assert not any(x in vis for x in
                                       ("target","target_slot","latent","condition","channel"))
                        assert vis["query_address"] not in [
                            item["address"] for item in vis["history"]+vis["writes"]
                        ]


def test_same_present_opposite_past_with_same_format_writes():
    p = make_pair(103,128,32,1.0,0.25)
    a,b = p.left,p.right
    assert a.query_address == b.query_address
    assert a.current_bit == b.current_bit
    assert a.writes == b.writes
    assert len(a.writes) == 32
    assert all(hasattr(x,"address") and hasattr(x,"value") for x in a.writes)
    assert a.target != b.target
    assert sum(x != y for x,y in zip(a.history,b.history)) == 1


def test_reproducible_noisy_generation():
    a = make_pair(556,32,4,0.25,0.08)
    b = make_pair(556,32,4,0.25,0.08)
    c = make_pair(557,32,4,0.25,0.08)
    assert a == b
    assert a != c


def test_unknown_generator_levels_fail_closed():
    for args in ((32,2,0.0,0.08),(6,4,0.0,0.08),
                 (32,4,0.75,0.08),(32,4,0.25,0.1)):
        try:
            make_pair(1,*args)
        except ValueError:
            pass
        else:
            raise AssertionError(f"invalid generator grid accepted: {args}")


def test_nonlearned_similarity_can_fail_even_if_oracle_is_exact():
    # Not a trained-model success claim: prove noisy observables actually
    # break the exact-match shortcut on at least one deterministic episode.
    errors=0
    for seed in range(40):
        p=make_pair(1_200_000+seed,32,32,1.0,0.25)
        for ep in (p.left,p.right):
            guess,_ = similarity_prediction(ep,metric="euclidean")
            errors += guess != ep.target
    assert errors > 0


def test_raw_similarity_never_reads_target_label_for_selection():
    p = make_pair(321,128,4,0.5,0.08)
    # Flip all target labels and slots in metadata only; raw predictions
    # must not depend on any benchmark-side oracle attributes.
    from dataclasses import replace
    ep = p.left
    altered = replace(ep,target=1-ep.target,target_slot=(ep.target_slot+1)%128)
    for metric in ("cosine","euclidean"):
        assert similarity_prediction(ep,metric=metric) == (
            similarity_prediction(altered,metric=metric)
        )


def test_noisy_development_diagnostic_is_separate_from_001_contract():
    rows = diagnostic_rows(seeds=(17,),histories=(32,128),delays=(4,),
                           fractions=(0.0,1.0),noises=(0.08,0.25),
                           pairs_per_cell=2)
    assert len(rows) == 8
    assert all(r["oracle_accuracy"] == 1 for r in rows)
    assert all(r["exact_model_visible_identity_reuse"] is False for r in rows)
    assert all(r["candidate_scores_computed_per_query"] == r["H"]+r["D"]
               for r in rows)
    assert all(0 <= r["cosine_accuracy"] <= 1 for r in rows)
    assert all(0 <= r["euclidean_accuracy"] <= 1 for r in rows)
