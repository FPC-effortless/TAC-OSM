"""Falsification and cost-transparency tests for Integration-001 development.

Passing these checks is not a confirmatory result; in fact an exact-key
baseline scoring 100% reveals a shortcut the learned system must beat in
meaningful generalization or cost rather than claiming accuracy novelty.
"""
from __future__ import annotations

import json
from pathlib import Path

from tac_osm.integration_001_benchmark import model_inputs, validate_pair
from tac_osm.integration_001_diagnostic import (
    INTERFERENCE_STRENGTHS, bit_distance, dense_query_cost, diagnostic_rows,
    exact_lookup_control, make_interference_pair, near_key, reset_control,
)


def test_same_format_near_neighbor_distractors_are_genuinely_present():
    for h in (32, 128, 512):
        for delay in (1, 4, 16, 32):
            for strength in INTERFERENCE_STRENGTHS:
                p = make_interference_pair(42 + h*113 + delay*19, h, delay, strength)
                count = sum(bit_distance(w.key, p.left.query_key) == 1
                            for w in p.left.writes)
                assert count == round(delay * strength)
                assert all(len(x.key) == 32 for x in p.left.writes)


def test_neighbor_generation_cannot_duplicate_target_identity():
    for strength in INTERFERENCE_STRENGTHS:
        p = make_interference_pair(103, 32, 32, strength)
        validate_pair(p, 32, 32)
        assert all(w.key != p.left.query_key for w in p.left.writes)
        assert len({x.key for x in p.left.history + p.left.writes}) == 64


def test_only_target_payload_differs_in_opposite_history_pairs():
    pair = make_interference_pair(17, 128, 16, 1.0)
    a, b = model_inputs(pair.left), model_inputs(pair.right)
    assert a["query_key"] == b["query_key"]
    assert a["current_bit"] == b["current_bit"]
    assert a["writes"] == b["writes"]
    assert sum(x != y for x, y in zip(a["history"], b["history"])) == 1
    assert pair.left.target != pair.right.target


def test_exact_lookup_shortcut_is_perfect_despite_confusable_writes():
    for h in (32, 128, 512):
        for delay in (1, 4, 16, 32):
            p = make_interference_pair(h*113 + delay, h, delay, 1.0)
            for episode in (p.left, p.right):
                trace = exact_lookup_control(episode)
                assert trace.target_found
                assert trace.output == episode.target
                assert trace.index_refresh_writes == delay
                assert trace.intervening_write_count == delay


def test_actual_distractor_writes_do_not_change_exact_key_solution():
    p = make_interference_pair(105, 512, 32, 1.0)
    for ep in (p.left, p.right):
        with_writes = exact_lookup_control(ep)
        without_writes = exact_lookup_control(ep, include_intervening=False)
        assert with_writes.output == without_writes.output == ep.target
        assert with_writes.total_primitive_actions - without_writes.total_primitive_actions == 32


def test_reset_control_exactly_half_on_opposite_history_pair():
    p = make_interference_pair(105, 128, 16, 1.0)
    guesses = [reset_control(ep) for ep in (p.left, p.right)]
    assert guesses[0] == guesses[1]
    assert sum(g == ep.target for g, ep in zip(guesses, (p.left, p.right))) == 1


def test_dense_topk_scoring_cannot_be_counted_as_sublinear():
    for h in (32, 128, 512):
        ep = make_interference_pair(h, h, 4, 0.5).left
        assert dense_query_cost(ep) == h + 4
        # A single-query index cost includes ALL construction and refresh.
        assert exact_lookup_control(ep).total_primitive_actions == h + 4 + 2


def test_development_diagnostics_not_interpreted_as_confirmatory():
    rows = diagnostic_rows(seeds=(11,), history_sizes=(32, 512),
                           delays=(1, 16), strengths=(0.0, 1.0),
                           pairs_per_cell=3)
    assert len(rows) == 8
    assert all(r["exact_key_lookup_accuracy"] == 1.0 for r in rows)
    assert all(r["no_intervening_write_accuracy"] == 1.0 for r in rows)
    assert all(r["reset_accuracy"] == 0.5 for r in rows)
    assert all(r["dense_candidate_checks_per_query"] == r["H"]+r["D"] for r in rows)
    assert all(r["exact_index_total_actions_per_query"] == r["H"]+r["D"]+2 for r in rows)
    contract = json.loads((Path(__file__).resolve().parents[1] /
                           "contracts/TACOSM-PLM-INTEGRATION-001.json").read_text())
    assert contract["thresholds"]["minimum_persistent_accuracy"] == 0.9
    assert contract["evaluation"]["seeds"] == [101, 211, 307, 401, 503]
