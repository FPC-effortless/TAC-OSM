#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from tac_osm.contract import load_contract

EXPERIMENT_ID = "TACOSM-GRAPH-CASM-EVIDENCE-INDEX-017B"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("result")
    args = parser.parse_args()

    result = json.loads(Path(args.result).read_text(encoding="utf-8"))
    contract = load_contract(EXPERIMENT_ID)

    assert result["experiment_id"] == EXPERIMENT_ID
    assert result["status"] == "measured"
    assert result["protocol"]["seeds"] == list(contract.seeds)
    assert result["protocol"]["M_levels"] == list(contract.h_levels)
    assert result["protocol"]["tasks_per_seed_M"] == contract.eval_steps
    assert result["protocol"]["primary_M"] == 512
    assert result["protocol"]["library_size"] == 512
    assert result["protocol"]["timing_repeats"] == 30

    expected_tasks = len(contract.seeds) * len(contract.h_levels) * contract.eval_steps
    raw = result["raw"]
    assert set(map(int, raw)) == set(contract.seeds)

    observed = 0
    for seed in contract.seeds:
        assert set(map(int, raw[str(seed)])) == set(contract.h_levels)
        for m in contract.h_levels:
            assert len(raw[str(seed)][str(m)]) == contract.eval_steps
            for row in raw[str(seed)][str(m)]:
                observed += 1
                assert row["selector_exact_match"] is True
                assert row["indexed_histogram_bin_reads"] > 0
                assert row["exhaustive_candidate_record_reads"] == 16 * m
                assert row["indexed_histogram_bin_reads"] <= 16 * (2 ** 6)
    assert observed == expected_tasks

    checks = result["checks"]
    for seed in contract.seeds:
        c = checks[str(seed)]
        assert c["train_eval_structure_disjoint"] is True
        assert c["train_eval_truth_disjoint"] is True
        assert c["target_identity_used_before_choice"] is False
        assert c["target_evidence_used_before_choice"] is False
        assert c["same_activation_trace_channel"] is True
        assert c["exact_selector_equivalence_required"] is True

    primary = result["primary"]
    assert primary["M"] == 512
    assert len(primary["seed_level_exhaustive_minus_indexed_us"]) == len(contract.seeds)
    assert primary["exact_selector_action_agreement"] == 1.0

    recomputed = []
    for seed in contract.seeds:
        recomputed.append(
            result["timing"][str(seed)]["512"]["exhaustive_query_us"]
            - result["timing"][str(seed)]["512"]["indexed_query_us"]
        )

    assert all(abs(a - b) < 1e-12 for a, b in zip(
        recomputed, primary["seed_level_exhaustive_minus_indexed_us"]
    ))
    assert abs(
        statistics.fmean(recomputed) - primary["mean_exhaustive_minus_indexed_us"]
    ) < 1e-12

    complexity = result["complexity"]
    for m in contract.h_levels:
        assert complexity["exhaustive_candidate_record_reads"][str(m)] == 16 * m
    assert complexity["indexed_query_upper_bound_bin_reads"] == 16 * (2 ** 6)

    assert result["scope"]["index_build_is_one_time_per_seed"] if "index_build_is_one_time_per_seed" in result["scope"] else True
    assert result["scope"]["index_build_remains_O_M"] is True
    assert result["scope"]["no_claim_of_C5_total_history_sublinearity"] is True

    leakage = result["audit"]["leakage"]
    assert "target identity" in leakage["forbidden_selector_inputs"]
    assert "target index" in leakage["forbidden_selector_inputs"]
    assert "test outcomes" in leakage["forbidden_selector_inputs"]

    print(
        json.dumps(
            {
                "experiment_id": EXPERIMENT_ID,
                "validated_trials": observed,
                "primary_mean_exhaustive_minus_indexed_us": primary["mean_exhaustive_minus_indexed_us"],
                "seed_bootstrap_95ci": primary["seed_bootstrap_95ci"],
                "exact_selector_agreement": primary["exact_selector_action_agreement"],
                "leakage_gates": "PASS",
                "complexity_accounting": "PASS",
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
