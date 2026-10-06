#!/usr/bin/env python3
"""Preregistered top-down PLM assembly with bottom-up MTSK ablation."""
from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path

import numpy as np

from tac_osm.contract import load_contract
from tac_osm.measurement.results import (
    Design,
    Gate,
    MeasurementRecord,
    Provenance,
    contract_fingerprint,
    now,
    results_dir_for,
    write_record,
)
from tac_osm.mtsk_topdown import (
    BENCHMARK_HASH,
    dependency_lock_hash,
    evaluate,
    featurize,
    make_balanced_pairs,
    MLPPolicy,
    STATE_ALPHAS,
)

EXPERIMENT_ID = "TACOSM-PLM-TDBU-MTSK-001"
TRAIN_PAIRS = 300
TEST_PAIRS = 100
LEARNING_RATE = 0.05
HIDDEN = 12
MATERIALITY = 0.10

TRAINED_ARMS = ("no_state", "single_timescale", "two_timescale", "mtsk")


def _seed_examples(seed: int, H: int, split: str) -> list:
    base = (10_000 if split == "train" else 20_000) + seed * 1009 + H * 17
    return make_balanced_pairs(base, H, TRAIN_PAIRS if split == "train" else TEST_PAIRS)


def _assert_pair_contract(examples: list) -> None:
    by_pair: dict[int, list] = {}
    for x in examples:
        by_pair.setdefault(x.pair_id, []).append(x)
    assert by_pair and all(len(v) == 2 for v in by_pair.values())
    for pair in by_pair.values():
        assert pair[0].current_observation == pair[1].current_observation == 0.0
        assert pair[0].label != pair[1].label
        assert np.array_equal(pair[1].history, -pair[0].history)


def _bootstrap_seed_difference(a: list[float], b: list[float], seed: int, rounds: int = 20_000) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    diffs = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    draws = rng.choice(diffs, size=(rounds, len(diffs)), replace=True).mean(axis=1)
    return float(np.mean(diffs)), float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def run(args: argparse.Namespace) -> dict:
    contract = load_contract(EXPERIMENT_ID)
    if tuple(args.h_levels) != contract.h_levels:
        contract.require_levels(args.h_levels)
    contract.require_seeds(args.seeds)
    contract.require_steps(args.steps)
    contract.require_eval_steps(args.eval_steps)
    contract.require_arms([a.name for a in contract.arms])
    assert HIDDEN == 12 and LEARNING_RATE == 0.05

    records = []
    initial_hashes = {}
    for H in contract.h_levels:
        for seed in contract.seeds:
            train = _seed_examples(seed, H, "train")
            test = _seed_examples(seed, H, "test")
            _assert_pair_contract(train)
            _assert_pair_contract(test)
            train_labels = np.asarray([x.label for x in train], dtype=np.int64)
            assert set(train_labels.tolist()) == {0, 1}
            for arm in TRAINED_ARMS:
                policy = MLPPolicy(seed=seed + 7, hidden=HIDDEN)
                initial_hashes[(H, seed, arm)] = policy.parameter_hash()
                X = featurize(train, arm)
                assert X.shape == (len(train), 4)
                updates = policy.fit(X, train_labels, contract.steps, LEARNING_RATE)
                assert updates == contract.steps
                eval_result = evaluate(policy, test, arm)
                train_state_work = 0 if arm == "no_state" else len(train[0].history) * len(train) * 9
                optimizer_forward_backward_work = contract.steps * len(train) * 4 * (4 * HIDDEN + 2 * HIDDEN)
                train_work = float(train_state_work + optimizer_forward_backward_work)
                reset_result = evaluate(policy, test, "mtsk", intervention="reset") if arm == "mtsk" else None
                shuffle_result = evaluate(policy, test, "mtsk", intervention="shuffle") if arm == "mtsk" else None
                if policy.parameter_hash() == initial_hashes[(H, seed, arm)]:
                    raise RuntimeError("evaluation checkpoint is identical to initialization")
                if arm == "mtsk":
                    if reset_result is None or shuffle_result is None:
                        raise RuntimeError("MTSK intervention records missing")
                records.append({
                    "H": H,
                    "seed": seed,
                    "arm": arm,
                    "success": eval_result.success,
                    "action_gap": eval_result.action_gap,
                    "state_footprint": eval_result.state_footprint,
                    "evaluation_work_per_episode": eval_result.evaluation_work_per_episode,
                    "train_compute_work_proxy": train_work,
                    "parameter_count": policy.parameter_count,
                    "checkpoint_hash": policy.parameter_hash(),
                    "reset_success": reset_result.success if reset_result else None,
                    "shuffle_success": shuffle_result.success if shuffle_result else None,
                    "reset_action_gap": reset_result.action_gap if reset_result else None,
                    "shuffle_action_gap": shuffle_result.action_gap if shuffle_result else None,
                })

    primary_mtsk = [r["success"] for r in records if r["H"] == 256 and r["arm"] == "mtsk"]
    primary_single = [r["success"] for r in records if r["H"] == 256 and r["arm"] == "single_timescale"]
    primary_no = [r["success"] for r in records if r["H"] == 256 and r["arm"] == "no_state"]
    delta, lo, hi = _bootstrap_seed_difference(primary_mtsk, primary_single, seed=991)
    no_delta = float(np.mean(primary_mtsk) - np.mean(primary_no))

    reset = [r["reset_success"] for r in records if r["H"] == 256 and r["arm"] == "mtsk"]
    shuffle = [r["shuffle_success"] for r in records if r["H"] == 256 and r["arm"] == "mtsk"]

    return MeasurementRecord(
        provenance=Provenance(
            experiment_id=EXPERIMENT_ID,
            contract_source=f"contracts/{EXPERIMENT_ID}.json",
            contract_sha256=contract_fingerprint(Path(__file__).resolve().parent.parent / "contracts" / f"{EXPERIMENT_ID}.json"),
            git_commit=args.commit,
            script=Path(__file__).name,
            python=platform.python_version(),
            recorded_at=now(),
        ),
        design=Design(
            steps=contract.steps,
            eval_steps=contract.eval_steps,
            seeds=tuple(contract.seeds),
            h_levels=tuple(contract.h_levels),
            k_levels=(),
            arms=tuple(a.name for a in contract.arms),
            smoke=False,
            contract_checked=True,
        ),
        gate=Gate(
            name="TDBU-MTSK-001-preconditions",
            tolerance="exact contract/benchmark/invariant checks",
            passed=True,
            cells=(),
        ),
        endpoints={
            "primary": {
                "H": 256,
                "mtsk_success_by_seed": primary_mtsk,
                "single_timescale_success_by_seed": primary_single,
                "no_state_success_by_seed": primary_no,
                "mtsk_minus_single_mean": delta,
                "seed_bootstrap_95ci": [lo, hi],
                "mtsk_minus_no_state_mean": no_delta,
                "materiality_threshold": MATERIALITY,
            },
            "interventions": {
                "mtsk_reset_success_by_seed": reset,
                "mtsk_shuffle_success_by_seed": shuffle,
                "reset_mean": float(np.mean(reset)),
                "shuffle_mean": float(np.mean(shuffle)),
            },
            "parameter_counts": sorted({r["parameter_count"] for r in records}),
            "state_alphas": STATE_ALPHAS,
            "generator_hash": BENCHMARK_HASH,
            "dependency_lock_hash": dependency_lock_hash(),
            "state_work_rule": "9 arithmetic ops per state-slot update; 3 slots per observation",
            "training_work_rule": "state-feature work plus four matrix-operation passes per optimizer step; reported only as a matched proxy",
        },
        decision_rule=tuple(
            {
                "condition": b.condition,
                "licenses": b.licenses,
                "does_not_license": b.does_not_license,
            }
            for b in contract.decision_rule
        ),
        audit={
            "benchmark": {
                "version": "tdbu-mtsk-v1",
                "generator_hash": BENCHMARK_HASH,
                "paired_histories": True,
                "current_observation_constant": True,
                "train_test_seed_streams_disjoint": True,
                "hidden_pair_type_in_router_features": False,
                "class_balanced_evaluation": True,
            },
            "model_state": {
                "initial_checkpoint_hashes": {
                    f"{H}:{seed}:{arm}": value
                    for (H, seed, arm), value in initial_hashes.items()
                },
                "all_parameter_counts": sorted({r["parameter_count"] for r in records}),
                "trained_arms": list(TRAINED_ARMS),
            },
        },
        per_seed={
            "rows": records,
            "primary_seed_values": {
                "mtsk": primary_mtsk,
                "single_timescale": primary_single,
                "no_state": primary_no,
            },
            "intervention_seed_values": {
                "reset": reset,
                "shuffle": shuffle,
            },
            "bootstrap": {
                "seed": 991,
                "rounds": 20000,
                "mtsk_minus_single": [delta, lo, hi],
            },
        },
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--eval-steps", type=int, default=200)
    parser.add_argument("--h-levels", type=int, nargs="+", default=[8, 64, 256])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--commit", default="UNKNOWN")
    parser.add_argument("--output", default="results/TACOSM-PLM-TDBU-MTSK-001.json")
    args = parser.parse_args()
    record = run(args)
    path = Path(args.output)
    if path == Path("results/TACOSM-PLM-TDBU-MTSK-001.json"):
        path = results_dir_for(__file__) / path.name
    write_record(record, path)
    payload = record.to_dict()
    print(json.dumps(payload["endpoints"]["primary"], indent=2, sort_keys=True))
    print(json.dumps(payload["endpoints"]["interventions"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
