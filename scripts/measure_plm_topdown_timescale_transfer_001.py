#!/usr/bin/env python3
"""Preregistered temporal-scale transfer measurement."""
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
    report_smoke,
    results_dir_for,
    write_record,
)
from tac_osm.timescale_transfer import (
    ARMS,
    BENCHMARK_HASH,
    EXTRAPOLATION_TEST_INDICES,
    INTERPOLATION_TEST_INDICES,
    POLICY_HIDDEN,
    ARM_ALPHAS,
    TEST_FILTER_ALPHAS,
    TRAIN_FILTER_ALPHAS,
    benchmark_hash,
    evaluate,
    make_pairs,
    make_policy,
    representability_witness,
    parameter_count,
    _features_for_state,
    PAIR_TYPES_TEST,
    PAIR_TYPES_TRAIN,
)

EXPERIMENT_ID = "TACOSM-PLM-TDBU-TIMESCALE-TRANSFER-001"
TRAIN_PAIRS = 300
TEST_PAIRS = 120
LEARNING_RATE = 0.05
MATERIALITY = 0.10
STEPS_DEFAULT = 500
EVAL_STEPS_DEFAULT = 200
TRAINED_ARMS = ARMS


def _seed_examples(seed: int, H: int, split: str) -> list:
    base = (30_000 if split == "train" else 40_000) + seed * 1009 + H * 17
    if split == "train":
        return make_pairs(
            base,
            H,
            TRAIN_PAIRS,
            alphas=TRAIN_FILTER_ALPHAS,
            pair_types=PAIR_TYPES_TRAIN,
        )
    return make_pairs(
        base,
        H,
        TEST_PAIRS,
        alphas=TEST_FILTER_ALPHAS,
        pair_types=PAIR_TYPES_TEST,
    )


def _assert_pair_contract(examples: list) -> None:
    by_pair: dict[int, list] = {}
    for x in examples:
        by_pair.setdefault(x.pair_id, []).append(x)
    assert by_pair and all(len(v) == 2 for v in by_pair.values())
    for pair in by_pair.values():
        assert pair[0].current_observation == pair[1].current_observation == 0.0
        assert pair[0].label != pair[1].label
        assert np.array_equal(pair[1].history, -pair[0].history)


def _assert_filter_separation() -> None:
    assert set(np.round(TRAIN_FILTER_ALPHAS, 12)).isdisjoint(
        set(np.round(TEST_FILTER_ALPHAS, 12))
    )
    assert np.all(np.diff(TRAIN_FILTER_ALPHAS) > 0)
    assert np.all(np.diff(TEST_FILTER_ALPHAS) > 0)


def _select_test_subset(
    examples: list,
    subset_indices: set[int],
) -> list:
    out = []
    for example in examples:
        pair_type = PAIR_TYPES_TEST[example.pair_id % len(PAIR_TYPES_TEST)]
        if pair_type[0] in subset_indices and pair_type[1] in subset_indices:
            out.append(example)
    if not out:
        raise RuntimeError("registered test subset produced no examples")
    return out


def _select_extrapolation_subset(examples: list) -> list:
    out = []
    for example in examples:
        pair_type = PAIR_TYPES_TEST[example.pair_id % len(PAIR_TYPES_TEST)]
        if pair_type[0] in EXTRAPOLATION_TEST_INDICES or pair_type[1] in EXTRAPOLATION_TEST_INDICES:
            out.append(example)
    if not out:
        raise RuntimeError("registered extrapolation subset produced no examples")
    return out


def _bootstrap_seed_difference(
    a: list[float],
    b: list[float],
    seed: int,
    rounds: int = 20_000,
) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    diffs = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    draws = rng.choice(diffs, size=(rounds, len(diffs)), replace=True).mean(axis=1)
    return float(np.mean(diffs)), float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def run(args: argparse.Namespace) -> MeasurementRecord:
    contract = load_contract(EXPERIMENT_ID)
    contract.require_levels(args.h_levels)
    contract.require_seeds(args.seeds)
    contract.require_steps(args.steps)
    contract.require_eval_steps(args.eval_steps)
    contract.require_arms([a.name for a in contract.arms])
    contract.require_k_levels(contract.k_levels)

    _assert_filter_separation()
    if benchmark_hash() != BENCHMARK_HASH:
        raise RuntimeError("benchmark version hash mismatch")

    representability = representability_witness(history_length=256)
    if min(representability.values()) < 0.99:
        raise RuntimeError(
            "distributed-eight representability gate failed: "
            + json.dumps(representability, sort_keys=True)
        )

    records = []
    initial_hashes = {}
    for H in contract.h_levels:
        for seed in contract.seeds:
            train = _seed_examples(seed, H, "train")
            test = _seed_examples(seed, H, "test")
            _assert_pair_contract(train)
            _assert_pair_contract(test)
            train_labels = np.asarray([x.label for x in train], dtype=np.int64)
            test_interp = _select_test_subset(test, set(INTERPOLATION_TEST_INDICES))
            test_extra = _select_extrapolation_subset(test)
            for arm in TRAINED_ARMS:
                policy = make_policy(seed + 7, arm)
                initial_hashes[(H, seed, arm)] = (
                    hashlib_sha(policy)
                )
                X = _features_for_state(train, arm)
                updates = policy.fit(X, train_labels, contract.steps, LEARNING_RATE)
                assert updates == contract.steps
                eval_result = evaluate(policy, test, arm)
                interp_result = evaluate(policy, test_interp, arm)
                extra_result = evaluate(policy, test_extra, arm)
                train_state_work = len(train) * len(train[0].history) * 3 * len(ARM_ALPHAS[arm])
                optimizer_passes = contract.steps * len(train) * 2 * (
                    X.shape[1] * POLICY_HIDDEN[arm] + 2 * POLICY_HIDDEN[arm]
                )
                train_work = float(train_state_work + optimizer_passes)
                reset_result = (
                    evaluate(policy, test, "distributed_eight", intervention="reset")
                    if arm == "distributed_eight"
                    else None
                )
                shuffle_result = (
                    evaluate(policy, test, "distributed_eight", intervention="shuffle")
                    if arm == "distributed_eight"
                    else None
                )
                records.append(
                    {
                        "H": H,
                        "seed": seed,
                        "arm": arm,
                        "success": eval_result.success,
                        "interpolation_success": interp_result.success,
                        "extrapolation_success": extra_result.success,
                        "action_gap": eval_result.action_gap,
                        "state_footprint": eval_result.state_footprint,
                        "evaluation_work_per_episode": eval_result.evaluation_work_per_episode,
                        "train_compute_work_proxy": train_work,
                        "parameter_count": policy.parameter_count,
                        "checkpoint_hash": hashlib_sha(policy),
                        "reset_success": reset_result.success if reset_result else None,
                        "shuffle_success": shuffle_result.success if shuffle_result else None,
                        "reset_action_gap": reset_result.action_gap if reset_result else None,
                        "shuffle_action_gap": shuffle_result.action_gap if shuffle_result else None,
                    }
                )

    primary_eight = [
        r["success"] for r in records if r["H"] == 256 and r["arm"] == "distributed_eight"
    ]
    primary_three = [
        r["success"] for r in records if r["H"] == 256 and r["arm"] == "three_timescale"
    ]
    primary_no = [
        r["success"] for r in records if r["H"] == 256 and r["arm"] == "no_state"
    ]
    delta, lo, hi = _bootstrap_seed_difference(primary_eight, primary_three, seed=1991)
    no_delta = float(np.mean(primary_eight) - np.mean(primary_no))
    reset = [
        r["reset_success"] for r in records
        if r["H"] == 256 and r["arm"] == "distributed_eight"
    ]
    shuffle = [
        r["shuffle_success"] for r in records
        if r["H"] == 256 and r["arm"] == "distributed_eight"
    ]

    return MeasurementRecord(
        provenance=Provenance(
            experiment_id=EXPERIMENT_ID,
            contract_source=f"contracts/{EXPERIMENT_ID}.json",
            contract_sha256=contract_fingerprint(
                Path(__file__).resolve().parent.parent
                / "contracts"
                / f"{EXPERIMENT_ID}.json"
            ),
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
            k_levels=tuple(contract.k_levels),
            arms=tuple(a.name for a in contract.arms),
            smoke=False,
            contract_checked=True,
        ),
        gate=Gate(
            name="TDBU-TIMESCALE-TRANSFER-001-preconditions",
            tolerance="exact contract, train/test filter separation, paired-history invariants, representability",
            passed=True,
            cells=(),
        ),
        endpoints={
            "primary": {
                "H": 256,
                "distributed_eight_success_by_seed": primary_eight,
                "three_timescale_success_by_seed": primary_three,
                "no_state_success_by_seed": primary_no,
                "distributed_eight_minus_three_mean": delta,
                "seed_bootstrap_95ci": [lo, hi],
                "distributed_eight_minus_no_state_mean": no_delta,
                "materiality_threshold": MATERIALITY,
            },
            "transfer": {
                "interpolation_alphas": TEST_FILTER_ALPHAS[list(INTERPOLATION_TEST_INDICES)].tolist(),
                "extrapolation_alphas": TEST_FILTER_ALPHAS[list(EXTRAPOLATION_TEST_INDICES)].tolist(),
                "H256_mean_by_arm_interpolation": {
                    arm: float(np.mean([r["interpolation_success"] for r in records if r["H"] == 256 and r["arm"] == arm]))
                    for arm in ARMS
                },
                "H256_mean_by_arm_extrapolation": {
                    arm: float(np.mean([r["extrapolation_success"] for r in records if r["H"] == 256 and r["arm"] == arm]))
                    for arm in ARMS
                },
            },
            "interventions": {
                "distributed_eight_reset_success_by_seed": reset,
                "distributed_eight_shuffle_success_by_seed": shuffle,
                "reset_mean": float(np.mean(reset)),
                "shuffle_mean": float(np.mean(shuffle)),
            },
            "parameter_counts": {arm: parameter_count(arm) for arm in ARMS},
            "state_footprints": {arm: len(ARM_ALPHAS[arm]) for arm in ARMS},
            "state_work_rule": "3 arithmetic operations per state-slot update; router arithmetic and verification are counted separately",
            "generator_hash": BENCHMARK_HASH,
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
                "version": GENERATOR_VERSION,
                "generator_hash": BENCHMARK_HASH,
                "train_filter_alphas": TRAIN_FILTER_ALPHAS.tolist(),
                "test_filter_alphas": TEST_FILTER_ALPHAS.tolist(),
                "filter_sets_disjoint": True,
                "test_extrapolation_indices": list(EXTRAPOLATION_TEST_INDICES),
                "paired_histories": True,
                "current_observation_constant": True,
                "train_test_seed_streams_disjoint": True,
                "hidden_pair_type_in_policy_input": True,
                "class_balanced_evaluation": True,
            },
            "representability": {
                "distributed_eight_impulse_kernel_r2": representability,
                "minimum_required": 0.99,
            },
            "model_state": {
                "initial_checkpoint_hashes": {
                    f"{H}:{seed}:{arm}": value
                    for (H, seed, arm), value in initial_hashes.items()
                },
                "all_parameter_counts": {arm: parameter_count(arm) for arm in ARMS},
                "trained_arms": list(TRAINED_ARMS),
            },
            "capacity": {
                "policy_hidden": dict(POLICY_HIDDEN),
                "primary_parameter_match": parameter_count("distributed_eight") == parameter_count("three_timescale"),
            },
        },
        per_seed={
            "rows": records,
            "primary_seed_values": {
                "distributed_eight": primary_eight,
                "three_timescale": primary_three,
                "no_state": primary_no,
            },
            "intervention_seed_values": {"reset": reset, "shuffle": shuffle},
            "bootstrap": {
                "seed": 1991,
                "rounds": 20000,
                "distributed_eight_minus_three": [delta, lo, hi],
            },
        },
    )


def hashlib_sha(policy: object) -> str:
    import hashlib
    h = hashlib.sha256()
    for name in ("W1", "b1", "W2", "b2"):
        h.update(np.ascontiguousarray(getattr(policy, name)).tobytes())
    return h.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=STEPS_DEFAULT)
    parser.add_argument("--eval-steps", type=int, default=EVAL_STEPS_DEFAULT)
    parser.add_argument("--h-levels", type=int, nargs="+", default=[64, 256, 512])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--commit", default="UNKNOWN")
    parser.add_argument("--output", default=f"results/{EXPERIMENT_ID}.json")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    contract = load_contract(EXPERIMENT_ID)
    if args.smoke:
        report_smoke(
            contract,
            EXPERIMENT_ID,
            steps=args.steps,
            eval_steps=args.eval_steps,
            h_levels=args.h_levels,
            seeds=args.seeds,
            arms=[a.name for a in contract.arms],
        )
        return 0
    record = run(args)
    path = Path(args.output)
    if path == Path(f"results/{EXPERIMENT_ID}.json"):
        path = results_dir_for(__file__) / path.name
    write_record(record, path)
    payload = record.to_dict()
    print(json.dumps(payload["endpoints"]["primary"], indent=2, sort_keys=True))
    print(json.dumps(payload["endpoints"]["interventions"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
