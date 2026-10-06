#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
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
from tac_osm.mtsk_topdown import MLPPolicy
from tac_osm.memory_isolation import (
    ARMS,
    BENCHMARK_HASH,
    benchmark_hash,
    dependency_hash,
    evaluate,
    featurize,
    make_pairs,
    representability_sign_accuracy,
)

EXPERIMENT_ID = "TACOSM-PLM-TDBU-MEMORY-ISOLATION-001"
TRAIN_PAIRS_DEFAULT = 300
LR_DEFAULT = 0.05
MATERIALITY = 0.10
STEPS_DEFAULT = 500
EVAL_STEPS_DEFAULT = 200
REGISTERED_ARMS = ("no_state", "single_timescale", "two_timescale", "mtsk")


def seed_examples(seed: int, history_length: int, split: str, n_pairs: int):
    base = (70_000 if split == "train" else 80_000) + seed * 1009 + history_length * 17
    return make_pairs(base, history_length, n_pairs)


def pairs_ok(ex):
    by = {}
    for e in ex:
        by.setdefault(e.pair_id, []).append(e)
    assert by and all(len(v) == 2 for v in by.values())
    for v in by.values():
        assert v[0].current_observation == v[1].current_observation == 0.0
        assert v[0].label != v[1].label
        assert np.array_equal(v[1].history, -v[0].history)


def checkpoint_hash(policy: MLPPolicy) -> str:
    h = hashlib.sha256()
    for array in (policy.W1, policy.b1, policy.W2, policy.b2):
        h.update(np.ascontiguousarray(array).tobytes())
    return h.hexdigest()


def boot(a, b, seed=3991, rounds=20_000):
    d = np.asarray(a) - np.asarray(b)
    rng = np.random.default_rng(seed)
    z = rng.choice(d, size=(rounds, len(d)), replace=True).mean(axis=1)
    return float(d.mean()), float(np.quantile(z, 0.025)), float(np.quantile(z, 0.975))


def run(args):
    contract = load_contract(EXPERIMENT_ID)
    contract.require_levels(args.h_levels)
    contract.require_seeds(args.seeds)
    contract.require_steps(args.steps)
    contract.require_eval_steps(args.eval_steps)
    contract.require_arms(REGISTERED_ARMS)

    if args.eval_steps % 2:
        raise RuntimeError("registered eval_steps must be even because evaluation examples are exact pairs")
    assert benchmark_hash() == BENCHMARK_HASH

    representation = representability_sign_accuracy()
    if representation < 0.90:
        raise RuntimeError(f"representability gate failed: {representation}")

    train_pairs = TRAIN_PAIRS_DEFAULT
    eval_pairs = contract.eval_steps // 2
    rows = []
    initials = {}

    for H in contract.h_levels:
        for seed in contract.seeds:
            train = seed_examples(seed, H, "train", train_pairs)
            test = seed_examples(seed, H, "test", eval_pairs)
            pairs_ok(train)
            pairs_ok(test)
            assert len(test) == contract.eval_steps
            y = np.asarray([e.label for e in train], dtype=np.int64)

            for arm in REGISTERED_ARMS:
                policy = MLPPolicy(seed=seed + 7, hidden=12)
                initials[(H, seed, arm)] = checkpoint_hash(policy)
                X = featurize(train, arm)
                assert X.shape == (len(train), 4)
                policy.fit(X, y, contract.steps, LR_DEFAULT)

                evaluation = evaluate(policy, test, arm)
                reset = evaluate(policy, test, "mtsk", "reset") if arm == "mtsk" else None
                shuffle = evaluate(policy, test, "mtsk", "shuffle") if arm == "mtsk" else None

                rows.append(
                    {
                        "H": H,
                        "seed": seed,
                        "arm": arm,
                        "success": evaluation["success"],
                        "action_gap": evaluation["action_gap"],
                        "evaluation_work_per_episode": evaluation["work"],
                        "parameter_count": policy.parameter_count,
                        "checkpoint_hash": checkpoint_hash(policy),
                        "initial_checkpoint_hash": initials[(H, seed, arm)],
                        "reset_success": reset["success"] if reset else None,
                        "shuffle_success": shuffle["success"] if shuffle else None,
                    }
                )

    m = [r["success"] for r in rows if r["H"] == 256 and r["arm"] == "mtsk"]
    s = [r["success"] for r in rows if r["H"] == 256 and r["arm"] == "single_timescale"]
    n = [r["success"] for r in rows if r["H"] == 256 and r["arm"] == "no_state"]
    d, lo, hi = boot(m, s)

    reset = [r["reset_success"] for r in rows if r["H"] == 256 and r["arm"] == "mtsk"]
    shuffle = [r["shuffle_success"] for r in rows if r["H"] == 256 and r["arm"] == "mtsk"]

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
            k_levels=(),
            arms=REGISTERED_ARMS,
            smoke=False,
            contract_checked=True,
        ),
        gate=Gate(
            name="TDBU-MEMORY-ISOLATION-001-preconditions",
            tolerance="exact contract and benchmark invariants",
            passed=True,
            cells=(),
        ),
        endpoints={
            "primary": {
                "H": 256,
                "mtsk_success_by_seed": m,
                "single_timescale_success_by_seed": s,
                "no_state_success_by_seed": n,
                "mtsk_minus_single_mean": d,
                "seed_bootstrap_95ci": [lo, hi],
                "mtsk_minus_no_state_mean": float(np.mean(m) - np.mean(n)),
                "materiality_threshold": MATERIALITY,
            },
            "interventions": {
                "mtsk_reset_success_by_seed": reset,
                "mtsk_shuffle_success_by_seed": shuffle,
            },
            "representation_sign_accuracy": representation,
            "generator_hash": BENCHMARK_HASH,
            "dependency_hash": dependency_hash(),
            "parameter_count": 86,
            "train_pairs_per_H_seed_arm": train_pairs,
            "evaluation_examples_per_H_seed_arm": contract.eval_steps,
        },
        decision_rule=tuple(
            {
                "condition": branch.condition,
                "licenses": branch.licenses,
                "does_not_license": branch.does_not_license,
            }
            for branch in contract.decision_rule
        ),
        audit={
            "benchmark": {
                "generator_hash": BENCHMARK_HASH,
                "ground_truth_alphas": [0.45, 0.985],
                "paired_histories": True,
                "current_observation_constant": True,
                "train_test_seed_streams_disjoint": True,
                "hidden_relation_family": False,
                "new_test_stream": True,
                "test_examples_are_exactly_registered_eval_steps": True,
            },
            "model_state": {
                "parameter_count": 86,
                "online_sequential_update": True,
                "state_is_reset_once_per_episode": True,
                "policy_receives_only_current_observation_and_terminal_state": True,
                "initial_checkpoint_hashes": {
                    f"{H}:{seed}:{arm}": value
                    for (H, seed, arm), value in initials.items()
                },
            },
            "leakage": {
                "featurizer_fields": ["current_observation", "history_via_state_update"],
                "excluded_fields": ["label", "pair_id", "pair_identity", "target_action"],
            },
        },
        per_seed={
            "rows": rows,
            "bootstrap": {
                "seed": 3991,
                "rounds": 20_000,
                "mtsk_minus_single": [d, lo, hi],
            },
        },
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=STEPS_DEFAULT)
    parser.add_argument("--eval-steps", type=int, default=EVAL_STEPS_DEFAULT)
    parser.add_argument("--h-levels", type=int, nargs="+", default=[64, 256, 1024])
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
            arms=list(REGISTERED_ARMS),
        )
        return 0

    record = run(args)
    path = Path(args.output)
    if path == Path(f"results/{EXPERIMENT_ID}.json"):
        path = results_dir_for(__file__) / path.name
    write_record(record, path)
    print(json.dumps(record.to_dict()["endpoints"]["primary"], indent=2, sort_keys=True))
    print(json.dumps(record.to_dict()["endpoints"]["interventions"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
