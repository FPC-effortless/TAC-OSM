#!/usr/bin/env python3
"""Posthoc DEV064B VALIDATION-only audit. Never imports a TEST namespace.

Pass an extracted artifact directory containing pilot.json and checkpoints/*.pt.
No fitting or model selection. This is an audit of already-seen validation data.
"""
import argparse
import collections
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from run_enabling_casm_dev064b import (
    ARMS, VAL_N, VAL_SEED, TEST_SEED, generate, features, model, oracle,
    transition, world,
)


def validation_worlds():
    rng = np.random.default_rng(VAL_SEED)
    worlds, states, budgets, xs, qs = [], [], [], [], []
    for _ in range(VAL_N):
        w = world(rng)
        state = tuple(int(z) for z in rng.integers(0, 2, 3))
        state += (int(rng.random() < 0.15),)
        if not any(state[:3]):
            # Preserves the EXISTING generator, including its per-j draw issue.
            state = tuple(int(j == int(rng.integers(0, 3))) for j in range(3)) + (state[3],)
        budget = int(rng.integers(2, 5))
        worlds.append(w)
        states.append(state)
        budgets.append(budget)
        xs.append(features(w, state, budget))
        qs.append(oracle(w, state, budget))
    X, Q = np.asarray(xs, dtype=np.float32), np.asarray(qs, dtype=np.float32)
    old_X, old_Q, source = generate(VAL_SEED, VAL_N, purpose="validation")
    assert np.array_equal(X, old_X.numpy())
    assert np.array_equal(Q, old_Q.numpy())
    assert source["seed"] == VAL_SEED
    return worlds, states, budgets, X, Q, source


def simulate(w, original_state, budget, policy, seed):
    """One repeated-seed physical start; common RNG isn't a causal oracle."""
    rng = np.random.default_rng(seed)
    state = original_state
    total, repairs, healthy_repairs, new_fault_bits = 0., 0, 0, 0
    stopped = False
    for step in range(budget):
        action = int(policy(w, state, budget - step))
        if action == 0:
            stopped = True
            break
        total += w.setup_cost if action == 1 else w.costs[action - 2]
        if action >= 2:
            repairs += 1
            healthy_repairs += int(not state[action - 2])
        branch = transition(w, state, action)
        threshold = rng.random()
        cumul = 0.
        next_state = branch[-1][1]
        for probability, value in branch:
            cumul += probability
            if threshold <= cumul + 1e-12:
                next_state = value
                break
        new_fault_bits += max(0, sum(next_state[:3]) - sum(state[:3]))
        state = next_state
    faults = sum(state[:3])
    total += w.penalty * faults
    return [total, int(stopped and not faults), faults,
            repairs, healthy_repairs, new_fault_bits, int(stopped)]


def simulate_summary(w, states, budgets, policy):
    records = np.asarray([
        simulate(wi, si, bi, policy, 90000 + i)
        for i, (wi, si, bi) in enumerate(zip(w, states, budgets))
    ])
    return {
        "mean_total_cost": float(records[:, 0].mean()),
        "completed_fraction": float(records[:, 1].mean()),
        "final_fault_bits": int(records[:, 2].sum()),
        "repairs": int(records[:, 3].sum()),
        "healthy_repairs": int(records[:, 4].sum()),
        "new_fault_bits": int(records[:, 5].sum()),
        "explicit_stop_count": int(records[:, 6].sum()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    root = args.artifact_dir
    manifest = json.loads((root / "pilot.json").read_text())
    assert manifest["study"] == "DEV064B-PREREQUISITE-CHAINED-CASM"
    assert manifest["test_opened"] is False
    assert manifest["status"] == "VALIDATION_PILOTS_ONLY"
    assert TEST_SEED == 641643
    assert len(manifest["fits"]) == 10

    worlds, states, budgets, xx, q, source = validation_worlds()
    assert source["x_sha256"] == manifest["validation"]["x_sha256"]
    assert source["y_sha256"] == manifest["validation"]["y_sha256"]
    labels = np.argmin(q, axis=1)
    result = {
        "study": "DEV064B-POSTHOC-VALIDATION-AUDIT",
        "status": "VALIDATION_REJECTED_NOT_CONFIRMATORY",
        "test_opened": False,
        "heldout_test_evaluations": 0,
        "pilot_artifact_count": 10,
        "source_validation_sha256": {
            "x": source["x_sha256"], "q": source["y_sha256"],
        },
        "actions": {str(k): int(v) for k, v in sorted(collections.Counter(labels).items())},
        "constant_prepare_accuracy": float(np.mean(labels == 1)),
        "initially_healthy_examples": int(sum(not any(s[:3]) for s in states)),
        "pilot_models": [],
    }
    for row in manifest["fits"]:
        arm, seed = row["arm"], row["seed"]
        assert arm in ARMS and seed in (641650, 641651)
        chk_path = root / "checkpoints" / f"{arm}-{seed}.pt"
        raw = chk_path.read_bytes()
        assert hashlib.sha256(raw).hexdigest() == row["checkpoint_sha256"]
        checkpoint = torch.load(chk_path, map_location="cpu", weights_only=True)
        assert checkpoint["seed"] == seed and checkpoint["arm"] == arm
        candidate = model(arm)
        candidate.load_state_dict(checkpoint["model"], strict=True)
        candidate.eval()
        with torch.inference_mode():
            predictions = candidate(torch.from_numpy(xx)).numpy()
        chosen = predictions.argmin(axis=-1)
        measured_accuracy = float(np.mean(chosen == labels))
        measured_regret = float(np.mean(q[np.arange(len(labels)), chosen] - q.min(axis=1)))
        assert abs(measured_accuracy - row["val_action_accuracy"]) < 1e-8
        assert abs(measured_regret - row["val_expected_regret"]) < 1e-6

        def candidate_policy(w, state, horizon):
            with torch.inference_mode():
                observation = torch.from_numpy(features(w, state, horizon)[None])
                return int(candidate(observation).argmin(-1).item())

        physical = simulate_summary(worlds, states, budgets, candidate_policy)
        result["pilot_models"].append({
            "arm": arm, "seed": seed,
            "initial_action_accuracy": measured_accuracy,
            "initial_oracle_regret": measured_regret,
            "mismatch_indices": np.flatnonzero(chosen != labels).tolist(),
            "checkpoint_sha256": row["checkpoint_sha256"],
            "validation_trajectories": physical,
        })
        print(f"{arm} seed={seed} complete={physical['completed_fraction']:.5f}"
              f" cost={physical['mean_total_cost']:.5f}", flush=True)

    def oracle_policy(w, state, horizon):
        return int(np.argmin(oracle(w, state, horizon)))

    result["known_law_cost_oracle"] = simulate_summary(worlds, states, budgets, oracle_policy)
    result["rejection_reasons"] = [
        "Validation learned initial decisions saturated: 9/10 arms 1017/1024",
        "No reliable closed-loop completion (all pilot arms and cost oracle <0.95)",
        "Same-world/seed pilot comparisons cannot establish CASM superiority",
        "Existing dataset fallback can generate all-healthy starting states",
        "Heldout TEST namespace was never opened",
    ]
    if max(r["validation_trajectories"]["completed_fraction"]
           for r in result["pilot_models"]) >= 0.95:
        raise AssertionError("Unexpected validation recovery; review scientific conclusion")
    if result["known_law_cost_oracle"]["completed_fraction"] >= 0.95:
        raise AssertionError("Unexpected reference recovery; review scientific conclusion")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n")
    print(f"audit={args.output} status={result['status']}; test_opened=False", flush=True)


if __name__ == "__main__":
    main()
