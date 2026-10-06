#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from tac_osm.contract import contract_fingerprint, load_contract
from tac_osm.memory_isolation import ARMS, BENCHMARK_HASH, GROUND_TRUTH_ALPHAS

EXPERIMENT_ID = "TACOSM-PLM-TDBU-MEMORY-ISOLATION-001"
ROOT = Path(__file__).resolve().parent.parent

parser = argparse.ArgumentParser()
parser.add_argument("result")
args = parser.parse_args()

x = json.loads(Path(args.result).read_text(encoding="utf-8"))
c = load_contract(EXPERIMENT_ID)

assert x["provenance"]["experiment_id"] == EXPERIMENT_ID
assert x["provenance"]["contract_sha256"] == contract_fingerprint(
    ROOT / "contracts" / f"{EXPERIMENT_ID}.json"
)
assert x["provenance"]["git_commit"] != "UNKNOWN"
assert x["endpoints"]["generator_hash"] == BENCHMARK_HASH
assert x["endpoints"]["parameter_count"] == 86

rows = x["per_seed"]["rows"]
expected_rows = len(c.h_levels) * len(c.seeds) * len(c.arms)
assert len(rows) == expected_rows
assert len({(r["H"], r["seed"], r["arm"]) for r in rows}) == expected_rows
assert {r["H"] for r in rows} == set(c.h_levels)
assert {r["seed"] for r in rows} == set(c.seeds)
assert {r["arm"] for r in rows} == set(ARMS)
assert all(r["parameter_count"] == 86 for r in rows)
assert all(np.isfinite(r["success"]) and np.isfinite(r["action_gap"]) for r in rows)
assert np.allclose(x["audit"]["benchmark"]["ground_truth_alphas"], GROUND_TRUTH_ALPHAS)
assert x["audit"]["benchmark"]["hidden_relation_family"] is False
assert x["audit"]["benchmark"]["test_examples_are_exactly_registered_eval_steps"] is True
assert x["audit"]["model_state"]["online_sequential_update"] is True
assert x["audit"]["model_state"]["state_is_reset_once_per_episode"] is True
assert x["audit"]["leakage"]["excluded_fields"] == [
    "label",
    "pair_id",
    "pair_identity",
    "target_action",
]

primary = x["endpoints"]["primary"]
assert len(primary["mtsk_success_by_seed"]) == len(c.seeds)
assert len(primary["single_timescale_success_by_seed"]) == len(c.seeds)

interventions = x["endpoints"]["interventions"]
assert len(interventions["mtsk_reset_success_by_seed"]) == len(c.seeds)
assert len(interventions["mtsk_shuffle_success_by_seed"]) == len(c.seeds)

assert x["endpoints"]["evaluation_examples_per_H_seed_arm"] == c.eval_steps
assert x["endpoints"]["train_pairs_per_H_seed_arm"] > 0
assert x["endpoints"]["representation_sign_accuracy"] >= 0.90

print("PASS: memory-isolation post-run invariants")
