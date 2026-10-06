#!/usr/bin/env python3
from __future__ import annotations
import argparse
import json
from pathlib import Path

from tac_osm.contract import load_contract

EXPERIMENT_ID = "TACOSM-PLM-INTEGRATED-E2E-002"

parser = argparse.ArgumentParser()
parser.add_argument("result")
args = parser.parse_args()
x = json.loads(Path(args.result).read_text(encoding="utf-8"))
c = load_contract(EXPERIMENT_ID)

assert x["experiment_id"] == EXPERIMENT_ID
assert x["status"] == "measured"
assert x["protocol"]["seeds"] == list(c.seeds)
assert x["protocol"]["steps"] == c.steps
assert x["protocol"]["evaluation_episodes_per_seed"] == c.eval_steps
assert x["protocol"]["registered_arms"] == [a.name for a in c.arms]
assert x["protocol"]["benchmark_generator_version"] == x["leakage_audit"]["benchmark_generator_version"]
assert x["protocol"]["benchmark_hash"] == x["leakage_audit"]["benchmark_hash"]
rows = x["seed_results"]
assert len(rows) == len(c.seeds)
assert {r["seed"] for r in rows} == set(c.seeds)
assert len({r["evaluation_episode_fingerprint"] for r in rows}) == len(rows)
for r in rows:
    assert 0.0 <= r["normal_q2"] <= 1.0
    assert 0.0 <= r["no_memory_q2"] <= 1.0
    assert 0.0 <= r["shuffle_image_q2"] <= 1.0
    assert r["normal_q2"] >= 0.0
assert x["leakage_audit"]["q1_entity_rule"] == "q1 target entity is entities[0]"
assert x["leakage_audit"]["q2_entity_rule"] == "q2 target entity is entities[1]"
assert x["leakage_audit"]["q1_q2_entities_distinct"] is True
assert x["leakage_audit"]["heldout_compositions_excluded_from_training"] is True
assert x["leakage_audit"]["train_eval_rng_streams_disjoint"] is True
assert x["leakage_audit"]["evaluation_generated_after_training"] is True
assert x["leakage_audit"]["controls_reuse_exact_same_episode_objects"] is True
assert set(x["leakage_audit"]["forbidden_pre_action_fields"]) == {"answer","environment_outcome","verifier_target"}
assert x["summary"]["primary_pass"] == bool(x["summary"]["normal_q2"] >= c.primary_endpoint.threshold)
assert x["summary"]["all_seed_min_q2"] == min(r["normal_q2"] for r in rows)
assert len(x["summary"]["primary_q2_seed_bootstrap_ci95"]) == 2
print("PASS: E2E-002 independent result and leakage validator")
