#!/usr/bin/env python3
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
from tac_osm.measurement.results import contract_fingerprint
from tac_osm.memory_isolation import ARMS,BENCHMARK_HASH,GROUND_TRUTH_ALPHAS

EXPERIMENT_ID="TACOSM-PLM-TDBU-MEMORY-ISOLATION-001"
ROOT=Path(__file__).resolve().parent.parent

p=argparse.ArgumentParser(); p.add_argument("result"); args=p.parse_args()
x=json.loads(Path(args.result).read_text(encoding="utf-8"))
assert x["provenance"]["experiment_id"]==EXPERIMENT_ID
assert x["provenance"]["contract_sha256"]==contract_fingerprint(ROOT/"contracts"/f"{EXPERIMENT_ID}.json")
assert x["provenance"]["git_commit"]!="UNKNOWN"
assert x["endpoints"]["generator_hash"]==BENCHMARK_HASH
assert x["endpoints"]["parameter_count"]==86
rows=x["per_seed"]["rows"]
assert len(rows)==3*5*4
assert len({(r["H"],r["seed"],r["arm"]) for r in rows})==len(rows)
assert {r["H"] for r in rows}=={64,256,1024}
assert {r["seed"] for r in rows}=={0,1,2,3,4}
assert {r["arm"] for r in rows}==set(ARMS)
assert all(r["parameter_count"]==86 for r in rows)
assert all(np.isfinite(r["success"]) and np.isfinite(r["action_gap"]) for r in rows)
assert np.allclose(x["audit"]["benchmark"]["ground_truth_alphas"],GROUND_TRUTH_ALPHAS)
assert x["audit"]["benchmark"]["hidden_relation_family"] is False
primary=x["endpoints"]["primary"]
assert len(primary["mtsk_success_by_seed"])==5 and len(primary["single_timescale_success_by_seed"])==5
interventions=x["endpoints"]["interventions"]
assert len(interventions["mtsk_reset_success_by_seed"])==5
assert len(interventions["mtsk_shuffle_success_by_seed"])==5
assert x["endpoints"]["representation_sign_accuracy"] >= 0.90
print("PASS: memory-isolation post-run invariants")
