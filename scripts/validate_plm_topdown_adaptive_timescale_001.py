#!/usr/bin/env python3
import argparse,json
from pathlib import Path
import numpy as np
from tac_osm.measurement.results import contract_fingerprint
from tac_osm.adaptive_timescale import BENCHMARK_HASH,TRAIN_FILTER_ALPHAS,TEST_FILTER_ALPHAS,ARMS
EXPERIMENT_ID="TACOSM-PLM-TDBU-ADAPTIVE-TIMESCALE-001"; ROOT=Path(__file__).resolve().parent.parent
p=argparse.ArgumentParser(); p.add_argument("result"); a=p.parse_args()
x=json.loads(Path(a.result).read_text())
assert x["provenance"]["experiment_id"]==EXPERIMENT_ID
assert x["provenance"]["contract_sha256"]==contract_fingerprint(ROOT/"contracts"/f"{EXPERIMENT_ID}.json")
assert x["provenance"]["git_commit"]!="UNKNOWN"
assert x["endpoints"]["generator_hash"]==BENCHMARK_HASH
assert len(x["per_seed"]["rows"])==3*5*4
assert len({(r["H"],r["seed"],r["arm"]) for r in x["per_seed"]["rows"]})==60
assert all(r["parameter_count"]==86 for r in x["per_seed"]["rows"])
assert set(np.round(TRAIN_FILTER_ALPHAS,12)).isdisjoint(set(np.round(TEST_FILTER_ALPHAS,12)))
for arm in ARMS: assert sum(1 for r in x["per_seed"]["rows"] if r["arm"]==arm)==15
assert len(x["endpoints"]["learned_alphas_by_seed"])==5
assert len(x["endpoints"]["interventions"]["adaptive_reset_success_by_seed"])==5
assert len(x["endpoints"]["interventions"]["adaptive_shuffle_success_by_seed"])==5
print("PASS: adaptive-timescale post-run invariants")
