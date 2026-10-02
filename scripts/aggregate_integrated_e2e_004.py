#!/usr/bin/env python3
from __future__ import annotations
import json,random,statistics,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/"src"))
from tac_osm.contract import load_contract
ARMS=("typed_learned","typed_oracle"); SEEDS=(0,1,2,3,4)
def bootstrap(values,samples=5000,seed=20261002):
    rng=random.Random(seed); n=len(values); draws=sorted(statistics.fmean(values[rng.randrange(n)] for _ in range(n)) for _ in range(samples))
    return [draws[int(.025*samples)],draws[int(.975*samples)-1]]
def main():
    c=load_contract("TACOSM-PLM-INTEGRATED-E2E-004"); c.require_levels((3,)); c.require_seeds(SEEDS); c.require_steps(300); c.require_eval_steps(400)
    if c.check_consistency(): raise RuntimeError("contract consistency failure")
    records={}
    for arm in ARMS:
        d=json.loads((ROOT/"artifacts"/f"TACOSM-PLM-INTEGRATED-E2E-004-{arm}.json").read_text())
        if d["experiment_id"]!="TACOSM-PLM-INTEGRATED-E2E-004" or d["arm"]!=arm: raise RuntimeError(f"artifact identity mismatch: {arm}")
        if tuple(d["protocol"]["seeds"])!=SEEDS or d["protocol"]["steps"]!=300 or d["protocol"]["eval_episodes"]!=400: raise RuntimeError(f"protocol mismatch: {arm}")
        if sorted(int(r["seed"]) for r in d["seed_results"])!=list(SEEDS): raise RuntimeError(f"seed coverage mismatch: {arm}")
        records[arm]=d
    commits={records[a]["provenance"]["git_commit"] for a in ARMS}
    if len(commits)!=1 or "unknown" in commits: raise RuntimeError(f"arm provenance mismatch: {sorted(commits)}")
    l=[float(r["typed_learned_q2_accuracy"]) for r in records["typed_learned"]["seed_results"]]; o=[float(r["typed_oracle_q2_accuracy"]) for r in records["typed_oracle"]["seed_results"]]
    n=[float(r["typed_learned_no_memory_q2_accuracy"]) for r in records["typed_learned"]["seed_results"]]; b=[float(r["typed_learned_bit_accuracy"]) for r in records["typed_learned"]["seed_results"]]
    lm=statistics.fmean(l); om=statistics.fmean(o); nm=statistics.fmean(n)
    out={"experiment_id":"TACOSM-PLM-INTEGRATED-E2E-004","status":"measured","provenance":{"arm_artifacts":{a:records[a]["provenance"] for a in ARMS}},"protocol":{"seeds":SEEDS,"steps":300,"eval_episodes":400,"heldout_compositions":records["typed_oracle"]["protocol"]["heldout_compositions"]},"arm_seed_results":{a:records[a]["seed_results"] for a in ARMS},"summary":{"typed_learned_q2_mean":lm,"typed_learned_q2_seed_bootstrap_ci95":bootstrap(l),"typed_learned_min_seed_q2":min(l),"typed_oracle_q2_mean":om,"typed_oracle_q2_seed_bootstrap_ci95":bootstrap(o),"typed_oracle_min_seed_q2":min(o),"typed_learned_bit_accuracy_mean":statistics.fmean(b),"typed_learned_no_memory_q2_mean":nm,"typed_learned_memory_drop":lm-nm,"frozen_e2e003_mean":0.52,"typed_learned_vs_e2e003_delta":lm-0.52,"oracle_ceiling_pass":om==1.0,"primary_pass":om==1.0 and lm>=0.80 and min(l)>=0.40,"material_improvement_vs_e2e003":lm-0.52>=0.05},"decision_rule":{"oracle":"typed_oracle q2 mean == 1.00","primary":"typed_oracle q2 mean == 1.00 AND typed_learned q2 mean >= 0.80 AND every typed_learned seed >= 0.40","materiality":"typed_learned q2 mean - 0.5200 >= 0.05"}}
    (ROOT / "artifacts/TACOSM-PLM-INTEGRATED-E2E-004-COMBINED.json").write_text(json.dumps(out, indent=2, sort_keys=True) + "\n"); print(json.dumps(out, indent=2, sort_keys=True))
"); print(json.dumps(out,indent=2,sort_keys=True))
if __name__=="__main__": main()
