#!/usr/bin/env python3
"""G-CASM-016D: exact objective comparison on the frozen 015 trace channel."""
from __future__ import annotations
import argparse, collections, hashlib, itertools, json, os, random, resource, statistics, sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src")); sys.path.insert(0,str(ROOT/"scripts"))
import run_graph_casm_structured_action_probe_015 as g15
from tac_osm.trace_capability import budget_capped_utility

EXPERIMENT_ID="TACOSM-GRAPH-CASM-BUDGET-UTILITY-SELECTION-016D"
SEEDS=(0,1,2,3,4); M_LEVELS=(32,64,128,256,512); TASKS_PER_M=32; PRIMARY_M=512; BUDGET=8
INPUT_ROWS=g15.INPUT_ROWS; GENERATOR_COMMIT=g15.GENERATOR_COMMIT

def task(seed,m,i):
    r=random.Random(seed*3000001+m*10007+i*1009+17)
    target=r.randrange(m)
    h=hashlib.sha256(repr((seed,m,i,target)).encode()).hexdigest()[:16]
    return target,f"g16d:{seed}:{m}:{i}:{h}"

def action_evaluate(cache,indices,channel):
    scores=g15.action_scores(cache,indices,"activation_trace")
    base=BUDGET/len(indices)
    if channel=="information_per_work":
        return g15.choose_best_action(scores), None
    if channel!="budget_utility_per_work":
        raise ValueError(channel)
    scored=[]
    for s in scores:
        row=int(s.action.parameters[0])
        evidence=tuple(cache.trace[(idx,row)][0] for idx in indices)
        ub=budget_capped_utility(evidence,BUDGET)
        gain=ub-base
        scored.append((gain/max(1e-12,s.expected_cost),ub,s))
    chosen=max(scored,key=lambda x:(x[0],x[1],x[2].information_gain_bits,-x[2].action.parameters[0]))
    return chosen[2], chosen[1]

def verify(target,candidates,shortlist):
    for idx in shortlist:
        cand=candidates[idx]
        for bits in INPUT_ROWS:
            got,_=g15.g010.execute_exact(cand,bits)
            if got!=int(target.truth_table[bits]):
                break
        else:
            return 1.0
    return 0.0

def trial(cache,candidates,target_index,channel):
    idxs=tuple(range(len(candidates)))
    chosen,ub_hint=action_evaluate(cache,idxs,channel)
    row=int(chosen.action.parameters[0])
    evidence=tuple(cache.trace[(i,row)][0] for i in idxs)
    target_evidence=cache.trace[(target_index,row)][0]
    bucket=tuple(i for i,e in enumerate(evidence) if e==target_evidence)
    shortlist=bucket[:BUDGET]
    success=verify(candidates[target_index],candidates,shortlist)
    ub=budget_capped_utility(evidence,BUDGET)
    return {
      "selected_row":row,
      "information_gain_bits":float(chosen.information_gain_bits),
      "information_per_work":float(chosen.information_per_work),
      "probe_environment_work_units":float(chosen.expected_cost),
      "budget_capped_utility":float(ub),
      "target_bucket_size":len(bucket),
      "verified_success":float(success),
      "shortlist_size":len(shortlist),
      "accounted_total_work_units":float(chosen.expected_cost),
      "target_in_shortlist":float(target_index in shortlist)
    }

def bootstrap(deltas,seed=16064,rounds=4000):
    rng=random.Random(seed); n=len(deltas)
    ds=sorted(statistics.fmean(deltas[rng.randrange(n)] for _ in range(n)) for _ in range(rounds))
    return [float(ds[int(.025*(rounds-1))]),float(ds[int(.975*(rounds-1))])]

def main(smoke=False):
    from tac_osm.contract import load_contract
    c=load_contract(EXPERIMENT_ID)
    if smoke: seeds=(0,); levels=(32,128); tasks=4
    else: seeds=SEEDS; levels=M_LEVELS; tasks=TASKS_PER_M
    raw={str(s):{str(m):[] for m in levels} for s in seeds}
    checks={}
    for s in seeds:
        trn=g15.g010.generate(s+100,g15.TRAIN_PROGRAMS)
        ts={g15.g010.structure_key(e) for e in trn}; tt={g15.g010.truth_signature(e) for e in trn}
        lib=g15.g010.generate(s+5000,g15.LIBRARY_SIZE,exclude_structures=ts,exclude_truths=tt)
        if len(lib)!=g15.LIBRARY_SIZE or ts & {g15.g010.structure_key(e) for e in lib} or tt & {g15.g010.truth_signature(e) for e in lib}: raise RuntimeError("split gate failed")
        cache,_=g15.build_cache(lib)
        checks[str(s)]={"train_eval_structure_disjoint":True,"train_eval_truth_disjoint":True,"target_identity_blind":True,"target_evidence_hidden_before_selection":True,"generator_commit":GENERATOR_COMMIT}
        for m in levels:
            cand=lib[:m]
            for i in range(tasks):
                ti,_=task(s,m,i)
                raw[str(s)][str(m)].append({"information_per_work":trial(cache,cand,ti,"information_per_work"),"budget_utility_per_work":trial(cache,cand,ti,"budget_utility_per_work")})
    seed_rows=[]
    deltas=[]
    if str(PRIMARY_M) in raw[str(seeds[0])]:
        for s in seeds:
            vals=raw[str(s)][str(PRIMARY_M)]
            seed_rows.append({
              "information_per_work":statistics.fmean(x["information_per_work"]["budget_capped_utility"] for x in vals),
              "budget_utility_per_work":statistics.fmean(x["budget_utility_per_work"]["budget_capped_utility"] for x in vals)
            })
        deltas=[x["budget_utility_per_work"]-x["information_per_work"] for x in seed_rows]
    primary = {"M":PRIMARY_M,"B":BUDGET,"available":bool(deltas)}
    if deltas:
        primary.update({
          "seed_level_deltas":deltas,
          "mean_delta":statistics.fmean(deltas),
          "seed_bootstrap_95ci":bootstrap(deltas)
        })
    else:
        primary["reason"]="smoke omitted primary M; no confirmatory inference"
    result={"experiment_id":EXPERIMENT_ID,"status":"measured","provenance":{"tacosm_commit":os.environ.get("GITHUB_SHA","local"),"run_id":os.environ.get("GITHUB_RUN_ID","local"),"generator_commit":GENERATOR_COMMIT},"protocol":{"seeds":list(seeds),"M_levels":list(levels),"tasks_per_seed_M":tasks,"B":BUDGET},"checks":checks,"results":raw,"summary":{"primary":primary,"by_M":{}},"scope":{"finite_domain_objective_comparison":True,"learned_probe_policy_claim":False,"submodularity_claim":False}}
    for m in levels:
        result["summary"]["by_M"][str(m)]={}
        for ch in ("information_per_work","budget_utility_per_work"):
            vals=[x[ch]["budget_capped_utility"] for s in seeds for x in raw[str(s)][str(m)]]
            result["summary"]["by_M"][str(m)][ch]={"U8_mean":statistics.fmean(vals),"verified_success_mean":statistics.fmean(x[ch]["verified_success"] for s in seeds for x in raw[str(s)][str(m)]),"probe_work_mean":statistics.fmean(x[ch]["probe_environment_work_units"] for s in seeds for x in raw[str(s)][str(m)])}
    p=Path("artifacts"); p.mkdir(exist_ok=True); (p/f"{EXPERIMENT_ID}.json").write_text(json.dumps(result,indent=2,sort_keys=True)+"\n"); print(json.dumps(result["summary"],indent=2,sort_keys=True))

if __name__=="__main__":
    p=argparse.ArgumentParser(); p.add_argument("--smoke",action="store_true"); main(p.parse_args().smoke)
