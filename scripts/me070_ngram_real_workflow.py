#!/usr/bin/env python3
"""Factual BPI2012 1/2/3/5-event n-gram reference, not new TAC memory evidence.

Authentic original source, FIT-only counts, frozen hyperparameters, prior-free
TEST hindsight midpoint comparison marked EXPOSED. A fit's original JSON count
state MUST be committed to the repository BEFORE score or replay.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
from hashlib import sha256
import json
import math
from pathlib import Path
import statistics
import subprocess
import sys

from scripts.me069_workflow_source import load

STATE=Path("research/replay-bundles/ME070/practical-ngrams/FIT_NGRAM_STATE.json")
SCORES=Path("research/replay-bundles/ME070/practical-ngrams/EXPOSED_ORIGINAL_SCORES.json")
ORDERS=(1,2,3,5)
ALPHA=5.0

def stable_bytes(record):
    return (json.dumps(record,sort_keys=True,separators=(",",":"),ensure_ascii=True)+"\n").encode()

def fit(fit_cases, pre, source):
    vocab=len(pre["activities"])+1
    base=[float(x) for x in pre["global_next_activity_probability"]]
    rows={str(k):{} for k in (2,3,5)}
    # All real observed FIT event transitions. Only original, FIT-causal.
    counters={k:defaultdict(lambda:defaultdict(int)) for k in (2,3,5)}
    for case in fit_cases:
        activity=[int(i) for i in case["activity"]]
        for t in range(len(activity)-1):
            target=activity[t+1]
            for k in (2,3,5):
                if t+1>=k:
                    context=",".join(map(str,activity[t-k+1:t+1]))
                    counters[k][context][str(target)]+=1
    for k,by_context in counters.items():
        rows[str(k)]={context:dict(sorted(counts.items())) for context,counts in sorted(by_context.items())}
    return {"schema":"tac-osm.me070.real-ngram-fit.v1",
            "scope":"FIT only; practical non-neural reference; exposed original TEST diagnostic, no TAC superiority claim",
            "real_source_sha256":source["source_sha256"],
            "FIT_case_ids_sha256":source["cohorts"]["FIT"]["ids_sha256"],
            "fit_cases":len(fit_cases), "fit_events":sum(int(x["n"]) for x in fit_cases),
            "trained_counts_orders":[2,3,5],"alpha_backoff":ALPHA,
            "vocabulary":vocab, "base_probability":base,
            "first_order_transition_probability":[[math.exp(float(x)) for x in row]
                                                   for row in pre["transition_log_probability"]],
            "context_transition_counts":rows,
            "training_equation":"P_k=(counts_k + alpha * P_lower)/(N_context+alpha); fallback P_lower",
            "first_order_equation":"Original ME069 FIT-only empirical Markov transition prior with pseudocount 0.5",
            "selected_without_DEV_or_TEST":True}

def predict(state,history,order):
    n=int(state["vocabulary"])
    if not history: return state["base_probability"]
    row=state["first_order_transition_probability"][history[-1]]
    if order==1:return row
    result=row
    for k in [2,3,5]:
        if k>order or len(history)<k:break
        key=",".join(map(str,history[-k:]))
        counts=state["context_transition_counts"][str(k)].get(key)
        if not counts: continue
        mass=sum(counts.values())
        result=[(int(counts.get(str(i),0))+ALPHA*result[i])/(mass+ALPHA)
                for i in range(n)]
    return result

def score(state,cohorts,source):
    if state["real_source_sha256"]!=source["source_sha256"]:
        raise ValueError("original source SHA mismatch")
    if state["FIT_case_ids_sha256"]!=source["cohorts"]["FIT"]["ids_sha256"]:
        raise ValueError("FIT original case IDs mismatch")
    summary={"scope":"EXPOSED_ME069_HISTORICAL_TEST_NO_MODEL_SELECTION_NO_TRAINING",
             "original_source_sha256":source["source_sha256"],
             "state_sha256":sha256(stable_bytes(state)).hexdigest(),
             "fit_case_count":state["fit_cases"],
             "seeded_comparative_architecture_superiority":False,
             "cohorts":{}}
    for cohort in ("DEV","TEST"):
        rows=cohorts[cohort]
        metrics={str(k):{"nll":[],"correct":0} for k in ORDERS}
        cases=0
        for case in rows:
            a=[int(i) for i in case["activity"]]
            if len(a)<2:continue
            t=(len(a)-2)//2
            history=a[:t+1]
            target=a[t+1]
            cases+=1
            for k in ORDERS:
                p=predict(state,history,k)
                metrics[str(k)]["nll"].append(-math.log(max(1e-12,p[target])))
                metrics[str(k)]["correct"]+=max(range(len(p)),key=lambda j:p[j])==target
        if cases==0:raise ValueError("empty eligible original evaluation cases")
        summary["cohorts"][cohort]={"cases":cases,"orders":{
            k:{"case_mean_NLL":statistics.mean(v["nll"]),
               "accuracy":v["correct"]/cases}
            for k,v in metrics.items()}}
    return summary

def must_be_owned(path:Path):
    if not path.is_file():raise ValueError("missing original committed count state")
    p=path.as_posix()
    content=subprocess.check_output(["git","show","HEAD:"+p])
    if content!=path.read_bytes():
        raise ValueError("original model state must be identical to Git-tracked checkpoint BEFORE scoring")

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--phase",choices=["fit","score","replay"],required=True)
    ap.add_argument("--cache",type=Path,default=Path("/tmp/me070-original-bpi2012"))
    a=ap.parse_args()
    cohorts,pre,source=load(a.cache)
    if a.phase=="fit":
        if STATE.exists():
            raise ValueError("original fitted state already present; refuse overwrite")
        state=fit(cohorts["FIT"],pre,source)
        STATE.parent.mkdir(parents=True,exist_ok=True)
        STATE.write_bytes(stable_bytes(state))
        print(json.dumps({"phase":"ORIGINAL_FIT_UNSCORED","state_path":STATE.as_posix(),
                          "state_sha256":sha256(STATE.read_bytes()).hexdigest(),
                          "fit_cases":state["fit_cases"],
                          "context_count":{k:len(v) for k,v in state["context_transition_counts"].items()}}))
    else:
        must_be_owned(STATE)
        state=json.loads(STATE.read_text())
        result=score(state,cohorts,source)
        if a.phase=="score":
            if SCORES.exists():raise ValueError("original score file already exists")
            SCORES.write_bytes(stable_bytes(result))
            print(json.dumps(result,indent=2))
        else:
            must_be_owned(SCORES)
            if stable_bytes(result)!=SCORES.read_bytes():
                raise ValueError("independent original FIT count-state replay mismatch")
            print(json.dumps({"phase":"REPO_COMMITTED_MODEL_SCORE_REPLAY_PASS",
                              "state_sha256":sha256(STATE.read_bytes()).hexdigest(),
                              "score_sha256":sha256(SCORES.read_bytes()).hexdigest(),
                              "frozen_test_was_already_exposed":True}))

if __name__=="__main__":
    main()
