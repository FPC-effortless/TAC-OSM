#!/usr/bin/env python3
"""Audit authentic original ME069 BPI2012 eligible 32/64/128/256 histories.
Do not re-fit, read case labels for model-selection, or promote exposed TEST.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from scripts.me069_workflow_source import load

def measure(cohorts,src):
    levels=(32,64,128,256)
    return {"scope":"ORIGINAL_REAL_BPI2012_SOURCE_ONLY_EXISTING_EXPOSED_SPLIT; NOT NEW TRAINING",
            "source_sha256":src["source_sha256"],
            "cohorts":{name:{"cases":len(rows),"events":sum(int(x["n"]) for x in rows),
                      "maximum_events":max(int(x["n"]) for x in rows),
                      "true_history_support":{str(n):{
                        "distinct_cases_with_n_observed_events_and_a_future":sum(int(x["n"])>n for x in rows),
                        "available_next_event_targets_after_n_history":sum(max(0,int(x["n"])-n) for x in rows)
                      } for n in levels}} for name,rows in cohorts.items()},
            "must_not_promote_model_claims":True}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--cache",type=Path,default=Path("/tmp/me070-original-bpi2012"))
    ap.add_argument("--output",type=Path,default=Path("/tmp/me070-bpi2012-eligibility.json"))
    a=ap.parse_args()
    cohorts,_,src=load(a.cache)
    stats=measure(cohorts,src)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(stats,indent=2,sort_keys=True)+"\n")
    print(json.dumps(stats,indent=2,sort_keys=True))

if __name__=="__main__":main()
