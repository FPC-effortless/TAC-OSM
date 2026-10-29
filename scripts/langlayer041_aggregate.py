#!/usr/bin/env python3
"""Full original 20 trained-checkpoint source-owned long-prefix control audit."""
from __future__ import annotations
import argparse,hashlib,json,math,statistics
from pathlib import Path
ARMS=("h48_graph","h48_plain","h59_graph","h59_plain")
SEEDS=(1701,1702,1703,1704,1705)
LENGTHS=("128","256","512")
def summarize(root):
    rows={};schedule=None
    for p in Path(root).rglob("*.json"):
        r=json.loads(p.read_text())
        if r.get("study")!="LANGLAYER041":raise ValueError("unexpected source diagnostic")
        arm=r.get("arm");seed=r.get("seed")
        if arm not in ARMS or seed not in SEEDS or (arm,seed) in rows:raise ValueError("bad or duplicate arm/seed")
        if r.get("target_count")!=32 or r.get("training_updates")!=0 or r.get("protected_TEST")!="NOT_ACCESSED":
            raise ValueError("unexpected label/training access")
        if set(r["lengths"])!=set(LENGTHS):raise ValueError("incomplete authentic context comparisons")
        if schedule is None:schedule=r["source_docs"]
        if r["source_docs"]!=schedule:raise ValueError("different publisher docs/target offsets per arm/seed")
        if len(schedule)!=4 or len({(v["size"],v["bytes_sha256"]) for v in schedule.values()})!=4:
            raise ValueError("original four source owner identities invalid")
        for L in LENGTHS:
            scores=r["lengths"][L]
            if scores["n"]!=32 or not all(math.isfinite(float(scores[k])) for k in (
                "genuine_nll","distant_donor_nll","history_utility_delta_donor_minus_genuine",
                "genuine_top1","distant_donor_top1")):
                raise ValueError("nonfinite/incomplete original source prediction")
            if abs(scores["distant_donor_nll"]-scores["genuine_nll"]-
                   scores["history_utility_delta_donor_minus_genuine"])>1e-8:
                raise ValueError("corrupt derived history sensitivity")
        if not len(r["original_checkpoint_sha256"])==64:raise ValueError("unidentified original source checkpoint")
        rows[arm,seed]=r
    if set(rows)!={(a,s) for a in ARMS for s in SEEDS}:
        raise ValueError("incomplete 20-checkpoint genuine history cohort")
    result={
       "study":"LANGLAYER041",
       "status":"COMPLETE_FROZEN_TWENTY_CHECKPOINTS_REAL_DOCUMENT_HISTORY_DIAGNOSTIC",
       "trained_models":0,"protected_TEST":"NOT_ACCESSED","same_target_offsets_across_models":True,
       "observed_target_count_per_model_seed":32,
       "context_lengths":[128,256,512],
       "original_source_documents":schedule,
       "no_true_persistent_streaming_claim":True,
       "warning":"Different context lengths change relative model positions; true historical context vs same-length donor is the stronger functional lesion. 32 already-exposed labels per seed and no independent TEST.",
       "per_model":{},"per_seed":{},"paired_control_effects":{}}
    for arm in ARMS:
        result["per_model"][arm]={}
        result["per_seed"][arm]={str(s):rows[arm,s]["lengths"] for s in SEEDS}
        for L in LENGTHS:
            metrics={}
            for key in ("genuine_nll","distant_donor_nll","history_utility_delta_donor_minus_genuine",
                        "genuine_top1","distant_donor_top1","cpu_process_seconds_two_conditions"):
                vals=[rows[arm,s]["lengths"][L][key] for s in SEEDS]
                metrics[key+"_mean"]=statistics.mean(vals)
                metrics[key+"_sd"]=statistics.stdev(vals)
            result["per_model"][arm][L]=metrics
    for L in LENGTHS:
        vals=[rows["h48_plain",s]["lengths"][L]["genuine_nll"]-
              rows["h59_plain",s]["lengths"][L]["genuine_nll"]
              for s in SEEDS]
        result["paired_control_effects"]["h48_plain_minus_h59_plain_L"+L]={
            "mean":statistics.mean(vals),"sd":statistics.stdev(vals),
            "by_seed":dict(zip(map(str,SEEDS),vals))}
    return result
def main():
    p=argparse.ArgumentParser()
    p.add_argument("--results",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args();r=summarize(a.results)
    if a.output.exists():raise FileExistsError("refuse overwrite")
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(r,indent=2,sort_keys=True,allow_nan=False)+"\n")
    print(json.dumps({k:v for k,v in r.items() if k in ("status","per_model","paired_control_effects")},sort_keys=True))
if __name__=="__main__":main()
