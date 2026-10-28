#!/usr/bin/env python3
"""Complete 15-model no-fit LANGLAYER036 diagnostic summary. Fail closed."""
import argparse
import json
import math
import statistics
from pathlib import Path
ARMS=("tac_context026","tac_layer2","tac_layer3")
SEEDS=(1701,1702,1703,1704,1705)
EXPECTED={(a,s) for a in ARMS for s in SEEDS}
LESIONS=("block_bypass","no_recurrence","no_concept_projection",
         "no_concept_history","uniform_concept_graph")

def summarize(directory):
    original=list(directory.rglob("*.json"))
    collected={}
    for path in original:
        record=json.loads(path.read_text())
        if record.get("study")!="LANGLAYER036":raise ValueError(f"unexpected file {path}")
        key=(record["arm"],record["seed"])
        if key not in EXPECTED or key in collected:raise ValueError("extra/duplicate result "+str(key))
        if record.get("new_optimizer_updates")!=0 or record.get("protected_TEST")!="NOT_ACCESSED":
            raise ValueError("not frozen allowed source")
        if record.get("fit_seen_targets")!=4096 or record.get("exposed_dev_targets")!=4096:
            raise ValueError("wrong targets")
        collected[key]=record
    if set(collected)!=EXPECTED:
        raise ValueError(f"incomplete cohort: {sorted(EXPECTED-set(collected))}")
    for seed in SEEDS:
        samples={collected[a,seed]["sample_order_sha256"] for a in ARMS}
        if len(samples)!=1:raise ValueError("unmatched source sample schedule")
    result={"study":"LANGLAYER036","status":"PASS_FROZEN_INTERNAL_DIAGNOSTIC_NOT_QUALIFICATION",
       "model_fits":0,"protected_TEST":"NOT_ACCESSED",
       "evidence":"source-verified publisher FIT and previously exposed DEV; 15 original trained checkpoints",
       "warning":"FIT is training-seen batches, DEV is a different publisher document cohort. Gap does not prove classic overfitting.",
       "arms":{},"layers":{},"per_seed":{}}
    for arm in ARMS:
        records=[collected[arm,s] for s in SEEDS]
        out={}
        for key in ("fit_seen_nll","exposed_dev_nll","dev_minus_seen_fit_nll"):
            vals=[float(row[key]) for row in records]
            if not all(math.isfinite(v) for v in vals):raise ValueError("invalid NLL")
            out[key+"_mean"]=statistics.mean(vals)
            out[key+"_sd"]=statistics.stdev(vals)
        result["arms"][arm]=out
        result["per_seed"][arm]={str(s):collected[arm,s] for s in SEEDS}
        if arm=="tac_context026":continue
        result["layers"][arm]={}
        depth=2 if arm=="tac_layer2" else 3
        for i in range(1,depth):
            rows=[r["layers"][str(i)] for r in records]
            a={}
            for key in ("ratio_residual_to_input_rms","cosine_input_to_output","gate"):
                vals=[float(row[key]) for row in rows]
                a[key+"_mean"]=statistics.mean(vals)
                a[key+"_sd"]=statistics.stdev(vals)
            a["lesions"]={}
            for name in LESIONS:
                vals=[float(row["lesion_delta_dev_nll"][name]) for row in rows]
                if any(not math.isfinite(x) for x in vals):raise ValueError("nonfinite lesion")
                a["lesions"][name]={
                    "mean_delta_nll":statistics.mean(vals),
                    "sd":statistics.stdev(vals),
                    "seeds_positive_delta_when_removed":sum(v>0 for v in vals),
                    "by_seed":dict(zip(map(str,SEEDS),vals))}
            for param_name in ("forget.weight","read_gate.weight","concept_q.weight",
                               "concept_k.weight","concept_gate.weight",
                               "concept_to_state.weight","output.weight","residual_gain"):
                grad_name=param_name+"_gradient_rms"
                vals=[float(r["gradient"][str(i)][grad_name]) for r in records]
                a[grad_name+"_mean"]=statistics.mean(vals)
                a[grad_name+"_seeds_nonzero"]=sum(v>0 for v in vals)
            result["layers"][arm][str(i)]=a
    return result

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--input",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    summary=summarize(a.input)
    if a.output.exists():raise FileExistsError("refuse to overwrite")
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(summary,indent=2,sort_keys=True,allow_nan=False)+"\n")
    print(json.dumps(summary["arms"],sort_keys=True))
    print(json.dumps(summary["layers"],sort_keys=True))
if __name__=="__main__":main()
