#!/usr/bin/env python3
"""LANGMEM088 strict original source paired 3-seed +1 slow bias aggregate."""
import argparse,hashlib,json,math,statistics
from pathlib import Path
SEEDS=(1701,1702,1703)
ARMS=("baseline","slow_bias_plus1")
SOURCE_PATHS={"src/ch05-01-defining-structs.md","src/ch08-01-vectors.md"}
def collect(folder):
    files=list(folder.rglob("EVAL_RECEIPT.json"))
    if len(files)!=3:raise ValueError("exact three original-model seed source reports required")
    receipts={}
    target_signatures={}
    for file in files:
        r=json.loads(file.read_text());seed=r.get("seed")
        if seed not in SEEDS or seed in receipts:raise ValueError("wrong/duplicate source-trained model seed")
        if (r.get("study")!="LANGMEM088" or r.get("status")!="FIRST_UNSEEN_RUST_PAIRED_MIX_BIAS_MATCHED640"
            or r.get("registered_seed_count")!=3 or r.get("trained_640_steps_per_arm") is not True
            or r.get("bias_arm_extra_parameters")!=0 or r.get("all_arms_trainable_parameters")!=100521
            or r.get("model_state_tensor_bytes")!=1920
            or set(r.get("source_new_docs",{}))!=SOURCE_PATHS
            or r.get("original_Rust_git_commit")!="2771df165f092a4fbb9f46b5b2e7450f19010adb"
            or r.get("original_nextbyte_targets_per_seed")!=3584
            or r.get("both_models_frozen_and_sha_verified_before_new_DEV") is not True
            or r.get("original_prefix_max_abs_error",math.inf)>3e-5
            or r.get("protected_TEST")!="NOT_ACCESSED" or r.get("gpu_minutes")!=0):
            raise ValueError("original matched train/eval model/source/compute contract mismatch")
        for k in ("source_nextbyte_target_sha256","original_baseline_model_checkpoint_sha256","bias_plus1_model_checkpoint_sha256"):
            h=r.get(k)
            if not isinstance(h,str) or len(h)!=64 or any(c not in "0123456789abcdef" for c in h):
                raise ValueError("real original model or source SHA missing")
        rows=r.get("rows")
        if not isinstance(rows,list) or len(rows)!=3584:raise ValueError("missing genuine source labels")
        ids=[]
        for x in rows:
            if x["owner"] not in SOURCE_PATHS or not 256<=x["clock"]<2048 or not 0<=x["actual_next_byte"]<=255:
                raise ValueError("actual original source future clock/owner unknown")
            ids.append((x["owner"],x["clock"],x["actual_next_byte"]))
            for arm in ARMS:
                for key in ("nll","old_slow_zero_minus_full_nll","slow_mix_share","old_slow_read_L2","current_slow_write_read_L2"):
                    y=x[arm+"_"+key]
                    if not isinstance(y,(float,int)) or not math.isfinite(y):
                        raise ValueError("nonfinite original source prediction or control")
                if not 0<=x[arm+"_slow_mix_share"]<=1 or x[arm+"_accuracy"] not in (0,1):
                    raise ValueError("learned slow mixture impossible")
        if len(set((a,t) for a,t,_ in ids))!=3584:
            raise ValueError("duplicate real nextbyte owner/clock source")
        target_signatures[seed]=tuple(ids)
        receipts[seed]=r
    if set(receipts)!=set(SEEDS) or len({r["source_nextbyte_target_sha256"] for r in receipts.values()})!=1:
        raise ValueError("all three source seeds not paired on exact true target sha")
    if not(target_signatures[1701]==target_signatures[1702]==target_signatures[1703]):
        raise ValueError("source true labels/owner/clock mismatch")
    docs=[r["source_new_docs"] for r in receipts.values()]
    if any(x!=docs[0] for x in docs):raise ValueError("new publisher raw original Git blob/SHA differs across seeds")
    if len({r["bias_plus1_model_checkpoint_sha256"] for r in receipts.values()})!=3 or len({r["original_baseline_model_checkpoint_sha256"] for r in receipts.values()})!=3:
        raise ValueError("duplicate original trained checkpoint")
    metrics={}
    for name in ARMS:
        metrics[name]={}
        for stratum,rule in (("overall",lambda t:True),("early",lambda t:t<512),("middle",lambda t:512<=t<1024),("late",lambda t:t>=1024)):
            cells=[x for r in receipts.values() for x in r["rows"] if rule(x["clock"])]
            metrics[name][stratum]={"repeated_prediction_count":len(cells),
                "distinct_source_target_clocks":len(cells)//3,
                **{k:statistics.mean(x[name+"_"+k] for x in cells)
                    for k in ("nll","accuracy","old_slow_zero_minus_full_nll","slow_mix_share","old_slow_read_L2","current_slow_write_read_L2")}}
        metrics[name]["by_seed"]={str(seed):{
                "NLL":r["metrics"][name]["overall"]["nll"],
                "old_slow_zero_minus_full_nll":r["metrics"][name]["overall"]["old_slow_zero_minus_full_nll"],
                "slow_mix_share":r["metrics"][name]["overall"]["slow_mix_share"]}
                for seed,r in receipts.items()}
    return {"study":"LANGMEM088","status":"THREE_ORIGINAL_SEED_MATCHED640_BIAS_PLUS1_FIRST_NEW_RUST",
        "seeds":list(SEEDS),"original_publisher":"Rust Project; two new same-publisher docs",
        "full_original_100521p_unchanged_graph":True,
        "new_extra_parameters":0,"matched_training_steps":640,
        "true_nextbyte_positions_per_seed":3584,
        "source_SHA256":receipts[1701]["source_nextbyte_target_sha256"],
        "new_source_files":docs[0],
        "same_true_source_labels_clocks_across_models":True,
        "metrics":metrics,
        "bias_minus_baseline_nll":metrics["slow_bias_plus1"]["overall"]["nll"]-metrics["baseline"]["overall"]["nll"],
        "bias_minus_baseline_old_slow_contribution":metrics["slow_bias_plus1"]["overall"]["old_slow_zero_minus_full_nll"]-metrics["baseline"]["overall"]["old_slow_zero_minus_full_nll"],
        "bias_minus_baseline_slow_mix_share":metrics["slow_bias_plus1"]["overall"]["slow_mix_share"]-metrics["baseline"]["overall"]["slow_mix_share"],
        "source_owner_count":2,"independently_qualified":False,"model_promoted":False,
        "protected_TEST":"NOT_ACCESSED","gpu_minutes":0,
        "limits":["two fresh Rust docs same publisher as 085/086/087, not independent cross-publisher suite",
          "old_slow attribution is one-query read lesion retaining current write and original full next state",
          "same source byte positions reused across 3 trained seeds, no iid point-level inference",
          "slow mix share and NLL improvement do not establish correct historical-byte/factual retrieval",
          "all 640-step original baseline model checkpoints from 087 reused; bias+1 models trained anew on same real FIT order",
          "no matched trained fast-only model; no broad architecture superiority conclusion"]}
def main():
    p=argparse.ArgumentParser();p.add_argument("--results",type=Path,required=True);p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError("original source result must remain immutable")
    r=collect(a.results);a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(r,sort_keys=True,indent=2,allow_nan=False)+"\n")
    print("LANGMEM088_AGGREGATE "+json.dumps({"bias_minus_baseline_nll":r["bias_minus_baseline_nll"],"old_slow_attribution_delta":r["bias_minus_baseline_old_slow_contribution"],"mix_delta":r["bias_minus_baseline_slow_mix_share"],"overall":{a:metrics["overall"] for a,metrics in r["metrics"].items()}},sort_keys=True),flush=True)
if __name__=="__main__":main()
