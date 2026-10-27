#!/usr/bin/env python3
"""Source-frozen, no-fit TAC-vs-GRU real exposed DEV comparison scorecard.

Reads original Git model manifests only. Strict cohort, checkpoint,
parameter count and exact FIT sampler parity checks; no training or future
TEST source opened. This is NOT FLOP, tuning search or fresh-site parity.
"""
import argparse,hashlib,json,math
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
TAC=ROOT/"research/replay-bundles/MODEL01-TAC-REAL-FIT-001/ORIGINAL_FIT.json"
GRU=ROOT/"research/replay-bundles/MODEL01-GRU-103K-CONTROL-001/ORIGINAL_FIT.json"
NATIVE_PARAMS=103252
GRU_PARAMS=102883
NATIVE_SHAS={
    1701:"2d33226a652304ad69770a2509be33b221cacfc424b5ad6b6c79dce3570b4d4f",
    1702:"9a37e4afb7cd9cc59912e437a4d706faecb393cb177dce0da5f6ea4837c5e0b9",
    1703:"86fbb16d42c697582a885891bc80274f2af26ae1e9c9f4cd2cf531f6e4e544d4"
}
GRU_SHAS={
    1701:"b88a4137c2e40c95c33fbdd447f46d9f50b8dcddfa6ae1c8d822f9a9fd39f3db",
    1702:"a08363b1329831b841a9b777097ff15d0127f9efc014e755bba23b4ec4a96013",
    1703:"ffc8334697de760072f2faca198d8f88b2c039544a8c1a8cc21586487c85b539"
}
METRICS=("mae_wh","rmse_wh","nominal90_observed_coverage","interval_score90","width90_wh")
SOURCE="2fccf354445d886e7917620b0195db1f3e3e34d5a067a93b844694a4c561255a"

class ScorecardError(ValueError):pass

def evaluate(native,gru):
    if (native.get("study")!="MODEL01-TAC-REAL-FIT-001" or
        gru.get("study")!="MODEL01-GRU-103K-CONTROL-001" or
        native.get("source_zip_sha256")!=SOURCE or gru.get("source_zip_sha256")!=SOURCE):
        raise ScorecardError("source/model identity not independently verified")
    if (native.get("trainable_params")!=NATIVE_PARAMS or
        gru.get("native_learned_parameters")!=NATIVE_PARAMS or
        gru.get("gru_learned_parameters")!=GRU_PARAMS or
        gru.get("relative_parameter_gap")>=.01 or
        native.get("train_updates_per_seed")!=gru.get("training_steps") or
        native.get("batch_size")!=gru.get("training_batch_size") or
        not gru.get("same_exact_source_minibatch_indices") or
        not gru.get("same_optimizer_loss_lr_weight_decay_and_grad_clip")):
        raise ScorecardError("declared parameter/source/optimization parity failed")
    if (gru.get("training_FLOPs_matched") is not False or
        gru.get("protected_future_test_rows_parsed") is not False or
        native.get("original_protected_tail_parsed") is not False or
        gru.get("independent_evaluation") is not False):
        raise ScorecardError("nonpromotion and unseen protected source guarantees altered")
    n_runs=native.get("runs",[])
    g_runs=gru.get("runs",[])
    if len(n_runs)!=3 or len(g_runs)!=3:
        raise ScorecardError("all original seeds mandatory")
    paired=[]
    for a,b in zip(n_runs,g_runs):
        seed=a.get("seed")
        if (seed not in NATIVE_SHAS or seed!=b.get("seed") or
            a.get("checkpoint_sha256")!=NATIVE_SHAS[seed] or
            b.get("checkpoint_sha256")!=GRU_SHAS[seed] or
            a.get("checkpoint_sha256")!=b.get("matched_native_checkpoint_sha256") or
            a.get("sample_order_sha256")!=b.get("sample_order_sha256")):
            raise ScorecardError("original weights or exact minibatch training parity changed")
        for metric in METRICS:
            av=a["dev"].get(metric)
            bv=b["dev"].get(metric)
            if not all(type(x) in (int,float) and math.isfinite(x) for x in (av,bv)):
                raise ScorecardError("original nonfinite/incompatible metric "+metric)
        if a["dev"]["n"]!=1968 or b["dev"]["n"]!=1968:
            raise ScorecardError("unequal original exposed DEV denominators")
        paired.append({
            "seed":seed,"real_DEV_forecast_count_per_model":1968,
            "native_checkpoint_sha256":a["checkpoint_sha256"],
            "gru_checkpoint_sha256":b["checkpoint_sha256"],
            "source_FIT_minibatch_sha256":a["sample_order_sha256"],
            "native_dev":{k:a["dev"][k] for k in METRICS},
            "gru_dev":{k:b["dev"][k] for k in METRICS},
            "gru_minus_native_dev":{k:b["dev"][k]-a["dev"][k] for k in METRICS}
        })
    def mean(model,key):
        return sum(p[model][key] for p in paired)/len(paired)
    avg={"native":{k:mean("native_dev",k) for k in METRICS},
         "gru":{k:mean("gru_dev",k) for k in METRICS}}
    return {
        "schema":"model01-genuine-exposed-dev-neural-controlled-comparison-v1",
        "source_zip_sha256":SOURCE,
        "original_native_model_path":str(TAC.relative_to(ROOT)),
        "original_gru_model_path":str(GRU.relative_to(ROOT)),
        "actual_native_parameters":NATIVE_PARAMS,
        "actual_gru_parameters":GRU_PARAMS,
        "relative_parameter_gap":abs(NATIVE_PARAMS-GRU_PARAMS)/NATIVE_PARAMS,
        "same_exact_fit_minibatch_indices_per_seed":True,
        "same_observation_history_target_split_batch_updates_optimization":True,
        "training_FLOPs_or_tuning_budget_equal":False,
        "fitting_during_comparison":False,
        "protected_future_test_opened":False,
        "trained_GRU_control_original_checkpoint_verified":True,
        "native_seed_summaries":paired,
        "three_seed_means":avg,
        "mean_GRU_MAE_relative_to_TAC":avg["gru"]["mae_wh"]/avg["native"]["mae_wh"]-1,
        "mean_GRU_interval_score_relative_to_TAC":
            avg["gru"]["interval_score90"]/avg["native"]["interval_score90"]-1,
        "metric_specific_conclusion":"reused development: GRU slightly lower mean point MAE, native slightly lower mean interval score, neither dominates",
        "independent_qualification":False,
        "neural_architecture_superiority_claim":False,
        "scientific_tier":"PARAMETER_SOURCE_UPDATE_MATCHED_EXPOSED_DEV_NOT_FLOP_OR_TUNING_MATCHED",
        "warnings":[
            "Source is same retrospective exposed single-household UCI374, not newly confirmed real-world skill",
            "Native TAC tuning preceded this comparison and had previously encountered exposed DEV",
            "Same FIT sampler, params and update count do not equalize different architectural FLOPs",
            "The confidence/interval score and point MAE have mixed orderings",
            "No isolated native associative memory benefit or long-retention result",
            "No deployed product, customer willingness to pay, independently observed source-time or causal decision value"
        ]
    }

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out",type=Path,required=True)
    args=p.parse_args()
    if args.out.exists():raise ScorecardError("no evidence overwrites")
    val=evaluate(json.loads(TAC.read_bytes()),json.loads(GRU.read_bytes()))
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(val,indent=2,sort_keys=True,allow_nan=False)+"\n")
    print(json.dumps({k:val[k] for k in ("schema","relative_parameter_gap","three_seed_means","mean_GRU_MAE_relative_to_TAC","mean_GRU_interval_score_relative_to_TAC","scientific_tier")},sort_keys=True))

if __name__=="__main__":
    main()
