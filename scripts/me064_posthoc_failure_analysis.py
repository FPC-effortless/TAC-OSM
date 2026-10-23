"""ME064 POSTHOC failure analysis from already frozen original score JSON.

NO new outcome data, no inference, no retraining, no threshold choice from
target labels, no model selection. Cannot qualify any new architecture or claim
unseen-target independence again. Physical MAE units, lower is better.
"""
from __future__ import annotations
import argparse, hashlib, json, math
from pathlib import Path

ROOT=Path("research/replay-bundles/ME064")
SEEDS=(64001,64002,64003,64004,64005)
# Fixed intuitive physical-MAE diagnostic bands chosen independently of
# target distribution. NOT a model threshold and never used for inference.
STATIC_BOUND=0.025
MOVE_BOUND=0.100


def mean(x):
    if not x: raise ValueError("empty real episode group")
    return sum(x)/len(x)


def analyse(source):
    if source["study"]!="ME064_GENUINELY_UNSEEN_SIXTH_TASK":
        raise ValueError("incorrect original frozen study")
    if source["task"]!="install_spray_nozzle_onto_bottle":
        raise ValueError("different from original frozen task")
    if not source["no_target_training"] or not source["no_target_normalization"] or (
            not source["no_target_calibration"]):
        raise ValueError("posthoc frozen scoring restrictions violated")
    scores=source["scores"]
    expected={f"{name}-s{seed}" for name in ("native_tac","gru_world") for seed in SEEDS}
    if set(scores)!=expected:raise ValueError("missing or unregistered paired seed")
    ids=sorted(scores["native_tac-s64001"],key=int)
    if len(ids)!=80: raise ValueError("all 80 original scored real episodes required")
    episodes=[]
    for ep in ids:
        tac=[scores[f"native_tac-s{seed}"][ep] for seed in SEEDS]
        gru=[scores[f"gru_world-s{seed}"][ep] for seed in SEEDS]
        baseline=tac[0]
        p=float(baseline["persistence_MAE"])
        s=float(baseline["servo_MAE"])
        if any(abs(float(r["persistence_MAE"])-p)>1e-6 for r in tac+gru) or any(
                abs(float(r["servo_MAE"])-s)>1e-6 for r in tac+gru):
            raise ValueError("baseline original episode not identical across seeds")
        if not math.isfinite(p) or not math.isfinite(s):
            raise ValueError("non-finite original real physical baseline")
        te=mean([float(r["MAE"]) for r in tac])
        ge=mean([float(r["MAE"]) for r in gru])
        if min(te,ge)<0 or not all(math.isfinite(v) for v in [te,ge]):
            raise ValueError("invalid physical predictive MAE")
        weights=[mean([float(x["regime_weights_mean"][j]) for x in tac]) for j in range(3)]
        label="near_stationary" if p<STATIC_BOUND else ("limited_motion" if p<MOVE_BOUND else "moving")
        episodes.append({"episode_id":int(ep),"block":int(ep)//4,"band":label,
                         "tac_mae":te,"gru_mae":ge,"persistence_mae":p,
                         "servo_mae":s,"tac_persistence_weight":weights[0],
                         "tac_regret_to_persistence":te-p,
                         "tac_outperformed_persistence":bool(te<p),
                         "tac_outperformed_gru":bool(te<ge)})
    by_band={}
    for label in ("near_stationary","limited_motion","moving"):
        subset=[r for r in episodes if r["band"]==label]
        by_band[label]={
            "episode_count":len(subset),
            "unique_original_blocks":len({r["block"] for r in subset}),
            "mean_tac_physical_MAE":None if not subset else mean([r["tac_mae"] for r in subset]),
            "mean_gru_physical_MAE":None if not subset else mean([r["gru_mae"] for r in subset]),
            "mean_persistence_physical_MAE":None if not subset else mean([r["persistence_mae"] for r in subset]),
            "mean_servo_physical_MAE":None if not subset else mean([r["servo_mae"] for r in subset]),
            "tac_oracle_regret_to_persistence":None if not subset else mean([r["tac_regret_to_persistence"] for r in subset]),
            "mean_model_persistence_routing_weight":None if not subset else mean([r["tac_persistence_weight"] for r in subset]),
            "tac_better_than_persistence_episode_count":sum(r["tac_outperformed_persistence"] for r in subset),
            "tac_better_than_gru_episode_count":sum(r["tac_outperformed_gru"] for r in subset),
        }
    seeds=[]
    for seed in SEEDS:
        tac=[scores[f"native_tac-s{seed}"][ep]["MAE"] for ep in ids]
        gru=[scores[f"gru_world-s{seed}"][ep]["MAE"] for ep in ids]
        seeds.append({"seed":seed,"tac_mae":mean(tac),"gru_mae":mean(gru),
                      "tac_wins":mean(tac)<mean(gru)})
    return {
        "study":"ME064_UNTOUCHED_TASK_POSTHOC_DIAGNOSTIC_FROM_FROZEN_SCORE_ONLY",
        "task":source["task"],"qualification":"POSTHOC_EXPLORATORY_ALREADY_EXPOSED",
        "no_new_trajectory_read":True,"no_model_fitting_or_selection":True,
        "not_actionable_prediction_threshold":True,
        "no_causal_counterfactual_inference":True,
        "diagnostic_bands_predefined_in_physical_mae":{
            "near_stationary":f"persistence MAE < {STATIC_BOUND}",
            "limited_motion":f"{STATIC_BOUND} <= persistence MAE < {MOVE_BOUND}",
            "moving":f"persistence MAE >= {MOVE_BOUND}",
            "warning":"Uses future original outcome labels AFTER evaluation exclusively for diagnosis, never permissible input to a real-time gate."
        },
        "all_original_target_episodes":len(episodes),
        "aggregates_by_motion_band":by_band,
        "per_seed_paired_scores":seeds,
        "paired_seed_tac_wins":sum(row["tac_wins"] for row in seeds),
        "tac_episode_win_counts":{
            "better_than_persistence":sum(r["tac_outperformed_persistence"] for r in episodes),
            "better_than_gru":sum(r["tac_outperformed_gru"] for r in episodes)},
        "overall_means":{
            "tac":mean([r["tac_mae"] for r in episodes]),
            "gru":mean([r["gru_mae"] for r in episodes]),
            "persistence":mean([r["persistence_mae"] for r in episodes]),
            "servo":mean([r["servo_mae"] for r in episodes])},
        "strongest_baseline_gate":"FAIL_PER_PREDECLARED_ORIGINAL_TEST",
    }


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--original-score",type=Path,
                   default=ROOT/"ORIGINAL_UNTOUCHED_TASK_SCORES.json")
    p.add_argument("--output",type=Path,
                   default=ROOT/"POSTHOC_FAILURE_DIAGNOSTIC.json")
    a=p.parse_args()
    payload=a.original_score.read_bytes()
    result=analyse(json.loads(payload))
    result["source_original_score_sha256"]=hashlib.sha256(payload).hexdigest()
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,sort_keys=True,indent=2)+"\n")
    print("ME064_SCORE_ONLY_POSTHOC="+json.dumps({
        "bands":result["aggregates_by_motion_band"],
        "seed_wins":result["paired_seed_tac_wins"],
        "no_target_retraining":True}),flush=True)


if __name__=="__main__":
    main()
