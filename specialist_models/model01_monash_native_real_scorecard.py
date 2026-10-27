#!/usr/bin/env python3
"""Client-cluster original real Monash source TAC-vs-naive DEV scorecard.

No model fitting, no original source re-download, no heldout test access.
Best DEV reference is hindsight descriptive; CAL-only selected reference is
separately disclosed. Bootstrap uses 36 physical clients, not hourly samples.
Only development claims; uncertainty does not turn exposed DEV independent.
"""
from __future__ import annotations
import argparse,hashlib,json,math,random
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
TRAIN=ROOT/"research/replay-bundles/MODEL01-MONASH-NATIVE-MULTIHORIZON-001/ORIGINAL_FIT.json"
BASE=ROOT/"research/source-evidence/MODEL01-MONASH321-ORIGINAL-CAL-DEV-SOURCE-BASELINES.json"
SOURCE_SHA="eff447075dde68dca0105ab7e2851c5637967ae3bb21556fd8b931f196d5968c"
SEEDS=[1801,1802,1803]
HORIZONS=("1","24")
METHODS=("last_observation","same_hour_previous_day","same_hour_previous_week","daily_weekly_blend")
class ScoreError(ValueError):pass

def quantile(xs,p):
    values=sorted(xs)
    return values[min(len(values)-1,int((len(values)-1)*p))]

def run_scorecard(a,b):
    if (a.get("study")!="MODEL01-MONASH-NATIVE-MULTIHORIZON-001" or
        a.get("source_zip_sha256")!=SOURCE_SHA or b.get("source_zip_sha256")!=SOURCE_SHA or
        a.get("native_learned_deployed_parameters")!=103381 or
        [r["seed"] for r in a.get("runs",[])]!=SEEDS or
        b.get("source_arm_counts")!={"FIT":176,"CAL":34,"DEV":36,"SEALED_TEST":75} or
        a.get("sealed_test_target_values_parsed") is not False or
        b.get("TEST_customer_numeric_targets_parsed") is not False or
        a.get("independent_confirmatory_test_performed") is not False):
        raise ScoreError("original model/source/heldout provenance violated")
    original_map={row["client_public_hash"]:row for row in b["per_client_CAL_DEV"]["DEV"]}
    if len(original_map)!=36:
        raise ScoreError("original customer identity duplicated")
    summary={}
    for h in HORIZONS:
        references={meth:{"cal_mae_kw":b["macro_references"]["CAL"][h][meth]["mean_client_mae_kw"],
                          "dev_mae_kw":b["macro_references"]["DEV"][h][meth]["mean_client_mae_kw"]}
                    for meth in METHODS}
        cal_rule=min(METHODS,key=lambda k:(references[k]["cal_mae_kw"],k))
        dev_oracle=min(METHODS,key=lambda k:(references[k]["dev_mae_kw"],k))
        individual=[]
        customer_deltas={}
        for r in a["runs"]:
            clients=r["dev_per_customer"]
            if set(clients)!=set(original_map):
                raise ScoreError("original per-client paired source identity mismatch")
            if r["dev_macro"][h]["total_decisions"]!=12096 or r["dev_macro"][h]["customers"]!=36:
                raise ScoreError("original source denominator changed")
            deltas=[]
            for identifier in sorted(clients):
                actual=clients[identifier][h]
                baseline=original_map[identifier]["source_baselines"][h]
                expected=baseline[cal_rule]["n"]
                if actual["n"]!=expected:
                    raise ScoreError("source forecast case mismatch")
                delta=actual["mae_kw"]-baseline[cal_rule]["mae_kw"]
                deltas.append(delta)
                customer_deltas.setdefault(identifier,[]).append(delta)
            actual_mae=r["dev_macro"][h]["mean_client_mae_kw"]
            if not math.isclose(sum(clients[c][h]["mae_kw"] for c in clients)/36,
                                actual_mae,abs_tol=1e-6):
                raise ScoreError("per-client macro score differs")
            individual.append({
                "seed":r["seed"],"source_model_checkpoint_sha256":r["checkpoint_sha256"],
                "dev_macro_mae_kw":actual_mae,
                "dev_macro_rmse_kw":r["dev_macro"][h]["mean_client_rmse_kw"],
                "dev_macro_coverage90":r["dev_macro"][h]["mean_client_coverage90"],
                "dev_interval_score90_kw":r["dev_macro"][h]["mean_client_interval_score90_kw"],
                "dev_interval_width90_kw":r["dev_macro"][h]["interval_width90_kw"],
                "DEV_improves_on_CAL_chosen_reference":actual_mae<references[cal_rule]["dev_mae_kw"],
                "DEV_improves_on_DEV_best_descriptive_reference":actual_mae<references[dev_oracle]["dev_mae_kw"],
                "number_of_36_client_better_than_CAL_chosen_ref":sum(x<0 for x in deltas)
            })
        cluster_differences=[sum(v)/len(v) for v in customer_deltas.values()]
        if len(cluster_differences)!=36:
            raise ScoreError("not 36 actual physical customer uncertainty units")
        rng=random.Random(20261027+int(h))
        boots=[]
        for _ in range(1000):
            boots.append(sum(cluster_differences[rng.randrange(36)] for _ in range(36))/36)
        avg=sum(r["dev_macro_mae_kw"] for r in individual)/3
        summary[h]={
            "CAL_selected_reference":cal_rule,
            "DEV_best_hindsight_reference":dev_oracle,
            "no_fit_reference_metrics":references,
            "mean_native_three_seed_mae_kw":avg,
            "mean_native_improvement_fraction_vs_CAL_rule":
                1-avg/references[cal_rule]["dev_mae_kw"],
            "mean_native_improvement_fraction_vs_hindsight_best":
                1-avg/references[dev_oracle]["dev_mae_kw"],
            "per_seed":individual,
            "paired_DEV_36_client_means_across_seeds":{
                "source_units":"separate physical customers, not 12096 hourly forecasts",
                "mean_client_delta_mae_kw_native_minus_CAL_rule":sum(cluster_differences)/36,
                "posthoc_cluster_bootstrap_95_percentile_delta_mae_kw":
                    [quantile(boots,.025),quantile(boots,.975)],
                "independent_confirmation":False
            }
        }
    return {
        "schema":"MODEL01-MONASH-NATIVE-MULTIHORIZON-001-SOURCE-DEV-SCORECARD-v1",
        "source_zip_sha256":SOURCE_SHA,
        "trained_parameters":103381,
        "original_training_manifest_path":str(TRAIN.relative_to(ROOT)),
        "original_source_baseline_path":str(BASE.relative_to(ROOT)),
        "CAL_and_DEV_previously_exposed":True,
        "FIT_CAL_DEV_customer_counts":[176,34,36],
        "sealed_TEST_customers":75,
        "test_numeric_outcomes_scored":False,
        "horizon_results":summary,
        "release_gate":"FAILED_24H_PRACTICAL_BASELINE_AND_BOTH_INTERVAL_COVERAGES",
        "independent_capability_gain":False,
        "neural_architecture_superiority":False,
        "posthoc_development_cluster_bootstrap":True,
        "warnings":[
            "CAL-selected reference is allowed for development decisions; best-on-DEV reference is descriptive hindsight only",
            "This is a newly trained two-horizon model, not transfer of old UCI374 appliance-only checkpoints",
            "Same source Monash real client identities, but DEV baseline labels were seen in prior source qualification",
            "One customer group, with many temporal correlated samples per customer, is the paired uncertainty unit",
            "Sampled 1000 bootstrap resamples after inspecting real scores are descriptive, not preregistered independent inference",
            "Nominal 90% empirical global CAL-residual bounds failed DEV coverage for every seed and horizon",
            "Hourly local DST correction, preactivation artificial zeros and physical available-at remain unverified",
            "No release, safety/financial effectiveness or demonstrated native memory architectural advantage"
        ]
    }

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out",type=Path,required=True)
    x=p.parse_args()
    if x.out.exists():raise ScoreError("refuse source report overwrite")
    evidence=run_scorecard(json.loads(TRAIN.read_bytes()),json.loads(BASE.read_bytes()))
    x.out.parent.mkdir(parents=True,exist_ok=True)
    x.out.write_text(json.dumps(evidence,sort_keys=True,indent=2,allow_nan=False)+"\n")
    print(json.dumps({"schema":evidence["schema"],"horizons":{
        h:{"mean_model_mae":v["mean_native_three_seed_mae_kw"],
           "CAL_selection":v["CAL_selected_reference"],
           "relative_change_vs_CAL_rule":v["mean_native_improvement_fraction_vs_CAL_rule"],
           "relative_change_vs_DEV_best":v["mean_native_improvement_fraction_vs_hindsight_best"],
           "per_seed":v["per_seed"]} for h,v in evidence["horizon_results"].items()},
        "release_gate":evidence["release_gate"]},sort_keys=True))

if __name__=="__main__":main()
