"""ME066 independent original real physical score replay with honest CPU float tolerance.

Exact native model + optimizer/RNG bytes are audited separately from score values.
All source/task/checkpoint identity, JSON schemas, booleans, counts and strings
MUST match exactly. Small CPU numeric variation is not bitwise-identical replay.
Never select/tune after opening a new authentic target.
"""
from __future__ import annotations
import argparse,hashlib,json,math
from pathlib import Path

FILES=("ORIGINAL_SOURCE_CAL_SCORES.json",
       "ORIGINAL_UNTOUCHED_TASK_SCORES.json",
       "REAL_WORLD_EVALUATION_SUMMARY.json")
ABS_TOL=1e-4
REL_TOL=1e-5
PRIMARY_SUMMARY_ABS_TOL=1e-5
class ReplayMismatch(ValueError):pass

def compare(a,b,key,stats):
    if type(a) is not type(b):
        raise ReplayMismatch(f"{key}: source/replay JSON types differ")
    if isinstance(a,dict):
        if set(a)!=set(b):
            raise ReplayMismatch(f"{key}: score output fields differ")
        for k in sorted(a):
            compare(a[k],b[k],f"{key}.{k}",stats)
    elif isinstance(a,list):
        if len(a)!=len(b):raise ReplayMismatch(f"{key}: array length changed")
        for i,(left,right) in enumerate(zip(a,b)):
            compare(left,right,f"{key}[{i}]",stats)
    elif isinstance(a,float):
        if not math.isfinite(a) or not math.isfinite(b):
            raise ReplayMismatch(f"{key}: nonfinite actual/replayed prediction")
        err=abs(a-b)
        tol=ABS_TOL+REL_TOL*max(abs(a),abs(b))
        if key.startswith("REAL_WORLD_EVALUATION_SUMMARY.json.") and (
                ".physical_MAE." in key or "block_bootstrap_95" in key or
                "tac_minus_reference_mean" in key):
            tol=PRIMARY_SUMMARY_ABS_TOL
        if err>tol:
            raise ReplayMismatch(f"{key}: actual {a} replay {b} diff {err} > {tol}")
        stats["total_float_fields"]+=1
        stats["different_float_fields"]+=int(err!=0.)
        stats["maximum_float_abs_difference"]=max(stats["maximum_float_abs_difference"],err)
    elif a!=b:
        raise ReplayMismatch(f"{key}: strict nonfloating source identity changed")

def run(published,recomputed):
    rows={}
    stats={"total_float_fields":0,"different_float_fields":0,"maximum_float_abs_difference":0.}
    for name in FILES:
        original=published/name;again=recomputed/name
        if not original.is_file() or not again.is_file():
            raise ReplayMismatch(f"{name}: original/independent score missing")
        expected=json.loads(original.read_text())
        result=json.loads(again.read_text())
        compare(expected,result,name,stats)
        rows[name]={
            "original_sha256":hashlib.sha256(original.read_bytes()).hexdigest(),
            "independent_replay_sha256":hashlib.sha256(again.read_bytes()).hexdigest(),
            "byte_identical":original.read_bytes()==again.read_bytes(),
        }
    result=json.loads((published/"REAL_WORLD_EVALUATION_SUMMARY.json").read_text())
    if result["study"]!="ME066_FOUR_ARM_GENUINE_ORIGINAL_ROBOT_FIT_EQUAL_BUDGET":
        raise ReplayMismatch("incorrect research experiment identity")
    if (result["results"]["task"]!="plug_charger_into_power_strip" or
            result["trained_original_checkpoints_replayed_from_clean_Git"]!=20 or
            result["heldout_training_updates"]!=0 or
            result["old_ME064_test_reused"] is not False):
        raise ReplayMismatch("original trained source/frozen target eligibility changed")
    return {
        "study":result["study"],"original_target":result["results"]["task"],
        "numeric_score_replay":"PASS_WITH_FLOAT_TOLERANCE",
        "byte_identical_score_JSON":all(f["byte_identical"] for f in rows.values()),
        "checkpoint_binary_load_verification":"PASS_SEPARATE_20_ORIGINAL_MODELS",
        "score_comparison":{"abs_tolerance":ABS_TOL,"relative_tolerance":REL_TOL,
                            "primary_summary_abs_tolerance":PRIMARY_SUMMARY_ABS_TOL},
        "floating_field_differences":stats,
        "files":rows,
        "scientific_superiority_gate_pass":result["results"]["registered_superiority_gate_pass"],
        "original_physical_MAE":result["results"]["physical_MAE"],
        "independent_robot_or_operator_proven":False,
        "source_CAL_or_target_fitting_performed":False,
        "target_model_updates":0,
    }

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--published",required=True,type=Path)
    parser.add_argument("--recomputed",required=True,type=Path)
    parser.add_argument("--receipt",required=True,type=Path)
    args=parser.parse_args()
    receipt=run(args.published,args.recomputed)
    args.receipt.parent.mkdir(parents=True,exist_ok=True)
    args.receipt.write_text(json.dumps(receipt,indent=2,sort_keys=True)+"\n")
    print("ME066_INDEPENDENT_REAL_SCORE_REPLAY_PASS="+json.dumps({
        "byte_identical":receipt["byte_identical_score_JSON"],
        "max_float_difference":receipt["floating_field_differences"]["maximum_float_abs_difference"],
        "scientific_gate":receipt["scientific_superiority_gate_pass"]},sort_keys=True),flush=True)
