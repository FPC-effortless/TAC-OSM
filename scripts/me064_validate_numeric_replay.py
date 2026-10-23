"""Independent ME064 real score replay with explicit floating-point tolerance.

All original trained checkpoint bytes are already checked bitwise by the
independent model-load audit. This tool verifies the publisher original
outcome-derived score JSON structures, exact identities, and finite numerical
tolerance; it never fits, selects checkpoints, or accesses the heldout itself.

An expected slight CPU float variation is NOT identical byte-for-byte replay.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

FILES = (
    "ORIGINAL_SOURCE_CAL_SCORES.json",
    "ORIGINAL_UNTOUCHED_TASK_SCORES.json",
    "REAL_WORLD_EVALUATION_SUMMARY.json",
)
ABS_TOL = 1e-4
REL_TOL = 1e-5
SUMMARY_ABS_TOL = 1e-5


class Mismatch(ValueError):
    pass


def check(expected, actual, key, stats):
    if type(expected) is not type(actual):
        raise Mismatch(f"{key}: numeric/object type changed")
    if isinstance(expected, dict):
        if set(expected) != set(actual):
            raise Mismatch(f"{key}: score JSON keys differ")
        for k in sorted(expected):
            check(expected[k],actual[k],f"{key}.{k}",stats)
        return
    if isinstance(expected,list):
        if len(expected)!=len(actual):
            raise Mismatch(f"{key}: list length changed")
        for n,(a,b) in enumerate(zip(expected,actual)):
            check(a,b,f"{key}[{n}]",stats)
        return
    if isinstance(expected,float):
        if not math.isfinite(expected) or not math.isfinite(actual):
            raise Mismatch(f"{key}: nonfinite source or replay number")
        diff=abs(expected-actual)
        stats["total_float_fields"]+=1
        stats["maximum_absolute_float_difference"]=max(
            stats["maximum_absolute_float_difference"],diff)
        tolerance=ABS_TOL+REL_TOL*max(abs(expected),abs(actual))
        if key.startswith("REAL_WORLD_EVALUATION_SUMMARY.json") and (
                key.endswith("MAE") or "physical_MAE" in key or
                "TAC_minus_each_comparator" in key):
            tolerance=SUMMARY_ABS_TOL
        if diff>tolerance:
            raise Mismatch(f"{key}: expected={expected} actual={actual} diff={diff} tolerance={tolerance}")
        if diff>0:
            stats["float_fields_with_nonidentical_values"]+=1
        return
    if expected!=actual:
        raise Mismatch(f"{key}: exact factual nonfloat field differs")


def validate(published, replay):
    published=Path(published)
    replay=Path(replay)
    rows={}
    stats={"total_float_fields":0,"float_fields_with_nonidentical_values":0,
           "maximum_absolute_float_difference":0.}
    for name in FILES:
        pub=published/name
        rec=replay/name
        if not pub.is_file() or not rec.is_file():
            raise Mismatch(f"{name}: required original score JSON not present")
        frozen=json.loads(pub.read_text())
        recomputed=json.loads(rec.read_text())
        check(frozen,recomputed,name,stats)
        rows[name]={
            "published_sha256":hashlib.sha256(pub.read_bytes()).hexdigest(),
            "recomputed_sha256":hashlib.sha256(rec.read_bytes()).hexdigest(),
            "byte_identical":pub.read_bytes()==rec.read_bytes(),
        }
    original=json.loads((published/"REAL_WORLD_EVALUATION_SUMMARY.json").read_text())
    if original["study"]!="ME064_MULTI_TASK_REAL_ROBOT_LEAVE_ONE_TASK_OUT":
        raise Mismatch("incorrect original task scientific study")
    if original["untouched_evaluation_task"]!="install_spray_nozzle_onto_bottle":
        raise Mismatch("unexpected target task")
    if original["trained_original_git_checkpoint_count"]!=10:
        raise Mismatch("not all ten frozen models")
    if original["results"]["registered_superiority_gate_pass"] is not False:
        raise Mismatch("cannot rewrite failed registered science result")
    receipt={
        "study":original["study"],"status":"PASS_WITH_FLOAT_TOLERANCE",
        "schema_and_nonfloat_metadata_exact":True,
        "original_checkpoint_sha256_bitwise_verification":"PASS_PRIOR_MODEL_AUDIT",
        "score_json_byte_identical":all(r["byte_identical"] for r in rows.values()),
        "numeric_result_tolerance":{"abs":ABS_TOL,"rel":REL_TOL,
                "primary_summary_abs":SUMMARY_ABS_TOL},
        "floating_point_difference":stats,
        "input_files":rows,
        "original_target":original["untouched_evaluation_task"],
        "target_fits":0,"real_original_TEST_opened":False,
        "new_independent_hardware_site":False,
        "scientific_gate_pass":False,
        "primary_MAE":original["results"]["physical_MAE"],
    }
    return receipt


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--published",type=Path,required=True)
    p.add_argument("--recomputed",type=Path,required=True)
    p.add_argument("--receipt",type=Path,required=True)
    args=p.parse_args()
    result=validate(args.published,args.recomputed)
    args.receipt.parent.mkdir(parents=True,exist_ok=True)
    args.receipt.write_text(json.dumps(result,sort_keys=True,indent=2)+"\n")
    print("ME064_ORIGINAL_REAL_SCORE_REPLAY_FLOAT_TOLERANCE_PASS "+
          json.dumps({"max_diff":result["floating_point_difference"]["maximum_absolute_float_difference"],
              "byte_identical":result["score_json_byte_identical"],
              "scientific_gate_pass":False}),flush=True)


if __name__=="__main__":
    main()
