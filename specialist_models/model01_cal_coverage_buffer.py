#!/usr/bin/env python3
"""Retrospective CAL-only coverage-buffer experiment on frozen real native TAC.

All width choices 90/94/96% are FIXED before this run and ALL reported,
NOT selected by old exposed DEV or future TEST. First chronological 60% CAL
learns signed source-phase residual bounds. Next 20% CAL and last 20% CAL
are separately reported; then *already exposed* DEV is scored for diagnosis.
No neural fitting, optimizer step, or independent coverage qualification.
"""
from __future__ import annotations
import argparse,json,math,sys
from pathlib import Path
import numpy as np
import torch
import model01_native_cal_only_intervals as previous

ROOT=Path(__file__).resolve().parents[1]
PUBLISHER_SHA=previous.native.SOURCE
LEVELS=(.90,.94,.96)
MIN_PHASE_COUNT=200

class ExperimentError(ValueError):pass

def buffered_intervals(signed_residuals,indices,central_mass):
    """Frozen conditional signed empirical bounds from fit-only CAL."""
    if central_mass not in LEVELS:
        raise ExperimentError("only predeclared nominal central masses are admissible")
    residuals=np.asarray(signed_residuals,dtype=np.float64)
    phase_indices=np.asarray(indices,dtype=np.int64)
    if (residuals.ndim!=1 or phase_indices.ndim!=1 or
        len(residuals)!=len(phase_indices) or not np.all(np.isfinite(residuals))):
        raise ExperimentError("nonfinite/mismatched CAL source")
    groups=previous.phase(phase_indices)
    lower=(1-central_mass)/2
    upper=1-lower
    offsets={}
    for j in range(4):
        block=residuals[groups==j]
        if len(block)<MIN_PHASE_COUNT:
            raise ExperimentError("not enough source observations in CAL phase group")
        a=previous.quantile(block,lower)
        b=previous.quantile(block,upper)
        if a>=b:
            raise ExperimentError("invalid signed phase range")
        offsets[str(j)]=[a,b]
    return offsets

def run():
    original=previous.immutable_checkpoints()
    raw=previous.native.uci.fetch_official_zip()
    if previous.native.uci.digest(raw)!=PUBLISHER_SHA:
        raise ExperimentError("publisher source changed")
    y=np.asarray(previous.native.uci.original_rows(raw),dtype=np.float64)
    mean=float(y[:previous.native.uci.FIT_END].mean())
    std=float(y[:previous.native.uci.FIT_END].std())
    if not math.isfinite(std) or std<=0:raise ExperimentError("invalid training scale")
    torch.set_num_threads(2)
    cal=previous.native.windows(
        y,previous.native.uci.eligible_indices(previous.native.uci.FIT_END,previous.native.uci.CAL_END),mean,std)
    dev=previous.native.windows(
        y,previous.native.uci.eligible_indices(previous.native.uci.CAL_END,previous.native.uci.DEV_END),mean,std)
    first,second,third=previous.partition(len(cal[2]))
    def piece(sl):
        return tuple(v[sl] for v in cal)
    records=[]
    for frozen in original["runs"]:
        net=previous.restore(frozen)
        fp,fy,fi=previous.predict(net,piece(first),mean,std)
        sp,sy,si=previous.predict(net,piece(second),mean,std)
        ap,ay,ai=previous.predict(net,piece(third),mean,std)
        dp,dy,di=previous.predict(net,dev,mean,std)
        prior=previous.score([-frozen["q90_wh"],frozen["q90_wh"]],dp,dy,di)
        if (not math.isclose(prior["interval_score90"],frozen["dev"]["interval_score90"],rel_tol=5e-6,abs_tol=1e-3)
            or not math.isclose(prior["coverage90"],frozen["dev"]["nominal90_observed_coverage"],abs_tol=1e-10)):
            raise ExperimentError("original neural DEV frozen score mismatch")
        configs={str(round(100*level)):buffered_intervals(fy-fp,fi,level) for level in LEVELS}
        rows={}
        for name,cfg in configs.items():
            rows[name]={
                "CAL_SELECT_report_only":previous.score(cfg,sp,sy,si),
                "CAL_AUDIT_report_only":previous.score(cfg,ap,ay,ai),
                "EXPOSED_DEV_report_only":previous.score(cfg,dp,dy,di)
            }
        records.append({
            "seed":frozen["seed"],
            "original_model_checkpoint_sha256":frozen["checkpoint_sha256"],
            "CAL_phase_offsets_wh":configs,
            "per_preset_level":rows,
            "original_full_CAL_90_interval_EXPOSED_DEV":prior,
            "no_candidate_chosen":True,
            "weights_unchanged":True,
            "original_point_MAE_Wh":frozen["dev"]["mae_wh"],
            "original_point_RMSE_Wh":frozen["dev"]["rmse_wh"]
        })
    return {
        "schema":"MODEL01-TAC-CAL-COVERAGE-BUFFER-001-original-real-source-v1",
        "study":"MODEL01-TAC-CAL-COVERAGE-BUFFER-001",
        "source_zip_sha256":PUBLISHER_SHA,
        "original_git_neural_checkpoint":"research/replay-bundles/MODEL01-TAC-REAL-FIT-001/ORIGINAL_FIT.json",
        "original_neural_trainable_params":103252,
        "all_prespecified_interval_central_masses":list(LEVELS),
        "CAL_chronological_fit_select_audit_ratios":[.6,.2,.2],
        "no_model_fitting":True,"no_dev_based_candidate_selection":True,
        "source_protected_tail_parsed":False,
        "source_available_at_timestamp_verified":False,
        "DEV_was_already_exposed_before_study":True,
        "independent_qualification":False,
        "global_release_coverage_gate_passed":False,
        "new_intervals_production_authorized":False,
        "confidence_guarantee":"NONE_TIME_SERIES_TEMPORAL_DEPENDENCE",
        "runs":records,
        "limitations":[
            "buffered 94/96% candidates designed after prior 90% approach undercovered exposed DEV",
            "all three central masses are reported, not selected on previously exposed outcomes",
            "chronologically neighboring CAL fragments and DEV from single repeatedly studied household",
            "quantiles assume nothing about future nonexchangeable sensor regimes",
            "coverage at nominal 90% and interval scores are retrospective, not verified operational uncertainty",
            "phase indices tied to original source origin; nontransferrable to independent site automatically",
            "no new customer, genuine fresh cohort, online source timestamp receipt or TAC memory benefit"
        ]
    }

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args(argv)
    try:
        if a.out.exists():raise ExperimentError("refusing original evidence overwrite")
        receipt=run()
        a.out.parent.mkdir(parents=True,exist_ok=True)
        a.out.write_text(json.dumps(receipt,sort_keys=True,indent=2,allow_nan=False)+"\n")
        print(json.dumps({"schema":receipt["schema"],
                          "runs":[{"seed":r["seed"],"levels":{
                              k:{"CAL_AUDIT":v["CAL_AUDIT_report_only"],
                                 "DEV":v["EXPOSED_DEV_report_only"]}
                              for k,v in r["per_preset_level"].items()},
                              "original_DEV":r["original_full_CAL_90_interval_EXPOSED_DEV"]}
                              for r in receipt["runs"]]},sort_keys=True,allow_nan=False))
        return 0
    except (ExperimentError,previous.IntegrityError,previous.native.StudyError,
            previous.native.uci.SourceError,OSError,ValueError,TypeError,RuntimeError) as ex:
        print(json.dumps({"status":"BLOCKED_OR_FAILED","reason":str(ex)},file=sys.stderr))
        return 2

if __name__=="__main__":
    sys.exit(main())
