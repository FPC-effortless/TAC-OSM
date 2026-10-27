#!/usr/bin/env python3
"""GP-UNC-006: original UCI501 historical-FIT neural residual diagnostic.

Actual original ME111 source/checkpoints (five frozen seeds). GP sidecars
adapt on already-exposed *training-period* target labels; descriptive in-FIT
point-error only, NEVER independent prediction, probability calibration,
superiority, source-time permission, neural refit or protected TEST.

Source FIT loader pre-parses 2013–2014 including future FIT values; event
admission enforces only software chronology based on recorded hourly indices.
The original frozen neural checkpoints were themselves FIT-trained on those
outcomes; even sequential GP updates CANNOT be considered true out-of-sample.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
ORIGINAL_COMMIT="77c753a69f10fa13cd7fdf849edb12bc53f03e6d"
SOURCE_SHA="d1b9261c54132f04c374f762f1e5e512af19f95c95fd6bfa1e8ac7e927e3b0b8"
SEEDS=(111001,111002,111003,111004,111005)
BASIS_ORIGINS=(11968,11984)
FORECAST_ORIGINS=tuple(range(12000,12128,8))  # 16 frozen FIT-ONLY issue hours
HISTORY=64
STATIONS=12
CHANNELS=6
M=2
NOISE=.5  # frozen before examining any derived scores


def audit(native, output):
    native=native.resolve()
    if output.exists():
        raise ValueError("refuse overwrite source-original diagnostic receipt")
    if subprocess.check_output(["git","-C",str(native),"rev-parse","HEAD"],
                               text=True).strip()!=ORIGINAL_COMMIT:
        raise ValueError("original native source not immutable-commit pinned")
    sys.path[:0]=[str(native),str(native/"src"),str(ROOT/"src")]
    import numpy as np
    import torch
    import tac_osm
    tac_osm.__path__.append(str(ROOT/"src"/"tac_osm"))
    from scripts import native_latent_001_diagnosis as forensic
    from scripts import me111_conditional_training as original
    from tac_osm.gp_historical_fit_diagnostic import HistoricalFITSidecar, ARMS

    torch.set_num_threads(1)
    d=original.fit_data()
    if d["phase"]!="fit" or original.data.SHA!=SOURCE_SHA:
        raise ValueError("authentic source is not original FIT-only")
    if hashlib.sha256(original.data.ARCHIVE.read_bytes()).hexdigest()!=SOURCE_SHA:
        raise ValueError("official source ZIP SHA mismatch")
    if min(BASIS_ORIGINS)>=min(FORECAST_ORIGINS) or not all(
            earlier<min(FORECAST_ORIGINS) for earlier in BASIS_ORIGINS):
        raise ValueError("inducing addresses must freeze before first issue")
    if FORECAST_ORIGINS[-1]+6>=original.data.FIT_STOP:
        raise ValueError("attempt to score outside original FIT prohibited")
    if any(y-x<7 for x,y in zip(FORECAST_ORIGINS,FORECAST_ORIGINS[1:])):
        raise ValueError("source hours need strictly later label eligibility")
    sha_index=forensic.checked_index(original)
    if set(sha_index)!=set(SEEDS):
        raise ValueError("source-native original checkpoint index is not exact")

    result=[]
    for seed in SEEDS:
        model,checkpoint,sha=forensic.model_for(original,d,seed,sha_index[seed])
        versions=model.versions()
        unmodified_cp_hash=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        native_basis={}
        for t in BASIS_ORIGINS:
            frame=forensic.read_frame(d,[(station,t) for station in range(STATIONS)],HISTORY)
            with torch.no_grad():
                _,state=model(*frame[:6])
            memories=state.memory.detach().cpu().reshape(STATIONS,64).tolist()
            for station in range(STATIONS):
                for channel in range(CHANNELS):
                    key=f"station-{station:02d}-channel-{channel}"
                    native_basis.setdefault(key,[]).append(tuple(memories[station]))
        if any(len(v)!=M for v in native_basis.values()) or len(native_basis)!=72:
            raise ValueError("original two-address frozen source-native basis incomplete")
        sidecar=HistoricalFITSidecar(source_sha256=SOURCE_SHA,
                                    checkpoint_sha256=sha,
                                    native_inducing=native_basis,
                                    noise_variance=NOISE,
                                    max_records=len(FORECAST_ORIGINS)*STATIONS*CHANNELS)
        expected_issues=0
        # Real original FIT label is accessed ONLY after all source forecasts
        # at the corresponding hour have been issued. The entire FIT archive
        # had already been pre-parsed by the original trainer/loader.
        for t in FORECAST_ORIGINS:
            frame=forensic.read_frame(d,[(station,t) for station in range(STATIONS)],HISTORY)
            with torch.no_grad():
                forecast,state=model(*frame[:6])
            if not torch.isfinite(forecast.log_location).all():
                raise ValueError("native trained forecast nonfinite")
            memories=state.memory.detach().cpu().reshape(STATIONS,64).tolist()
            point=forecast.log_location.detach().cpu().tolist()
            for station in range(STATIONS):
                for channel in range(CHANNELS):
                    key=f"station-{station:02d}-channel-{channel}"
                    rowid=f"{seed}:{station}:{channel}:{t}"
                    sidecar.issue(dict(
                        kind="issue",row_id=rowid,entity_id=key,
                        features=tuple(memories[station]),
                        base_prediction=float(point[station][channel]),
                        feature_at=float(t),issued_at=float(t),
                        checkpoint_sha256=sha))
                    expected_issues+=1
            # Later original archived observation event, never in issue input.
            for station in range(STATIONS):
                for channel in range(CHANNELS):
                    rid=f"{seed}:{station}:{channel}:{t}"
                    raw=float(d["raw"][station,t+6,channel])
                    if not math.isfinite(raw):
                        sidecar.missing_label(rid)
                    else:
                        if raw<0:
                            raise ValueError("source negative pollutant concentration")
                        actual=(math.log1p(raw)-float(d["mean"][channel])) / float(d["std"][channel])
                        sidecar.label(dict(kind="label_arrived",row_id=rid,
                                           truth=actual,available_at=float(t+6)))
        report=sidecar.report()
        if (len(sidecar.seen)!=expected_issues or expected_issues!=1152 or
                sum((report["original_label_events_scored"],
                     report["original_missing_label_receipts"]))!=expected_issues or
                sidecar.updates < 1):
            raise ValueError("zero or mismatched true original FIT residual events")
        if model.versions()!=versions or hashlib.sha256(checkpoint.read_bytes()).hexdigest()!=unmodified_cp_hash:
            raise ValueError("original native checkpoint or parameters mutated")
        report.update(seed=seed,original_native_checkpoint_sha256=sha,
                      original_native_learned_parameters=sum(p.numel() for p in model.parameters()),
                      original_fit_issue_rows=expected_issues,
                      model_weight_optimizer_updates=0)
        result.append(report)
        print("GP_UNC006_SEED="+json.dumps({
            "seed":seed,"scored":report["original_label_events_scored"],
            "gp_updates":report["gp_real_fit_residual_update_events"],
            "mae_by_arm":{x:report["arms"][x]["mae_standardized_log_units"] for x in ARMS}},
            sort_keys=True,allow_nan=False),flush=True)
    pooled={a:{"n":sum(x["arms"][a]["n"] for x in result),
               "mae_standardized_log_units":sum(
                   x["arms"][a]["mae_standardized_log_units"]*x["arms"][a]["n"] for x in result)/
                    sum(x["arms"][a]["n"] for x in result),
               "rmse_standardized_log_units":math.sqrt(
                   sum(x["arms"][a]["rmse_standardized_log_units"]**2*x["arms"][a]["n"]
                       for x in result)/sum(x["arms"][a]["n"] for x in result))}
            for a in ARMS}
    receipt={
        "study":"GP-UNC-006",
        "evidence":"REAL_ORIGINAL_HISTORICAL_FIT_IN_SAMPLE_NONQUALIFYING",
        "original_source_commit":ORIGINAL_COMMIT,
        "original_publisher_archive_sha256":SOURCE_SHA,
        "original_trained_native_checkpoint_count":len(result),
        "original_trained_params_each":104663,
        "source_streams":STATIONS,
        "forecast_channels":CHANNELS,
        "basis_origin_hours":list(BASIS_ORIGINS),
        "evaluation_fit_only_issue_hours":list(FORECAST_ORIGINS),
        "original_fit_archive_was_preparsed":True,
        "original_neural_models_trained_on_scored_fit_targets":True,
        "label_availability_is_unverified_historical_hour_assumption":True,
        "gp_inducing_points_per_source_channel":M,
        "gp_feature_dimension":64,
        "noise_variance_fixed_prior_to_this_run":NOISE,
        "real_fit_issued_predictions":sum(x["original_fit_issue_rows"] for x in result),
        "real_fit_observed_labels":sum(x["original_label_events_scored"] for x in result),
        "real_fit_missing_outcomes":sum(x["original_missing_label_receipts"] for x in result),
        "real_fit_posterior_update_events":sum(x["gp_real_fit_residual_update_events"] for x in result),
        "arms":pooled,"models":result,
        "new_neural_model_fit":False,
        "original_checkpoints_unchanged":True,
        "no_source_fitted_hyperparameter_search":True,
        "matched_two_inducing_x64_GP_reference_arms":True,
        "independent_cohort":False,
        "protected_test_access":False,
        "probabilistic_calibration_tested":False,
        "six_family_comparative_qualification":False,
        "scientific_promotion":"NOT_QUALIFIED",
    }
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open("x",encoding="utf-8") as stream:
        json.dump(receipt,stream,sort_keys=True,indent=2,allow_nan=False)
    print("GP_UNC006_RESULT="+json.dumps(
        {k:v for k,v in receipt.items() if k!="models"},sort_keys=True,allow_nan=False))
    return receipt


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pinned-native-checkout",type=Path,required=True)
    parser.add_argument("--out",type=Path,required=True)
    args=parser.parse_args()
    audit(args.pinned_native_checkout,args.out)
