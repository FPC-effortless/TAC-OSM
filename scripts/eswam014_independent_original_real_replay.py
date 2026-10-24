"""Independent no-optimizer publisher-real replay of two guarded ESWAM014 fits."""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
import torch

from scripts.eswam012_train_original_multiagent import original_cases, all_original_case_rows
from scripts.eswam014_guarded_original_real import MANIFEST, OUT, SEEDS, ARCH, PARAMETERS
from tac_osm.parameter_budget import file_sha, validate_runtime_evidence
from tac_osm.variants.eswam014_quadratic_motion import QuadraticMotionNative


def replay_original_real(archive:Path, output:Path):
    torch.set_num_threads(2)
    source, rejected = original_cases(archive)
    dev = source[3]
    original_fit_result=json.loads((OUT/"FINAL_RESULTS.json").read_text())
    if len(original_fit_result["original_guarded_fits"])!=2:
        raise RuntimeError("no complete original real fitted source checkpoint set")
    report={"study":"ESWAM014_INDEPENDENT_ZERO_TRAINING_REAL_SOURCE_REPLAY",
            "original_fitted_checkpoint_count":2,
            "original_publisher_sha256":original_fit_result["source_archive_sha256"],
            "original_recording_is_PREVIOUSLY_EXPOSED_DEV":True,
            "source_independent_site_TEST":False,
            "new_optimizer_steps":0,
            "tolerance_meters":0.000005,
            "all_original_source_actor_predictions_checked":True,
            "results":[]}
    for frozen in original_fit_result["original_guarded_fits"]:
        seed=frozen["seed"]
        if seed not in SEEDS or frozen["architecture"]!=ARCH:
            raise RuntimeError("fitted source identity changed")
        ck=OUT/f"seed-{seed}.pt"
        old=OUT/f"seed-{seed}-ORIGINAL-REAL-DEV-CASES.json"
        receipt=OUT/f"live-guard-{seed}.json"
        if file_sha(ck)!=frozen["fitted_checkpoint_sha256"]:
            raise RuntimeError("original real trained checkpoint bytes changed")
        if file_sha(old)!=frozen["original_real_DEV_predictions_sha256"]:
            raise RuntimeError("original real predicted source score bytes changed")
        guard=validate_runtime_evidence({
            "seed":seed,"architecture":ARCH,
            "trainable_parameters":PARAMETERS,"deployed_parameters":PARAMETERS,
            "checkpoint_sha256":file_sha(ck),
            "_prefit_registration_sha256":file_sha(MANIFEST),
            "_expected_optimizer_steps":320,
            "runtime_evidence":{
                "receipt_path":str(receipt),"receipt_sha256":file_sha(receipt),
                "checkpoint_path":str(ck),"checkpoint_sha256":file_sha(ck)},
        })
        serialized=torch.load(ck,map_location="cpu",weights_only=False)
        if serialized["seed"]!=seed or serialized["actual_guarded_optimizer_steps"]!=320:
            raise RuntimeError("guarded optimization receipt mismatched")
        model=QuadraticMotionNative()
        model.load_state_dict(serialized["weights"],strict=True)
        fresh=all_original_case_rows(model,dev,"eswam014")
        saved=json.loads(old.read_text())
        if len(fresh)!=len(saved):
            raise RuntimeError("independent real original-case count changed")
        maxdiff=0.
        for actual,record in zip(fresh,saved):
            for label in ("source_original_recording","source_original_frame",
                          "source_original_timestamp_ms","source_original_actor_id",
                          "horizon_frames","source_actor_cluster"):
                if actual[label]!=record[label]:
                    raise RuntimeError("source original case ID/ordering is not reproducible")
            for label in ("true_original_future_xy_m","observed_current_xy_m",
                          "model_predicted_xy_m","predicted_laplace_scale_xy"):
                if len(actual[label])!=len(record[label]):
                    raise RuntimeError("original physical coordinate shape changed")
                for new,previous in zip(actual[label],record[label]):
                    if not (math.isfinite(new) and math.isfinite(previous)):
                        raise RuntimeError("nonfinite original source predicted state")
                    maxdiff=max(maxdiff,abs(new-previous))
            for name in ("model","prior","p0","p8","p256"):
                maxdiff=max(maxdiff,abs(actual["l2_meters"][name]-
                                        record["l2_meters"][name]))
        if maxdiff>report["tolerance_meters"]:
            raise RuntimeError(f"cross-execution real case score drift above preregistered tolerance: {maxdiff}")
        outcome={"seed":seed,"architecture":ARCH,"original_REAL_CASES":len(fresh),
                 "original_actor_tracks":len({r["source_actor_cluster"] for r in fresh}),
                 "original_checkpoint_sha256":file_sha(ck),
                 "independent_full_optimizer_guard_receipt_id":guard,
                 "max_abs_original_predictions_scales_and_errors_meters":maxdiff,
                 "PASS":True}
        print("ESWAM014_INDEPENDENT_REAL_REPLAY="+json.dumps(outcome,sort_keys=True),flush=True)
        report["results"].append(outcome)
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(report,sort_keys=True,indent=2)+"\n")
    print("ESWAM014_INDEPENDENT_REPLAY_ALL_PASS",flush=True)


if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--archive",type=Path,required=True)
    ap.add_argument("--output",type=Path,required=True)
    args=ap.parse_args()
    replay_original_real(args.archive,args.output)
