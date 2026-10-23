"""Compact checksum-linked summary of ORIGINAL no-fit DIAGKEY002 diagnosis.

Never score/retrain new data; read immutable original Git-owned all-case JSON.
"""
from __future__ import annotations
import hashlib,json
from pathlib import Path
import numpy as np

ROOT=Path("research/replay-bundles/DIAGKEY002")
SOURCE=ROOT/"ORACLE_AND_FROZEN_LADDER.json"
OUT=ROOT/"COMPACT_AUDITED_RESULTS.json"
DELAYS=("0","1","2","4","8","16","64","256")
ARMS=("tac","tac_noassoc","gru","transformer")
SEEDS=("75001","75002","75003","75004","75005")
def run():
    if OUT.exists():
        print("DIAGKEY002 original compact receipt exists; refuse overwrite",flush=True)
        return
    original=SOURCE.read_bytes()
    full=json.loads(original)
    ladder=full["original_frozen_ladder"]
    measured={}
    for delay in DELAYS:
        measured[delay]={}
        for arm in ARMS:
            rows=[ladder["original_seed_scores"][seed][arm][delay] for seed in SEEDS]
            correct=sum(row["correct"] for row in rows)
            n=sum(row["trials"] for row in rows)
            measured[delay][arm]={
                "five_original_seed_accuracy":[r["accuracy"] for r in rows],
                "mean_accuracy":float(np.mean([r["accuracy"] for r in rows])),
                "pooled_correct":correct,"pooled_trials":n,
                "pooled_fraction":correct/n,
                "mean_NLL":float(np.mean([r["mean_NLL"] for r in rows])),
                "note":"Five original synthetic case streams, no runtime training-guard qualification, not independent real observations",
            }
    oracle={}
    for scenario,rows in full["oracle"].items():
        oracle[scenario]={delay:{
            "accuracy":rows[delay]["accuracy"],
            "mean_true_read_amplitude":rows[delay]["mean_true_read_amplitude"],
            "mean_value_L2_error":rows[delay]["mean_value_L2_error"],
            "n":rows[delay]["n"],
        } for delay in DELAYS}
    summary={
        "study":"DIAGKEY002_NO_FIT_SOFTWARE_MEMORY_DIAGNOSTIC",
        "original_detailed_file":str(SOURCE),
        "original_detailed_SHA256":hashlib.sha256(original).hexdigest(),
        "original_study_qualifications":{
            "new_optimizer_steps":0,
            "real_world_data_used":False,
            "pilot_original_optimizer_budget_guard":"FAIL",
            "parameter_match_alone_is_not_compute_parity":True,
            "original_pilot_FLOP_parity":"FAIL",
            "original_trained_sequence_lengths":[16,64,256],
            "new_short_ladder_0_through_8":"OUT_OF_TRAINING_DISTRIBUTION",
            "new_model_training_success_claim":False,
        },
        "oracle":oracle,"original_frozen_model_ladder":measured,
        "original_frozen_gradient_paths":full["original_frozen_gradient_paths"],
        "scientific_decision":"Software mechanism diagnosis only. No real-world skill or qualified architectural superiority established.",
    }
    OUT.write_text(json.dumps(summary,sort_keys=True,indent=2)+"\n")
    print("DIAGKEY002_COMPACT_ACTUAL="+json.dumps({
        "original_SHA256":summary["original_detailed_SHA256"],
        "ladder":{delay:{arm:round(measured[delay][arm]["mean_accuracy"],5)
                         for arm in ARMS} for delay in DELAYS},
        "gradient_paths":summary["original_frozen_gradient_paths"]},sort_keys=True),flush=True)

if __name__=="__main__":run()
