"""DEV-052 independent native Veritas action+receipt audit of frozen DEV-051 transfer."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import statistics
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
import torch
from tac_osm.development.noisy_veritas_transfer import study_cohort,summarize
from tac_osm.development.learned_veritas_policy import digest
from run_noisy_veritas_dev052 import PROTOCOL,POLICIES,load_frozen,evaluate


def audit_one(directory,dev051):
    folder=Path(directory)
    report=json.loads((folder/"report.json").read_text())
    proto=json.loads(PROTOCOL.read_text())
    assert report["study"]=="DEV-052A" and not report["smoke"]
    assert report["status"]=="FROZEN_TRANSFER_DEVELOPMENT_NOT_CONFIRMATORY"
    assert report["protocol_sha256"]==digest(PROTOCOL)
    seed=report["seed"]
    assert seed in proto["source"]["trained_seeds"]
    nets,original=load_frozen(seed,dev051)
    assert original==report["dev051_checkpoint"]
    eval_seeds=proto["data"]["evaluation_seeds"]
    worlds=proto["data"]["evaluation_worlds_per_seed"]
    evaluation=list(study_cohort(eval_seeds,worlds))
    cal=list(study_cohort(proto["data"]["calibration_seeds"],
                          proto["data"]["calibration_worlds_per_seed"]))
    assert [x.episode_id for _,_,x in cal]==report["split_ids"]["calibration"]
    assert [x.episode_id for _,_,x in evaluation]==report["split_ids"]["evaluation"]
    assert set(report["split_ids"]["calibration"]).isdisjoint(report["split_ids"]["evaluation"])
    again=evaluate(evaluation,nets,original["original_gate"])
    assert set(again)==set(report["evaluation"])==set(POLICIES)
    for name in POLICIES:
        old=report["evaluation"][name]
        fresh=again[name]
        assert len(old["rows"])==len(fresh["rows"])==128
        # No statistical tolerance for native discrete actions, receipts or
        # verifier. Only wall clock excluded from strict replay.
        assert fresh==old,(seed,name,"native physical replay drift")
        assert fresh["summary"]==summarize(fresh["rows"])
    return dict(seed=seed,sha=report["source"],
                policies={p:report["evaluation"][p]["summary"] for p in POLICIES},
                original_checkpoint=original["original_sha256"])


def run(paths,dev051,out):
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    a=[audit_one(p,dev051) for p in paths]
    assert sorted(x["seed"] for x in a)==[7101,7102,7103]
    assert len({x["sha"] for x in a})==1
    result={}
    for policy in POLICIES:
        vals=[x["policies"][policy] for x in a]
        result[policy]=dict(mean_cost_per_entity=statistics.mean(
             x["cost_per_entity"] for x in vals),
             residual_faults_across_model_seeds=sum(x["residual_faults"] for x in vals),
             complete_worlds_across_model_seeds=sum(x["completed_worlds"] for x in vals),
             blocked_actions_across_model_seeds=sum(x["blocked_actions"] for x in vals),
             mean_sensing_calls=statistics.mean(x["mean_inspections"] for x in vals))
    reference=result["bayesian_dp"]
    guards={}
    for p in ("bc_frozen","dagger_frozen","dagger_old_gate_frozen"):
        x=result[p]
        guards[p]=(x["residual_faults_across_model_seeds"]<=reference["residual_faults_across_model_seeds"] and
                    x["blocked_actions_across_model_seeds"]<=reference["blocked_actions_across_model_seeds"] and
                    x["mean_cost_per_entity"]<reference["mean_cost_per_entity"])
    summary=dict(study="DEV-052A",status="AUDITED_ADAPTIVE_DEVELOPMENT_NOT_CONFIRMATORY",
        seeds=a,summary=result,passed_cost_and_failure_guard=guards,
        source_frozen_and_verified=True,independent_training_replay=False,
        no_hidden_state_at_policy_boundary=True,
        same_worlds_reused_across_model_seeds=True,
        note="A frozen DEV-051 BC/DAgger policy is not a learned OSM trained on noisy repairs.")
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    (out/"analysis.json").write_text(json.dumps(summary,indent=2,allow_nan=False)+"\n")
    print(json.dumps(summary,indent=2,allow_nan=False),flush=True)


if __name__=="__main__":
    q=argparse.ArgumentParser()
    q.add_argument("seed_paths",nargs=3)
    q.add_argument("--dev051",required=True)
    q.add_argument("--output",required=True)
    x=q.parse_args()
    run(x.seed_paths,x.dev051,x.output)
