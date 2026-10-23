"""ME063 original frozen model physical source+transfer scoring; NO REFITTING.

Loads checksum-verified Git-native checkpoint BYTES from clean checkout.
CAL calibrates the source-only intervals; historical DEV is NOT independent.
A separately metadata-registered different task is frozen before trajectory open.
"""
from __future__ import annotations

import argparse
import io
import json
import math
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch

from scripts.audit_real_hil_trajectory import audit_frame_table, download_file
from scripts.me060_train_trajectory import cohorts, evaluate, digest, HISTORY, HORIZON
from scripts.me057_train_fused_anchor_robot import load_part, normalize, sha_bytes, sha_file, write_json
from scripts.me063_prefit_original_fit import model_for, SEEDS, CONFIG
from scripts.me055_reserve_independent_robot_task import REVISION

BUNDLE = Path("research/replay-bundles/ME063")
RESEARCH = Path("research/mathematical-engineering/me063")


def complete_checkpoint_inventory():
    trained = json.loads((BUNDLE / "TRAINING_RESULT.json").read_text())
    audited = json.loads((BUNDLE / "CLEAN_CHECKOUT_MODEL_REPLAY.json").read_text())
    if audited["model_replay"] != "PASS" or audited["checkpoints_restored"] != 10:
        raise RuntimeError("independent Git checkpoint restore has not passed")
    recorded = trained["checkpoints"]
    expected = {(seed, arm) for seed in SEEDS for arm in CONFIG}
    if len(recorded) != 10 or {(r["seed"],r["arm"]) for r in recorded} != expected:
        raise RuntimeError("required original real-FIT models missing")
    for record in recorded:
        if sha_file(BUNDLE/"checkpoints"/Path(record["path"]).name) != record["checkpoint_sha256"]:
            raise RuntimeError("original frozen checkpoint bytes changed")
    return recorded


def load_model(record):
    checkpoint = torch.load(record["path"], map_location="cpu", weights_only=False)
    if checkpoint["scoring_status"] != "NOT_EVALUATED_HOLDOUT_UNOPENED":
        raise RuntimeError("checkpoint did not freeze before any scoring")
    identified={key:np.asarray(value,dtype=np.float32)
                for key,value in checkpoint["identified"].items()}
    model=model_for(record["arm"],identified)
    model.load_state_dict(checkpoint["model"],strict=True)
    model.eval()
    return model, identified


def evaluate_with_regime_receipts(model, original, identified):
    """Same original physical evaluator plus causal router weights for each episode."""
    prediction = evaluate(model, original, identified)
    model.eval()
    with torch.no_grad():
        for ep, measured in sorted(prediction.items(), key=lambda x: int(x[0])):
            row = original[int(ep)]
            origin = measured["origin"]
            belief = model.initial(1)
            for j in range(origin-HISTORY, origin):
                belief=model.observe(
                    belief, row["state"][j:j+1],row["action"][j:j+1],
                    row["state"][j+1:j+2],row["human"][j:j+1,None])
            forecast=model.forecast(
                belief, row["state"][origin:origin+1],
                row["action"][None,origin:origin+HORIZON])
            weights=forecast.weights[0]
            measured["regime_weights_mean"]=weights.mean(0).tolist()
            measured["regime_weights_by_horizon"]=weights.tolist()
    return prediction


def source_scores(cache, records):
    split, metadata = cohorts(cache)
    scores = {"CAL":{}, "DEV":{}}
    source_hashes={}
    for group in ("CAL","DEV"):
        original=load_part(cache,split[group],metadata,"ME063_FROZEN_"+group)
        source_hashes[group]=digest({str(e):r["sha256"] for e,r in sorted(original.items())})
        for record in records:
            model,identified=load_model(record)
            scores[group][f'{record["arm"]}-s{record["seed"]}']=evaluate_with_regime_receipts(
                model,normalize(original,identified),identified)
        print(json.dumps({"event":"ME063_SOURCE_EVALUATION_FINISHED",
                          "group":group,"episodes":len(original)}),flush=True)
    write_json(BUNDLE/"ORIGINAL_SOURCE_SCORES.json",{
        "study":"ME063_HISTORICALLY_EXPOSED_SOURCE_DEVELOPMENT",
        "scores":scores,"source_receipts":source_hashes,
        "old_TEST_opened":False,"refit_updates":0,"independent_qualification":False})
    return scores


def frozen_target(cache, records):
    registration=json.loads((RESEARCH/"TRANSFER_PROTOCOL.json").read_text())
    if registration["target_trajectory_files_opened"] or registration["target_outcome_data_opened"]:
        raise RuntimeError("prospective metadata-only registration invalid")
    task=registration["task"]
    meta_bytes=download_file(REVISION,f"{task}/meta/episodes.jsonl",
                             cache/task/"meta/episodes.jsonl",2_000_000)
    if sha_bytes(meta_bytes)!=registration["episodes_metadata_sha256"]:
        raise RuntimeError("frozen real target metadata SHA256 mismatch")
    info_bytes=download_file(REVISION,f"{task}/meta/info.json",
                             cache/task/"meta/info.json",1_000_000)
    if sha_bytes(info_bytes)!=registration["info_sha256"]:
        raise RuntimeError("frozen target physical schema/info changed")
    meta={row["episode_index"]:row for row in (json.loads(s) for s in meta_bytes.splitlines())}
    original={}
    for index,(e,n) in enumerate(registration["cohort"],1):
        path=f"{task}/data/chunk-{e//1000:03d}/episode_{e:06d}.parquet"
        raw=download_file(REVISION,path,cache/path,15_000_000)
        table=pq.read_table(io.BytesIO(raw))
        audit_frame_table(table,meta[e])
        if table.num_rows!=n:
            raise RuntimeError("frozen target episode length changed")
        original[e]={
            "state":np.asarray(table["observations.state.qpos"].to_pylist(),dtype=np.float32),
            "action":np.asarray(table["action.qpos"].to_pylist(),dtype=np.float32),
            "human":np.asarray(table["intervention"].to_pylist(),dtype=np.bool_),
            "length":n,"sha256":sha_bytes(raw)}
        if index%20==0:
            print(json.dumps({"event":"ME063_FROZEN_REAL_TRANSFER_SOURCE",
                              "episodes_downloaded":index,"total":len(registration["cohort"])}),flush=True)
    receipt=digest({str(e):r["sha256"] for e,r in sorted(original.items())})
    outputs={}
    for record in records:
        model,identified=load_model(record)
        outputs[f'{record["arm"]}-s{record["seed"]}']=evaluate_with_regime_receipts(
            model,normalize(original,identified),identified)
    write_json(BUNDLE/"ORIGINAL_TRANSFER_SCORES.json",{
        "study":"ME063_FROZEN_DIFFERENT_TASK_TRANSFER",
        "task":task, "original_episode_count":len(original),
        "source_sha256":receipt,
        "metadata_registration":registration,
        "checkpoint_sha256":[r["checkpoint_sha256"] for r in records],
        "refit_updates":0,"target_preprocessing_fitted":False,
        "target_calibration_fitted":False,"source_TEST_opened":False,
        "scores":outputs,
        "independent_site_operator_qualification":False})
    return outputs,receipt,task


def summarize(episode_tables, group, source_cal=None, target=False):
    # Mean seed error is computed for each episode, followed by an episode
    # bootstrap. Five fitted seeds are not treated as 5 independent populations.
    ids=sorted(next(iter(episode_tables.values())),key=int)
    errors={arm:np.asarray([
        [episode_tables[f"{arm}-s{s}"][e]["MAE"] for e in ids] for s in SEEDS])
        for arm in CONFIG}
    by_ep={name:data.mean(axis=0) for name,data in errors.items()}
    means={name:float(values.mean()) for name,values in by_ep.items()}
    reference=next(iter(episode_tables.values()))
    means["persistence"]=float(np.mean([reference[e]["persistence_MAE"] for e in ids]))
    means["servo"]=float(np.mean([reference[e]["servo_MAE"] for e in ids]))
    # Adjacent episodes share recording blocks. Cluster bootstrap by original
    # 4-episode provenance block, not falsely independent episode rows.
    rng=np.random.default_rng(63063 if not target else 63064)
    blocks=sorted({int(e)//4 for e in ids})
    if len(blocks)<10:
        raise ValueError("insufficient independent recording blocks for registered CI")
    reference_episodes={
        "gru_world":by_ep["gru_world"],
        "persistence":np.asarray([reference[e]["persistence_MAE"] for e in ids]),
        "servo":np.asarray([reference[e]["servo_MAE"] for e in ids]),
    }
    intervals={}
    for comparator,reference_values in reference_episodes.items():
        diff=by_ep["native_tac"]-reference_values
        block_means=np.asarray([
            diff[np.asarray([int(e)//4==block for e in ids])].mean()
            for block in blocks])
        sampled=rng.integers(len(block_means),size=(20000,len(block_means)))
        interval=np.quantile(block_means[sampled].mean(axis=1),[.025,.975]).tolist()
        intervals[comparator]=interval
    ci=intervals["gru_world"]
    seed_wins=sum(bool(a<b) for a,b in zip(
        errors["native_tac"].mean(axis=1),errors["gru_world"].mean(axis=1)))
    gate=bool(all(
        means["native_tac"]<=.98*means[name] and intervals[name][1]<0
        for name in reference_episodes))
    routing={}
    for arm in CONFIG:
        seed_route=[]
        for seed in SEEDS:
            observations=episode_tables[f"{arm}-s{seed}"]
            weights=np.asarray([observations[e]["regime_weights_mean"] for e in ids])
            seed_route.append(weights.mean(0))
        routing[arm]={
            "mean_weights_persistence_servo_neural":np.mean(seed_route,axis=0).tolist(),
            "per_seed_weights":[row.tolist() for row in seed_route],
        }
    result={
        "population":group,"episodes":len(ids),"mean_physical_MAE":means,
        "TAC_paired_seed_wins":seed_wins,
        "TAC_minus_GRU_episode_bootstrap_95_interval":ci,
        "TAC_minus_each_comparator_recording_block_95_intervals":intervals,
        "recording_block_count":len(blocks),
        "registered_performance_gate_pass":gate,
        "per_seed_MAE":{arm:values.mean(axis=1).tolist() for arm,values in errors.items()},
        "development_not_independent":not target,
        "transfer_new_different_task_not_separate_site":target,
        "trained_checkpoints":10,"transfer_training_updates":0,
        "calibration":{},
        "routing":routing,
    }
    if source_cal is not None:
        for arm in CONFIG:
            rows=[]
            for seed in SEEDS:
                key=f"{arm}-s{seed}"
                cal=list(source_cal[key].values())
                cal_scores=np.asarray([max(r["scaled_errors"]) for r in cal])
                rank=min(len(cal),math.ceil((len(cal)+1)*.90))
                q=float(np.sort(cal_scores)[rank-1])
                eval_rows=[episode_tables[key][e] for e in ids]
                rows.append({
                    "seed":seed,"source_CAL_episodes":len(cal),
                    "nominal_90_whole_episode":.90,"quantile":q,
                    "episode_coverage":float(np.mean([max(r["scaled_errors"])<=q for r in eval_rows])),
                    "mean_physical_interval_width":float(np.mean([2*q*r["scale_mean"] for r in eval_rows])),
                })
            result["calibration"][arm]={
                "per_seed":rows,
                "mean_whole_episode_coverage":float(np.mean([r["episode_coverage"] for r in rows])),
                "mean_physical_interval_width":float(np.mean([r["mean_physical_interval_width"] for r in rows])),
                "coverage_guarantee":"UNQUALIFIED_EXCHANGEABILITY_NOT_VERIFIED",
            }
    return result


def run(cache):
    torch.set_num_threads(1)
    records=complete_checkpoint_inventory()
    source=source_scores(cache,records)
    source_result=summarize(source["DEV"],"HISTORICALLY_EXPOSED_SOURCE_DEV",source["CAL"])
    transfer,source_hash,task=frozen_target(cache,records)
    transfer_result=summarize(transfer,"FROZEN_DIFFERENT_PHYSICAL_TASK",source["CAL"],target=True)
    summary={
        "study":"ME063_REAL_PHYSICAL_FORECASTING_AFTER_GIT_CHECKPOINT_FREEZE",
        "registered_training_resource_parity":"PASS",
        "original_checkpoints_git_clean_checkout_verified":10,
        "source_dev":source_result,"frozen_transfer":transfer_result,
        "transfer_task":task,"transfer_source_sha256":source_hash,
        "original_source_test_opened":False,
        "memory_necessity_qualified":False,
        "policy_value_or_autonomy_evaluated":False,
        "true_all_operator_flops_certified":False,
        "qualification":"UNQUALIFIED_UNLESS_EACH_PREDECLARED_GATE_PASSES",
    }
    write_json(BUNDLE/"REAL_WORLD_EVALUATION_SUMMARY.json",summary)
    print("ME063_REAL_WORLD_SCORE="+json.dumps({
        "source":source_result["mean_physical_MAE"],
        "source_gate":source_result["registered_performance_gate_pass"],
        "transfer":transfer_result["mean_physical_MAE"],
        "transfer_gate":transfer_result["registered_performance_gate_pass"],
        "source_CI":source_result["TAC_minus_GRU_episode_bootstrap_95_interval"],
        "transfer_CI":transfer_result["TAC_minus_GRU_episode_bootstrap_95_interval"],
        "calibration_source":source_result["calibration"],
        "calibration_transfer":transfer_result["calibration"],
        "independent_test":False,
    },sort_keys=True),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--publisher-cache",type=Path,required=True)
    a=p.parse_args()
    run(a.publisher_cache)
