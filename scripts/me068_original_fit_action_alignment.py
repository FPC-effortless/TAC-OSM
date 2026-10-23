"""ME068 real original FIT-only robot command/state temporal alignment diagnostic.

No training, no CAL, no TEST, no task outside original five FIT sources.
Audit factual robot joint transitions against FIT-only identified absolute
14D commanded action qpos using fixed candidate temporal offsets.
Negative lag uses FUTURE logged actions strictly as a hindsight timestamp
DIAGNOSTIC and is never available to a deployment forecaster.
"""
from __future__ import annotations
import argparse, json, hashlib
from collections import defaultdict
from pathlib import Path
import numpy as np

from scripts.me064_mixed_real_fit import fit_metadata

OUT=Path("research/mathematical-engineering/me068")
LAGS=(-2,-1,0,1,2,4,8)
SOURCE_SHA256="4d01753b2de68ef2eb863b94aeb790f6b2f98c6068aede3bf60e315f72a8b776"
MIN_VALID_SAMPLES=300

def individual_episode(row, identified):
    x=row["state"].numpy().astype(np.float64)
    a=row["action"].numpy().astype(np.float64)
    human=row["human"].numpy().astype(bool)
    n=len(x)
    # Fixed common origins for all candidates; no cross-episode transitions,
    # no authority transition in the whole {-2,...,8} audit window.
    tt=np.arange(8,n-3)
    if not len(tt):
        return None
    valid=np.asarray([not bool(human[t-8:t+3].any()) for t in tt])
    tt=tt[valid]
    if not len(tt):
        return None
    current=x[tt]
    actual=x[tt+1]
    scale=np.asarray(identified["state_scale"],np.float64)
    mean=np.asarray(identified["state_mean"],np.float64)
    astd=np.asarray(identified["action_scale"],np.float64)
    amean=np.asarray(identified["action_mean"],np.float64)
    gain=np.asarray(identified["servo_gain"],np.float64)
    limit=np.asarray(identified["servo_step_limit"],np.float64)
    abs_move=np.abs((actual-current)*scale)
    persist_errors=abs_move.mean(axis=1)
    metrics={
        "valid_original_autonomous_transitions":len(tt),
        "persistence_one_step_physical_MAE":float(persist_errors.mean()),
        "mean_observed_physical_qpos_change":float(abs_move.mean()),
        "fraction_physically_static_0_025_mean_joint_change":float(np.mean(
            abs_move.mean(axis=1)<.025)),
        "lag_scores":{},
    }
    for lag in LAGS:
        # Our sign convention: +k is an action logged k frames BEFORE t;
        # -k is a future logged action and MUST NEVER be used at inference.
        past_or_future=a[tt-lag]
        target=(past_or_future*astd+amean-mean)/scale
        delta=np.clip(gain*(target-current),-limit,limit)
        forecast=current+delta
        error=np.abs((forecast-actual)*scale).mean(axis=1)
        norm=np.linalg.norm(delta,axis=1)*np.linalg.norm(actual-current,axis=1)
        cos=np.sum(delta*(actual-current),axis=1)/np.maximum(norm,1e-12)
        mask=norm>1e-10
        metrics["lag_scores"][str(lag)]={
            "one_step_servo_physical_MAE":float(error.mean()),
            "mean_gain_over_persistence_MAE":float((persist_errors-error).mean()),
            "mean_motion_direction_cosine_nonzero":(
                float(cos[mask].mean()) if bool(mask.any()) else None),
            "usable_by_causal_forecaster":lag>=0,
        }
    return metrics

def aggregate(episodes):
    if not episodes:
        return None
    total=sum(d["valid_original_autonomous_transitions"] for d in episodes)
    if not total:
        return None
    weights=np.asarray([d["valid_original_autonomous_transitions"] for d in episodes])
    weights=weights/weights.sum()
    avg=lambda values:float(np.dot(weights,np.asarray(values,np.float64)))
    lags={}
    for lag in LAGS:
        rows=[d["lag_scores"][str(lag)] for d in episodes]
        lags[str(lag)]={
            "original_frame_weighted_servo_physical_MAE":avg([
                d["one_step_servo_physical_MAE"] for d in rows]),
            "original_frame_weighted_servo_minus_persistence_MAE":avg([
                -d["mean_gain_over_persistence_MAE"] for d in rows]),
            "lagged_action_is_causally_available":lag>=0,
        }
    causal=min((lag for lag in LAGS if lag>=0),
               key=lambda lag:lags[str(lag)]["original_frame_weighted_servo_physical_MAE"])
    hindsight=min(LAGS,key=lambda lag:lags[str(lag)]["original_frame_weighted_servo_physical_MAE"])
    return {
        "genuine_original_fit_episode_count":len(episodes),
        "valid_autonomous_transitions":int(total),
        "one_step_persistence_MAE":avg([d["persistence_one_step_physical_MAE"] for d in episodes]),
        "near_stationary_origin_fraction_threshold_is_diagnostic_only":avg([
            d["fraction_physically_static_0_025_mean_joint_change"] for d in episodes]),
        "per_action_offset":lags,
        "best_FIT_observational_causal_action_lag":causal,
        "best_FIT_observational_hindsight_action_lag":hindsight,
        "best_lag_is_not_heldout_validated":True,
    }

def main(cache):
    plan,identified,source,receipt,orders=fit_metadata(cache)
    if receipt["fit_original_sha256"]!=SOURCE_SHA256 or len(source)!=160:
        raise RuntimeError("ME068 original source FIT-only cohort identity not verified")
    by_task=defaultdict(list)
    all_data=[]
    provenance={}
    for (task,episode),row in sorted(source.items()):
        measurements=individual_episode(row,identified)
        if measurements is None:
            continue
        by_task[task].append(measurements)
        all_data.append(measurements)
        provenance[f"{task}:{episode}"]={
            "original_parquet_sha256":receipt["per_task_fit_sha256"][task],
            "valid_unintervened_transitions":measurements["valid_original_autonomous_transitions"]
        }
    summary=aggregate(all_data)
    if not summary or summary["valid_autonomous_transitions"]<MIN_VALID_SAMPLES:
        raise RuntimeError("original real autonomous physical transitions insufficient")
    result={
        "study":"ME068_ORIGINAL_REAL_FIT_ACTION_STATE_TIME_ALIGNMENT",
        "status":"FIT_ONLY_OBSERVATIONAL_DIAGNOSTIC_NOT_INDEPENDENT",
        "original_source_fit_sha256":receipt["fit_original_sha256"],
        "original_fit_episode_count":len(source),
        "actual_scored_source_episode_count":len(all_data),
        "original_CAL_opened":False,
        "old_exposed_TEST_opened":False,
        "reserved_tissue_box_task_opened":False,
        "fitted_new_models":0,
        "optimizer_updates":0,
        "original_train_only_identification":{"servo_gain":np.asarray(identified["servo_gain"]).tolist(),
            "servo_step_limit":np.asarray(identified["servo_step_limit"]).tolist(),
            "state_scale":np.asarray(identified["state_scale"]).tolist()},
        "per_task":{task:aggregate(rows) for task,rows in by_task.items()},
        "overall_original_FIT":summary,
        "original_publisher_episode_counts":{task:len(rows) for task,rows in by_task.items()},
        "source_profile":"Original 14D robot joint qpos observation and 14D absolute action.qpos, real takeover flags; five original robot tasks",
        "temporal_lag_sign_rule":"positive k uses action at t-k, 0 uses action at t, negative uses logged action from after t, never legitimate input for deployed model",
        "information_leakage_policy":"Future action offsets used ONLY to diagnose retrospective publisher timestamp semantics on already-training FIT; never eligible as prediction features or basis for a superiority claim",
        "interpretation_limits":[
            "Original FIT-only observed action-state correlations are not independent heldout forecasting performance.",
            "FIT-identified servo_gain learned on original FIT; any in-FIT lag ranking may be optimistically biased.",
            "Human-free filtering and common origin windows alter the distribution; this does not identify physical actuator latency as a causal mechanism.",
            "Near-stationary threshold groups use FUTURE movement and may only be diagnostic.",
            "Do not change action timing without an independently preregistered fresh real-data experiment and new untouched task."
        ],
    }
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/"SOURCE_FIT_ACTION_ALIGNMENT.json").write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
    print("ME068_GENUINE_ORIGINAL_FIT_ACTION_TIMING="+json.dumps({
        "real_fit_episodes":len(source),
        "scored_autonomous_transitions":summary["valid_autonomous_transitions"],
        "per_task_best_lag":{k:v["best_FIT_observational_causal_action_lag"]
            for k,v in result["per_task"].items()},
        "overall_MAE":{k:v["original_frame_weighted_servo_physical_MAE"]
            for k,v in summary["per_action_offset"].items()},
        "persistent_baseline":summary["one_step_persistence_MAE"],
        "training_updates":0},sort_keys=True),flush=True)

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--cache",type=Path,required=True)
    args=parser.parse_args()
    main(args.cache)
