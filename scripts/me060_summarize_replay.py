"""Replay every frozen ME060 forecast and summarize registered development gates."""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
import numpy as np
import torch

from scripts.me060_train_trajectory import (
    ROOT, SEEDS, cohorts, build, digest, evaluate)
from scripts.me057_train_fused_anchor_robot import (
    load_part, normalize, sha_file, write_json)
from scripts.validate_research_parity import check_manifest


def summarize(folder,cache,model_factory=build):
    torch.set_num_threads(1)
    report=json.loads((folder/"results.json").read_text())
    final_plan=json.loads((folder/"FAIR_COMPARISON_FINAL.json").read_text())
    seeds=sorted({r["seed"] for r in final_plan["arms"]})
    final=check_manifest(final_plan,"final")
    split,metadata=cohorts(cache)
    first=torch.load(folder/report["checkpoints"][0]["path"],weights_only=False)
    identified={k:np.asarray(v,dtype=np.float32) for k,v in first["identified"].items()}
    scalar_replays=0;source_receipts={}
    for part in ("CAL","DEV"):
        raw=load_part(cache,split[part],metadata,"ME060_REPLAY_"+part)
        source_receipts[part]=digest({str(e):r["sha256"] for e,r in sorted(raw.items())})
        source=normalize(raw,identified)
        for record in report["checkpoints"]:
            path=folder/record["path"]
            if sha_file(path)!=record["measured"]["checkpoint_sha256"]:
                raise RuntimeError("trained checkpoint bytes changed")
            model=model_factory(record["arm"],identified)
            model.load_state_dict(torch.load(path,weights_only=False)["model"])
            key=f'{record["arm"]}-s{record["seed"]}'
            replay=evaluate(model,source,identified)
            if replay!=report["evaluation"][part][key]:
                raise RuntimeError("frozen forecast replay mismatch")
            scalar_replays+=sum(len(row["scaled_errors"]) for row in replay.values())
    evaluation=report["evaluation"];dev=evaluation["DEV"]
    ids=sorted(next(iter(dev.values())),key=int)
    by_arm={};uncertainty={};per_seed={}
    for arm in ("native_tac","gru_world"):
        episode=np.asarray([[dev[f"{arm}-s{s}"][e]["MAE"] for e in ids] for s in seeds])
        by_arm[arm]=episode.mean(0)
        per_seed[arm]=episode.mean(1).tolist()
        uncertainty[arm]=[]
        for seed in seeds:
            key=f"{arm}-s{seed}"
            cal=list(evaluation["CAL"][key].values())
            cal_scores=np.asarray([max(row["scaled_errors"]) for row in cal])
            rank=min(len(cal),math.ceil((len(cal)+1)*.90))
            q=float(np.sort(cal_scores)[rank-1])
            devrows=[dev[key][e] for e in ids]
            errors=np.concatenate([row["scaled_errors"] for row in devrows])
            uncertainty[arm].append({"seed":seed,"calibration_episodes":len(cal),
                "quantile":q,"nominal_episode_coverage":.90,
                "DEV_episode_coverage":float(np.mean([max(r["scaled_errors"])<=q for r in devrows])),
                "DEV_marginal_coverage":float(np.mean(errors<=q)),
                "DEV_mean_physical_interval_width":float(np.mean([2*q*r["scale_mean"] for r in devrows]))})
    reference=next(iter(dev.values()))
    persistence=np.asarray([reference[e]["persistence_MAE"] for e in ids])
    servo=np.asarray([reference[e]["servo_MAE"] for e in ids])
    rng=np.random.default_rng(60060)
    difference=by_arm["native_tac"]-by_arm["gru_world"]
    boot=difference[rng.integers(len(ids),size=(20000,len(ids)))].mean(1)
    interval=np.quantile(boot,[.025,.975]).tolist()
    means={name:float(values.mean()) for name,values in by_arm.items()}
    means.update(persistence=float(persistence.mean()),servo=float(servo.mean()))
    gate=bool(means["native_tac"]<=.98*min(means["gru_world"],means["persistence"],means["servo"]) and interval[1]<0)
    summary={"study":final_plan["study_id"],
        "source_commit":report.get("source_commit","ca259e59c832cbc7096d29d343493ef34915d4b2"),"independent_qualification":False,
        "old_TEST_opened":False,"DEV_episode_count":len(ids),"mean_physical_MAE":means,
        "per_seed_MAE":per_seed,"TAC_minus_GRU_episode_bootstrap_95_interval":interval,
        "TAC_paired_seed_wins":int(sum(a<b for a,b in zip(per_seed["native_tac"],per_seed["gru_world"]))),
        "registered_development_gate_pass":gate,
        "uncertainty":uncertainty,"final_numerical_parity":final,
        "true_full_training_FLOPs_certified":False,
        "replay":{"all_ten_checkpoints_hash_verified":True,"forecast_scores_exact":True,
            "standardized_absolute_error_scalar_replays":scalar_replays,
            "CAL_DEV_source_receipts":source_receipts},
        "limitations":["historically exposed development population","episode bootstrap conditional on five fitted seeds",
            "episode blocks do not establish operator/session independence","observed future actions are supplied factual covariates",
            "conformal exchangeability unverified","PyTorch supported-operator FLOPs omit some operations",
            "no observed action utility, causal policy value, or autonomous control evaluation"]}
    write_json(folder/"SUMMARY.json",summary)
    print(json.dumps({k:summary[k] for k in ("mean_physical_MAE","TAC_minus_GRU_episode_bootstrap_95_interval", "TAC_paired_seed_wins","registered_development_gate_pass","replay")}),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output",type=Path,default=Path("results/me060"))
    p.add_argument("--source-cache",type=Path,required=True)
    a=p.parse_args();summarize(a.output,a.source_cache)
