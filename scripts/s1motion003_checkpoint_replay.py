#!/usr/bin/env python3
"""S1MOTION003 read-only original frozen checkpoint replay and gradient diagnostics.

Never calls optimizer.step. No retraining, parameter mutation, alternate gate,
new research success or retroactive tuning of first DEV score.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch
from torch.nn import functional as F

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/"src")]
from scripts.s1motion003_real_tape_fit import (
    FIT,DEV,SEEDS,MatchedGRU,VelocityAnchoredSystem1,
    source_episodes,prepare,score,batch,windows,outputs,
)

LEDGER=ROOT/"research/parallel-six/S1MOTION003-ORIGINAL-LEDGER.json"


def original_model(src,model_record,seed,arm,norm):
    path=src/f"{arm}-{seed}.pt"
    blob=path.read_bytes()
    digest=hashlib.sha256(blob).hexdigest()
    if digest != model_record["checkpoint_sha256"]:
        raise RuntimeError(f"original checkpoint SHA mismatch {arm} {seed}")
    payload=torch.load(path,map_location="cpu",weights_only=False)
    if (payload["seed"]!=seed or payload["architecture"]!=arm or
        payload["updates"]!=192 or payload["batch"]!=8 or
        payload["frames"]!=32 or "torch_rng" not in payload or
        "optimizer" not in payload):
        raise RuntimeError("missing original optimizer/RNG/source trainer state")
    scale=torch.tensor(norm["train_delta_scale_normalized"],dtype=torch.float32)
    if arm=="system1":
        model=VelocityAnchoredSystem1()
        model.set_train_delta_scale(scale)
    else:
        model=MatchedGRU(scale)
    model.load_state_dict(payload["model"],strict=True)
    return model.eval(),digest


def safe_cos(a,b):
    v1=a.detach().reshape(-1).double()
    v2=b.detach().reshape(-1).double()
    d=v1.norm()*v2.norm()
    return float(torch.dot(v1,v2)/d) if d>1e-20 else None


def gradient_roles(model,fit,seed):
    selected,starts=windows(fit,seed)[0]
    x,u,y,event,previous=batch(fit,selected,starts)
    model.zero_grad(set_to_none=True)
    prediction,logit,scale=outputs(model,x,u,previous)
    point=(prediction-y).square().mean()
    risk=F.binary_cross_entropy_with_logits(logit,event)
    distribution=(.5*((prediction-y)/scale).square()+scale.log()).mean()
    terms={"point":point,"risk_weighted":.1*risk,"uncertainty_weighted":.01*distribution}
    if isinstance(model,VelocityAnchoredSystem1):
        named=dict(model.named_parameters())
        names=["core.core.encoder.weight",
               "core.core.memory_gates.weight",
               "core.reactive_head.0.weight",
               "core.reactive_head.2.weight"]
    else:
        named=dict(model.named_parameters())
        names=["recurrent.weight_ih_l0","recurrent.weight_hh_l0",
               "readout.weight"]
    grads={}
    for role,loss in terms.items():
        gg=torch.autograd.grad(loss,[named[n] for n in names],
              allow_unused=True,retain_graph=True)
        grads[role]={
            n:(p if p is not None else torch.zeros_like(named[n]))
            for n,p in zip(names,gg)
        }
    blocks={}
    for n in names:
        blocks[n]=dict(weighted_gradient_norm={
            role:float(gg[n].detach().double().norm())
            for role,gg in grads.items()},
            cosine_point_vs_risk=safe_cos(grads["point"][n],
                                         grads["risk_weighted"][n]),
            cosine_point_vs_uncertainty=safe_cos(grads["point"][n],
                                                grads["uncertainty_weighted"][n]))
    return dict(fit_only_gradient_sample_seed=seed,fit_windows=8,
                model_training_not_resumed=True,
                loss_values={name:float(x.detach()) for name,x in terms.items()},
                blocks=blocks)


@torch.no_grad()
def correction_diagnostics(model,dev,norm):
    """Purely descriptive; does NOT select/tune a correction on DEV."""
    xscale=np.asarray(norm["state_sd"])
    per=[]
    for ep in DEV:
        z=dev[ep]
        p,_,_=outputs(model,z["x"][None],z["prior_u"][None],
                       z["previous_x"][:1])
        pred=p[0].detach().numpy().astype(np.float64)
        norm_obs=z["x"].numpy().astype(np.float64)
        norm_prev=z["previous_x"].numpy().astype(np.float64)
        base=2*norm_obs-norm_prev
        y=z["y"].numpy().astype(np.float64)
        innovation=(pred-base)*xscale
        true_residual=(y-base)*xscale
        improvement=(innovation**2-2*innovation*true_residual)
        per.append(dict(episode=ep,
            correction_energy=float(np.mean(innovation**2)),
            signed_alignment=float(np.mean(innovation*true_residual)),
            expected_mse_difference_vs_velocity=float(np.mean(improvement)),
            mean_true_residual_square=float(np.mean(true_residual**2))))
    return per


def replay(original_dir,cache,outdir):
    ledger=json.loads(LEDGER.read_text())
    before=json.loads((original_dir/"before-dev-score.json").read_text())
    assert tuple(before["trained_seeds"])==SEEDS
    assert before["source"]["revision"]==ledger["source_snapshot"]
    assert before["source"]["full_parquet_sha256"]==ledger["source_parquet_sha256"]
    episodes,source=source_episodes(cache)
    if source["full_parquet_sha256"]!=ledger["source_parquet_sha256"]:
        raise RuntimeError("real source changed")
    fit,dev,norm=prepare({e:episodes[e] for e in FIT},
                         {e:episodes[e] for e in DEV})
    for k,v in norm.items():
        if not np.array_equal(np.asarray(v),np.asarray(before["normalization"][k])):
            raise RuntimeError(f"source-normalization mismatch {k}")
    rows=[]
    for original in ledger["original_source_logged_scores"]:
        seed=original["seed"]
        report={}
        for arm in ("system1","gru"):
            model,digest=original_model(original_dir,original[arm],seed,arm,norm)
            actual=score(model,dev,norm)
            expected=original[arm]
            for key,value in [
                ("mse",actual["raw_joint_MSE"]["tac"]),
                ("high_motion_mse",actual["raw_joint_high_motion_MSE"]["tac"]),
                ("proxy_AUPRC",actual["proxy_AUPRC"]),
                ("proxy_AUROC",actual["proxy_AUROC"]),
                ("proxy_Brier",actual["proxy_Brier"])]:
                if not np.isclose(value,expected[key],rtol=1e-6,atol=1e-11):
                    raise RuntimeError(f"original score mismatch {seed}:{arm}:{key}: replay={value:.17g}, original={expected[key]:.17g}, delta={value-expected[key]:.17g}")
            roles=gradient_roles(model,fit,seed)
            per=correction_diagnostics(model,dev,norm)
            report[arm]=dict(checkpoint_sha256=digest,
                             verified_original=expected,
                             full_per_episode_first_dev=actual["per_episode"],
                             correction_decomposition=per,
                             fit_gradient_diagnostics=roles)
            print("S1MOTION003_COLD_REPLAY_PASS",seed,arm,digest,flush=True)
        rows.append(dict(seed=seed,models=report,registered_gate_pass=False))
    outdir.mkdir(parents=True,exist_ok=True)
    result=dict(study="S1MOTION003",source=source,original_run_id=40277450986,
                original_artifact_id=12630707934,unchanged_original_seed_scores=True,
                full_original_3_seed_gate_pass=False,original_pass_seed_count=0,
                all_checkpoints_frozen_before_first_DEV=True,no_optimizer_or_retrain=True,
                outputs_purely_retrospective_diagnostics=True,
                true_adverse_labels_exist=False,full_train_compute_parity_established=False,
                truly_independent_heldout_site_tested=False,
                protected_original_tape_TEST_scored=False,records=rows)
    (outdir/"original-checkpoint-cold-replay.json").write_text(
        json.dumps(result,indent=2,sort_keys=True)+"\n")
    print("S1MOTION003_SIX_ORIGINAL_CHECKPOINTS_VERIFIED=PASS",flush=True)
    print("S1MOTION003_ORIGINAL_QUALIFICATION=FAIL_0_OF_3",flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--original-dir",required=True)
    p.add_argument("--cache",default="/tmp/s1motion003-replay-source")
    p.add_argument("--out",default="research/diagnostics/s1motion003-read-only-replay")
    args=p.parse_args()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    replay(Path(args.original_dir),Path(args.cache),ROOT/args.out)
