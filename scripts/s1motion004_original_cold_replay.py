#!/usr/bin/env python3
"""S1MOTION004 SHA-preserving read-only original 12-checkpoint replication.

No optimizer; no model update; re-evaluate only the already opened original
DEV episodes. Compare full original observations with transparent tolerances.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/"src")]
from scripts.s1motion004_real_towel_fit import (
    ARMS,SEEDS,DEV,FIT,real_source,prepare_fit_only,
    encode_source_episode,initialization,windows,batch,score
)
from tac_osm.variants.s1motion004_gradient_routes import losses

ORIGINAL=ROOT/"research/parallel-six/S1MOTION004-ORIGINAL-LEDGER.json"


def check_tangent_path(model,arm,fit,seed):
    """Actually check frozen real FIT gradient reachability, no optimizer.step."""
    sel,beg=windows(fit,seed)[0]
    x,u,y,event,previous=batch(fit,sel,beg)
    isolated=arm.endswith("_isolated")
    _,terms=losses(model,x,u,y,event,previous,isolated=isolated)
    auxiliary=.1*terms["proxy_bce"]+.01*terms["dispersion_nll"]
    if arm.startswith("tac"):
        core=model.core.core.memory_gates.weight
        point_core=model.core.core.readout[0].weight
        aux_head=model.core.reactive_head[0].weight
    else:
        core=model.recurrent.weight_ih_l0
        point_core=model.readout.weight
        aux_head=model.readout.weight
    grad=torch.autograd.grad(auxiliary,core,allow_unused=True,
                             retain_graph=True)[0]
    point_grad=torch.autograd.grad(terms["point"],core,
                                  allow_unused=True,retain_graph=True)[0]
    if point_grad is None or not torch.isfinite(point_grad).all().item() or (
        point_grad.abs().sum().item()==0):
        raise RuntimeError(f"point gradient missing from real FIT core {arm}")
    if isolated and grad is not None:
        raise RuntimeError(f"auxiliary leak into isolated memory core {arm}")
    if not isolated and (grad is None or
         not torch.isfinite(grad).all().item() or grad.abs().sum().item()==0):
        raise RuntimeError(f"shared auxiliary source gradients unexpectedly zero {arm}")
    return dict(isolated=isolated,real_fit_batch_frames=256,
        actual_aux_gradient_reaches_backbone=bool(grad is not None),
        point_gradient_norm=float(point_grad.norm()),
        shared_aux_gradient_norm=(float(grad.norm()) if grad is not None else 0.0))


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--original-dir",required=True)
    p.add_argument("--cache",default="/tmp/s1motion004-original-cold-replay")
    p.add_argument("--out",default="research/diagnostics/s1motion004-cold-replay")
    a=p.parse_args()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    original=Path(a.original_dir)
    frozen=json.loads((original/"first-dev-scores.json").read_text())
    before=json.loads((original/"before-dev-score.json").read_text())
    ledger=json.loads(ORIGINAL.read_text())
    if (tuple(before["trained_seed_list"])!=SEEDS or
        not before["all_twelve_model_states_frozen_before_first_DEV"] or
        not before["all_twelve_optimizer_rng_states_frozen_before_first_DEV"] or
        before["source"]["parquet_sha256"]!=ledger["pinned_parquet_sha256"] or
        frozen["all_three_seed_preregistered_predictive_gate_pass"]):
        raise RuntimeError("original evidence/gate inconsistency")
    fit_src,dev_src,source=real_source(a.cache,release_dev=False)
    if dev_src is not None or source["parquet_sha256"]!=ledger["pinned_parquet_sha256"]:
        raise RuntimeError("replay source provenance failed")
    fit,norm=prepare_fit_only(fit_src)
    for k,v in norm.items():
        if not np.array_equal(np.asarray(v),
                              np.asarray(before["train_stats_only"][k])):
            raise RuntimeError("FIT-only normalizer fingerprint changed")
    _,dev_src,other=real_source(a.cache,release_dev=True)
    if other["revision"]!=source["revision"]:
        raise RuntimeError("source changed after fit reconstitution")
    dev={ep:encode_source_episode(dev_src[ep],norm) for ep in DEV}
    scale=torch.tensor(norm["train_delta_scale_normalized"],dtype=torch.float32)
    output=[];summary=[]
    max_AUPRC_delta=0.;max_mse_delta=0.
    for observed_seed in frozen["original_first_dev_real_episode_scores"]:
        seed=observed_seed["seed"]
        setup=initialization(seed,scale)
        for arm in ARMS:
            ledger_entry=next(z for z in ledger["original_fits"]
                              if z["seed"]==seed and z["model"]==arm)
            original_record=next(z for z in before["model_fit_receipts"]
                                 if z["seed"]==seed and z["arm"]==arm)
            filename=original/f"{arm}-{seed}.pt"
            digest=hashlib.sha256(filename.read_bytes()).hexdigest()
            if digest!=ledger_entry["sha"] or digest!=original_record["checkpoint_sha256"]:
                raise RuntimeError("SHA mismatch or model replaced during original replay")
            # This original source artifact belongs to the same authorized repo
            # and was SHA-pinned before loading Python pickle checkpoint fields.
            ckpt=torch.load(filename,map_location="cpu",weights_only=False)
            if (ckpt["arm"]!=arm or ckpt["seed"]!=seed or
                ckpt["original_updates"]!=192 or
                ckpt["source_sha"]!=source["revision"] or
                ckpt["source_parquet_sha256"]!=source["parquet_sha256"] or
                "optimizer" not in ckpt or "torch_rng" not in ckpt):
                raise RuntimeError("original trained checkpoint/optimizer receipt changed")
            model=setup[arm]
            model.load_state_dict(ckpt["model"],strict=True)
            model.eval()
            actual=score(model,dev,norm)
            frozen_score=observed_seed["scores"][arm]["dev_scores"]
            mse=float(actual["raw_joint_MSE"]["tac"])
            original_mse=float(frozen_score["raw_joint_MSE"]["tac"])
            auprc=float(actual["proxy_AUPRC"])
            original_auprc=float(frozen_score["proxy_AUPRC"])
            original_risk=float(frozen_score["proxy_Brier"])
            diff_mse=abs(mse-original_mse)
            diff_auprc=abs(auprc-original_auprc)
            max_mse_delta=max(max_mse_delta,diff_mse)
            max_AUPRC_delta=max(max_AUPRC_delta,diff_auprc)
            if not np.isclose(mse,original_mse,rtol=1.e-5,atol=1.e-11):
                raise RuntimeError(f"nonrepeatable original DEV physical MSE {arm}/{seed}")
            # Numerical rank ties can perturb AP more than squared-error
            # while leaving model weights and baseline unchanged. Record drift.
            if (diff_auprc>0.005 or
                abs(float(actual["proxy_Brier"])-original_risk)>1.e-4):
                raise RuntimeError(f"nonrepeatable original proxy score {arm}/{seed}; AP delta={diff_auprc}")
            model.train()
            real_fit_gradient=check_tangent_path(model,arm,fit,seed)
            model.eval()
            output.append(dict(seed=seed,arm=arm,model_sha256=digest,
                original_MSE=original_mse,replay_MSE=mse,
                delta_physical_MSE=diff_mse,
                original_proxy_AUPRC=original_auprc,replay_proxy_AUPRC=auprc,
                absolute_ranking_score_drift=diff_auprc,
                true_FIT_gradient_routes=real_fit_gradient,
                actual_model_bytes_unmodified=True))
            print("S1MOTION004_ORIGINAL_COLD_REPLAY_PASS",
                  json.dumps(output[-1],sort_keys=True),flush=True)
    o=ROOT/a.out
    o.mkdir(parents=True,exist_ok=True)
    report=dict(study="S1MOTION004",original_run_id=40280359434,
        original_artifact_id=12632067673,
        all_twelve_original_model_hashes_verified=True,
        original_score_delta_max_source_mse=max_mse_delta,
        original_score_delta_max_proxy_AUPRC=max_AUPRC_delta,
        original_full_scientific_gate_pass=False,
        original_full_gate_seed_pass_count=1,
        models_retrained=0,training_optimizer_steps=0,
        numerical_replay_not_bitwise_certification=True,
        full_training_FLOPs_matched=False,
        original_TAPE_and_TOWEL_protected_tests_unaccessed=True,
        authentic_adverse_outcome_labels_absent=True,
        records=output)
    (o/"cold-replay.json").write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    print("S1MOTION004_ALL_12_SHA_CHECKPOINTS_COLD_REPLAY=PASS",flush=True)
    print("S1MOTION004_ORIGINAL_SCIENTIFIC_GATE=FAIL_2_OF_3_SEEDS",flush=True)


if __name__=="__main__":
    main()
