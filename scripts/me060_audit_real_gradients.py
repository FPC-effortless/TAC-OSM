#!/usr/bin/env python3
"""ME-060 genuine original FIT parameter-reachability audit, not model fitting.

Records which nominal trainable parameters actually receive the supervised
gradient under *exactly the frozen ME059 objective*. The one optimizer update
is a discarded diagnostic pilot, never a scored trained model.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np
import torch

from scripts.me058_profile_original_fit import (
    read_verified_fit, identify_fit_only, schedule_fit_only, original_batch,
    construct, hashed, SOURCE_MANIFEST_SHA256, SEEDS, FROZEN_CAPACITY)
from scripts.me056_profile_paired_execution import boundary_objective
from scripts.me055_profile_prefit import objective
from scripts.validate_research_parity import check_manifest


def gradient_reachability(model):
    records=[]
    for name,parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        grad=parameter.grad
        present=grad is not None
        finite=present and bool(torch.isfinite(grad).all())
        nonzero=finite and bool((grad.detach()!=0).any())
        records.append({"parameter":name,"elements":parameter.numel(),
            "gradient_present":present,"gradient_finite":finite,
            "gradient_nonzero":nonzero,
            "gradient_abs_max":float(grad.detach().abs().max()) if finite else None})
    counts={k:sum(row["elements"] for row in records if row[k])
            for k in ("gradient_present","gradient_finite","gradient_nonzero")}
    counts["declared_trainable"]=sum(row["elements"] for row in records)
    counts["gradient_absent"]=counts["declared_trainable"]-counts["gradient_present"]
    return records,counts


def exact_source_loss(name,model,batch):
    if name=="native_tac":
        kernels=(model._burnin_training,model._burnin_boundary_training,
            model._observe_training,model._predict_training,
            model._dense_transition_auxiliary_training)
        return boundary_objective(model,batch,kernels)
    return objective(model,batch)


def one_true_fit_gradient_receipt(name,batch,physical_prior):
    torch.manual_seed(SEEDS[0])
    model=construct(name,physical_prior)
    if sum(p.numel() for p in model.parameters() if p.requires_grad)!=FROZEN_CAPACITY[name]:
        raise RuntimeError("actual original FIT audited model capacity changed")
    optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
    optimizer.zero_grad(set_to_none=True)
    loss=exact_source_loss(name,model,batch)
    if not torch.isfinite(loss):
        raise RuntimeError("original physical FIT gradient loss nonfinite")
    loss.backward()
    rows,count=gradient_reachability(model)
    before={n:p.detach().clone() for n,p in model.named_parameters() if p.requires_grad}
    torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
    optimizer.step()
    for row in rows:
        parameter=dict(model.named_parameters())[row["parameter"]]
        row["weight_changed"] = bool(torch.any(parameter.detach()!=before[row["parameter"]]))
    count["weight_changed"]=sum(r["elements"] for r in rows if r["weight_changed"])
    return {"architecture":name,"original_FIT_loss":float(loss.detach()),
            "gradient_coverage":count,"parameter_details":rows,
            "absent_parameter_names":[r["parameter"] for r in rows
                                     if not r["gradient_present"]],
            "nonzero_gradient_parameter_names":[r["parameter"] for r in rows
                                      if r["gradient_nonzero"]],
            "one_discarded_real_FIT_optimizer_pilot":True,
            "model_fits_started":0}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--cache",default="results/me060/private-real-FIT")
    parser.add_argument("--source-registration",
        default="research/me055-independent-robot-task-source.json")
    parser.add_argument("--output",default="results/me060/ORIGINAL_FIT_GRADIENT_REACHABILITY.json")
    args=parser.parse_args()
    torch.set_num_threads(1)
    originals,groups,source_sha=read_verified_fit(args.cache,args.source_registration)
    if source_sha!=SOURCE_MANIFEST_SHA256:
        raise RuntimeError("ME060 original FIT SHA mismatch before update")
    stats=identify_fit_only(originals)
    selection,_=schedule_fit_only(originals,SEEDS[0])
    sample=original_batch(originals,selection[0],stats)
    manifests=json.loads(Path("research/mathematical-engineering/me059/FAIR_COMPARISON.json").read_text())
    if manifests.get("expected_physical_prior_sha256")!=hashed(stats):
        raise RuntimeError("ME060 actual original FIT physical prior differs from frozen ME058")
    expected=manifests["frozen_sample_orders"][str(SEEDS[0])]
    if expected!=schedule_fit_only(originals,SEEDS[0])[1]["schedule_sha256"]:
        raise RuntimeError("ME060 original frozen FIT pairing changed")
    if check_manifest(manifests,"preflight")["status"]!="PASS":
        raise RuntimeError("ME060 source prerequisite comparison registration failed")
    records=[one_true_fit_gradient_receipt(arm,sample,stats)
             for arm in ("native_tac","gru_world")]
    result={"study":"ME060_REAL_FIT_GRADIENT_REACHABILITY_DIAGNOSTIC",
            "source_original_FIT_manifest_sha256":source_sha,
            "frozen_parent_registration":"research/mathematical-engineering/me059/FAIR_COMPARISON.json",
            "same_real_FIT_sample_SHA256":expected,
            "physical_prior_sha256":hashed(stats),
            "model_fits_started":0,
            "scored_model_checkpoints":0,
            "DEV_TEST_trajectory_files_opened":0,
            "full_architecture_intelligence_claim":False,
            "software_fixtures_not_used_for_measured_results":True,
            "arms":records}
    path=Path(args.output);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
    for item in records:
        print(json.dumps({"event":"ME060_FIT_GRADIENT_COVERAGE",
             "architecture":item["architecture"],
             "counts":item["gradient_coverage"],
             "params_without_gradient":item["absent_parameter_names"]}),flush=True)


if __name__=="__main__":
    main()
