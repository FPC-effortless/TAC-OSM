#!/usr/bin/env python3
"""LANGCOMPARE029 guarded future per-model/seed training launcher.

Default dry-run is CPU-only. Real fits require ALL source-controlled science
and operator quota gates, two approved BF16 SM80+ GPUs, original FIT source and
a complete four-arm parity manifest. No DEV scores or protected TEST access.
No GPU workflow invokes fit on this draft. Runs only after an explicit future
matched-venue amendment and operator approval in a protected environment.
"""
from __future__ import annotations
import argparse, json, os, random
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from scripts.langcompare029_preflight import (
    load_tac,load_mamba3,load_bdh,ByteTransformer,parameter_count,
    ARM_ORDER,EXPECTED)
from scripts.langcompare029_sampling import (
    SEEDS,UPDATES,BATCH,fit_sources,iter_batches,schedule_receipt)
from scripts.langcompare029_checkpoint import create_bytes,sha

ROOT=Path(__file__).resolve().parents[1]
READINESS=ROOT/"research/mathematical-engineering/langcompare029/COMPUTE_AND_READINESS.json"
PARITY=ROOT/"research/mathematical-engineering/langcompare029/FAIR_COMPARISON.json"

def blockers():
    """Pure CPU reads; no device allocation or optimizer."""
    r=json.loads(READINESS.read_text())
    issues=[]
    if r["compute"].get("approved_gpu_sm80_compatible") is not True:
        issues.append("NO_APPROVED_SM80_PLUS_BF16_TWO_GPU_COMPARISON_VENUE")
    if r["compute"]["actual_mamba_parameters"]=="NOT_MEASURED":
        issues.append("MAMBA3_ACTUAL_PARAMETERS_NOT_VERIFIED")
    if r["compute"]["matched_FLOPs"]=="NOT_MEASURED":
        issues.append("MATCHED_TRAINING_FLOPS_NOT_AUDITED")
    if r["compute"]["gpu_execution_approval"]!="GRANTED":
        issues.append("OPERATOR_GPU_QUOTA_NOT_APPROVED")
    if not PARITY.is_file():issues.append("FOUR_ARM_PARITY_MANIFEST_MISSING")
    return issues

def run(*,mode,arm,seed,checkpoint=None,bdh_source=None):
    if arm not in ARM_ORDER or seed not in SEEDS:
        raise ValueError("unregistered model/seed")
    if mode=="dry-run":
        return {"status":"BLOCKED_NO_FIT","study_id":"LANGCOMPARE029",
                "arm":arm,"seed":seed,"blocking_requirements":blockers(),
                "optimizer_steps":0,"gpu_minutes":0,"DEV_NLL":"NOT_RUN",
                "protected_TEST":"NOT_ACCESSED"}
    if mode!="fit":raise ValueError("unknown mode")
    missing=blockers()
    if missing:raise PermissionError("no fit permitted: "+", ".join(missing))
    if os.environ.get("PHONE_LAB_GPU_EXECUTION_ENABLED")!="true":
        raise PermissionError("protected operator quota approval unavailable")
    if checkpoint is None:raise ValueError("unique checkpoint destination required")
    from scripts.validate_research_parity import check_manifest
    manifest=json.loads(PARITY.read_text())
    if manifest.get("study_id")!="LANGCOMPARE029" or manifest.get("parameter_budget_contract")!="owner-2026-10-28-500k":
        raise PermissionError("registered 500K four-way comparison manifest absent")
    result=check_manifest(manifest,"preflight")
    if result.get("status")!="PASS":raise PermissionError("source/compute parity fail")
    if torch.cuda.device_count()!=2:
        raise RuntimeError("exactly two approved physical CUDA devices required")
    for gpu in range(2):
        cap=torch.cuda.get_device_capability(gpu)
        if cap[0]<8:raise RuntimeError(f"official Mamba-3 BF16 incompatible with SM{cap[0]}{cap[1]}")
    gpu=SEEDS.index(seed)%2
    torch.cuda.set_device(gpu)
    if not torch.cuda.is_bf16_supported():raise RuntimeError("native BF16 unavailable")
    torch.set_num_threads(2)
    torch.manual_seed(seed);np.random.seed(seed);random.seed(seed)
    expected=dict(zip(ARM_ORDER,EXPECTED))[arm]
    if arm=="tac_context026":model=load_tac()
    elif arm=="causal_transformer":model=ByteTransformer()
    elif arm=="bdh_official":
        if bdh_source is None:raise PermissionError("pinned official BDH implementation missing")
        model=load_bdh(bdh_source)
    else:model=load_mamba3()
    counts=parameter_count(model)
    if counts!={"trainable":expected,"deployed":expected}:
        raise ValueError("actual model size differs from registered arm")
    schedule=schedule_receipt(fit_sources(),seed)
    arms=[x for x in manifest["arms"] if x.get("seed")==seed and x.get("architecture")==arm]
    if len(arms)!=1 or arms[0].get("training_sample_order_sha256")!=schedule["sample_order_sha256"]:
        raise PermissionError("registered genuine FIT sample schedule differs")
    if any(arms[0][k]!=expected for k in ("trainable_parameters","deployed_parameters")):
        raise PermissionError("actual model count differs from frozen manifest")
    dest=Path(checkpoint)
    if dest.exists():raise FileExistsError("would overwrite original checkpoint")
    model=model.to(f"cuda:{gpu}").train()
    optimizer=torch.optim.AdamW(model.parameters(),lr=0.001)
    steps=0
    fit=fit_sources()
    for step,items in iter_batches(fit,seed):
        raw=torch.tensor([list(x[2]) for x in items],dtype=torch.long,device=f"cuda:{gpu}")
        logits=model(raw[:,:-1])
        if not torch.is_tensor(logits):
            logits=logits.logits if hasattr(logits,"logits") else logits[0]
        if logits.shape!=(BATCH,128,256):raise ValueError("future target or logits shape mismatch")
        loss=F.cross_entropy(logits.float().reshape(-1,256),raw[:,1:].reshape(-1))
        if not torch.isfinite(loss).item():raise ValueError("nonfinite next-original-byte loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gs=[p.grad for p in model.parameters() if p.requires_grad]
        if any(g is None or not torch.isfinite(g).all() for g in gs):
            raise ValueError("untrained or nonfinite gradient parameter")
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
        optimizer.step()
        steps=step+1
    if steps!=UPDATES:raise RuntimeError("incomplete matched source exposure")
    ck=create_bytes(model=model,optimizer=optimizer,model_name=arm,seed=seed,
        source_order_sha=schedule["sample_order_sha256"],updates=steps)
    dest.parent.mkdir(parents=True,exist_ok=True)
    dest.write_bytes(ck)
    return {"status":"FIT_CHECKPOINT_FROZEN_NOT_EVALUATED",
            "arm":arm,"seed":seed,"steps":steps,"checkpoint_sha256":sha(ck),
            "source_order_sha256":schedule["sample_order_sha256"],
            "DEV_NLL":"NOT_RUN","protected_TEST":"NOT_ACCESSED",
            "scientific_result":"NOT_RANKED_WITHOUT_POSTFIT_COMPUTE_AND_REPLAY"}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode",choices=("dry-run","fit"),default="dry-run")
    p.add_argument("--arm",choices=ARM_ORDER,default=ARM_ORDER[0])
    p.add_argument("--seed",type=int,choices=SEEDS,default=SEEDS[0])
    p.add_argument("--bdh-source",type=Path)
    p.add_argument("--checkpoint-output",type=Path)
    p.add_argument("--output",type=Path)
    a=p.parse_args()
    r=run(mode=a.mode,arm=a.arm,seed=a.seed,
          bdh_source=a.bdh_source,checkpoint=a.checkpoint_output)
    raw=json.dumps(r,sort_keys=True,indent=2)+"\n"
    if a.output:
        if a.output.exists():raise FileExistsError("receipt already exists")
        a.output.parent.mkdir(parents=True,exist_ok=True)
        a.output.write_text(raw)
    print(raw,end="")
if __name__=="__main__":main()
