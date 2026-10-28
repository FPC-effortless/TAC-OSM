#!/usr/bin/env python3
"""Prospective LANGCOMPARE029 full-cohort DEV evaluation and replay gate.

NO scoring is permitted before all 20 genuine train checkpoints and measured
matched-compute final manifest pass. The 4 DEV docs have already been exposed:
even passing scores only support an INTERNAL development comparison. This module
never touches protected TEST or trains any model. Default = CPU dry-run only.
"""
from __future__ import annotations
import argparse
import json
import math
import os
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from torch.nn import functional as F
from scripts.langcompare029_preflight import (
    ARM_ORDER,EXPECTED,load_tac,load_bdh,load_mamba3,ByteTransformer)
from scripts.langcompare029_sampling import SEEDS,fit_sources,schedule_receipt
from scripts.langcompare029_checkpoint import verify_bytes

ROOT=Path(__file__).resolve().parents[1]
PARITY=ROOT/"research/mathematical-engineering/langcompare029/FAIR_COMPARISON.json"

def cohort_readiness():
    """File system checks only: never opens DEV labels or CUDA resources."""
    if not PARITY.is_file():
        return {"status":"BLOCKED_NO_SCORING","missing":["FINAL_RESOURCE_PARITY_MANIFEST"],
                "checkpoint_count":0,"DEV_NLL":"NOT_RUN","TEST":"NOT_ACCESSED"}
    manifest=json.loads(PARITY.read_text())
    rows=manifest.get("arms",[])
    expected={(a,s) for a in ARM_ORDER for s in SEEDS}
    actual={(r.get("architecture"),r.get("seed")) for r in rows}
    issues=[]
    if actual!=expected or len(rows)!=20:
        issues.append("INCOMPLETE_FOUR_ARM_FIVE_SEED_COHORT")
    paths=set()
    count=0
    for row in rows:
        loc=row.get("checkpoint_path")
        digest=row.get("measured",{}).get("checkpoint_sha256")
        if not isinstance(loc,str) or not loc or not isinstance(digest,str) or len(digest)!=64:
            issues.append("MISSING_ORIGINAL_CHECKPOINT_PATH_OR_SHA")
            continue
        path=Path(loc)
        if path.is_symlink() or not path.is_file() or path.as_posix() in paths:
            issues.append("MISSING_UNSAFE_OR_DUPLICATE_ORIGINAL_CHECKPOINT")
        else:count+=1
        paths.add(path.as_posix())
    return {"status":"BLOCKED_NO_SCORING" if issues else "FILE_COHORT_PRESENT_UNVALIDATED",
            "missing":sorted(set(issues)),"checkpoint_count":count,
            "DEV_NLL":"NOT_RUN","TEST":"NOT_ACCESSED"}

def instantiate(arch,bdh_source):
    if arch=="tac_context026":return load_tac()
    if arch=="causal_transformer":return ByteTransformer()
    if arch=="bdh_official":
        if bdh_source is None:raise ValueError("official BDH source missing")
        return load_bdh(bdh_source)
    if arch=="mamba3_siso_official":return load_mamba3()
    raise ValueError("unexpected benchmark architecture")

@torch.no_grad()
def authentic_dev_metrics(model,windows):
    model.eval()
    by_doc={}
    total={"n":0,"nll":0.,"brier":0.,"top1":0,"top5":0}
    device=next(model.parameters()).device
    for name,blocks in sorted(windows.items()):
        x=torch.tensor([list(raw) for _,raw in blocks],dtype=torch.long,device=device)
        if x.shape!=(8,129):raise ValueError("different genuine original DEV windows")
        outs=model(x[:,:-1])
        if not torch.is_tensor(outs):
            outs=outs.logits if hasattr(outs,"logits") else outs[0]
        if outs.shape!=(8,128,256):raise ValueError("wrong model next-byte forecast shape")
        logits=outs.float()
        targets=x[:,1:]
        logp=F.log_softmax(logits,dim=-1)
        lp=logp.gather(-1,targets.unsqueeze(-1)).squeeze(-1)
        prob=logp.exp()
        correct=(logits.argmax(-1)==targets)
        top5=(logits.topk(5,dim=-1).indices==targets.unsqueeze(-1)).any(-1)
        brier=1. - 2.*prob.gather(-1,targets.unsqueeze(-1)).squeeze(-1) + prob.square().sum(-1)
        n=targets.numel()
        row={"n":n,"nll_nats":float(-lp.double().sum()),
             "brier_sum":float(brier.double().sum()),
             "top1_hits":int(correct.sum()),"top5_hits":int(top5.sum())}
        if any(not math.isfinite(row[k]) for k in ("nll_nats","brier_sum")):
            raise ValueError("nonfinite original DEV scoring")
        by_doc[name]={
            "n":n,"nll":row["nll_nats"]/n,"brier":row["brier_sum"]/n,
            "top1":row["top1_hits"]/n,"top5":row["top5_hits"]/n}
        total["n"]+=n
        total["nll"]+=row["nll_nats"]
        total["brier"]+=row["brier_sum"]
        total["top1"]+=row["top1_hits"]
        total["top5"]+=row["top5_hits"]
    if total["n"]!=4096 or len(by_doc)!=4:
        raise ValueError("original 4096 exposed DEV target contract changed")
    n=total["n"]
    loss=total["nll"]/n
    return {"n":n,"nll":loss,"bits_per_original_byte":loss/math.log(2),
            "brier":total["brier"]/n,"top1":total["top1"]/n,
            "top5":total["top5"]/n,"per_document":by_doc}

def evaluate(*,bdh_source):
    # All protected operator, source and compute checks happen BEFORE DEV source.
    if os.environ.get("PHONE_LAB_GPU_EXECUTION_ENABLED")!="true":
        raise PermissionError("operator GPU quota approval missing")
    state=cohort_readiness()
    if state["status"]!="FILE_COHORT_PRESENT_UNVALIDATED":
        raise PermissionError(f"not all original trained model checkpoints: {state['missing']}")
    from scripts.validate_research_parity import check_manifest
    manifest=json.loads(PARITY.read_text())
    gate=check_manifest(manifest,"final")
    if gate.get("status")!="PASS":
        raise PermissionError("missing complete verified postfit source/compute parity")
    rows={(r["architecture"],r["seed"]):r for r in manifest["arms"]}
    if torch.cuda.device_count()!=2 or any(torch.cuda.get_device_capability(i)[0]<8 for i in range(2)):
        raise RuntimeError("official Mamba3 BF16 requires two approved SM80+ GPUs")
    # Cold checkpoint replay BEFORE opening DEV labels.
    checkpoint_models={}
    fit=fit_sources()
    for arch in ARM_ORDER:
        for seed in SEEDS:
            row=rows[(arch,seed)]
            torch.cuda.set_device(SEEDS.index(seed)%2)
            model=instantiate(arch,bdh_source)
            expected_count=dict(zip(ARM_ORDER,EXPECTED))[arch]
            if sum(p.numel() for p in model.parameters())!=expected_count:
                raise ValueError("scored architecture/source differs")
            optimizer=torch.optim.AdamW(model.parameters(),lr=.001)
            raw=Path(row["checkpoint_path"]).read_bytes()
            r=verify_bytes(raw,fresh_model=model,fresh_optimizer=optimizer,
                expected_name=arch,seed=seed,
                sample_order_sha=schedule_receipt(fit,seed)["sample_order_sha256"],
                expected_digest=row["measured"]["checkpoint_sha256"],
                diagnostic_only=False)
            if not r["status"].startswith("PASS_SOURCE_PINNED"):
                raise ValueError("original trained checkpoint not replayable")
            device=f"cuda:{SEEDS.index(seed)%2}"
            checkpoint_models[(arch,seed)]=model.to(device).eval()
    # Only now open exactly the prior-exposed DEV partition.
    sys.path.insert(0,str(ROOT/"research/mathematical-engineering/langbase024"))
    sys.path.insert(0,str(ROOT/"research/mathematical-engineering/langtrain025"))
    from byte_next_reference import original_source
    from train_byte_native import windows
    fit_orig,dev,supp=original_source(ROOT)
    assert len(fit_orig)==13 and len(dev)==4 and supp[0] in fit
    cases=windows(dev)
    if len(cases)!=4:raise ValueError("DEV source changed")
    results={}
    for (arch,seed),model in checkpoint_models.items():
        results[f"{arch}/{seed}"]=authentic_dev_metrics(model,cases)
    return {"status":"INTERNAL_EXPOSED_DEV_COMPLETE_NOT_INDEPENDENT_QUALIFICATION",
            "study":"LANGCOMPARE029","cohort_size":len(results),
            "sources":"PSF CPython 3.13.0 frozen original 4 exposed DEV owners",
            "score":results,"protected_TEST":"NOT_ACCESSED",
            "superiority":"NOT_CERTIFIED_BY_DEV"}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode",choices=("dry-run","score"),default="dry-run")
    p.add_argument("--bdh-source",type=Path)
    p.add_argument("--output",type=Path)
    args=p.parse_args()
    r=cohort_readiness() if args.mode=="dry-run" else evaluate(bdh_source=args.bdh_source)
    data=json.dumps(r,sort_keys=True,indent=2)+"\n"
    if args.output:
        if args.output.exists():raise FileExistsError("refuse overwriting original result")
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(data)
    print(data,end="")

if __name__=="__main__":
    main()
