#!/usr/bin/env python3
"""LANGSCALE075: authentic publisher byte TAC fixed-recurrence FFN capacity pilot 160x3 CPU fit.

This run FITS for real, on GitHub Actions CPU only, using an explicitly
preregistered *single_model_training* exploratory lane: 160 AdamW steps/seed,
common source windows, different active parameter/FLOP budgets, no scaling law or architecture win.
All original LANGLAYER039 / 035 / 037 models and checkpoints remain unchanged.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import random
import resource
import sys
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/"src"))
sys.path.insert(0,str(ROOT/"research/mathematical-engineering/langbase024"))
sys.path.insert(0,str(ROOT/"research/mathematical-engineering/langtrain025"))
sys.path.insert(0,str(ROOT/"research/mathematical-engineering/context026"))
sys.path.insert(0,str(ROOT/"research/mathematical-engineering/langscale075"))

import numpy as np
import torch
from torch.nn import functional as F
from native_matrix import NativeMatrixTAC075, COUNTS, WIDTHS
def parameter_count(model):
    return {"trainable":sum(p.numel() for p in model.parameters() if p.requires_grad),
            "deployed":sum(p.numel() for p in model.parameters())}
from scripts.langlayer035_sampling import fit_sources,iter_batches,schedule_receipt
from train_byte_native import windows as authentic_dev_windows, score_model
def authentic_dev_metrics(model, dev_windows):
    return score_model(model, dev_windows)["aggregate"]
from scripts.validate_research_parity import check_manifest
from byte_next_reference import original_source

REGISTRATION=ROOT/"research/mathematical-engineering/langscale075/FAIR_COMPARISON.json"
STUDY="LANGSCALE075"
ARMS={k:v for k,v in COUNTS.items() if k!="m8"}
SEEDS=(1701,1702,1703)
STEPS=160
BATCH=8
WINDOW=129
SOURCE_COMMIT="60403a5409ff2c3f3b07dd2ca91a7a3e096839c7"

def digest(blob):
    return hashlib.sha256(blob).hexdigest()

def logits_of(model,tokens):
    output=model(tokens)
    if torch.is_tensor(output):return output
    if hasattr(output,"logits"):return output.logits
    return output[0]

SOURCE_MODEL_SHA256="c88f3f7c1b628cb733a4a58dcd168a75411d8bb3d80e382068593713e887510a"
SCALING_MODEL_GIT_BLOB="a9fdf6119a70081288294e33e77e84c0b4a1152f"

def model_source_preflight():
    source=(ROOT/"research/mathematical-engineering/context026/context_transition.py").read_bytes()
    upper=(ROOT/"research/mathematical-engineering/langscale075/native_matrix.py").read_bytes()
    if hashlib.sha256(source).hexdigest()!=SOURCE_MODEL_SHA256:
        raise ValueError("CONTEXT026 source differs from audited pinned original")
    source_digest=hashlib.sha1(b"blob "+str(len(upper)).encode()+b"\0"+upper).hexdigest()
    if source_digest!=SCALING_MODEL_GIT_BLOB:
        raise ValueError("LANGSCALE075 scaling constructor differs from pinned blob")

def model_for(arm,bdh_source=None):
    if arm not in ARMS: raise ValueError("unregistered capacity scale")
    return NativeMatrixTAC075(tier=arm)

def prefit_registration(arm,seed,steps):
    if arm not in ARMS or seed not in SEEDS or steps!=STEPS:
        raise ValueError("pilot architecture/seed/step schedule differs from preregistration")
    m=json.loads(REGISTRATION.read_text())
    result=check_manifest(m,"preflight")
    if result.get("status")!="PASS_PREFIT_SINGLE_MODEL":
        raise ValueError("original single-model training preregistration failed")
    rows=m.get("models",[])
    if len(rows)!=6 or {(x.get("name"),x.get("seed")) for x in rows}!={(a,s) for a in ARMS for s in SEEDS}:
        raise ValueError("incomplete model matrix")
    if any(x["trainable_parameters"]!=ARMS[x["name"]] or
           x["deployed_parameters"]!=ARMS[x["name"]] for x in rows):
        raise ValueError("registered model parameter parity mismatch")
    # Approximate parameter parity (<1%) is maintained with fixed FFN455/upper48;
    # native and aligned address matrices become wider. FLOPs NOT matched.
    if tuple(sorted(ARMS.values()))!=(500175,500663):
        raise ValueError("invalid fixed capacity-scaling geometry")
    if m["budget"]["training_updates"]!=STEPS or m["budget"]["batch_size"]!=BATCH:
        raise ValueError("pilot budget differs")
    return m

def _safe_rng_numpy():
    state=np.random.get_state()
    return {"name":state[0],"keys":state[1].tolist(),
        "pos":int(state[2]),"has_gauss":int(state[3]),"cached_gauss":float(state[4])}

def checkpoint_blob(model,optimizer,arm,seed,steps,sample_hash):
    metadata={
        "study":STUDY,"arm":arm,"seed":seed,"updates":steps,
        "source_commit":SOURCE_COMMIT,"sample_order_sha256":sample_hash,
        "parameters":parameter_count(model),
        "model":model.state_dict(),
        "optimizer":optimizer.state_dict(),
        "rng_torch":torch.get_rng_state(),
        "rng_numpy":_safe_rng_numpy(),
        "rng_python":random.getstate(),
    }
    sink=io.BytesIO()
    torch.save(metadata,sink)
    return sink.getvalue()

def cold_reload(raw,arm,seed,bdh_source,sample_hash):
    d=torch.load(io.BytesIO(raw),map_location="cpu",weights_only=True)
    if (d["study"],d["arm"],d["seed"],d["updates"],d["source_commit"],d["sample_order_sha256"])!=(
        STUDY,arm,seed,STEPS,SOURCE_COMMIT,sample_hash):
        raise ValueError("original checkpoint/sample metadata mismatch")
    if any(k not in d for k in ("rng_torch","rng_numpy","rng_python","optimizer","model")):
        raise ValueError("missing original optimizer/RNG")
    m=model_for(arm,bdh_source)
    counts=parameter_count(m)
    if counts!=d["parameters"] or counts!={"trainable":ARMS[arm],"deployed":ARMS[arm]}:
        raise ValueError("model count changed upon cold reload")
    m.load_state_dict(d["model"],strict=True)
    o=torch.optim.AdamW(m.parameters(),lr=0.001)
    o.load_state_dict(d["optimizer"])
    if not o.state:raise ValueError("fitted optimizer moments not preserved")
    return m

def run(arm,seed,steps,bdh_source,out):
    if out.exists():raise FileExistsError("refuse overwrite evidence")
    model_source_preflight()
    prefit_registration(arm,seed,steps)
    # Both native capacity arms must operate on identical source and original byte sampling.
    fit=fit_sources()
    fs=schedule_receipt(fit,seed,steps=steps)
    if fs["examples_seen"]!=steps*BATCH or fs["next_byte_targets_seen"]!=steps*BATCH*128:
        raise ValueError("real source exposure incorrect")
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    torch.set_num_threads(2)
    model=model_for(arm,bdh_source).to("cpu").train()
    expected={"trainable":ARMS[arm],"deployed":ARMS[arm]}
    if parameter_count(model)!=expected:raise ValueError("actual model count mismatch")
    optimizer=torch.optim.AdamW(model.parameters(),lr=0.001)
    log=[]
    import datetime
    started=time.perf_counter()
    cpu_started=time.process_time()
    h=hashlib.sha256()
    observed_steps=0
    for step,indices in iter_batches(fit,seed,steps=steps):
        windows=[]
        for owner,offset,raw in indices:
            if raw!=fit[owner][offset:offset+WINDOW]:raise ValueError("source owner corruption")
            h.update(f"{owner}:{offset};".encode())
            windows.append(list(raw))
        b=torch.tensor(windows,dtype=torch.long)
        out_logits=logits_of(model,b[:,:-1])
        if out_logits.shape!=(BATCH,128,256):raise ValueError("next-byte logits geometry")
        loss=F.cross_entropy(out_logits.float().reshape(-1,256),b[:,1:].reshape(-1))
        if not torch.isfinite(loss).item():raise ValueError("nonfinite language loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if any(p.grad is None or not torch.isfinite(p.grad).all()
               for p in model.parameters() if p.requires_grad):
            raise ValueError("untrained or nonfinite model tensor gradient")
        norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
        if not math.isfinite(float(norm)):raise ValueError("nonfinite model gradient norm")
        optimizer.step()
        observed_steps=step+1
        log.append(float(loss.detach()))
        print(f"CPU_FIT {STUDY} arch={arm} seed={seed} update={observed_steps}/{steps} loss={log[-1]:.6f}",flush=True)
    if h.hexdigest()!=fs["sample_order_sha256"] or observed_steps!=steps:
        raise ValueError("FIT source replay digest/step budget mismatch")
    elapsed=time.perf_counter()-started
    cpu_seconds=time.process_time()-cpu_started
    ck=checkpoint_blob(model,optimizer,arm,seed,steps,fs["sample_order_sha256"])
    out.mkdir(parents=True,exist_ok=False)
    cp=out/f"{arm}_seed{seed}_fit{steps}.pt"
    cp.write_bytes(ck)
    ck_hash=digest(ck)
    restored=cold_reload(ck,arm,seed,bdh_source,fs["sample_order_sha256"])
    # The exposed DEV set is visited ONCE, only after frozen checkpoint created,
    # using identical exact original 4x8 129-byte windows. No TEST access.
    _,dev,_=original_source(ROOT)
    dev_windows=authentic_dev_windows(dev)
    score=authentic_dev_metrics(model,dev_windows)
    cold=authentic_dev_metrics(restored,dev_windows)
    for key in ("nll","brier","top1","top5"):
        if abs(score[key]-cold[key])>1e-6:
            raise ValueError("trained model original checkpoint cold replay changed DEV score")
    receipt={
      "study":STUDY,"arm":arm,"seed":seed,"updates":observed_steps,
      "status":"ACTUAL_CPU_FIT_NATIVE_MATRIX_WIDTH_UNMATCHED_FLOPS",
      "scientific_claim":"NATIVE_AND_ALIGNED_MEMORY_CAPACITY_8_12_16_RESET_ONLY_NOT_STREAMING",
      "source":"PSF CPython3.13.0 original hash-authenticated FIT bytes",
      "source_commit":SOURCE_COMMIT,
      "source_sample_order_sha256":fs["sample_order_sha256"],
      "source_bytes_sha256":fs["sample_bytes_sha256"],
      "examples_seen":fs["examples_seen"],
      "target_bytes_seen":fs["next_byte_targets_seen"],
      "parameters":expected,
      "optimizer":"AdamW lr=.001 no hyperparameter tuning, grad_norm_clip=1",
      "scale_axis":"NATIVE_ASSOCIATIVE_KEY_VALUE_AND_ALIGNED_ADDRESS",
      "ffn_width":455,"upper_state_width":48,
      "native_matrix_dim":WIDTHS[arm],
      "native_matrix_FP32_active_bytes_per_owner":4*WIDTHS[arm]*WIDTHS[arm],
      "aligned_matrix_FP32_bytes_per_owner":2*16*WIDTHS[arm]*4,
      "native_and_aligned_address_scaled":True,
      "loss_first":log[0],"loss_last":log[-1],
      "loss_mean":sum(log)/len(log),
      "DEV_EXPOSED_4096_ONLY":score,
      "checkpoint_filename":cp.name,"checkpoint_sha256":ck_hash,
      "checkpoint_reloaded_identical_scored_DEV":True,
      "measured_cpu_process_seconds":cpu_seconds,
      "measured_wall_seconds":elapsed,
      "peak_rss_kb_linux":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
      "matched_training_FLOPs":"NOT_MATCHED_DIFFERENT_NATIVE_MATRIX_WIDTH","GPU_minutes":0,
      "protected_TEST":"NOT_ACCESSED",
      "independently_qualified":False,
    }
    fp=out/"FIT_AND_EXPOSED_DEV_RECEIPT.json"
    fp.write_text(json.dumps(receipt,indent=2,sort_keys=True,allow_nan=False)+"\n")
    print("REAL_CPU_FIT_COMPLETE "+json.dumps({k:receipt[k] for k in
         ("arm","seed","updates","status","loss_last","measured_cpu_process_seconds","checkpoint_sha256")}),flush=True)
    return receipt

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--arm",choices=tuple(ARMS),required=True)
    p.add_argument("--seed",type=int,choices=SEEDS,required=True)
    p.add_argument("--steps",type=int,default=STEPS)
    p.add_argument("--bdh-source",type=Path)
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args()
    run(a.arm,a.seed,a.steps,a.bdh_source,a.out)
if __name__=="__main__":
    main()
