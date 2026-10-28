#!/usr/bin/env python3
"""LANGCOMPARE029 content-verified safe checkpoint and no-fit replay primitives.

This module DOES NOT train; use original pinned model+optimizer identities.
Bytes are loaded with torch.load(weights_only=True), without executing pickles.
Current CI probes only zero-update diagnostic snapshots and genuine FIT prefixes.
Final trained-model promotion requires a separate full clean Git replay and
matched-compute receipts. Existing LANGTRAIN025 originals are immutable.
"""
from __future__ import annotations
import hashlib
import io
import json
import random
from pathlib import Path
import numpy as np
import torch
from scripts.langcompare029_preflight import parameter_count

STUDY="LANGCOMPARE029"
SOURCE_COMMIT="60403a5409ff2c3f3b07dd2ca91a7a3e096839c7"
SEEDS=(1701,1702,1703,1704,1705)
MODELS=("tac_context026","mamba3_siso_official","bdh_official","causal_transformer")
PARAM_LIMIT=(475000,525000)
EXPECTED_EXPOSURES=327680

def sha(b: bytes)->str:
    return hashlib.sha256(b).hexdigest()

def _safe_numpy_rng():
    s=np.random.get_state()
    return {"generator":s[0],"state":[int(x) for x in s[1].tolist()],
            "position":int(s[2]),"has_gaussian":int(s[3]),"cached":float(s[4])}

def create_bytes(*,model,optimizer,model_name,seed,source_order_sha,updates,
                 diagnostic_only=False):
    if model_name not in MODELS or seed not in SEEDS:
        raise ValueError("unregistered architecture/seed")
    if not isinstance(source_order_sha,str) or len(source_order_sha)!=64:
        raise ValueError("missing real FIT sample-order SHA256")
    if not isinstance(updates,int) or not 0<=updates<=320:
        raise ValueError("invalid optimizer update count")
    if diagnostic_only:
        if updates!=0:raise ValueError("diagnostic snapshot MUST be untrained")
    elif updates!=320:
        raise ValueError("trained record MUST contain all registered 320 steps")
    if optimizer is None:raise ValueError("original optimizer required")
    counts=parameter_count(model)
    if counts["trainable"]!=counts["deployed"] or not all(PARAM_LIMIT[0]<=v<=PARAM_LIMIT[1] for v in counts.values()):
        raise ValueError("checkpoint has wrong active/deployed parameter count")
    if not diagnostic_only and not optimizer.state:
        raise ValueError("trained optimizer has no original state")
    data={
        "kind":"UNTRAINED_REPLAY_DIAGNOSTIC" if diagnostic_only else "FULL_TRAINED_CHECKPOINT_NOT_QUALIFIED",
        "study":STUDY,"model_name":model_name,"seed":seed,"updates":updates,
        "source_commit":SOURCE_COMMIT,"source_order_sha256":source_order_sha,
        "counts":counts,
        "model_state":model.state_dict(),
        "optimizer_state":optimizer.state_dict(),
        "rng_torch":torch.get_rng_state(),
        "rng_cuda":torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        "rng_python":random.getstate(),
        "rng_numpy":_safe_numpy_rng(),
    }
    out=io.BytesIO()
    torch.save(data,out)
    return out.getvalue()

def verify_bytes(raw:bytes,*,fresh_model,fresh_optimizer,expected_name,seed,
                 sample_order_sha,expected_digest=None,diagnostic_only=False):
    if expected_digest is not None and sha(raw)!=expected_digest:
        raise ValueError("checkpoint raw bytes SHA256 mismatch")
    original=torch.load(io.BytesIO(raw),map_location="cpu",weights_only=True)
    wanted_kind="UNTRAINED_REPLAY_DIAGNOSTIC" if diagnostic_only else "FULL_TRAINED_CHECKPOINT_NOT_QUALIFIED"
    if any([
        original.get("kind")!=wanted_kind,
        original.get("study")!=STUDY,
        original.get("model_name")!=expected_name,
        original.get("seed")!=seed,
        original.get("updates")!=(0 if diagnostic_only else 320),
        original.get("source_commit")!=SOURCE_COMMIT,
        original.get("source_order_sha256")!=sample_order_sha,
        original.get("counts")!=parameter_count(fresh_model),
    ]):
        raise ValueError("checkpoint metadata / active model / original FIT schedule mismatch")
    for key in ("rng_torch","rng_cuda","rng_python","rng_numpy","optimizer_state","model_state"):
        if key not in original:raise ValueError("checkpoint RNG/optimizer/model state incomplete")
    fresh_model.load_state_dict(original["model_state"],strict=True)
    fresh_optimizer.load_state_dict(original["optimizer_state"])
    if not diagnostic_only and not fresh_optimizer.state:
        raise ValueError("missing trained optimizer moment history")
    return {"status":"PASS_SAME_SOURCE_COLD_LOAD_UNTRAINED" if diagnostic_only else
        "PASS_SOURCE_PINNED_CHECKPOINT_LOAD_NOT_EVALUATED",
        "sha256":sha(raw),"seed":seed,"model":expected_name,
        "updates":original["updates"],
        "weights_original_check":True,
        "optimizer_original_check":True,
        "rng_original_check":True,
        "DEV_scored":False,"TEST_opened":False}

def compare_no_fit_logits(original,replayed,prefix:torch.Tensor,*,atol=0.):
    """FIT-only no-fit snapshot equivalence, NOT a prediction score."""
    original.eval();replayed.eval()
    with torch.no_grad():
        x=original(prefix)
        y=replayed(prefix)
        if not torch.is_tensor(x):x=x.logits if hasattr(x,"logits") else x[0]
        if not torch.is_tensor(y):y=y.logits if hasattr(y,"logits") else y[0]
        if x.shape!=y.shape or x.shape[-1]!=256:
            raise ValueError("cold replay model output shape mismatch")
        worst=float((x-y).abs().max())
        if not np.isfinite(worst) or worst>atol:
            raise ValueError(f"cold replay logits differ: {worst}")
    return worst
