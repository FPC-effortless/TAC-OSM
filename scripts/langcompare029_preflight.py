#!/usr/bin/env python3
"""LANGCOMPARE029: NO-FIT exact source/parameter/causal preflight.

Never trains, scores DEV targets or checks protected TEST. Official Mamba-3
and BDH must be loaded from source-pinned external installations; missing
implementations remain BLOCKED, not replaced with simulated substitutes.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
import math
import sys
from pathlib import Path

import torch
from torch import nn

ROOT=Path(__file__).resolve().parents[1]
SOURCE_SHA="c88f3f7c1b628cb733a4a58dcd168a75411d8bb3d80e382068593713e887510a"
BDH_GIT_BLOB="4cfff79df97fa6faafb33bf20739a027edd8f360"
SEED=1701
TAC=500122
MAMBA_ANALYTICAL=501884
BDH_ANALYTICAL=501504
TRANSFORMER=496960
ARM_ORDER=("tac_context026","mamba3_siso_official","bdh_official","causal_transformer")
EXPECTED=(TAC,MAMBA_ANALYTICAL,BDH_ANALYTICAL,TRANSFORMER)

class ByteTransformer(nn.Module):
    """Strictly causal independent embedding/head, deterministic positions."""
    def __init__(self):
        super().__init__()
        d=96
        self.embedding=nn.Embedding(256,d)
        layer=nn.TransformerEncoderLayer(d_model=d,nhead=4,dim_feedforward=384,
                                        dropout=0.0,batch_first=True,norm_first=True)
        self.layers=nn.TransformerEncoder(layer,num_layers=4,enable_nested_tensor=False)
        self.norm=nn.LayerNorm(d)
        self.head=nn.Linear(d,256)
    def forward(self,tokens):
        b,t=tokens.shape
        idx=torch.arange(t,device=tokens.device,dtype=self.embedding.weight.dtype)
        inv=torch.exp(-math.log(10000.0)*torch.arange(0,96,2,device=tokens.device,dtype=idx.dtype)/96)
        angles=idx[:,None]*inv[None,:]
        pe=torch.stack((angles.sin(),angles.cos()),dim=-1).reshape(t,96)
        x=self.embedding(tokens)+pe[None,:,:]
        mask=torch.ones(t,t,device=tokens.device,dtype=torch.bool).triu(1)
        return self.head(self.norm(self.layers(x,mask=mask)))

def parameter_count(model):
    train=sum(p.numel() for p in model.parameters() if p.requires_grad)
    deployed=sum(p.numel() for p in model.parameters())
    return {"trainable":train,"deployed":deployed}

def compare_prefix(model, device, atol):
    torch.manual_seed(SEED)
    a=torch.tensor([[66,97,115,105,99,32,65,66]],dtype=torch.long,device=device)
    b=a.clone();b[:,5:]=torch.tensor([109,110,111],device=device)
    model.eval()
    with torch.no_grad():
        x=model(a);y=model(b)
        if not torch.is_tensor(x): x=x.logits if hasattr(x,"logits") else x[0]
        if not torch.is_tensor(y): y=y.logits if hasattr(y,"logits") else y[0]
        if x.shape!=(1,8,256) or y.shape!=(1,8,256):raise ValueError("wrong logits geometry")
        err=(x[:,:5]-y[:,:5]).abs().max().item()
        if not math.isfinite(err) or err>atol:raise ValueError(f"future poison changed prefix outputs: {err}")
    return float(err)

def load_tac():
    p=ROOT/"research/mathematical-engineering/context026/context_transition.py"
    if hashlib.sha256(p.read_bytes()).hexdigest()!=SOURCE_SHA:
        raise ValueError("CONTEXT026 source changed; do not benchmark silently")
    spec=importlib.util.spec_from_file_location("context_transition029",str(p))
    mod=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.ContextTransition026(vocab=256,ffn_width=501)

def load_bdh(path):
    if path is None:return None
    p=Path(path)
    raw=p.read_bytes()
    git_sha=hashlib.sha1(b"blob "+str(len(raw)).encode()+b"\\0"+raw).hexdigest()
    if git_sha!=BDH_GIT_BLOB:raise ValueError("BDH file differs from audited official Git blob")
    spec=importlib.util.spec_from_file_location("bdh_official_029",str(p))
    mod=importlib.util.module_from_spec(spec)
    sys.modules[spec.name]=mod
    spec.loader.exec_module(mod)
    return mod.BDH(mod.BDHConfig(n_layer=6,n_embd=48,dropout=0.1,
        n_head=4,mlp_internal_dim_multiplier=69,vocab_size=256))

def load_mamba3():
    if not torch.cuda.is_available():return None
    # This imports the *official* Mamba-3 layer and correct LM wrapper.
    from mamba_ssm.models.config_mamba import MambaConfig
    from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
    config=MambaConfig(d_model=80,n_layer=6,d_intermediate=128,vocab_size=256,
        ssm_cfg={"layer":"Mamba3","d_state":32,"expand":2,"headdim":32,
                 "is_mimo":False,"chunk_size":64,"is_outproj_norm":False},
        attn_layer_idx=[],rms_norm=False,residual_in_fp32=True,
        fused_add_norm=False,tie_embeddings=False,pad_vocab_size_multiple=8)
    return MambaLMHeadModel(config,dtype=torch.bfloat16,device="cuda")

def source_audit():
    src=ROOT/"research/mathematical-engineering/langbase024"
    sys.path.insert(0,str(src))
    from byte_next_reference import original_source
    fit,dev,supplement=original_source(ROOT)
    if len(fit)!=13 or len(dev)!=4:raise ValueError("frozen source partition changed")
    return {"publisher":"CPython 3.13.0 60403a5409ff2c3f3b07dd2ca91a7a3e096839c7",
            "fit_docs":len(fit)+1,"exposed_dev_docs":len(dev),
            "source_receipt_verified":True,"utf8_fit_supplement":supplement[0],
            "test_access":"NONE","dev_scoring":"NONE"}

def preflight(*,bdh_path=None,attempt_mamba=False,require_complete=False,check_causality=True):
    torch.set_num_threads(2)
    doc=source_audit()
    models={"tac_context026":load_tac(),"causal_transformer":ByteTransformer()}
    if bdh_path is not None:models["bdh_official"]=load_bdh(bdh_path)
    if attempt_mamba:models["mamba3_siso_official"]=load_mamba3()
    rows={}
    for name,expected in zip(ARM_ORDER,EXPECTED):
        model=models.get(name)
        if model is None:
            rows[name]={"status":"PENDING_OFFICIAL_SOURCE_OR_ACCELERATOR",
                        "analytical_parameters":expected}
            continue
        actual=parameter_count(model)
        if actual["trainable"]!=expected or actual["deployed"]!=expected:
            raise ValueError(f"{name} ACTUAL parameter count {actual}, expected {expected}")
        row={"status":"PARAMETER_COUNT_VERIFIED","actual_parameters":actual}
        if check_causality:
            device="cuda" if name=="mamba3_siso_official" else "cpu"
            model=model.to(device)
            try: row["future_poison_prefix_abs_error"]=compare_prefix(
                model,device,3e-3 if device=="cuda" else 1e-5)
            except Exception as exc:raise RuntimeError(f"{name} causality preflight failed") from exc
        rows[name]=row
    spread=(max(EXPECTED)-min(EXPECTED))/max(EXPECTED)
    if not spread<=0.01:raise ValueError("analytical parameter-parity failed")
    ready=all(row["status"]=="PARAMETER_COUNT_VERIFIED" for row in rows.values())
    if require_complete and not ready:raise RuntimeError("four-arm native preflight INCOMPLETE")
    return {"study_id":"LANGCOMPARE029","status":"NO_FIT_FOUR_ARM_PREFLIGHT_VERIFIED" if ready
        else "NO_FIT_PARTIAL_PREFLIGHT_NOT_READY_TO_TRAIN",
        "parameter_spread_analytical_fraction":spread,"models":rows,"source":doc,
        "gradient_checks":"NOT_RUN","optimizer_steps":0,"gpu_minutes":0,
        "DEV_NLL":"NOT_RUN","protected_TEST":"NOT_ACCESSED",
        "training_authorized":False,"need_owner_scale_rule_amendment":True,
        "need_matched_compute_receipts":True,"need_official_dependency_pins":True}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bdh-source",type=Path,default=None)
    p.add_argument("--attempt-official-mamba3",action="store_true")
    p.add_argument("--require-complete",action="store_true")
    p.add_argument("--output",type=Path)
    args=p.parse_args()
    r=preflight(bdh_path=args.bdh_source,attempt_mamba=args.attempt_official_mamba3,
                require_complete=args.require_complete)
    raw=json.dumps(r,indent=2,sort_keys=True,allow_nan=False)+"\\n"
    if args.output:
        if args.output.exists():raise ValueError("refuse to overwrite prior receipts")
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(raw)
    print(raw,end="")
if __name__=="__main__":main()
