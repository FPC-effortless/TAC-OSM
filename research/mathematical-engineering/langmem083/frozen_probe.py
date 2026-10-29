"""LANGMEM083 frozen original TAC slow-matrix historical-byte readout probe.

All feature vectors derive from ACTUAL original published source prefix:
M_t: original 079 slow8x8 state after observed t, original old writer
w_(a-1) stored at historical event a=t-age. Future byte raw[t+1],
old byte raw[a], and isolated source event Q_a never enter feature.
An auxiliary 9120-parameter MLP is trained on ORIGINAL FIT only.
The two NEW licensed publisher DEV documents are loaded only after all
three heads and their optimizer/RNG checkpoint hashes are frozen.

This is a privileged *supplied old address* probe, not autonomous TAC
history retrieval or new full architecture generalization.
"""
from __future__ import annotations
import hashlib
import json
import math
import random
from pathlib import Path
import sys
import time
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

ROOT=Path(__file__).resolve().parents[1]
for name in ("langmem083","langmem082","langmem081","langmem080","langmem079","langmem078",
             "langscale075","langscale070","context026","langmemory042","langbase024"):
    x=str(ROOT/"research/mathematical-engineering"/name)
    if x not in sys.path:sys.path.insert(0,x)
sys.path[:0]=[str(ROOT),str(ROOT/"src")]

from scripts.langmem079_carried_eval import restore_original
from scripts.langmem080_event_trace import source_factors
from dual_streaming import DualNativeStreamingTAC079
from scripts.langlayer035_sampling import fit_sources

AGES=(8,32,128,256)
ARMS=("correct","address_only","wrong_key")
HIDDEN=32
INPUT=20
NUM_CLASSES=256
STEPS=120
BATCH=128
LR=0.003
SEEDS=(1701,1702,1703)
CLOCKS=range(128,640,4)
FEATURE_COUNT=9120
DEPLOYED_PARAMETERS=508747
SOURCE_VERSION="LANGMEM083-FROZEN-PROBE-v1"

class FrozenReadout083(nn.Module):
    def __init__(self):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(INPUT,HIDDEN),nn.ReLU(),nn.Linear(HIDDEN,NUM_CLASSES))
        if sum(p.numel() for p in self.parameters())!=FEATURE_COUNT:
            raise ValueError("registered source probe param count not 9120")
    def forward(self,x):
        if x.shape[-1]!=INPUT:raise ValueError("source feature width must be20")
        return self.net(x)


def source_tensor_digest(model):
    digest=hashlib.sha256()
    for name,parameter in sorted(model.state_dict().items()):
        digest.update(name.encode()+b"\0")
        digest.update(parameter.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def wrong_writer_key(keys,prefix,a,t,correct):
    """Find most similar *wrong 4-byte observed history* writer.

    Source history access limited to actual bytes 0..t already observed.
    Selection ignores the historical byte raw[a] (training label) and any
    future raw[t+1] byte; compares old preceding source contexts only.
    """
    old=prefix[a-4:a]
    candidates=[b for b in range(4,t) if b!=a and prefix[b-4:b]!=old]
    if not candidates:
        raise ValueError("not enough distinct source context for wrong-key control")
    writer=max(candidates,key=lambda b:(float((keys[b]*correct).sum()),b))
    return keys[writer]


@torch.no_grad()
def causal_feature_set(model,sources,*,source_role):
    """Only genuine original source bytes; never mixes source document owners.

    FIT and unseen DEV call this separately. Caller must not open unseen DEV
    before saving and SHA-sealing three trained probe checkpoints.
    """
    if source_role not in ("FIT","UNSEEN_DEV"):raise ValueError("source role")
    engine=DualNativeStreamingTAC079(model,"cap_all")
    out={arm:[] for arm in ARMS}
    labels=[];metadata=[];original_hash=hashlib.sha256()
    for owner,raw in sorted(sources.items()):
        if type(raw) is not bytes or len(raw)<641:
            raise ValueError("source owner bytes missing/too short")
        state=engine.start((owner,))
        keys={}
        for t in range(640):
            if state.clock!=t or state.tensor_bytes()!=4160:
                raise ValueError("original trained source carry not a genuine 4160B state")
            observed=torch.tensor([raw[t]],dtype=torch.long)
            f=source_factors(model,state,observed,mode="cap_all")
            _,next_state=engine.step(observed,state,owners=(owner,))
            if t>=4:keys[t]=f["write_key"].reshape(8).detach().clone()
            state=next_state
            if t not in CLOCKS:continue
            # All vectors are determined by the original observed source
            # 0..t and trained frozen model. The old *target* byte raw[a]
            # is only accessed below after constructing every feature arm.
            prefix=raw[:t+1]
            for i,age in enumerate(AGES):
                a=t-age
                if a<4:continue
                key=keys[a]
                wrong=wrong_writer_key(keys,prefix,a,t,key)
                M=state.tensors["slow_native"][0]
                correct_read=(M@key).reshape(8)
                wrong_read=(M@wrong).reshape(8)
                onehot=torch.zeros(4,dtype=torch.float32)
                onehot[i]=1.
                out["correct"].append(torch.cat((correct_read,key,onehot)).clone())
                out["address_only"].append(torch.cat((torch.zeros_like(correct_read),key,onehot)))
                out["wrong_key"].append(torch.cat((wrong_read,wrong,onehot)).clone())
                # Historical byte label exists in genuine already-observed
                # history. It is intentionally *not* in model inputs.
                y=int(raw[a])
                labels.append(y)
                metadata.append((owner,t,a,age))
                original_hash.update(owner.encode()+b":"+str(t).encode()+b":"+
                                     str(a).encode()+b":"+bytes([y])+b";")
    if not labels or any(len(v)!=len(labels) for v in out.values()):
        raise ValueError("registered genuine source target/feature count mismatch")
    xs={arm:torch.stack(parts).float() for arm,parts in out.items()}
    y=torch.tensor(labels,dtype=torch.long)
    if any(x.shape!=(len(labels),20) or not torch.isfinite(x).all() for x in xs.values()):
        raise ValueError("source readout feature value invalid")
    if source_role=="FIT" and len(sources)!=14:
        raise ValueError("original 14 FIT source owners changed")
    if source_role=="UNSEEN_DEV" and len(sources)!=2:
        raise ValueError("new independent source documents missing")
    return xs,y,metadata,original_hash.hexdigest()


def run_training(seed,original_model,fit_source,output):
    if seed not in SEEDS:raise ValueError("unregistered pretrained seed")
    if output.exists():raise FileExistsError("no original source fit output overwrite")
    output.mkdir(parents=True)
    for p in original_model.parameters():p.requires_grad_(False)
    original_model.eval()
    frozen_before=source_tensor_digest(original_model)
    cpu_start=time.process_time()
    x,y,meta,source_sha=causal_feature_set(original_model,fit_source,source_role="FIT")
    # Exactly same genuine target rows and sequence for all auxiliary arms.
    sample_gen=torch.Generator().manual_seed(seed+83000)
    schedules=torch.randint(0,len(y),(STEPS,BATCH),generator=sample_gen)
    schedule_hash=hashlib.sha256(schedules.numpy().tobytes()).hexdigest()
    counts=torch.zeros((len(AGES),256),dtype=torch.int64)
    for i,(_,t,a,age) in enumerate(meta):
        counts[AGES.index(age),int(y[i])]+=1
    receipt={"study":"LANGMEM083","seed":seed,"source_role":"ORIGINAL_PUBLISHER_FIT",
       "original_model_checkpoint_frozen":True,"original_model_param_count":499627,
       "probe_learned_parameters_per_arm":FEATURE_COUNT,
       "deployed_total_parameters_with_frozen_original":DEPLOYED_PARAMETERS,
       "num_original_fit_owners":len(fit_source),"fit_historical_labels":len(y),
       "fit_historical_target_sha256":source_sha,
       "fit_source_sampler_sha256":schedule_hash,
       "trained_updates_per_arm":STEPS,"batch_size_per_arm":BATCH,
       "optimizer":"AdamW","learning_rate":LR,
       "fit_frequency_by_age":counts.tolist(),
       "original_encoder_weights_sha256_before":frozen_before,
       "checkpoint_receipts":{},"gpu_minutes":0,
       "protected_TEST":"NOT_ACCESSED",
       "unseen_DEV_accessed_before_probes_frozen":False,
       "no_original_model_updates":True}
    for arm in ARMS:
        # same initialized probe weights, same SGD samples, same real labels
        torch.manual_seed(seed+83110)
        probe=FrozenReadout083()
        optimizer=torch.optim.AdamW(probe.parameters(),lr=LR)
        pweights=hashlib.sha256(b"".join(
            t.detach().numpy().tobytes() for t in probe.state_dict().values())).hexdigest()
        probe.train()
        for row in schedules:
            logits=probe(x[arm][row])
            loss=F.cross_entropy(logits,y[row])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if not (math.isfinite(float(loss.item())) and all(
                par.grad is not None and torch.isfinite(par.grad).all() for par in probe.parameters())):
                raise ValueError("probe fit gradient/loss invalid")
            optimizer.step()
        probe.eval()
        path=output/f"PROBE_{arm}.pt"
        state={"study":"LANGMEM083","seed":seed,"arm":arm,"steps":STEPS,
               "probe_model":probe.state_dict(),"probe_optimizer":optimizer.state_dict(),
               "original_model_weight_sha256":frozen_before,
               "source_fit_target_sha256":source_sha,
               "fit_sampler_sha256":schedule_hash,
               "initial_probe_weights_sha256":pweights,
               "python_rng_state":random.getstate(),
               "numpy_rng_state":np.random.get_state()[1].tolist(),
               "torch_rng_state":torch.get_rng_state()}
        torch.save(state,path)
        h=hashlib.sha256(path.read_bytes()).hexdigest()
        replay=torch.load(path,map_location="cpu",weights_only=True)
        restored=FrozenReadout083()
        restored.load_state_dict(replay["probe_model"],strict=True)
        with torch.no_grad():
            for k in (0,min(23,len(y)-1)):
                if not torch.equal(probe(x[arm][k:k+1]),restored(x[arm][k:k+1])):
                    raise ValueError("original auxiliary trained probe cold replay different logits")
        receipt["checkpoint_receipts"][arm]={"checkpoint_sha256":h,
             "name":path.name,"actual_trainable_parameters":FEATURE_COUNT,
             "training_updates":STEPS,"initial_probe_weights_sha256":pweights,
             "cold_replay_logits_bitwise":True}
    receipt["original_encoder_weights_sha256_after"]=source_tensor_digest(original_model)
    if receipt["original_encoder_weights_sha256_after"]!=frozen_before:
        raise ValueError("original frozen 078 TAC model weights mutated")
    if len({v["initial_probe_weights_sha256"] for v in receipt["checkpoint_receipts"].values()})!=1:
        raise ValueError("matched probe arm neural weight initialization mismatch")
    receipt["fit_cpu_process_seconds"]=time.process_time()-cpu_start
    receipt_path=output/"FIT_RECEIPT.json"
    receipt_path.write_text(json.dumps(receipt,sort_keys=True,indent=2,allow_nan=False)+"\n")
    print("LANGMEM083_TRAINED "+json.dumps({"seed":seed,"fit_rows":len(y),
        "checkpoints":receipt["checkpoint_receipts"],
        "old_model_weights_unchanged":True},sort_keys=True),flush=True)
    return receipt
