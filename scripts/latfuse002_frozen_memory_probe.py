"""LATFUSE002 trained original FIT-only no-update memory tomography.

Reading only the original trained checkpoints and genuine allowed FIT observations;
no optimizer, no source targets and no protected source numerical years. Lesion
output deltas are LOCAL sensitivity not causal world-model proof or recall.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import torch

from latfuse002_real_fit import (SOURCE_SHA,OUT,source_frames,calendar,
    FusedLatentFixedState,strict_budget,hour,dt)


def summary(x):
    x=x.detach().reshape(-1).float()
    return dict(mean=round(float(x.mean()),8),min=round(float(x.min()),8),
                max=round(float(x.max()),8))


def trace(source):
    d=source_frames(Path(source),phase="fit")
    date=dt.datetime(2014,6,1)
    start=hour(date)
    end=start+256
    assert end < hour(dt.datetime(2014,9,1))
    original=json.loads((OUT/"RESULTS.json").read_text())
    models=[]
    with torch.no_grad():
        for row in original["results"]:
            seed=row["seed"]
            path=OUT/f"seed-{seed}"/"final.pt"
            sha=hashlib.sha256(path.read_bytes()).hexdigest()
            if sha != row["original_checkpoint_sha256"]:
                raise RuntimeError("checkpoint identity changed")
            cp=torch.load(path,map_location="cpu",weights_only=False)
            model=FusedLatentFixedState()
            model.load_state_dict(cp["model"],strict=True)
            strict_budget(model)
            model.eval()
            ids=torch.arange(12)
            state=model.initial(12)
            alphas=[];betas=[]; keys=[]
            for t in range(start,end):
                x=d["values"][:,t]
                mask=d["valid"][:,t]
                age=d["age"][:,t]
                last=d["last"][:,t]
                u=model.encode(x,mask,age,last)
                alphas.append(torch.sigmoid(model.retention_map(u)).clone())
                betas.append(torch.sigmoid(model.write_map(u)).clone())
                keys.append(torch.nn.functional.normalize(model.key_map(u),dim=-1,eps=1e-8))
                state=model.step(state,x,mask,age,last,ids)
            query=calendar([end+6]*12).unsqueeze(1)
            base=model.forecast(state,ids,query)[0][:,0]
            lesions={}
            for part in ("fast","slow","associative","entities"):
                new_state=state._replace(**{part:torch.zeros_like(getattr(state,part))})
                pred=model.forecast(new_state,ids,query)[0][:,0]
                lesions[part]=round(float((pred-base).abs().mean()),8)
            # A single position's memory rank can be low even if matrix values are nonzero.
            singular=torch.linalg.svdvals(state.associative)
            rank=(singular>singular.max(dim=-1,keepdim=True).values.clamp_min(1e-8)*.001).sum(-1)
            k=torch.stack(keys)
            # Key collision within one stream across last32 real source hours.
            recent=k[-32:].permute(1,0,2)
            gram=(recent@recent.transpose(1,2)).abs()
            off=gram.masked_select(~torch.eye(32,dtype=torch.bool)[None].expand(12,-1,-1))
            a=torch.cat(alphas);b=torch.cat(betas)
            models.append(dict(seed=seed,original_checkpoint_sha256=sha,
                learned_retention=summary(a),learned_write=summary(b),
                scaled_key_pair_offdiagonal_abs_mean=round(float(off.mean()),8),
                matrix_effective_rank=summary(rank.float()),
                mean_absolute_local_forecast_lesion_delta=lesions,
                memory_weights_modified=False,train_updates_during_probe=0,
                note="source-conditioned read-only FIT path sensitivity; neither retrieval accuracy nor model necessity proof"))
    return dict(study="LATFUSE002-FROZEN-CHECKPOINT-READONLY",source_sha256=SOURCE_SHA,
       scope="original UCI501 FIT real observed 2014-06 first256 hourly steps per site; no outcomes",
       float_memory_bytes_per_stream=6800,model_count=len(models),models=models,
       scientific_capability_claim=False,model_fit_performed=False)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--source",required=True)
    v=p.parse_args()
    torch.set_num_threads(1)
    print("LATFUSE002_MEMORY_TRACE_JSON="+json.dumps(trace(v.source),sort_keys=True),flush=True)
