"""DEV-013: matched native CASM supervision-transfer study (development-only).

All treatment arms have identical model structure, parameter budgets, PLM
capacity and starting tensors. Auxiliary losses optionally tap the actual
native CASM readout, not a sidecar only. Causality, paired present inputs,
historical target separation and full-state reset checks are explicit.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import torch
import torch.nn.functional as F

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from tac_osm.development.casm_readout_alignment import (  # noqa: E402
    ARMS, CASMAlignedModel, matching_models,
)
from tac_osm.development.noisy_memory import parameter_hash  # noqa: E402
from tac_osm.development.protection_benchmark import (  # noqa: E402
    make_pair, fingerprint, seed_for, tensors, euclidean,
)
from run_rehearsal_writer_dev import (  # noqa: E402
    train_case, sample_rehearsal_inputs, conditions,
)

SEEDS=(43,59,71)
EVAL_SEEDS=(241,257,269)


def train(seed: int, steps: int):
    models=matching_models(seed)
    initials={n:parameter_hash(m) for n,m in models.items()}
    if len(set(initials.values()))!=1:
        raise AssertionError("initial weights not identical")
    opts={name:torch.optim.AdamW(
        (p for p in m.parameters() if p.requires_grad),
        lr=.003,weight_decay=.0001)
        for name,m in models.items()}
    metadata={"initial_hashes":initials,
              "train_fingerprints":[],"writer_gradient":{},
              "casm_aux_gradient":{},"last_loss":{}}
    writer_grad={name:False for name in ARMS}
    aux_grad={name:False for name in ARMS}
    for step in range(steps):
        a,v,q,c,y,fps,h,d,noise=train_case(seed,step)
        metadata["train_fingerprints"].extend(fps)
        (oldq,oldbit),(freshq,freshbit)=sample_rehearsal_inputs(
            a,v,h,d,noise,seed,step)
        g=torch.Generator().manual_seed(
            seed_for("dev013-shuffled-null",seed,step)%(2**63-1))
        no_old=torch.randint(0,2,oldbit.shape,generator=g)
        no_fresh=torch.randint(0,2,freshbit.shape,generator=g)
        for name,m in models.items():
            m.train()
            opts[name].zero_grad(set_to_none=True)
            state=m.encode(a,v,trace=False)
            task=F.cross_entropy(m.predict(state,q,c),y)
            aux=task.new_zeros(())
            if name=="separate_aux":
                aux=(F.cross_entropy(m.rehearsal_logits(state,oldq),oldbit)+
                     F.cross_entropy(m.rehearsal_logits(state,freshq),freshbit))/2
            elif name in ("aligned_aux","aligned_shuffled","aligned_detached"):
                detach=(name=="aligned_detached")
                t0=no_old if name=="aligned_shuffled" else oldbit
                t1=no_fresh if name=="aligned_shuffled" else freshbit
                aux=(F.cross_entropy(m.aligned_historical_logits(
                    state,oldq,detach_features=detach),t0)+
                     F.cross_entropy(m.aligned_historical_logits(
                    state,freshq,detach_features=detach),t1))/2
            loss=task+aux
            loss.backward()
            wgrad=m.memory.write.weight.grad
            writer_grad[name] |= bool(wgrad is not None and
                                      torch.isfinite(wgrad).all() and
                                      wgrad.abs().sum().item()>0)
            if name=="aligned_aux":
                selector_grad=m.read_selector[-1].weight.grad
                executor_grad=m.compute.selector[-1].weight.grad
                aux_grad[name] |= bool(
                    selector_grad is not None and torch.isfinite(selector_grad).all()
                    and selector_grad.abs().sum().item()>0
                    and executor_grad is not None and torch.isfinite(executor_grad).all()
                    and executor_grad.abs().sum().item()>0)
            torch.nn.utils.clip_grad_norm_(m.parameters(),1.0)
            opts[name].step()
            metadata["last_loss"][name]={
                "task_ce":float(task.detach()),
                "aux_ce":float(aux.detach()),
                "total_ce":float(loss.detach())}
    if not all(writer_grad.values()) or not aux_grad["aligned_aux"]:
        raise AssertionError("no task gradient to writer or aligned CASM routing")
    metadata["writer_gradient"]=writer_grad
    metadata["casm_aux_gradient"]=aux_grad
    metadata["trained_hashes"]={n:parameter_hash(m) for n,m in models.items()}
    if any(metadata["trained_hashes"][n]==initials[n] for n in ARMS):
        raise AssertionError("model did not train")
    for m in models.values():m.eval()
    return models,metadata


@torch.no_grad()
def evaluate(m: CASMAlignedModel, eval_seed: int, arm: str, smoke: bool):
    rows=[]
    for h,d,near,noise,role,rep in conditions(smoke):
        seed=seed_for("dev013-eval",eval_seed,h,d,near,noise,role,rep)
        pair=make_pair(seed,h,d,near,noise,role)
        eps=(pair.left,pair.right)
        a,v,q,c=tensors(list(eps))
        state=m.encode(a,v,trace=False)
        if state.shape!=(2,4,8,24):
            raise AssertionError("768-scalar PLM state changed")
        targets=torch.tensor([x.target for x in eps])
        bit_targets=targets^c
        decision=m.predict(state,q,c).argmax(-1)
        bit=m.rehearsal_logits(state,q) if arm=="separate_aux" else (
            m.aligned_historical_logits(state,q))
        bits=bit.argmax(-1)
        reset=m.predict(torch.zeros_like(state),q,c).argmax(-1)
        swapped=m.predict(state.flip(0),q,c).argmax(-1)
        bit_reset=(m.rehearsal_logits(torch.zeros_like(state),q)
                   if arm=="separate_aux" else
                   m.aligned_historical_logits(torch.zeros_like(state),q)).argmax(-1)
        # Deliberately NOT a TAC-OSM decision: the deterministic XOR is only
        # a diagnostic of how much error comes after historical-bit retrieval.
        diagnostic=(bits ^ c).tolist()
        classic=[euclidean(ep)[0] for ep in eps]
        rows.append({
            "condition":{"h":h,"d":d,"near":near,"noise":noise,
                         "role":role,"rep":rep},
            "fingerprints":[fingerprint(ep) for ep in eps],
            "targets":targets.tolist(),"bit_targets":bit_targets.tolist(),
            "task_pred":decision.tolist(),"bit_pred":bits.tolist(),
            "diagnostic_xor_pred":diagnostic,
            "reset_task_pred":reset.tolist(),
            "reset_bit_pred":bit_reset.tolist(),
            "swap_task_pred":swapped.tolist(),
            "full_history_euclidean_pred":classic,
            "full_history_scanned":[h+d,h+d],
            "pair_state_l2":float((state[0]-state[1]).norm()),
            "state_scalars_per_episode":state[0].numel(),
        })
    return rows


def summarize(rows):
    assert rows
    count=2*len(rows)
    res={"episodes":count}
    fields={
        "native_task_accuracy":("targets","task_pred"),
        "aux_bit_accuracy":("bit_targets","bit_pred"),
        "diagnostic_xor_accuracy":("targets","diagnostic_xor_pred"),
        "reset_task_accuracy":("targets","reset_task_pred"),
        "reset_bit_accuracy":("bit_targets","reset_bit_pred"),
        "swap_task_accuracy":("targets","swap_task_pred"),
        "euclidean_accuracy":("targets","full_history_euclidean_pred"),
    }
    for metric,(truth,pred) in fields.items():
        res[metric]=sum(int(x==y) for row in rows
                        for x,y in zip(row[truth],row[pred]))/count
    for role in ("retention","acquisition"):
        subset=[r for r in rows if r["condition"]["role"]==role]
        res[role+"_native_accuracy"]=(sum(
            int(x==y) for row in subset for x,y in
            zip(row["targets"],row["task_pred"]))/(2*len(subset))) if subset else None
        res[role+"_aux_accuracy"]=(sum(
            int(x==y) for row in subset for x,y in
            zip(row["bit_targets"],row["bit_pred"]))/(2*len(subset))) if subset else None
    long=[r for r in rows if r["condition"]["h"]==512]
    res["H512_native_accuracy"]=(sum(int(x==y) for row in long
        for x,y in zip(row["targets"],row["task_pred"]))/(2*len(long))) if long else None
    res["H512_aux_accuracy"]=(sum(int(x==y) for row in long
        for x,y in zip(row["bit_targets"],row["bit_pred"]))/(2*len(long))) if long else None
    return res


def run(out: Path, smoke: bool):
    torch.set_num_threads(2)
    out.mkdir(parents=True,exist_ok=True)
    steps=2 if smoke else 120
    seeds=SEEDS[:1] if smoke else SEEDS
    source_files=(
        "src/tac_osm/development/casm_readout_alignment.py",
        "src/tac_osm/development/rehearsal_writer.py",
        "src/tac_osm/development/noisy_memory.py",
        "src/tac_osm/development/protection_benchmark.py",
        "scripts/run_rehearsal_writer_dev.py",
        "scripts/run_casm_alignment_dev.py",
        "scripts/validate_casm_alignment_dev.py",
    )
    result={"study":"DEV-013","status":"DEVELOPMENT_ONLY_NOT_CONFIRMATORY",
            "seeds":list(seeds),"steps":steps,"smoke":smoke,
            "arms":list(ARMS),"source_sha256":{
                p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest()
                for p in source_files},"records":[]}
    for seed,eval_seed in zip(seeds,EVAL_SEEDS):
        models,train_meta=train(seed,steps)
        record={"seed":seed,"eval_seed":eval_seed,"training":train_meta,
                "models":{},"rows":{},"summaries":{}}
        for name,m in models.items():
            path=out/f"{name}-seed{seed}.pt"
            torch.save(m.state_dict(),path)
            record["models"][name]={
                "file":path.name,
                "checkpoint_sha256":hashlib.sha256(path.read_bytes()).hexdigest(),
                "parameter_count":sum(p.numel() for p in m.parameters()),
                "trainable_parameters":sum(p.numel() for p in m.parameters() if p.requires_grad)}
            rows=evaluate(m,eval_seed,name,smoke)
            record["rows"][name]=rows
            record["summaries"][name]=summarize(rows)
            summ=record["summaries"][name]
            print(f"seed {seed} {name}: task={summ['native_task_accuracy']:.4f} "
                  f"aux={summ['aux_bit_accuracy']:.4f} "
                  f"reset={summ['reset_task_accuracy']:.4f}",flush=True)
        first=[r["fingerprints"] for r in record["rows"][ARMS[0]]]
        if any(first!=[r["fingerprints"] for r in record["rows"][arm]]
               for arm in ARMS):
            raise AssertionError("arms evaluated on different observations")
        if len(set(train_meta["train_fingerprints"]))!=steps*4:
            raise AssertionError("repeated train case fingerprint")
        seen=set(train_meta["train_fingerprints"])
        eval_seen={fp for row in record["rows"][ARMS[0]] for fp in row["fingerprints"]}
        if seen & eval_seen:
            raise AssertionError("train/test overlap")
        if any((s["reset_task_accuracy"]!=.5 or
                s["reset_bit_accuracy"]!=.5) for s in record["summaries"].values()):
            raise AssertionError("zero-memory control differs from 50%")
        result["records"].append(record)
    (out/"report.json").write_text(json.dumps(result,indent=2))
    return result


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--output",type=Path,default=Path("artifacts/DEV-013"))
    p.add_argument("--smoke",action="store_true")
    a=p.parse_args()
    run(a.output,a.smoke)
