"""DEV-012 writer training: matched task/rehearsal/protection ablations.

Do not claim general memory capability or Gated DeltaNet-2 replication.
Rehearsal uses observed historical payloads *only in offline loss*, never as
query-time memory. Exact 768 recurrent scalars, unchanged native TAC-OSM
conditional operator path, opposite-past task and reset controls.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import torch
from torch import Tensor
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"src"))
from tac_osm.development.noisy_memory import parameter_hash  # noqa: E402
from tac_osm.development.rehearsal_writer import make_matching_models  # noqa: E402
from tac_osm.development.protection_benchmark import (  # noqa: E402
    make_pair, seed_for, fingerprint, tensors, euclidean,
)
from run_finite_ced_dev import eval_conditions  # noqa: E402

SEEDS = (43, 59, 71)
EVAL_SEEDS = (241, 257, 269)
ARMS = ("task_only","rehearsal","shuffled_rehearsal",
        "fixed_rehearsal","learned_rehearsal")


def train_case(seed: int, step: int):
    h=(32,128)[step%2]
    d=(4,16)[(step//2)%2]
    near=(0.0,1.0)[(step//4)%2]
    noise=(.08,.25)[(step//8)%2]
    episodes=[]
    for j,role in enumerate(("retention","acquisition")):
        p=make_pair(seed_for("dev012-train",seed,step,j),h,d,near,noise,role)
        episodes.extend((p.left,p.right))
    a,v,q,c=tensors(episodes)
    y=torch.tensor([ep.target for ep in episodes],dtype=torch.long)
    return a,v,q,c,y,tuple(fingerprint(ep) for ep in episodes),h,d,noise


def sample_rehearsal_inputs(addresses: Tensor, values: Tensor,
                            h: int, d: int, noise: float,
                            seed: int, step: int):
    """Training-only timestamps, chosen without evaluator queried slot.

    Samples one early event and one new write per episode. The index is used
    only for constructing offline labels and supervision queries. It is never
    an input to model.forward or an evaluation selector.
    """
    batch=addresses.shape[0]
    g=torch.Generator().manual_seed(
        seed_for("dev012-rehearsal-noise",seed,step)%(2**63-1))
    idx=torch.arange(batch)
    early=torch.tensor([(step*11+j//2*7)%h for j in range(batch)])
    fresh=torch.tensor([h+(step*7+j//2*3)%d for j in range(batch)])
    def entry(positions):
        reference=addresses[idx,positions]
        jitter=torch.randn((batch,addresses.shape[-1]),generator=g)*noise
        query=reference+jitter
        label=values[idx,positions]
        return query,label
    return entry(early),entry(fresh)


def train(seed: int, steps: int):
    models=make_matching_models(seed)
    initials={name:parameter_hash(model) for name,model in models.items()}
    optim={name:torch.optim.AdamW((p for p in m.parameters() if p.requires_grad),
                                   lr=.003,weight_decay=.0001)
           for name,m in models.items()}
    gradient_memory={name:False for name in ARMS}
    gradient_protection=False
    losses={name:{} for name in ARMS}
    train_fps=[]
    for step in range(steps):
        a,v,q,c,y,fps,h,d,noise=train_case(seed,step)
        train_fps.extend(fps)
        (oldq,oldbit),(newq,newbit)=sample_rehearsal_inputs(
            a,v,h,d,noise,seed,step)
        # Independent deterministic null target; distribution balanced
        # across batches but unrelated to the observed payload at timestamp.
        rng=torch.Generator().manual_seed(
            seed_for("dev012-null-label",seed,step)%(2**63-1))
        random_old=torch.randint(0,2,oldbit.shape,generator=rng)
        random_new=torch.randint(0,2,newbit.shape,generator=rng)
        for name,model in models.items():
            model.train()
            optimizer=optim[name]
            optimizer.zero_grad(set_to_none=True)
            state=model.encode(a,v,trace=False)
            task=F.cross_entropy(model.predict(state,q,c),y)
            aux=task.new_zeros(())
            if name != "task_only":
                negative=name=="shuffled_rehearsal"
                aux=(F.cross_entropy(model.rehearsal_logits(state,oldq),
                                     random_old if negative else oldbit)
                     +F.cross_entropy(model.rehearsal_logits(state,newq),
                                      random_new if negative else newbit))/2
            loss=task+aux
            loss.backward()
            grad=model.memory.write.weight.grad
            gradient_memory[name] |= bool(grad is not None
                        and torch.isfinite(grad).all()
                        and grad.abs().sum().item()>0)
            if name=="learned_rehearsal":
                gategrad=model.memory.protection[-1].weight.grad
                gradient_protection |= bool(gategrad is not None
                                and torch.isfinite(gategrad).all()
                                and gategrad.abs().sum().item()>0)
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            optimizer.step()
            losses[name]={"task_ce":float(task.detach()),"rehearsal_ce":float(aux.detach()),
                          "total_ce":float(loss.detach()),
                          "final_gate_mean":float(model.memory.last_gate.detach().mean())}
    if not all(gradient_memory.values()) or not gradient_protection:
        raise AssertionError("required recurrent writer or protection gate gradient missing")
    for name,model in models.items():
        if parameter_hash(model)==initials[name]:
            raise AssertionError("untrained writer")
        model.eval()
    return models, {
        "initial_hashes":initials,
        "trained_hashes":{n:parameter_hash(m) for n,m in models.items()},
        "writer_gradients":gradient_memory,
        "learned_protection_gradient":gradient_protection,
        "last_losses":losses,
        "training_fingerprints":train_fps,
        "samples_per_arm":len(train_fps),
        "extra_rehearsal_reads_per_episode_per_step":2,
    }


def conditions(smoke: bool):
    if smoke:
        return [(32,4,0.0,.08,"retention",0)]
    return [(h,d,near,noise,role,rep)
            for rep in range(2)
            for h,d,near,noise,role,_ in eval_conditions(False)]


@torch.no_grad()
def evaluate(model, seed: int, smoke: bool):
    model.eval()
    rows=[]
    for h,d,near,noise,role,rep in conditions(smoke):
        pair=make_pair(seed_for("dev012-eval",seed,h,d,near,noise,role,rep),
                       h,d,near,noise,role)
        eps=(pair.left,pair.right)
        a,v,q,c=tensors(list(eps))
        state=model.encode(a,v,trace=False)
        if state.shape!=(2,4,8,24):
            raise AssertionError("state no longer 768 scalars")
        labels=torch.tensor([ep.target for ep in eps])
        bit_labels=labels ^ c
        task_logits=model.predict(state,q,c)
        bit_logits=model.rehearsal_logits(state,q)
        reset_task=model.predict(torch.zeros_like(state),q,c).argmax(-1)
        reset_bit=model.rehearsal_logits(torch.zeros_like(state),q).argmax(-1)
        swapped_task=model.predict(state.flip(0),q,c).argmax(-1)
        swapped_bit=model.rehearsal_logits(state.flip(0),q).argmax(-1)
        classic=[euclidean(ep) for ep in eps]
        rows.append({
            "condition":{"h":h,"d":d,"near":near,"noise":noise,"role":role,"rep":rep},
            "fingerprints":[fingerprint(ep) for ep in eps],
            "targets":labels.tolist(),
            "bit_targets":bit_labels.tolist(),
            "task_predictions":task_logits.argmax(-1).tolist(),
            "bit_predictions":bit_logits.argmax(-1).tolist(),
            "reset_task_predictions":reset_task.tolist(),
            "reset_bit_predictions":reset_bit.tolist(),
            "swapped_task_predictions":swapped_task.tolist(),
            "swapped_bit_predictions":swapped_bit.tolist(),
            "oracle_full_history_predictions":[int(x[0]) for x in classic],
            "oracle_full_history_inspected":[h+d,h+d],
            "state_pair_l2":float((state[0]-state[1]).norm()),
            "memory_scalars_per_episode":state[0].numel(),
        })
    return rows


def summarize(rows):
    if not rows:raise AssertionError("no evaluation rows")
    n=len(rows)*2
    output={"episodes":n}
    fields={
        "task_accuracy":("targets","task_predictions"),
        "bit_accuracy":("bit_targets","bit_predictions"),
        "reset_task_accuracy":("targets","reset_task_predictions"),
        "reset_bit_accuracy":("bit_targets","reset_bit_predictions"),
        "swapped_task_accuracy":("targets","swapped_task_predictions"),
        "swapped_bit_accuracy":("bit_targets","swapped_bit_predictions"),
        "full_history_accuracy":("targets","oracle_full_history_predictions"),
    }
    for name,(t,p) in fields.items():
        output[name]=sum(sum(int(u==v) for u,v in
                         zip(row[t],row[p])) for row in rows)/n
    for role in ("retention","acquisition"):
        sub=[r for r in rows if r["condition"]["role"]==role]
        output[role+"_task_accuracy"]=(sum(sum(int(u==v) for u,v in
                   zip(r["targets"],r["task_predictions"])) for r in sub)/(2*len(sub))
                   if sub else None)
        output[role+"_bit_accuracy"]=(sum(sum(int(u==v) for u,v in
                   zip(r["bit_targets"],r["bit_predictions"])) for r in sub)/(2*len(sub))
                   if sub else None)
    long=[r for r in rows if r["condition"]["h"]==512]
    output["long512_task_accuracy"]=(sum(sum(int(u==v) for u,v in
               zip(r["targets"],r["task_predictions"])) for r in long)/(2*len(long))
               if long else None)
    output["state_pair_l2_mean"]=sum(r["state_pair_l2"] for r in rows)/len(rows)
    return output


def run(out: Path, smoke: bool=False):
    torch.set_num_threads(2)
    out.mkdir(parents=True,exist_ok=True)
    seeds=SEEDS[:1] if smoke else SEEDS
    steps=2 if smoke else 120
    files=("src/tac_osm/development/rehearsal_writer.py",
           "src/tac_osm/development/noisy_memory.py",
           "src/tac_osm/development/protected_memory.py",
           "src/tac_osm/development/protection_benchmark.py",
           "scripts/run_rehearsal_writer_dev.py",
           "scripts/validate_rehearsal_writer_dev.py")
    result={"study":"DEV-012","status":"DEVELOPMENT_ONLY_NOT_CONFIRMATORY",
            "smoke":smoke,"steps":steps,"seeds":list(seeds),
            "arms":list(ARMS),
            "source_sha256":{p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest()
                             for p in files},
            "protocol":"same initial tensors, task-only vs rehearsal vs shuffled vs fixed/learned protected writer",
            "records":[]}
    for seed,eval_seed in zip(seeds,EVAL_SEEDS):
        models,meta=train(seed,steps)
        record={"seed":seed,"eval_seed":eval_seed,"training":meta,"models":{},"rows":{},"summaries":{}}
        for name in ARMS:
            model=models[name]
            path=out/f"{name}-seed{seed}.pt"
            torch.save(model.state_dict(),path)
            record["models"][name]={
                "file":path.name,"sha256":hashlib.sha256(path.read_bytes()).hexdigest(),
                "trainable_parameters":sum(p.numel() for p in model.parameters() if p.requires_grad),
                "total_parameters":sum(p.numel() for p in model.parameters()),
                "mode":model.mode,
            }
            rows=evaluate(model,eval_seed,smoke)
            record["rows"][name]=rows
            record["summaries"][name]=summarize(rows)
            print(f"seed={seed} {name}: task={record['summaries'][name]['task_accuracy']:.3f}"
                  f" bit={record['summaries'][name]['bit_accuracy']:.3f}"
                  f" reset={record['summaries'][name]['reset_task_accuracy']:.3f}",flush=True)
        if any([r["fingerprints"] for r in record["rows"][name]] !=
               [r["fingerprints"] for r in record["rows"][ARMS[0]]] for name in ARMS):
            raise AssertionError("evaluation mismatch across arms")
        trainfps=set(meta["training_fingerprints"])
        evalfps={f for row in record["rows"][ARMS[0]] for f in row["fingerprints"]}
        if trainfps & evalfps or len(trainfps)!=4*steps:
            raise AssertionError("train/eval overlap or duplicate training examples")
        if any(rec["reset_task_accuracy"]!=.5 or rec["reset_bit_accuracy"]!=.5
               for rec in record["summaries"].values()):
            raise AssertionError("reset-memory leakage")
        result["records"].append(record)
    (out/"report.json").write_text(json.dumps(result,indent=2))
    return result


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--output",type=Path,default=Path("artifacts/DEV-012"))
    p.add_argument("--smoke",action="store_true")
    args=p.parse_args()
    run(args.output,smoke=args.smoke)
