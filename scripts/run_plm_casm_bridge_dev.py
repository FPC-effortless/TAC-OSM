"""DEV-015 finite PLM -> CASM task bridge: paired training and OOD axis test.

End-to-end gradients are contrasted with frozen-writer, frozen-reader and
independent random-label auxiliary controls. No true bit is supplied to
inference. Native 768-scalar memory and TAC-OSM conditional operators persist.
This is synthetic adaptive DEVELOPMENT evidence, not a new confirmed model.
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
from tac_osm.development.plm_casm_bridge import (  # noqa: E402
    PLMCASMBridge,ARMS,
)
from tac_osm.development.noisy_memory import parameter_hash  # noqa: E402
from tac_osm.development.protection_benchmark import (  # noqa: E402
    fingerprint,seed_for,tensors,
)
from run_casm_action_transfer_dev import (  # noqa: E402
    PARENT_SHA256,TRAIN_SEEDS,EVAL_SEEDS,read_parent_report,open_parent,
    generated_train_cases,generated_eval_cases,
)

TRAIN_STEPS=96


def make_models(reference,seed:int):
    out={}
    for arm in ARMS:
        torch.manual_seed(seed_for("dev015-initialization",seed)%(2**31))
        out[arm]=PLMCASMBridge(reference,mode=arm,top_k=1)
    hashes={arm:parameter_hash(m) for arm,m in out.items()}
    if len(set(hashes.values()))!=1:
        raise AssertionError("unmatched initial tensors across arms")
    return out,hashes


def train(reference,seed:int,smoke:bool):
    steps=2 if smoke else TRAIN_STEPS
    models,initial=make_models(reference,seed)
    opts={n:torch.optim.AdamW((p for p in m.parameters() if p.requires_grad),
                              lr=0.002,weight_decay=.0001)
          for n,m in models.items()}
    writer_grad={n:False for n in ARMS}
    read_grad={n:False for n in ARMS}
    actor_grad={n:False for n in ARMS}
    train_fps=[]
    last={}
    original=parameter_hash(reference)
    for step,episodes in generated_train_cases(seed,smoke):
        if step>=steps:break
        a,v,q,c=tensors(episodes)
        target=torch.tensor([e.target for e in episodes],dtype=torch.long)
        bit=target^c
        train_fps.extend(fingerprint(e) for e in episodes)
        g=torch.Generator().manual_seed(
            seed_for("dev015-null-bit",seed,step)%(2**63-1))
        shuffled=torch.randint(0,2,bit.shape,generator=g)
        for arm,m in models.items():
            m.train()
            opts[arm].zero_grad(set_to_none=True)
            logits,bit_logits,_=m(a,v,q,c)
            task_ce=F.cross_entropy(logits,target)
            aux=task_ce.new_zeros(())
            if arm in ("reader_aux","joint_aux","joint_shuffled"):
                aux=F.cross_entropy(bit_logits,
                                   shuffled if arm=="joint_shuffled" else bit)
            loss=task_ce+aux
            loss.backward()
            w=m.reader.memory.write.weight.grad
            r=m.reader.shared_head.weight.grad
            operator=m.actor.compute.selector[-1].weight.grad
            writer_grad[arm] |= bool(w is not None and
                                    torch.isfinite(w).all() and w.abs().sum()>0)
            read_grad[arm] |= bool(r is not None and
                                  torch.isfinite(r).all() and r.abs().sum()>0)
            actor_grad[arm] |= bool(operator is not None and
                                   torch.isfinite(operator).all() and operator.abs().sum()>0)
            torch.nn.utils.clip_grad_norm_(
                (p for p in m.parameters() if p.requires_grad),1.0)
            opts[arm].step()
            last[arm]={"task_ce":float(task_ce.detach()),
                       "aux_ce":float(aux.detach()),"total_ce":float(loss.detach())}
    if not all(actor_grad.values()) or not all(read_grad[x] for x in ARMS if x!="frozen_task"):
        raise AssertionError("action/reader gradients missing")
    if any(writer_grad[x] for x in ("frozen_task","reader_aux")):
        raise AssertionError("frozen writer received gradient")
    if not all(writer_grad[x] for x in ("joint_task","joint_aux","joint_shuffled")):
        raise AssertionError("joint writer did not receive task/aux gradient")
    if parameter_hash(reference)!=original:
        raise AssertionError("immutable reference checkpoint changed")
    for m in models.values():m.eval()
    return models,{
        "steps":steps,"initial_hashes":initial,
        "final_hashes":{n:parameter_hash(m) for n,m in models.items()},
        "frozen_parent_hash":original,"writer_gradients":writer_grad,
        "reader_gradients":read_grad,"actor_gradients":actor_grad,
        "train_fingerprints":train_fps,"final_losses":last,
    }


def change_basis(a,q,case_seed:int):
    """Orthogonal signed coordinate permutation, preserving Euclidean task."""
    g=torch.Generator().manual_seed(seed_for(
        "dev015-basis-permutation",case_seed)%(2**63-1))
    order=torch.randperm(12,generator=g)
    signs=(torch.randint(0,2,(12,),generator=g)*2-1).to(a.dtype)
    return a[...,order]*signs,q[...,order]*signs


@torch.no_grad()
def evaluate(m,seed:int,smoke:bool):
    m.eval()
    rows=[]
    for cond,episodes in generated_eval_cases(seed,smoke):
        publicfps=[fingerprint(e) for e in episodes]
        a0,v,q0,c=tensors(list(episodes))
        labels=torch.tensor([e.target for e in episodes],dtype=torch.long)
        bit=labels^c
        for axis in ("original","signed_permuted"):
            a,q=(a0,q0) if axis=="original" else change_basis(
                a0,q0,seed_for("dev015-shiftcase",seed,*cond))
            # Non-oracle, no stream index. Full-history exact Euclidean is an
            # expensive evaluator reference and scans every address.
            distance=((a-q[:,None,:])**2).sum(-1)
            i=distance.argmin(1)
            ref=(v.gather(1,i[:,None]).squeeze(1)^c).tolist()
            answer,bit_logits,posterior=m(a,v,q,c)
            calls=m.actor.last_operator_sample_calls
            no_compute=m.actor(posterior,c,ablate_compute=True)
            neutral=m.with_neutral_bit(a,v,q,c)
            flipped=m.actor(posterior.flip(0),c)
            row={
                "condition":dict(zip(("h","d","near","noise","role","rep"),cond)),
                "coordinate_family":axis,
                "fingerprints":publicfps,
                "target":labels.tolist(),
                "historical_bit_target":bit.tolist(),
                "original_action":answer.argmax(-1).tolist(),
                "retrieved_bit":bit_logits.argmax(-1).tolist(),
                "bit_posterior":posterior.tolist(),
                "no_casm_action":no_compute.argmax(-1).tolist(),
                "no_bit_action":neutral.argmax(-1).tolist(),
                "swapped_bit_action":flipped.argmax(-1).tolist(),
                "full_history_euclidean":ref,
                "full_history_records_scanned":[int(a.shape[1])]*2,
                "casm_operator_sample_calls":calls,
                "plm_scalars_per_episode":768,
            }
            if any(not torch.isfinite(t).all() for t in (answer,bit_logits,posterior)):
                raise AssertionError("nonfinite inference output")
            rows.append(row)
    return rows


def summarize(rows):
    assert rows
    out={}
    for family in ("original","signed_permuted"):
        cells=[r for r in rows if r["coordinate_family"]==family]
        n=2*len(cells)
        metrics={
            "action":("target","original_action"),
            "bit":("historical_bit_target","retrieved_bit"),
            "no_casm":("target","no_casm_action"),
            "no_bit":("target","no_bit_action"),
            "swap":("target","swapped_bit_action"),
            "euclidean":("target","full_history_euclidean"),
        }
        d={"episodes":n}
        for k,(a,b) in metrics.items():
            d[k]=sum(int(x==y) for row in cells
                     for x,y in zip(row[a],row[b]))/n
        d["mean_operator_sample_calls"]=sum(
            row["casm_operator_sample_calls"] for row in cells)/len(cells)
        for role in ("retention","acquisition"):
            subset=[r for r in cells if r["condition"]["role"]==role]
            d[role+"_action"]=(sum(int(x==y) for row in subset
                for x,y in zip(row["target"],row["original_action"]))/(2*len(subset))
                if subset else None)
        stress=[r for r in cells if r["condition"]["h"]==512]
        d["H512_action"]=(sum(int(x==y) for row in stress
            for x,y in zip(row["target"],row["original_action"]))/(2*len(stress))
            if stress else None)
        out[family]=d
    return out


def run(parent_archive:Path,out:Path,smoke:bool=False):
    torch.set_num_threads(2)
    out.mkdir(parents=True,exist_ok=True)
    parent=read_parent_report(parent_archive)
    seeds=TRAIN_SEEDS[:1] if smoke else TRAIN_SEEDS
    source=(
        "src/tac_osm/development/plm_casm_bridge.py",
        "src/tac_osm/development/casm_action_transfer.py",
        "src/tac_osm/development/casm_readout_alignment.py",
        "scripts/run_plm_casm_bridge_dev.py",
        "scripts/validate_plm_casm_bridge_dev.py",
        "scripts/run_casm_action_transfer_dev.py",
        "src/tac_osm/development/protection_benchmark.py",
    )
    report={
        "study":"DEV-015","status":"DEVELOPMENT_ONLY_NOT_CONFIRMATORY",
        "smoke":smoke,"seeds":list(seeds),"arms":list(ARMS),
        "parent_artifact_sha256":PARENT_SHA256,
        "source_sha256":{p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest()
                         for p in source},
        "records":[],
    }
    for seed,evseed in zip(seeds,EVAL_SEEDS):
        reference=open_parent(parent_archive,parent,seed)
        models,training=train(reference,seed,smoke)
        record={"seed":seed,"eval_seed":evseed,"training":training,
                "models":{},"rows":{},"summaries":{}}
        baseline=None
        for arm,m in models.items():
            f=out/f"{arm}-seed{seed}.pt"
            torch.save(m.state_dict(),f)
            record["models"][arm]={
                "file":f.name,"sha256":hashlib.sha256(f.read_bytes()).hexdigest(),
                "total_parameters":sum(p.numel() for p in m.parameters()),
                "trainable_parameters":sum(p.numel() for p in m.parameters()
                                            if p.requires_grad)}
            raw=evaluate(m,evseed,smoke)
            record["rows"][arm]=raw
            record["summaries"][arm]=summarize(raw)
            public=[(r["condition"],r["coordinate_family"],r["fingerprints"])
                    for r in raw]
            if baseline is None:baseline=public
            elif public!=baseline:
                raise AssertionError("mismatched evaluation observations")
            s=record["summaries"][arm]
            print(f"seed={seed} {arm}: task={s['original']['action']:.3f}"
                  f" OOD={s['signed_permuted']['action']:.3f}"
                  f" bit={s['original']['bit']:.3f}"
                  f" reset={s['original']['no_bit']:.3f}",flush=True)
        trainset=set(training["train_fingerprints"])
        evalset={fp for r in record["rows"][ARMS[0]] for fp in r["fingerprints"]}
        if trainset & evalset or len(trainset)!=4*training["steps"]:
            raise AssertionError("overlapping/duplicated train evaluation")
        for arm in ARMS:
            s=record["summaries"][arm]
            if any(s[f]["no_bit"]!=.5 for f in s):
                raise AssertionError("no-memory paired baseline not exactly 50%")
        report["records"].append(record)
    (out/"report.json").write_text(json.dumps(report,indent=2))
    return report


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--parent",required=True,type=Path)
    p.add_argument("--output",type=Path,default=Path("artifacts/DEV-015"))
    p.add_argument("--smoke",action="store_true")
    args=p.parse_args()
    run(args.parent,args.output,args.smoke)
