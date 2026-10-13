"""DEV-014: CASM action-rule isolation on frozen DEV-013 finite-memory bits.

Oracle-bit arms have privileged gold historical bits at both train AND TEST
and CANNOT be reported as realistic TAC-OSM inference. Predicted-bit arms
consume only frozen DEV-013 model bit probabilities (from bounded state and
public events). Same initial CASM weights, identical data and supervision,
sparse top-1 vs dense top-4 execution, and no-CASM nonlinear ablation.

No hard-coded XOR appears in any trainable model forward pass.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import sys
import zipfile

import torch
from torch import Tensor
import torch.nn.functional as F

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from tac_osm.development.casm_action_transfer import (  # noqa: E402
    CASMActionAdapter, frozen_memory_posterior, teacher_bit_distribution,
)
from tac_osm.development.casm_readout_alignment import CASMAlignedModel  # noqa: E402
from tac_osm.development.noisy_memory import parameter_hash  # noqa: E402
from tac_osm.development.protection_benchmark import (  # noqa: E402
    make_pair,seed_for,fingerprint,tensors,
)
from run_rehearsal_writer_dev import conditions  # noqa: E402

PARENT_SHA256="880d56e177b78d44961a8fc75e6e0a4151ebe6eeb7a55ce81afaa6c358172d65"
TRAIN_SEEDS=(43,59,71)
EVAL_SEEDS=(241,257,269)
ARMS=("oracle_sparse","oracle_dense","predicted_sparse","predicted_dense")
TOPK={"oracle_sparse":1,"oracle_dense":4,
      "predicted_sparse":1,"predicted_dense":4}


def read_parent_report(archive: Path):
    if hashlib.sha256(archive.read_bytes()).hexdigest()!=PARENT_SHA256:
        raise AssertionError("wrong DEV-013 parent archive SHA-256")
    with zipfile.ZipFile(archive) as z:
        names=set(z.namelist())
        expected={"report.json"}|{
            f"{mode}-seed{seed}.pt"
            for mode in ("task_only","separate_aux","aligned_aux",
                         "aligned_shuffled","aligned_detached")
            for seed in TRAIN_SEEDS}
        if names!=expected:
            raise AssertionError("parent files missing or unexpected")
        report=json.loads(z.read("report.json"))
        if (report.get("study")!="DEV-013" or
            report.get("seeds")!=list(TRAIN_SEEDS) or
            report.get("steps")!=120 or len(report["records"])!=3):
            raise AssertionError("wrong parent experiment identity")
        for rec in report["records"]:
            for name,arm in rec["models"].items():
                member=f"{name}-seed{rec['seed']}.pt"
                if hashlib.sha256(z.read(member)).hexdigest()!=arm["checkpoint_sha256"]:
                    raise AssertionError("parent checkpoint altered")
    return report


def open_parent(archive: Path, report: dict, seed: int):
    rec=next(x for x in report["records"] if x["seed"]==seed)
    with zipfile.ZipFile(archive) as z:
        data=z.read(f"aligned_aux-seed{seed}.pt")
    m=CASMAlignedModel()
    m.load_state_dict(torch.load(io.BytesIO(data),map_location="cpu",weights_only=True),strict=True)
    if parameter_hash(m)!=rec["training"]["trained_hashes"]["aligned_aux"]:
        raise AssertionError("parent model tensor digest mismatch")
    for p in m.parameters():p.requires_grad_(False)
    m.eval()
    return m


def generated_train_cases(seed: int, smoke: bool):
    steps=3 if smoke else 120
    for step in range(steps):
        h=(32,128)[step%2]; d=(4,16)[(step//2)%2]
        near=(0.0,1.0)[(step//4)%2]
        noise=(.08,.25)[(step//8)%2]
        episodes=[]
        for j,role in enumerate(("retention","acquisition")):
            p=make_pair(seed_for("dev014-train",seed,step,j),
                        h,d,near,noise,role)
            episodes.extend((p.left,p.right))
        yield step,episodes


def generated_eval_cases(seed: int, smoke: bool):
    for h,d,near,noise,role,rep in conditions(smoke):
        p=make_pair(seed_for("dev014-eval",seed,h,d,near,noise,role,rep),
                    h,d,near,noise,role)
        yield (h,d,near,noise,role,rep),(p.left,p.right)


@torch.no_grad()
def make_features(parent, episodes: list):
    a,v,q,c=tensors(episodes)
    y=torch.tensor([ep.target for ep in episodes],dtype=torch.long)
    bit=(y^c.long())
    predicted=frozen_memory_posterior(parent,a,v,q)
    oracle=teacher_bit_distribution(bit)
    return {
        "oracle":oracle.detach(),
        "predicted":predicted.detach(),
        "current":c.detach(),
        "target":y.detach(),
        "bit":bit.detach(),
        "fingerprints":[fingerprint(ep) for ep in episodes],
    }


def build_adapters(parent,seed:int):
    models={}
    initial=None
    for name in ARMS:
        torch.manual_seed(seed_for("dev014-model-init",seed)%(2**31))
        m=CASMActionAdapter(parent,top_k=TOPK[name])
        state=m.state_dict()
        if initial is None:initial={k:v.detach().clone() for k,v in state.items()}
        elif set(state)!=set(initial) or any(
                not torch.equal(v,initial[k]) for k,v in state.items()):
            raise AssertionError("unequal model initialization")
        models[name]=m
    return models


def train_models(parent,seed:int,smoke:bool):
    models=build_adapters(parent,seed)
    initials={name:parameter_hash(m) for name,m in models.items()}
    assert len(set(initials.values()))==1
    opts={name:torch.optim.AdamW(m.parameters(),lr=.002,weight_decay=.0001)
          for name,m in models.items()}
    source_fps=[]
    losses={}
    gradients={name:False for name in ARMS}
    batches=[]
    parenthash=parameter_hash(parent)
    # Build each training case from the *frozen* prior model only once.
    for _,episodes in generated_train_cases(seed,smoke):
        case=make_features(parent,episodes)
        source_fps.extend(case["fingerprints"])
        batches.append(case)
    parent.eval()
    for case in batches:
        for name,m in models.items():
            m.train()
            bits=case["oracle" if name.startswith("oracle") else "predicted"]
            loss=F.cross_entropy(m(bits,case["current"]),case["target"])
            opts[name].zero_grad(set_to_none=True)
            loss.backward()
            grad=m.compute.selector[-1].weight.grad
            gradients[name] |= bool(grad is not None and
                                    torch.isfinite(grad).all() and grad.abs().sum()>0)
            torch.nn.utils.clip_grad_norm_(m.parameters(),1.)
            opts[name].step()
            losses[name]=float(loss.detach())
    if not all(gradients.values()):
        raise AssertionError("training did not reach the real CASM selector")
    if parameter_hash(parent)!=parenthash:
        raise AssertionError("frozen parent was changed during training")
    for m in models.values():m.eval()
    return models,{
        "initial_hashes":initials,
        "final_hashes":{name:parameter_hash(m) for name,m in models.items()},
        "parent_tensor_hash":parenthash,
        "train_fingerprints":source_fps,
        "train_episodes":len(source_fps),
        "final_train_ce":losses,
        "real_casm_selector_gradient":gradients,
    }


@torch.no_grad()
def evaluate(m,parent,eval_seed:int,arm:str,smoke:bool):
    rows=[]
    parent.eval();m.eval()
    for cond,episodes in generated_eval_cases(eval_seed,smoke):
        case=make_features(parent,list(episodes))
        oracle=case["oracle"];predicted=case["predicted"]
        provided=oracle if arm.startswith("oracle") else predicted
        logits=m(provided,case["current"])
        calls=m.last_operator_sample_calls
        ablated=m(provided,case["current"],ablate_compute=True)
        if m.last_operator_sample_calls!=0:
            raise AssertionError("no-CASM control executed operators")
        nul=torch.full_like(provided,.5)
        reset=m(nul,case["current"])
        # Paired *memory* swap, not public current or gold. For oracle arms
        # this is a teacher-information swap and is only a diagnostic.
        swap=provided.reshape(-1,2,2).flip(1).reshape_as(provided)
        swapped=m(swap,case["current"])
        parent_preds=predicted.argmax(-1)
        bit_accuracy=(parent_preds==case["bit"]).tolist()
        rows.append({
            "condition":dict(zip(("h","d","near","noise","role","rep"),cond)),
            "fingerprints":case["fingerprints"],
            "target":case["target"].tolist(),
            "historical_bit":case["bit"].tolist(),
            "frozen_parent_bit_pred":parent_preds.tolist(),
            "frozen_parent_bit_probs":predicted.tolist(),
            "frozen_parent_bit_correct":bit_accuracy,
            "casm_prediction":logits.argmax(-1).tolist(),
            "no_casm_prediction":ablated.argmax(-1).tolist(),
            "uninformative_bit_prediction":reset.argmax(-1).tolist(),
            "swapped_bit_prediction":swapped.argmax(-1).tolist(),
            "casm_operator_sample_calls":calls,
            "teacher_bit_leak_control_only":arm.startswith("oracle"),
        })
    return rows


def summarize(rows):
    if not rows:raise AssertionError("empty evaluator")
    count=len(rows)*2
    summary={"episodes":count}
    for metric,pfield in (
        ("casm_accuracy","casm_prediction"),
        ("no_casm_accuracy","no_casm_prediction"),
        ("uninformative_bit_accuracy","uninformative_bit_prediction"),
        ("swapped_bit_accuracy","swapped_bit_prediction"),
        ("parent_bit_accuracy","frozen_parent_bit_pred"),
    ):
        truefield="historical_bit" if metric=="parent_bit_accuracy" else "target"
        summary[metric]=sum(int(x==y) for row in rows for x,y in zip(
            row[truefield],row[pfield]))/count
    summary["mean_executed_operator_calls"]=sum(
        row["casm_operator_sample_calls"] for row in rows)/len(rows)
    for role in ("retention","acquisition"):
        sub=[r for r in rows if r["condition"]["role"]==role]
        summary[role+"_casm_accuracy"]=(sum(int(x==y) for row in sub
             for x,y in zip(row["target"],row["casm_prediction"]))/(2*len(sub))
             if sub else None)
    long=[r for r in rows if r["condition"]["h"]==512]
    summary["H512_casm_accuracy"]=(sum(int(x==y) for row in long
         for x,y in zip(row["target"],row["casm_prediction"]))/(2*len(long))
         if long else None)
    return summary


def run(parent_archive:Path,out:Path,smoke:bool=False):
    torch.set_num_threads(2)
    out.mkdir(parents=True,exist_ok=True)
    parent_report=read_parent_report(parent_archive)
    seeds=TRAIN_SEEDS[:1] if smoke else TRAIN_SEEDS
    files=(
        "src/tac_osm/development/casm_action_transfer.py",
        "src/tac_osm/development/casm_readout_alignment.py",
        "scripts/run_casm_action_transfer_dev.py",
        "scripts/validate_casm_action_transfer_dev.py",
        "src/tac_osm/development/protection_benchmark.py",
        "scripts/run_rehearsal_writer_dev.py",
    )
    report={
        "study":"DEV-014","status":"DEVELOPMENT_ONLY_NOT_CONFIRMATORY",
        "parent_artifact_sha256":PARENT_SHA256,
        "smoke":smoke,"seeds":list(seeds),"arms":list(ARMS),
        "source_sha256":{p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest()
                         for p in files},
        "records":[],
    }
    for seed,eval_seed in zip(seeds,EVAL_SEEDS):
        parent=open_parent(parent_archive,parent_report,seed)
        models,meta=train_models(parent,seed,smoke)
        record={"seed":seed,"eval_seed":eval_seed,
                "training":meta,"models":{},"rows":{},"summaries":{}}
        for name,m in models.items():
            path=out/f"{name}-seed{seed}.pt"
            torch.save(m.state_dict(),path)
            record["models"][name]={
                "file":path.name,"sha256":hashlib.sha256(path.read_bytes()).hexdigest(),
                "parameter_count":sum(p.numel() for p in m.parameters()),
                "top_k":m.top_k}
            rows=evaluate(m,parent,eval_seed,name,smoke)
            record["rows"][name]=rows
            record["summaries"][name]=summarize(rows)
            s=record["summaries"][name]
            print(f"seed={seed} {name}: casm={s['casm_accuracy']:.3f}"
                  f" noCASM={s['no_casm_accuracy']:.3f}"
                  f" bit={s['parent_bit_accuracy']:.3f}",flush=True)
        first=[r["fingerprints"] for r in record["rows"][ARMS[0]]]
        if any(first!=[r["fingerprints"] for r in record["rows"][a]] for a in ARMS):
            raise AssertionError("different evaluation cases across models")
        trainset=set(meta["train_fingerprints"])
        evalset={fp for row in record["rows"][ARMS[0]] for fp in row["fingerprints"]}
        if len(trainset)!=meta["train_episodes"] or trainset & evalset:
            raise AssertionError("train/eval inputs overlap or duplicate")
        if any(s["uninformative_bit_accuracy"]!=.5
               for s in record["summaries"].values()):
            raise AssertionError("paired no-bit input violates chance control")
        report["records"].append(record)
    (out/"report.json").write_text(json.dumps(report,indent=2))
    return report


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--parent",type=Path,required=True)
    p.add_argument("--output",type=Path,default=Path("artifacts/DEV-014"))
    p.add_argument("--smoke",action="store_true")
    a=p.parse_args()
    run(a.parent,a.output,a.smoke)
