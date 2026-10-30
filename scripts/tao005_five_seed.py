#!/usr/bin/env python3
"""TAOFINAL005: five-seed original licensed TAC optimizer study (internal).

The optimizer rates are read strictly from the immutable separately trained and
scored TAOTUNE004 selection. Each seed writes five complete original model,
optimizer, RNG checkpoints to Git BEFORE independent report-DEV reads.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import random
import resource
import statistics
import sys
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/"src"),
              str(ROOT/"research/mathematical-engineering/langmem086"),
              str(ROOT/"research/mathematical-engineering/langbase024")]
import numpy as np
import torch
from torch.nn import functional as F
from compact_tac import CompactDualClippedTAC086
from scripts import langmem086_run as original
from scripts import tao004_tune as tune
from scripts.tao003_pilot import independent_numeric_replay
from scripts.tao002_prefit import SOURCE_SHA, MODEL_SHA, SIZE, ARMS
from scripts.validate_research_parity import check_manifest
from byte_next_reference import read_sha_source

REG=ROOT/"research/mathematical-engineering/taofinal005/FAIR_COMPARISON.json"
TUNING=ROOT/"research/evidence/taotune004/seed1701/TUNING_RESULTS.json"
PREFIT=ROOT/"research/mathematical-engineering/tao002/ORIGINAL_PREFIT_RECEIPT.json"
SEEDS=(1701,1702,1703,1704,1705)
STEPS=64
BATCH=4
LENGTH=256
TARGETS=STEPS*BATCH*LENGTH

def sha(data):return hashlib.sha256(data).hexdigest()
def weight_hash(model):return original.weight_hash(model)

def registry():
    m=json.loads(REG.read_text())
    x=check_manifest(m,"preflight")
    if x["status"]!="PASS_PREFIT_SINGLE_MODEL" or m["study_id"]!="TAOFINAL005":
        raise ValueError("prospective final source registration failed")
    if m["budget"]["updates_per_arm_seed"]!=STEPS or m["budget"]["real_targets_per_arm_seed"]!=TARGETS:
        raise ValueError("final fit budget drift")
    names={(f"{arm}_seed{seed}",seed) for seed in SEEDS for arm in ARMS}
    if {(r["name"],r["seed"]) for r in m["models"]}!=names or len(m["models"])!=25:
        raise ValueError("frozen seed x optimizer original model matrix differs")
    if sha((ROOT/"research/mathematical-engineering/langmem086/compact_tac.py").read_bytes())!=MODEL_SHA:
        raise ValueError("original graph hash drift")
    if m["real_world_source"]["sha256"]!=SOURCE_SHA:
        raise ValueError("publisher source identity drift")
    return m

def selection():
    cfg=registry()
    if not TUNING.exists():
        raise FileNotFoundError("15 original LR checkpoint selection not yet committed in Git")
    result=json.loads(TUNING.read_text())
    if result["study"]!="TAOTUNE004" or result["phase"]!="EARLY_REAL_SOURCE_LR_SELECTION":
        raise ValueError("wrong or unsealed original LR selection")
    if result["original_15_git_checkpoints_exact_hash"] is not True or result["independent_original_optimizer_state_replay"] is not True:
        raise ValueError("tuning originals did not pass independent replay")
    if set(result["selected"])!=set(ARMS) or set(result["trial_scores"])!=set(ARMS):
        raise ValueError("optimizer LR selection is incomplete")
    if result["tuning_owners"]!=sorted(["Doc/tutorial/controlflow.rst","Doc/tutorial/index.rst"]):
        raise ValueError("original tuning DEV boundary differs")
    if result["reporting_document_owners_unread_by_tuner"]!=["Doc/tutorial/introduction.rst","Doc/tutorial/venv.rst"]:
        raise ValueError("tuning used reserved reporting document")
    tuner=tune.registered()
    for arm in ARMS:
        scores=result["trial_scores"][arm]
        if len(scores)!=3:
            raise ValueError("unequal original LR tuning opportunity")
        for i,score in enumerate(scores):
            if score["index"]!=i or score["lr"]!=tuner["learning_rate_grid"][arm][i]:
                raise ValueError("tuning original trial LR changed")
        expected=min(scores,key=lambda row:(row["nll"],row["lr"],row["index"]))
        if result["selected"][arm]!=expected:
            raise ValueError("selected LR was not deterministic min tune DEV NLL")
    return cfg,result,sha(TUNING.read_bytes())

def frozen_real_fits(seed):
    docs,source=original.fit_sources086()
    if source!=SOURCE_SHA or len(docs)!=14:
        raise ValueError("original publisher source bytes changed")
    receipt=json.loads(PREFIT.read_text())
    frozen=next(x for x in receipt["samples"]["frozen_sample_schedules"] if x["seed"]==seed)
    if original.schedule(docs,seed)!=frozen:
        raise ValueError("seed source/schedule fails immutable original plan")
    batches=list(itertools.islice(original.batches(docs,seed),STEPS))
    h=hashlib.sha256();d=hashlib.sha256()
    for _,row in batches:
        for owner,start,raw in row:
            h.update(f"{owner}:{start};".encode());d.update(raw)
    schedule={"seed":seed,"steps":STEPS,"batch":BATCH,"context":LENGTH,"targets":TARGETS,
              "sample_order_sha256":h.hexdigest(),"sample_bytes_sha256":d.hexdigest()}
    return docs,batches,schedule

def fit_seed(seed,out:Path):
    if seed not in SEEDS:raise ValueError("unregistered seed")
    if out.exists():raise FileExistsError("original trained model bundle exists: no overwrite")
    if torch.cuda.is_available():raise ValueError("study registered CPU-only")
    torch.set_num_threads(2)
    cfg,sel,lr_sha=selection()
    docs,rows,schedule=frozen_real_fits(seed)
    fixture=torch.tensor([list(next(iter(docs.values()))[:16])],dtype=torch.long)
    initial=set();matrix={}
    for mode in ARMS:
        lr=sel["selected"][mode]["lr"]
        torch.manual_seed(seed);np.random.seed(seed);random.seed(seed)
        model=CompactDualClippedTAC086()
        if model.counts()["trainable"]!=SIZE or model.counts()["deployed"]!=SIZE:
            raise ValueError("actual live model count differs")
        start_hash=weight_hash(model);initial.add(start_hash)
        opt=tune.optimizer_for(mode,model,lr)
        altered=fixture.clone();altered[:,8:]=(altered[:,8:]+113)%256
        with torch.no_grad():
            a,_=model(fixture,owners=("original-FIT-poison",))
            b,_=model(altered,owners=("original-FIT-poison",))
        if not torch.equal(a[:,:8],b[:,:8]):
            raise ValueError("actual source future suffix poison detected")
        loss_trace=[];optimizer_seconds=0.
        begin=time.perf_counter()
        for step,row in rows:
            x=torch.tensor([list(raw[:-1]) for _,_,raw in row],dtype=torch.long)
            y=torch.tensor([list(raw[1:]) for _,_,raw in row],dtype=torch.long)
            opt.zero_grad(set_to_none=True)
            logits,_=model(x,owners=tuple(f"source-owner-{i}" for i in range(BATCH)))
            loss=F.cross_entropy(logits.reshape(-1,256),y.reshape(-1))
            if not bool(torch.isfinite(loss)):raise ValueError("nonfinite real source learning")
            loss.backward()
            if any(p.grad is None or not bool(torch.isfinite(p.grad).all()) for p in model.parameters()):
                raise ValueError("missing or nonfinite real-source gradient")
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            t=time.perf_counter();opt.step();optimizer_seconds+=time.perf_counter()-t
            if not all(bool(torch.isfinite(p).all()) for p in model.parameters()):
                raise ValueError("post-update model not finite")
            loss_trace.append(float(loss.detach()))
        wall=time.perf_counter()-begin
        if len(loss_trace)!=STEPS or start_hash==weight_hash(model):
            raise ValueError("genuine publisher fit incomplete")
        if any(int(s["step"])!=STEPS for s in opt.state.values()):
            raise ValueError("original optimizer update count differs")
        with torch.no_grad():
            cold_logits,_=model(fixture,owners=("cold-check",))
        path=out/mode
        path.mkdir(parents=True,exist_ok=False)
        ns=np.random.get_state()
        cp={"study":"TAOFINAL005","mode":mode,"seed":seed,"steps":STEPS,"lr":lr,
            "model":model.state_dict(),"optimizer":opt.state_dict(),
            "torch_rng":torch.get_rng_state(),"python_rng":random.getstate(),
            "numpy_rng":(ns[0],ns[1].tolist(),ns[2],ns[3],ns[4]),
            "fit_source_sha256":SOURCE_SHA,"model_source_sha256":MODEL_SHA,
            "training_sample_schedule":schedule,
            "selection_sha256":lr_sha,"original_initial_weight_hash":start_hash,
            "trained_model_weight_hash":weight_hash(model),
            "original_fit_loss":loss_trace,"cold_fixture":fixture,"cold_logits":cold_logits}
        file=path/"MODEL.pt";torch.save(cp,file);cp_sha=sha(file.read_bytes())
        reloaded=torch.load(file,map_location="cpu",weights_only=True)
        check=CompactDualClippedTAC086().eval();check.load_state_dict(reloaded["model"],strict=True)
        check_opt=tune.optimizer_for(mode,check,lr);check_opt.load_state_dict(reloaded["optimizer"])
        with torch.no_grad():check_logits,_=check(fixture,owners=("cold-check",))
        if not torch.equal(check_logits,cold_logits) or weight_hash(check)!=cp["trained_model_weight_hash"]:
            raise ValueError("original saved checkpoint exact same-machine model replay failed")
        if any(int(s["step"])!=STEPS for s in check_opt.state.values()):
            raise ValueError("original optimizer checkpoint did not restore")
        data={"study":"TAOFINAL005","mode":mode,"seed":seed,"lr":lr,
              "steps":STEPS,"targets":TARGETS,"source_sha256":SOURCE_SHA,
              "selection_sha256":lr_sha,"sample_receipt":schedule,
              "checkpoint_sha256":cp_sha,"trained_model_sha":cp["trained_model_weight_hash"],
              "initial_weight_sha":start_hash,"model_trainable_parameters":SIZE,
              "model_deployed_parameters":SIZE,
              "exact_same_machine_model_and_optimizer_cold_replay":True,
              "total_train_wall_seconds":wall,"optimizer_wall_seconds":optimizer_seconds,
              "peak_rss_bytes":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
              "fit_first_nll":loss_trace[0],"fit_last_nll":loss_trace[-1],
              "fit_trace":loss_trace,"FIT_future_poison":"PASS",
              "report_DEV_read":False,"protected_TEST":"NOT_ACCESSED","gpu_minutes":0}
        (path/"FIT_RECEIPT.json").write_text(json.dumps(data,indent=2,sort_keys=True)+"\n")
        matrix[mode]={"lr":lr,"checkpoint_sha256":cp_sha,
                      "final_fit_minibatch_nll":loss_trace[-1],
                      "train_wall_seconds":wall,"optimizer_wall_seconds":optimizer_seconds}
        print(json.dumps({"phase":"ORIGINAL_FIT_SEALED","seed":seed,"mode":mode,
                          "lr":lr,"FIT_last_nll":loss_trace[-1],
                          "checkpoint_sha256":cp_sha}),flush=True)
    if len(matrix)!=5 or len(initial)!=1:
        raise ValueError("seed model/treatment initial checkpoint parity differs")
    (out/"FIT_MATRIX.json").write_text(json.dumps({
        "study":"TAOFINAL005","seed":seed,"source_sha256":SOURCE_SHA,
        "sample_receipt":schedule,"original_initial_model_hash":next(iter(initial)),
        "selection_sha256":lr_sha,"all_five_original_checkpoints":True,
        "original_models":matrix,"report_DEV_read":False,"protected_TEST":"NOT_ACCESSED"},
        indent=2,sort_keys=True)+"\n")
    print(json.dumps({"phase":"SEED_ALL_5_ORIGINALS_SEALED","seed":seed}),flush=True)

@torch.no_grad()
def report(root:Path):
    torch.set_num_threads(2)
    cfg,selection_result,lr_sha=selection()
    model_cache={}
    rows={}
    replay={}
    final=json.loads(REG.read_text())
    for seed in SEEDS:
        record=json.loads((root/f"seed{seed}"/"FIT_MATRIX.json").read_text())
        if not record["all_five_original_checkpoints"] or record["selection_sha256"]!=lr_sha:
            raise ValueError("missing complete frozen seed source and selected LR checkpoint set")
        for mode in ARMS:
            folder=root/f"seed{seed}"/mode
            receipt=json.loads((folder/"FIT_RECEIPT.json").read_text())
            path=folder/"MODEL.pt"
            if sha(path.read_bytes())!=receipt["checkpoint_sha256"]:
                raise ValueError("actual Git model checkpoint bytes/hash differ")
            cp=torch.load(path,map_location="cpu",weights_only=True)
            if (cp["study"],cp["seed"],cp["mode"],cp["steps"])!=("TAOFINAL005",seed,mode,STEPS):
                raise ValueError("incorrect original model type/owner/update count")
            if cp["lr"]!=selection_result["selected"][mode]["lr"] or cp["selection_sha256"]!=lr_sha:
                raise ValueError("winner LR selected using non-registered data")
            if cp["fit_source_sha256"]!=SOURCE_SHA or cp["model_source_sha256"]!=MODEL_SHA or cp["training_sample_schedule"]!=record["sample_receipt"]:
                raise ValueError("original trained source/order contract differs")
            model=CompactDualClippedTAC086().eval();model.load_state_dict(cp["model"],strict=True)
            if weight_hash(model)!=cp["trained_model_weight_hash"]:
                raise ValueError("original learned model weights differ")
            opt=tune.optimizer_for(mode,model,cp["lr"]);opt.load_state_dict(cp["optimizer"])
            if any(int(s["step"])!=STEPS for s in opt.state.values()):
                raise ValueError("original trained optimizer moment mismatch")
            logits,_=model(cp["cold_fixture"],owners=("cold-check",))
            replay[f"{seed}_{mode}"]=independent_numeric_replay(logits,cp["cold_logits"])
            model_cache[(seed,mode)]=model
            row=next(r for r in final["models"] if r["name"]==f"{mode}_seed{seed}")
            row["measured"]={"trainable_parameters":SIZE,"deployed_parameters":SIZE,
                             "checkpoint_sha256":receipt["checkpoint_sha256"]}
    if len(model_cache)!=25 or check_manifest(final,"final")["status"]!="PASS_SINGLE_MODEL":
        raise ValueError("original final gate or 25 original trained checkpoints incomplete")
    # ZERO report-DEV access occurs before all 25 actual Git-backed original
    # model+optimizer source/replay checks above pass.
    src=json.loads((ROOT/"research/mathematical-engineering/language_source023/SOURCE_RECEIPT.json").read_text())
    report_files=[r for r in src["files"] if r["path"] in cfg["report_dev_owners"]]
    if len(report_files)!=2:raise ValueError("wrong frozen reporting source owner count")
    docs={r["path"]:read_sha_source(ROOT,r["path"],r) for r in report_files}
    target_h=hashlib.sha256()
    for owner,raw in sorted(docs.items()):
        target_h.update(owner.encode()+b"\x00"+raw[1:257])
    by_mode={name:[] for name in ARMS}
    for seed in SEEDS:
        for mode in ARMS:
            model=model_cache[(seed,mode)]
            nll=0.;brier=0.;correct=0;number=0
            for owner,raw in sorted(docs.items()):
                x=torch.tensor([list(raw[:256])],dtype=torch.long)
                y=torch.tensor([list(raw[1:257])],dtype=torch.long)
                logits,_=model(x,owners=(owner,))
                lp=logits.log_softmax(-1);pr=lp.exp()
                target_p=pr.gather(-1,y.unsqueeze(-1)).squeeze(-1)
                nll+=float((-lp.gather(-1,y.unsqueeze(-1)).squeeze(-1)).sum())
                brier+=float((1+pr.square().sum(-1)-2*target_p).sum())
                correct+=int((logits.argmax(-1)==y).sum());number+=y.numel()
            if number!=512:raise ValueError("genuine report source target count differs")
            row={"seed":seed,"lr":selection_result["selected"][mode]["lr"],
                 "report_DEV_nll":nll/number,"report_DEV_accuracy":correct/number,
                 "report_DEV_brier":brier/number,"report_DEV_targets":number,
                 "fit_wall_seconds":float(rows[(seed,mode)]["total_train_wall_seconds"]) if (seed,mode) in rows else None}
            receipt=json.loads((root/f"seed{seed}"/mode/"FIT_RECEIPT.json").read_text())
            row["fit_wall_seconds"]=receipt["total_train_wall_seconds"]
            row["optimizer_wall_seconds"]=receipt["optimizer_wall_seconds"]
            by_mode[mode].append(row)
    control={r["seed"]:r["report_DEV_nll"] for r in by_mode["adamw"]}
    aggregate={}
    tcrit=2.7764451051977987
    for mode,rs in by_mode.items():
        ns=[x["report_DEV_nll"] for x in rs]
        diffs=[x["report_DEV_nll"]-control[x["seed"]] for x in rs]
        mean=statistics.mean(diffs);sd=statistics.stdev(diffs)
        aggregate[mode]={"mean_dev_nll":statistics.mean(ns),
            "mean_top1_accuracy":statistics.mean(x["report_DEV_accuracy"] for x in rs),
            "mean_multiclass_brier":statistics.mean(x["report_DEV_brier"] for x in rs),
            "mean_fit_seconds":statistics.mean(x["fit_wall_seconds"] for x in rs),
            "mean_optimizer_seconds":statistics.mean(x["optimizer_wall_seconds"] for x in rs),
            "mean_paired_nll_minus_adamw":mean,
            "sd_paired_difference":sd,
            "paired_difference_95pct_t_interval_descriptive":[mean-tcrit*sd/math.sqrt(5),mean+tcrit*sd/math.sqrt(5)]}
    result={"study":"TAOFINAL005","phase":"FIVE_SEED_REPORTING_EXPOSED_DEV",
            "research_tier":"INTERNAL_DESCRIPTIVE_PREVIOUSLY_EXPOSED_DEV_NOT_INDEPENDENT",
            "source_sha256":SOURCE_SHA,"tuning_results_sha256":lr_sha,
            "tuning_originals_independently_replayed":True,"report_owners":sorted(docs),
            "report_target_sha256":target_h.hexdigest(),
            "report_targets_per_seed_arm":512,"seed_count":5,"training_updates_per_arm_seed":STEPS,
            "training_targets_per_arm_seed":TARGETS,"selected_LR":{k:v["lr"] for k,v in selection_result["selected"].items()},
            "rows":by_mode,"aggregate":aggregate,
            "independent_all_25_original_model_optimizer_source_replay":replay,
            "all_25_bitwise_logits":all(x["bitwise"] for x in replay.values()),
            "prior_TAO003_cross_host_bitwise_failure_preserved":True,
            "original_DEV_was_exposed_in_prior_TAO003":True,
            "no_architecture_or_independent_superiority_claim":True,
            "protected_TEST":"NOT_ACCESSED","gpu_minutes":0,
            "limits":["Full model training compute identical; optimizer extra FLOPs differ",
                      "Previously exposed same-publisher DEV is not new independent evidence",
                      "Paired interval over only five fixed training seeds is descriptive",
                      "Three LR candidates and one selection seed not globally tuned optimizer",
                      "PNDS whole-source equivalence, BF16/Triton/FSDP and sealed TEST still OPEN"]}
    p=root/"REPORT_RESULTS.json"
    if p.exists():raise FileExistsError("do not replace original first five-seed report")
    p.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
    (root/"FINAL_COMPARISON.json").write_text(json.dumps(final,indent=2,sort_keys=True)+"\n")
    print(json.dumps({"phase":"REAL_SOURCE_FIVE_SEED_REPORT","aggregate":aggregate}),flush=True)

def main():
    p=argparse.ArgumentParser()
    p.add_argument("phase",choices=("fit","report"))
    p.add_argument("--seed",type=int,choices=SEEDS)
    p.add_argument("--directory",type=Path,required=True)
    a=p.parse_args()
    if a.phase=="fit":
        if a.seed is None:raise ValueError("fit requires registered seed")
        fit_seed(a.seed,a.directory/f"seed{a.seed}")
    else:
        if a.seed is not None:raise ValueError("report all fixed seeds together only")
        report(a.directory)

if __name__=="__main__":main()
