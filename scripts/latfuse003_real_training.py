"""LATFUSE003 original UCI501 genuine source real fit, original bytes & clean replay.

Single new trained candidate, NOT a fair same-FLOPs neural architecture comparison.
No synthetic or fabricated source outcomes, no causal do(action) claims. Source
FIT 2013-2014, 2014 validation, already-exposed 2015 DEV only after BOTH
original checkpoint weights are frozen, protected 2016-2017 NEVER PARSED.

Read registration before fits. Two seeds × exactly 128 update steps; no tuning,
retries or checkpoint selection. Original sources/weights never overwritten.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import time

import torch
from scripts import latfuse002_real_fit as legacy
from scripts.validate_research_parity import check_manifest
from tac_osm.latent_fusion_003 import FusedLatentFourHead,exact_state_contract

ROOT=Path(__file__).resolve().parents[1]
REG=ROOT/"research/mathematical-engineering/latfuse003/REAL_FIT_REGISTRATION.json"
OUT=ROOT/"research/replay-bundles/LATFUSE003"
SEEDS=(310301,310302)
STEPS=128
BATCH=8
HISTORY=24
HORIZON=6
LAMBDA_KEYS=.25
LAMBDA_BALANCE=.03


def source_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def registered():
    j=json.loads(REG.read_text())
    check_manifest(j,"preflight")
    if (j.get("study_id")!="LATFUSE003-REAL-SINGLE-001" or
        [(m["seed"],m["trainable_parameters"],m["deployed_parameters"])
            for m in j["models"]] != [(seed,99747,99747) for seed in SEEDS] or
        j["real_world_source"]["sha256"]!=legacy.SOURCE_SHA):
        raise RuntimeError("original preregistration or live parameter budget mismatch")
    return j


def origins(seed):
    rng=random.Random(seed)
    return [(rng.randrange(12),rng.randrange(HISTORY-1,
            legacy.hour(legacy.TRAIN_END)-HORIZON))
            for _ in range(STEPS*BATCH)]


def recorded_batch(model,d,examples):
    sid=torch.tensor([x[0] for x in examples],dtype=torch.long)
    ix=torch.tensor([x[1] for x in examples],dtype=torch.long)
    if int(ix.min())<HISTORY-1 or int(ix.max())>=legacy.hour(legacy.TRAIN_END)-HORIZON:
        raise RuntimeError("observational source origin outside permitted train time")
    memory=model.initial(len(examples))
    enc=[]
    handle=model.embed.register_forward_hook(lambda _m,_args,out:enc.append(out))
    try:
        for t in range(HISTORY):
            original=ix-HISTORY+1+t
            memory=model.step(memory,d["values"][sid,original],d["valid"][sid,original],
                              d["age"][sid,original],d["last"][sid,original],sid)
    finally:
        handle.remove()
    # Only target-time known calendar — NOT physical future sensor/action.
    q=legacy.calendar([int(t+HORIZON) for t in ix]).unsqueeze(1)
    forecast,scale=model.forecast(memory,sid,q)
    # Must not use future observed measurements as model, state, router or query input.
    y=d["values"][sid,ix+HORIZON]
    valid=d["valid"][sid,ix+HORIZON]
    mu=forecast[:,0,:];sd=scale[:,0,:]
    target=torch.where(valid,y,mu.detach())
    nll=.5*(math.log(2*math.pi)+2*sd.log()+((target-mu)/sd).square())
    point=(target-mu).abs()
    observation=((.8*nll+.2*point)*valid).sum()/valid.sum().clamp_min(1)
    encoded=torch.stack(enc,dim=1)
    hist_ix=torch.stack([ix-HISTORY+1+t for t in range(HISTORY)],dim=1)
    past=d["values"][sid[:,None],hist_ix]
    seen=d["valid"][sid[:,None],hist_ix]
    separator=model.source_key_separation_loss(encoded,past,seen)
    balance=model.head_utilization_loss(encoded,seen)
    loss=observation+LAMBDA_KEYS*separator+LAMBDA_BALANCE*balance
    return loss,(observation,separator,balance)


def load_model(seed,idx):
    torch.manual_seed(seed)
    m=FusedLatentFourHead()
    contract=exact_state_contract(m)
    if contract["nominal_bytes"]!=6800 or contract["trainable_parameters"]!=99747:
        raise RuntimeError("trained model breaks fixed-state budget")
    if not all(p.dtype==torch.float32 for p in m.parameters()):
        raise RuntimeError("non-FP32 trained parameters")
    return m,contract


def nofit_preflight(d):
    """Already registered real source, zero optimizer steps and no heldout."""
    registered()
    model,c=load_model(SEEDS[0],None)
    init=[p.detach().clone() for p in model.parameters()]
    first=origins(SEEDS[0])[:BATCH]
    loss,components=recorded_batch(model,d,first)
    if not torch.isfinite(loss):
        raise RuntimeError("original genuine source prefit loss is not finite")
    loss.backward()
    for name,p in model.named_parameters():
        if p.grad is None or not bool(torch.isfinite(p.grad).all()):
            raise RuntimeError("unconnected/nonfinite original source model gradient: "+name)
    if any(not torch.equal(p,v) for p,v in zip(model.parameters(),init)):
        raise RuntimeError("no-fit source-gradient probe mutated weights")
    return dict(status="PASS_NO_FIT_TRUE_SOURCE",candidate_contract=c,
        source_original_sha256=legacy.SOURCE_SHA,source_channel_mask_consistent=True,
        source_FIT_only=True,forward_trainable_loss=float(loss.detach()),
        zero_optimizer_steps=True)


def fit(source):
    registered()
    if OUT.exists():
        raise RuntimeError("LATFUSE003 original source fit artifacts already exist; no overwrite")
    start=time.monotonic()
    d=legacy.source_frames(source,phase="fit")
    nofit=nofit_preflight(d)
    val=legacy.eval_origins(legacy.hour(legacy.TRAIN_END),legacy.hour(legacy.VAL_END))
    dev=legacy.eval_origins(legacy.hour(legacy.VAL_END),legacy.hour(legacy.DEV_END))
    srcfiles=["src/tac_osm/latent_fusion_001.py",
        "src/tac_osm/latent_fusion_002.py","src/tac_osm/latent_fusion_003.py",
        "scripts/latfuse002_real_fit.py","scripts/latfuse003_real_training.py",
        "scripts/validate_research_parity.py"]
    src_hash={s:source_sha(ROOT/s) for s in srcfiles}
    OUT.mkdir(parents=True,exist_ok=False)
    (OUT/"NOFIT_RECEIPT.json").write_text(json.dumps(nofit,indent=2,sort_keys=True)+"\n")
    rows=[]
    for seed in SEEDS:
        model,contract=load_model(seed,None)
        randplan=origins(seed)
        plan_sha=hashlib.sha256(json.dumps(randplan,separators=(",",":")).encode()).hexdigest()
        opt=torch.optim.AdamW(model.parameters(),lr=.0005,weight_decay=.001)
        losses=[]
        optimizer_steps=0
        before=[p.detach().clone() for p in model.parameters()]
        for i in range(STEPS):
            model.train()
            opt.zero_grad(set_to_none=True)
            cases=randplan[i*BATCH:(i+1)*BATCH]
            loss,components=recorded_batch(model,d,cases)
            if not torch.isfinite(loss):
                raise RuntimeError(f"seed {seed}, step {i}: nonfinite genuine source loss")
            loss.backward()
            if not all(p.grad is not None and torch.isfinite(p.grad).all()
                       for p in model.parameters()):
                raise RuntimeError("nonfinite/disconnected true-data model gradients")
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            opt.step()
            optimizer_steps+=1
            losses.append(float(loss.detach()))
        if optimizer_steps!=STEPS or not any(
                not torch.equal(p,z) for p,z in zip(model.parameters(),before)):
            raise RuntimeError("original model did not receive exact registered optimization")
        if exact_state_contract(model)!=contract:
            raise RuntimeError("source actual postfit parameter-state mismatch")
        folder=OUT/f"seed-{seed}"
        folder.mkdir()
        checkpoint=dict(model=model.state_dict(),optimizer=opt.state_dict(),
            seed=seed,steps=STEPS,batch_size=BATCH,history=HISTORY,horizon=HORIZON,
            lambda_key_separation=LAMBDA_KEYS,lambda_head_balance=LAMBDA_BALANCE,
            torch_rng=torch.get_rng_state(),plan_sha256=plan_sha,
            original_source_sha256=legacy.SOURCE_SHA,
            train_only_normalization=dict(center=d["center"],spread=d["spread"]),
            source_python_sha256=src_hash,actual_contract=contract,
            first_training_loss=losses[0],final_sampled_training_loss=losses[-1],
            full_training_losses=losses)
        f=folder/"final.pt"
        torch.save(checkpoint,f)
        cp_sha=source_sha(f)
        fitness=dict(seed=seed,training_steps=optimizer_steps,
            actual_training_cases=STEPS*BATCH,train_source_plan_sha256=plan_sha,
            original_checkpoint_sha256=cp_sha,actual_contract=contract,
            first_training_loss=losses[0],final_sampled_training_loss=losses[-1],
            all_parameter_tensors_updated=sum(int(not torch.equal(p,z))
                                   for p,z in zip(model.parameters(),before)))
        (folder/"FIT_RECEIPT.json").write_text(json.dumps(fitness,indent=2,sort_keys=True)+"\n")
        validation=legacy.evaluate(model,d,val)
        rows.append(dict(**fitness,validation_2014=validation))
        print("LATFUSE003_ORIGINAL_FIT_SEED="+json.dumps(rows[-1],sort_keys=True),flush=True)
    # Original weights, optimizer, histories, exact SHA for BOTH seeds frozen
    # before parsing any historically-exposed 2015 REPORT numeric field.
    reportdata=legacy.source_frames(source,phase="report")
    for row in rows:
        seed=row["seed"];folder=OUT/f"seed-{seed}"
        f=folder/"final.pt"
        if source_sha(f)!=row["original_checkpoint_sha256"]:
            raise RuntimeError("one original model mutated before previously-exposed DEV")
        cp=torch.load(f,map_location="cpu",weights_only=False)
        model,c=load_model(seed,None)
        model.load_state_dict(cp["model"],strict=True)
        development,arrays=legacy.evaluate(model,reportdata,dev,store=True)
        torch.save(arrays,folder/"DEV_PREDICTIONS.pt")
        row["development_2015"]=development
        row["previously_exposed_dev_only"]=True
        print("LATFUSE003_EXPOSED_DEV_SEED="+json.dumps(row,sort_keys=True),flush=True)
    report=dict(study_id="LATFUSE003-REAL-SINGLE-001",
        original_source_sha256=legacy.SOURCE_SHA,
        source_files_sha256=src_hash,parameters=99747,
        fp32_recurrent_state_bytes=6800,trained_seeds=list(SEEDS),
        actual_total_optimizer_steps=STEPS*len(SEEDS),
        original_training_data_scope="2013-03 to 2014-08 source FIT original",
        out_of_time_validation_scope="2014-09 to 2014-12 source original",
        development_scope="2015 source already-exposed DEV, all numerical targets parsed only AFTER original trained checkpoints saved",
        original_source_2016_2017="numeric records NEVER parsed",
        relative_compute_to_Latfuse002_not_equal=True,
        prefit_partial_forward_backward_flops_parent=58640568,
        prefit_partial_forward_backward_flops_candidate_with_aux=61714798,
        evidence_status="SINGLE_CANDIDATE_ORIGINAL_REAL_DATA_ONLY_NO_MATCHED_COMPUTE_ARCH_WIN",
        total_fit_pipeline_wall_seconds=time.monotonic()-start,results=rows)
    (OUT/"RESULTS.json").write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    print("LATFUSE003_ORIGINAL_FIT_RESULT="+json.dumps(report,sort_keys=True),flush=True)


def replay(source):
    registered()
    report=json.loads((OUT/"RESULTS.json").read_text())
    if (report["original_source_sha256"]!=legacy.SOURCE_SHA or
        report["trained_seeds"]!=list(SEEDS)):
        raise RuntimeError("registered original model source/seed changed")
    d=legacy.source_frames(source,phase="report")
    origins_=legacy.eval_origins(legacy.hour(legacy.VAL_END),legacy.hour(legacy.DEV_END))
    for result in report["results"]:
        seed=result["seed"]
        folder=OUT/f"seed-{seed}"
        f=folder/"final.pt"
        if source_sha(f)!=result["original_checkpoint_sha256"]:
            raise RuntimeError("stored original trained source weight SHA changed")
        checkpoint=torch.load(f,map_location="cpu",weights_only=False)
        if checkpoint["steps"]!=STEPS or checkpoint["plan_sha256"] != hashlib.sha256(
             json.dumps(origins(seed),separators=(",",":")).encode()).hexdigest():
            raise RuntimeError("registered real source trained exposure changed")
        model,c=load_model(seed,None)
        model.load_state_dict(checkpoint["model"],strict=True)
        score,raw=legacy.evaluate(model,d,origins_,store=True)
        original=torch.load(folder/"DEV_PREDICTIONS.pt",map_location="cpu",weights_only=True)
        for key in ("origin","forecast","truth","eligible","persistence"):
            if not torch.equal(raw[key],original[key]):
                raise RuntimeError(f"original real source prediction array mismatch: {key}")
        for k,v in score.items():
            previous=result["development_2015"][k]
            if v is None:
                if previous is not None:raise RuntimeError("missing original score discrepancy")
            elif isinstance(v,float):
                if abs(v-previous)>1e-8:raise RuntimeError(f"score mismatch: {k}")
            elif v!=previous:raise RuntimeError("integer original source mismatch")
    print("LATFUSE003_CLEAN_REPLAY_PASS="+json.dumps(
        dict(original_seeds=list(SEEDS),source_sha256=legacy.SOURCE_SHA,
             original_weights_verified=True,no_optimizer=True,
             all_2015_previous_DEV_arrays_bit_exact=True,
             original_protected_numeric_rows_unparsed=True),sort_keys=True))


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--mode",choices=("prefit","fit","replay"),required=True)
    ap.add_argument("--source",type=Path,required=True)
    args=ap.parse_args()
    torch.set_num_threads(1)
    if source_sha(args.source)!=legacy.SOURCE_SHA:
        raise RuntimeError("invalid original source archive")
    if args.mode=="prefit":
        registered()
        d=legacy.source_frames(args.source,phase="fit")
        print("LATFUSE003_ORIGINAL_SOURCE_PREFIT="+json.dumps(nofit_preflight(d),sort_keys=True))
    elif args.mode=="fit":
        fit(args.source)
    else:replay(args.source)
if __name__=="__main__":main()
