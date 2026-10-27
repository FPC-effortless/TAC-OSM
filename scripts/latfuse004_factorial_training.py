"""LATFUSE004 2x2 factorial original real-source exploratory fits and clean replay.

No architecture superiority because prefit FLOPs exceed exact <=1% comparative
tolerance (1.121%). No invented actions or synthetic capability evidence.
Exact same source cases/sampling order and optimizer schedule per seed, but
different computational work; honest DESCRIPTIVE comparisons only.
No protected numerical rows 2016-2017, original 2015 DEV opened only AFTER
all EIGHT original trained weights and optimizer histories are fixed to bytes.
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

from scripts import latfuse002_real_fit as source
from scripts import latfuse004_factorial_prefit as probes
from scripts.validate_research_parity import check_manifest
from tac_osm.latent_fusion_004_factorial import ARMS,FactorialFusion,contract

ROOT=Path(__file__).resolve().parents[1]
REG=ROOT/"research/mathematical-engineering/latfuse004/REAL_FIT_REGISTRATION.json"
OUT=ROOT/"research/replay-bundles/LATFUSE004"
SEEDS=(40401,40402)
STEPS=128
BATCH=8
WEIGHTS={"4h-hard":99747,"4h-soft":99747,"2h-hard":99261,"2h-soft":99261}


def digest(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def plan(seed):
    rng=random.Random(seed)
    return [(rng.randrange(12),rng.randrange(24,source.hour(source.TRAIN_END)-7))
            for _ in range(STEPS*BATCH)]


def audit_prereg():
    info=json.loads(REG.read_text())
    assert check_manifest(info,"preflight")["status"]=="PASS_PREFIT_SINGLE_MODEL"
    expected={(arm,seed,WEIGHTS[arm]) for seed in SEEDS for arm in ARMS}
    real={(row["architecture"],row["seed"],row["trainable_parameters"])
          for row in info["models"]}
    if real!=expected or len(info["models"])!=len(expected):
        raise RuntimeError("8-arm original source preregistration mismatch")
    if info["real_world_source"]["sha256"]!=source.SOURCE_SHA:
        raise RuntimeError("preregistered source hash mismatch")
    if info["protocol"]["training_compute_caveat"]["fully_verified"]:
        raise RuntimeError("cannot pretend full FLOP parity")
    return info


def new_model(seed,arm):
    torch.manual_seed(seed)
    m=FactorialFusion(arm)
    got=contract(arm)
    if (got["per_stream_nominal_bytes"]!=6800 or
        got["active_parameters"]!=WEIGHTS[arm]):
        raise RuntimeError("original trained budget violated")
    if not all(p.dtype==torch.float32 and p.requires_grad for p in m.parameters()):
        raise RuntimeError("candidate requires fully trained FP32 parameters")
    return m,got


def channel_calibration(a):
    m=a["forecast"][:,0,:]
    sd=a["forecast"][:,1,:]
    truth=a["truth"];observed=a["eligible"]
    safe=torch.where(observed,truth,m)
    rows=[]
    names=("PM2.5","PM10","SO2","NO2","CO","O3","TEMP","PRES","DEWP","RAIN","WSPM","WD_sin","WD_cos")
    for idx,name in enumerate(names):
        valid=observed[:,idx]
        count=int(valid.sum())
        if count==0:
            rows.append(dict(channel=name,n=0))
            continue
        mu=m[:,idx][valid];sigma=sd[:,idx][valid];y=safe[:,idx][valid]
        standardized=(y-mu).abs()/sigma
        nl=.5*(math.log(2*math.pi)+2*sigma.log()+standardized.square())
        rows.append(dict(channel=name,n=count,
                         normalized_mae=float((y-mu).abs().mean()),
                         normalized_gaussian_nll=float(nl.mean()),
                         nominal_90_coverage=float((standardized<=1.6448536269514722).float().mean()),
                         abs_standardized_residual_over_3_fraction=float((standardized>3).float().mean())))
    return rows


def fitting(source_path):
    audit_prereg()
    if OUT.exists():raise RuntimeError("original LATFUSE004 checkpoint output already exists; never overwrite")
    d=source.source_frames(source_path,phase="fit")
    val=source.eval_origins(source.hour(source.TRAIN_END),source.hour(source.VAL_END))
    dev=source.eval_origins(source.hour(source.VAL_END),source.hour(source.DEV_END))
    original_source_files=[
        "src/tac_osm/latent_fusion_001.py",
        "src/tac_osm/latent_fusion_002.py",
        "src/tac_osm/latent_fusion_003.py",
        "src/tac_osm/latent_fusion_004_factorial.py",
        "scripts/latfuse002_real_fit.py",
        "scripts/latfuse004_factorial_prefit.py",
        "scripts/latfuse004_factorial_training.py"]
    hashes={p:digest(ROOT/p) for p in original_source_files}
    OUT.mkdir(parents=True,exist_ok=False)
    report=[]
    for seed in SEEDS:
        original_cases=plan(seed)
        sample_sha=hashlib.sha256(json.dumps(original_cases,separators=(",",":")).encode()).hexdigest()
        for arm in ARMS:
            model,original_count=new_model(seed,arm)
            opt=torch.optim.AdamW(model.parameters(),lr=.0005,weight_decay=.001)
            learned_before=[p.detach().clone() for p in model.parameters()]
            losses=[];optimizer_updates=0
            begun=time.monotonic()
            for i in range(STEPS):
                opt.zero_grad(set_to_none=True)
                model.train()
                examples=original_cases[i*BATCH:(i+1)*BATCH]
                loss,components=probes.source_batch(d,model,examples)
                if not bool(torch.isfinite(loss)):
                    raise RuntimeError(f"source FIT nonfinite at {arm} seed {seed} update {i}")
                loss.backward()
                if not all(p.grad is not None and bool(torch.isfinite(p.grad).all())
                           for p in model.parameters()):
                    raise RuntimeError("source FIT gradient invalid/disconnected")
                torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
                opt.step()
                optimizer_updates+=1
                losses.append(float(loss.detach()))
            elapsed=time.monotonic()-begun
            if optimizer_updates!=STEPS or not any(
                not torch.equal(p,v) for p,v in zip(model.parameters(),learned_before)):
                raise RuntimeError("source optimizer/update count did not advance as preregistered")
            if contract(arm)!=original_count:
                raise RuntimeError("trained state/parameter architecture changed")
            folder=OUT/f"seed-{seed}"/arm
            folder.mkdir(parents=True)
            cp=folder/"final.pt"
            torch.save(dict(model=model.state_dict(),optimizer=opt.state_dict(),
                seed=seed,arm=arm,steps=STEPS,batch_size=BATCH,history=24,horizon=6,
                original_source_sha256=source.SOURCE_SHA,
                original_training_case_order_sha256=sample_sha,
                original_frozen_source_hashes=hashes,
                FIT_normalizer=dict(center=d["center"],spread=d["spread"]),
                torch_rng=torch.get_rng_state(),first_loss=losses[0],
                last_loss=losses[-1],training_loss_trace=losses,
                exact_original_recurrent_state=original_count),cp)
            sha=digest(cp)
            # Chronological 2014 validation only. The 2015 numerical report
            # is forbidden until all 8 weights + receipts have been frozen.
            val_metrics,val_arrays=source.evaluate(model,d,val,store=True)
            receipt=dict(arm=arm,seed=seed,
                optimizer_steps=optimizer_updates,original_training_examples=STEPS*BATCH,
                training_case_order_sha256=sample_sha,
                original_checkpoint_sha256=sha,
                trained_parameter_tensors_changed=sum(int(not torch.equal(p,v))
                       for p,v in zip(model.parameters(),learned_before)),
                state_contract=original_count,
                fit_CPU_wall_seconds=elapsed,
                sampled_train_loss_first=losses[0],
                sampled_train_loss_last=losses[-1],
                2014_validation=val_metrics,
                2014_channel_calibration=channel_calibration(val_arrays))
            (folder/"FIT_RECEIPT.json").write_text(json.dumps(receipt,indent=2,sort_keys=True)+"\n")
            report.append(receipt)
            print("LATFUSE004_ORIGINAL_FIT_ARM="+json.dumps({
                "arm":arm,"seed":seed,"optimizer_steps":optimizer_updates,
                "checkpoint_sha256":sha,"validation":val_metrics,
                "train_loss_last":losses[-1]},sort_keys=True),flush=True)

    # The original numerical DEV2015 source remains sealed until all eight
    # checkpoint bytes have been SHA-frozen and all FIT receipts persisted.
    for r in report:
        cp=OUT/f"seed-{r['seed']}"/r["arm"]/"final.pt"
        if digest(cp)!=r["original_checkpoint_sha256"]:
            raise RuntimeError("one original checkpoint changed before previously exposed DEV")
    d_dev=source.source_frames(source_path,phase="report")
    for r in report:
        folder=OUT/f"seed-{r['seed']}"/r["arm"]
        cp=torch.load(folder/"final.pt",map_location="cpu",weights_only=False)
        model,_=new_model(r["seed"],r["arm"])
        model.load_state_dict(cp["model"],strict=True)
        result,arrays=source.evaluate(model,d_dev,dev,store=True)
        torch.save(arrays,folder/"DEV_PREDICTIONS.pt")
        r["2015_exposed_DEV"]=result
        r["2015_channel_calibration"]=channel_calibration(arrays)
        print("LATFUSE004_EXPOSED_DEV_ARM="+json.dumps(
             {"arm":r["arm"],"seed":r["seed"],"dev":result},sort_keys=True),flush=True)
    original=dict(study_id="LATFUSE004-REAL-FACTORIAL-EXPLORATORY",
        source_sha256=source.SOURCE_SHA,original_code_sha256=hashes,
        parameter_budget_max_gap=(99747-99261)/99747,
        prefit_partial_flops_gap=0.01121155949233175,
        fair_head_to_head_eligible=False,
        original_real_fit_start="2013-03-01",original_real_fit_end_exclusive="2014-09-01",
        original_validation_end_exclusive="2015-01-01",
        exposed_2015_Dev_numerical_read_only_after_eight_cp=True,
        protected_2016_2017_numeric_untouched=True,
        real_source_future_known_calendar_not_action=True,
        all_eight_original_optimizations_finished=True,
        per_arm_original_optimizer_updates=STEPS,
        total_original_optimizer_updates=8*STEPS,
        trained_arms=list(ARMS),original_seeds=list(SEEDS),
        study_is_not_qualifying_matched_compute_comparison=True,
        models=report)
    (OUT/"RESULTS.json").write_text(json.dumps(original,indent=2,sort_keys=True)+"\n")
    print("LATFUSE004_RESULT="+json.dumps(
       {"status":"TRAINING_COMPLETED_AND_ORIGINAL_STORED","models":len(report),
       "fair_comparison":False},sort_keys=True),flush=True)


def replay(source_path):
    reg=audit_prereg()
    report=json.loads((OUT/"RESULTS.json").read_text())
    if not report["all_eight_original_optimizations_finished"] or len(report["models"])!=8:
        raise RuntimeError("incomplete source-trained factorial original evidence")
    d=source.source_frames(source_path,phase="report")
    dev=source.eval_origins(source.hour(source.VAL_END),source.hour(source.DEV_END))
    expected={(seed,arm) for seed in SEEDS for arm in ARMS}
    assert {(r["seed"],r["arm"]) for r in report["models"]}==expected
    for r in report["models"]:
        path=OUT/f"seed-{r['seed']}"/r["arm"]
        checkpoint=path/"final.pt"
        if digest(checkpoint)!=r["original_checkpoint_sha256"]:
            raise RuntimeError("original source-trained checkpoint SHA mismatch")
        cp=torch.load(checkpoint,map_location="cpu",weights_only=False)
        order=hashlib.sha256(json.dumps(plan(r["seed"]),separators=(",",":")).encode()).hexdigest()
        if (cp["seed"]!=r["seed"] or cp["arm"]!=r["arm"] or
            cp["steps"]!=STEPS or cp["original_source_sha256"]!=source.SOURCE_SHA or
            cp["original_training_case_order_sha256"]!=order or
            r["training_case_order_sha256"]!=order):
            raise RuntimeError("original learned experiment input/sample plan mismatch")
        model,_=new_model(r["seed"],r["arm"])
        model.load_state_dict(cp["model"],strict=True)
        actual,s=source.evaluate(model,d,dev,store=True)
        original=torch.load(path/"DEV_PREDICTIONS.pt",map_location="cpu",weights_only=True)
        for k in ("origin","forecast","truth","eligible","persistence"):
            if not torch.equal(s[k],original[k]):
                raise RuntimeError("original source numerical replay changed: "+k)
        for metric,val in actual.items():
            old=r["2015_exposed_DEV"][metric]
            if val is None:
                if old is not None:raise RuntimeError("missing original DEV scalar mismatch")
            elif isinstance(val,float):
                if abs(val-old)>1e-8:raise RuntimeError("original DEV metric changed: "+metric)
            elif val!=old:raise RuntimeError("original DEV metric changed: "+metric)
        calibrated=channel_calibration(s)
        for new,old in zip(calibrated,r["2015_channel_calibration"]):
            for k in new:
                x=new[k];y=old[k]
                if isinstance(x,float):
                    if abs(x-y)>1e-8:raise RuntimeError("channel calibration mismatch")
                elif x!=y:raise RuntimeError("channel identity mismatch")
    print("LATFUSE004_CLEAN_REPLAY_PASS="+json.dumps(
        {"original_models":8,"seeds":list(SEEDS),"arms":list(ARMS),
         "source_sha256":source.SOURCE_SHA,"no_optimizer_or_refit":True,
         "original_prediction_arrays_bitwise":True,
         "fair_equal_compute_comparison_proven":False},sort_keys=True),flush=True)


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--mode",choices=["prefit","fit","replay"],required=True)
    p.add_argument("--source",type=Path,required=True)
    a=p.parse_args()
    torch.set_num_threads(1)
    if digest(a.source)!=source.SOURCE_SHA:raise RuntimeError("original source invalid")
    if a.mode=="fit":fitting(a.source)
    elif a.mode=="replay":replay(a.source)
    else:
        audit_prereg()
        d=source.source_frames(a.source,phase="fit")
        ex=plan(SEEDS[0])[:BATCH]
        for arm in ARMS:
            model,_=new_model(SEEDS[0],arm)
            loss,_=probes.source_batch(d,model,ex)
            loss.backward()
            assert torch.isfinite(loss)
            assert all(p.grad is not None and torch.isfinite(p.grad).all()
                       for p in model.parameters())
        print("LATFUSE004_REGISTERED_SOURCE_NOFIT_PASS",flush=True)
if __name__=="__main__":
    main()
