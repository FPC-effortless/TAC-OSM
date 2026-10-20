#!/usr/bin/env python3
"""ME017: same native TAC and both GRUs; test optimizer EMA, not new components.

Original real KuaiRec FIT/previously inspected DEV only; user%16==4 remains unopened.
"""
from __future__ import annotations
import argparse,copy,hashlib,json,math,time
from pathlib import Path
import torch
from torch.nn import functional as F
from scripts.me016_native_slow_identity import (
    EngineeredAgent,read_early,build_examples,empirical_baselines,binary_scores,
    evaluate,gate_telemetry,parameters,batch,FIT_CUT,DEV_CUT,SEEDS,
    SOURCE_SHA,PIN_MD5,HISTORY)

ARMS=("native_chrono_anchor","gru_original","gru_anchor")
EMA_BETA=0.99
STEPS=600
BATCH=128
PARAMETERS=135050

def train(out):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    records,source=read_early()
    if source["zip_sha256"]!=SOURCE_SHA or source["zip_md5"]!=PIN_MD5:
        raise ValueError("official authentic data checksum changed")
    if source["future_test_labels_parsed"] or source["later_rows_skipped_before_label_parsing"]<1:
        raise ValueError("original user/time test privacy violated")
    train=build_examples(records,0,FIT_CUT)
    dev=build_examples(records,FIT_CUT,DEV_CUT)
    if (len(train["y"]),len(dev["y"]))!=(68825,106245):
        raise ValueError("original FIT/DEV cohort changed")
    baselines=empirical_baselines(train,dev)
    results={
      "study":"ME017_ORIGINAL_REAL_EMA_STABILITY",
      "official_source":source,
      "authentic_train_windows":len(train["y"]),
      "historically_opened_dev_windows":len(dev["y"]),
      "historically_opened_dev_not_fresh_test":True,
      "user4_and_previous_future_test_outcomes_parsed":False,
      "original_seed_list":list(SEEDS),
      "optimizer_updates_per_model":STEPS,
      "batch_size":BATCH,
      "ema_decay_preregistered":EMA_BETA,
      "active_parameters_per_model":PARAMETERS,
      "original_trainonly_baselines":{
         "causal_prior32":binary_scores(baselines["history_mean_predictions"],dev["y"].numpy()),
         "original_item_rate":binary_scores(baselines["item_predictions"],dev["y"].numpy())},
      "models":[],
      "test_evidence_status":"NOT_RUN",
      "architecture_promoted":False}
    torch.set_num_threads(1)
    for seed in SEEDS:
        for kind in ARMS:
            torch.manual_seed(seed)
            net=EngineeredAgent(kind)
            if parameters(net)!=PARAMETERS:raise ValueError("mismatched active parameters")
            weights={name:p.detach().clone() for name,p in net.named_parameters()}
            optim=torch.optim.AdamW(net.parameters(),lr=0.001,weight_decay=0.0001)
            gen=torch.Generator().manual_seed(seed+300)
            clock=time.perf_counter();curve=[]
            for step in range(1,STEPS+1):
                ids=torch.randint(len(train["y"]),(BATCH,),generator=gen)
                logit=net(*batch(train,ids))
                loss=F.binary_cross_entropy_with_logits(logit,train["y"][ids])
                if not bool(torch.isfinite(loss)):raise ValueError("nonfinite loss")
                optim.zero_grad(set_to_none=True);loss.backward()
                norm=torch.nn.utils.clip_grad_norm_(net.parameters(),1.0)
                if not bool(torch.isfinite(norm)):raise ValueError("nonfinite gradient")
                optim.step()
                with torch.no_grad():
                    for name,p in net.named_parameters():
                        weights[name].lerp_(p,1-EMA_BETA)
                if step in (1,150,300,450,600):
                    curve.append({"step":step,"train_batch_BCE":float(loss.detach()),"grad_norm":float(norm.detach())})
            training_seconds=round(time.perf_counter()-clock,3)
            final_score=evaluate(net,dev)
            final_diag=gate_telemetry(net,dev)
            ema_net=copy.deepcopy(net)
            with torch.no_grad():
                for name,p in ema_net.named_parameters():
                    p.copy_(weights[name])
            ema_score=evaluate(ema_net,dev)
            ema_diag=gate_telemetry(ema_net,dev)
            folder=out/"checkpoints";folder.mkdir(parents=True,exist_ok=True)
            saved={}
            for suffix,model,beta in (("final",net,None),("ema",ema_net,EMA_BETA)):
                file=folder/f"{kind}-{seed}-{suffix}.pt"
                torch.save({"kind":kind,"seed":seed,"steps":STEPS,
                   "source_sha256":SOURCE_SHA,"ema_beta":beta,
                   "state_dict":model.state_dict()},file)
                saved[suffix]={"path":str(file.relative_to(out)),
                    "sha256":hashlib.sha256(file.read_bytes()).hexdigest()}
            entry={"model":kind,"seed":seed,"trainable_parameters":PARAMETERS,
                   "true_training_cpu_seconds":training_seconds,
                   "final_on_historic_dev":final_score,
                   "ema_on_historic_dev":ema_score,
                   "historical_final_gate_measures":final_diag,
                   "historical_ema_gate_measures":ema_diag,
                   "training_loss_trace":curve,"checkpoints":saved}
            results["models"].append(entry)
            print(json.dumps({"arm":kind,"seed":seed,
                "historical_final_brier":final_score["brier"],
                "historical_ema_brier":ema_score["brier"],
                "training_cpu_seconds":training_seconds},sort_keys=True),flush=True)
    path=out/"results.json"
    path.write_text(json.dumps(results,sort_keys=True,indent=2)+"\n")
    print(json.dumps({"status":"ME017_EMA_REAL_FIT_COMPLETE",
       "fit_count":len(results["models"]),"checkpoint_count":2*len(results["models"]),
       "new_cohort_labels_read":False,"promoted":False}),flush=True)
    return results

def preflight():
    torch.manual_seed(17017);torch.set_num_threads(1)
    assert len(ARMS)==3 and EMA_BETA==0.99
    x=torch.randint(16000,(3,HISTORY))
    feedback=torch.rand(3,HISTORY,3)
    feedback[:,:,1]=(feedback[:,:,1]>.5).float()
    action=torch.randint(16000,(3,))
    duration=torch.rand(3,1)
    for kind in ARMS:
        m=EngineeredAgent(kind)
        assert parameters(m)==PARAMETERS
        p=m(x,feedback,action,duration)
        assert p.shape==(3,1) and bool(torch.isfinite(p).all())
        clone=copy.deepcopy(m)
        with torch.no_grad():
            for (_,v),(_,q) in zip(m.named_parameters(),clone.named_parameters()):
                assert torch.equal(v,q)
        F.binary_cross_entropy_with_logits(p,torch.rand_like(p)).backward()
        assert all(bool(torch.isfinite(t.grad).all()) for t in m.parameters() if t.grad is not None)
    print("ME017_EQUIVALENT_INITIAL_MODEL_EMA_SOFTWARE_PASS")

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--out",default="results/me017")
    parser.add_argument("--self-test",action="store_true")
    args=parser.parse_args()
    preflight() if args.self_test else train(args.out)
