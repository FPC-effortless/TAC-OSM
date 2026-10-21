#!/usr/bin/env python3
"""ME045 authentic full-model training, hard routing, latency and exact replay.

Original ME044 remains untouched. All source observations are authentic,
publisher-sha-pinned FIT and historically reused DEV measurements. No opened
new test cohort or simulated performance evidence.
"""
from __future__ import annotations
import argparse,hashlib,json,math,platform,resource,subprocess,time
from pathlib import Path
import numpy as np
import torch
from scripts.me043_train_adaptive import DOMAINS, save, sha
from tac_osm.me044_multiscale_decoder import build_model as prior_model, objective as prior_objective
from tac_osm.me045_conditional_multiscale import ARMS, make_model

HORIZONS=(96,720)
SEEDS=(45001,45002)
LANES=("exposure","cpu")
MODES=("adaptive","always","cheap")
UPDATES=1000
CPU_BUDGET=10.
MAX_CPU_UPDATES=10000
BATCH=4
SOURCE="ME045_REAL_FROZEN_CONTEXT_ROUTED_MULTISCALE"
torch.set_num_threads(1)


def period(domain):return 144 if domain=="tetouan" else 24


def input_examples(z,mod,h,seed,generator):
    anchors=torch.randint(96,mod.TR-h+1,(BATCH,),generator=generator)
    return mod.samples(z,anchors,h)


def pretrain_context(z,mod,domain,h,seed,parent,prec,out):
    """Genuine 1000-update separately saved common context F0, no DEV training."""
    torch.manual_seed(seed)
    old=prior_model("original_context",h,seed,period(domain))
    old.initialize_ar(parent)
    opt=torch.optim.AdamW([
        {"params":list(old.ar.parameters()),"lr":.0001},
        {"params":[p for k,p in old.named_parameters() if not k.startswith("ar.")],
         "lr":.001}],weight_decay=1e-4)
    rng=torch.Generator().manual_seed(seed+2000)
    t0=time.process_time();cpu=0.
    for step in range(1,UPDATES+1):
        x,y=input_examples(z,mod,h,seed,rng)
        tic=time.process_time()
        old.train();opt.zero_grad(set_to_none=True)
        loss,_=prior_objective(old,x,y,step)
        if not bool(torch.isfinite(loss)):raise ArithmeticError("nonfinite cheap FIT")
        loss.backward();torch.nn.utils.clip_grad_norm_(old.parameters(),1.)
        opt.step();cpu+=time.process_time()-tic
    old.eval()
    f=out/f"frozen-context-h{h}-s{seed}.pt"
    torch.save({"source":SOURCE,"seed":seed,"domain":domain,"horizon":h,
        "state_dict":old.state_dict(),"optimizer":opt.state_dict(),
        "torch_rng":torch.get_rng_state(),"sampler_rng":rng.get_state(),
        "steps":UPDATES,"parent_step":prec["selected_DEV_parent_step"]},f)
    rec={"horizon":h,"seed":seed,"steps":UPDATES,"real_FIT_exposures":UPDATES*BATCH,
         "optimizer_CPU":cpu,"full_process_CPU":time.process_time()-t0,
         "checkpoint":f.name,"checkpoint_sha256":sha(f),
         "parent_total_CPU":prec["total_CPU_including_DEV"],
         "trained_only_on_real_FIT":True}
    save(out/f"frozen-context-h{h}-s{seed}.json",rec)
    return old,rec


@torch.no_grad()
def evaluate(net,z,mod,h,mode,artifact=None):
    net.eval()
    origins=np.arange(mod.TR,mod.VA-h+1,48,dtype=np.int64)
    ps=[];ys=[];qs=[];sel=[];cheap=[]
    for j in range(0,len(origins),16):
        x,y=mod.samples(z,origins[j:j+16],h)
        p,c,q,chosen=net.routed(x,mode)
        if not torch.isfinite(p).all():raise ArithmeticError("nonfinite original DEV")
        ps.append(p.numpy());ys.append(y.numpy());cheap.append(c.numpy())
        qs.append(q.numpy());sel.append(chosen.numpy())
    p=np.concatenate(ps);y=np.concatenate(ys)
    q=np.concatenate(qs);chosen=np.concatenate(sel)
    c=np.concatenate(cheap)
    d=np.ascontiguousarray(p,dtype=np.float64)-np.ascontiguousarray(y,dtype=np.float64)
    mse=(d*d).mean((0,1));mae=np.abs(d).mean((0,1))
    start=-3 if z.shape[1]==8 else -1
    metrics={"primary_MSE":float(mse[start:].mean()),"primary_MAE":float(mae[start:].mean()),
        "all_channel_MSE":float(mse.mean()),"all_channel_MAE":float(mae.mean()),
        "channel_MSE":mse.tolist(),"channel_MAE":mae.tolist(),
        "forecast_scalars":int(p.size),"windows":len(origins),
        "executed_fraction":float(chosen.mean()),
        "selected_channels":int(chosen.sum()),
        "total_channel_windows":int(chosen.size),
        "mean_router_probability":float(q.mean()),"mode":mode}
    if artifact is not None:
        np.savez_compressed(artifact,pred=p,truth=y,cheap=c,
                            router=q,executed=chosen,origin=origins)
        metrics["artifact"]=artifact.name
        metrics["artifact_sha256"]=sha(artifact)
    return metrics


@torch.no_grad()
def latency(net,z,mod,h,mode):
    net.eval();x,_=mod.samples(z,[mod.TR],h)
    for _ in range(5):net.routed(x,mode)
    times=[]
    for _ in range(5):
        start=time.process_time()
        for _ in range(20):net.routed(x,mode)
        times.append((time.process_time()-start)/20)
    return {"median_full_forecast_CPU_ms":1000*float(np.median(times)),
            "per_repetition_CPU_ms":[1000*float(q) for q in times],
            "includes_context_router_normalization_full_AR":True,
            "batch":1,"horizon":h,"channel_count":z.shape[1],
            "original_first_DEV_window":int(mod.TR),"mode":mode}


def fit(z,mod,domain,h,seed,arm,lane,old,backbone,folder):
    model=make_model(arm,h,seed,period(domain),old)
    assert all(not p.requires_grad for n,p in model.named_parameters()
               if not n.startswith(("branch.","branch_projection.",
                                    "coefficients.","trust.","router.")))
    with torch.no_grad():
        x,_=mod.samples(z,[mod.TR],h)
        assert torch.equal(model.routed(x,"cheap")[0],old(x))
        assert torch.equal(model.routed(x,"always")[0],old(x))
    active=model.active_parameters()
    opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                           lr=.001,weight_decay=1e-4)
    rng=torch.Generator().manual_seed(seed+2000)
    n=UPDATES if lane=="exposure" else MAX_CPU_UPDATES
    trace=[];charged=0.;begin=time.process_time();wall=time.perf_counter()
    for step in range(1,n+1):
        x,y=input_examples(z,mod,h,seed,rng)
        tic=time.process_time();model.train();opt.zero_grad(set_to_none=True)
        loss,expert,router=model.expert_router_loss(x,y)
        if not bool(torch.isfinite(loss)):raise ArithmeticError("nonfinite training FIT")
        loss.backward()
        grad=torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad],1.)
        if not bool(torch.isfinite(grad)):raise ArithmeticError("nonfinite model gradient")
        opt.step();charged+=time.process_time()-tic
        if step==1 or step%100==0 or (lane=="cpu" and charged>=CPU_BUDGET):
            trace.append({"update":step,"real_FIT_loss":float(loss.detach()),
                          "expert_MSE":float(expert.detach()),
                          "router_objective":float(router.detach()),
                          "optimizer_CPU":charged})
        if lane=="cpu" and charged>=CPU_BUDGET:break
    metrics={mode:evaluate(model,z,mod,h,mode,folder/f"DEV-{mode}.npz") for mode in MODES}
    cpu_lat={mode:latency(model,z,mod,h,mode) for mode in MODES}
    cp=folder/"original-checkpoint.pt"
    torch.save({"study":SOURCE,"arm":arm,"seed":seed,"domain":domain,"horizon":h,
       "lane":lane,"updates":step,"active_parameter_count":active,
       "frozen_backbone_sha256":backbone["checkpoint_sha256"],
       "state_dict":model.state_dict(),"optimizer":opt.state_dict(),
       "sampler_rng":rng.get_state(),"torch_rng":torch.get_rng_state(),
       "source_csv_sha256":mod.CSV_SHA},cp)
    save(folder/"trace.json",trace)
    rec={"arm":arm,"seed":seed,"domain":domain,"horizon":h,"lane":lane,
         "steps":step,"real_training_histories":step*BATCH,
         "real_target_scalars_seen":step*BATCH*h*z.shape[1],
         "total_parameter_count":sum(p.numel() for p in model.parameters()),
         "active_trainable_parameters":active,
         "optimizer_CPU":charged,
         "whole_fit_process_CPU":time.process_time()-begin,
         "whole_fit_wall_seconds":time.perf_counter()-wall,
         "peak_process_RSS_KiB":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
         "parent_plus_frozen_pretraining_CPU":backbone["parent_total_CPU"]+
                                               backbone["full_process_CPU"],
         "metrics":metrics,"latency":cpu_lat,
         "checkpoint":cp.name,"checkpoint_sha256":sha(cp),
         "no_new_test_read":True,"threshold":.5,"training_expert_dense":True,
         "conditional_training_speedup_claim":False}
    save(folder/"summary.json",rec)
    print(json.dumps({"event":"ME045_REAL_FIT","domain":domain,"h":h,"seed":seed,
                      "lane":lane,"arm":arm,"steps":step,"MSE":metrics["adaptive"]["primary_MSE"],
                      "native_fraction":metrics["adaptive"]["executed_fraction"],
                      "latency_ms":cpu_lat["adaptive"]["median_full_forecast_CPU_ms"]}),
          flush=True)
    return rec


def summarize(data):
    out={}
    for domain,entry in data.items():
        fits=entry["fits"]
        if len(fits)!=24:raise ValueError("must complete all 24 fits per domain")
        summary={}
        for lane in LANES:
            summary[lane]={}
            for arm in ARMS:
                pairs=[q for q in fits if q["arm"]==arm and q["lane"]==lane]
                assert len(pairs)==4
                per_mode={}
                for mode in MODES:
                    per_mode[mode]={
                      "mean_primary_MSE":float(np.mean([q["metrics"][mode]["primary_MSE"] for q in pairs])),
                      "mean_all_channel_MSE":float(np.mean([q["metrics"][mode]["all_channel_MSE"] for q in pairs])),
                      "mean_CPU_ms":float(np.mean([q["latency"][mode]["median_full_forecast_CPU_ms"] for q in pairs])),
                      "mean_executed_fraction":float(np.mean([q["metrics"][mode]["executed_fraction"] for q in pairs])),
                      "by_horizon":{str(h):float(np.mean([q["metrics"][mode]["primary_MSE"]
                                for q in pairs if q["horizon"]==h])) for h in HORIZONS},
                      "by_seed":{str(seed):float(np.mean([q["metrics"][mode]["primary_MSE"]
                                for q in pairs if q["seed"]==seed])) for seed in SEEDS}}
                summary[lane][arm]={"modes":per_mode,
                   "mean_optimizer_CPU":float(np.mean([q["optimizer_CPU"] for q in pairs])),
                   "mean_training_updates":float(np.mean([q["steps"] for q in pairs])),
                   "active_parameter_counts":sorted(set(q["active_trainable_parameters"] for q in pairs))}
        out[domain]=summary
    e=out["etth1"];t=out["tetouan"]
    def item(data,lane,arm,mode):return data[lane][arm]["modes"][mode]
    en=item(e,"exposure","conditional_native","adaptive")
    tn=item(t,"cpu","conditional_native","adaptive")
    latency=float(np.mean([
        out[d][l]["conditional_native"]["modes"]["adaptive"]["mean_CPU_ms"]
        for d in ("etth1","tetouan") for l in LANES]))
    accuracy_by_domain={}
    for domain,entry in out.items():
        scores={}
        for lane in LANES:
            native=item(entry,lane,"conditional_native","adaptive")["mean_primary_MSE"]
            gru=item(entry,lane,"conditional_gru40","adaptive")["mean_primary_MSE"]
            cheap=item(entry,lane,"conditional_native","cheap")["mean_primary_MSE"]
            scores[lane]={"native_MSE":native,"matched_gru40_MSE":gru,
                          "frozen_context_MSE":cheap,
                          "native_better_than_both":native<gru and native<cheap}
        accuracy_by_domain[domain]=scores
    gates={
      "ETTh1_720_exposure_MSE_below_0p190500":
           en["by_horizon"]["720"]<.1905,
      "Tetouan_CPU_MSE_below_0p101500":tn["mean_primary_MSE"]<.1015,
      "native_adaptive_full_forecast_latency_below_0p45ms":latency<.45,
      "native_fraction_nontrivial":0<en["mean_executed_fraction"]<1,
      "ETTh1_cross_lane_beats_matched_GRU40_and_frozen_context":
           all(accuracy_by_domain["etth1"][l]["native_better_than_both"] for l in LANES),
      "Tetouan_cross_lane_beats_matched_GRU40_and_frozen_context":
           all(accuracy_by_domain["tetouan"][l]["native_better_than_both"] for l in LANES)}
    gates["all_registered_dev_gates_pass"]=all(gates.values())
    return {"domains":out,"gates":gates,"comparison":accuracy_by_domain,
            "average_native_batch_one_latency_ms":latency,
            "registered_threshold":.5,
            "all_DEVs_already_exposed":True,
            "sealed_cross_hardware_architecture_stop_rule_not_tested":True,
            "no_promotion_or_merge":True}


def run(path):
    path=Path(path);path.mkdir(parents=True,exist_ok=False)
    output={"study":SOURCE,"source_commit":subprocess.check_output(
         ["git","rev-parse","HEAD"],text=True).strip(),"complete":False,
        "original_avx512_cross_isa_exception_retained":True,
        "nonindependent_dev":True,"new_test_outcomes_opened":0,
        "all_three_way_resource_parity_qualified":False,
        "domains":{},"environment":{"python":platform.python_version(),
        "torch":torch.__version__,"numpy":np.__version__,
        "processor":platform.processor(),"cpu_capability":torch.backends.cpu.get_cpu_capability()}}
    save(path/"results.json",output)
    for domain,mod in DOMAINS.items():
        destination=path/domain;destination.mkdir()
        z,provenance=mod.get_source(destination)
        data={"provenance":provenance,"parents":[],"frozen_contexts":[],"fits":[]}
        output["domains"][domain]=data
        for h in HORIZONS:
            for seed in SEEDS:
                parent,parent_rec=mod.fit_parent(z,h,seed,destination)
                data["parents"].append(parent_rec)
                original,backbone=pretrain_context(z,mod,domain,h,seed,parent,parent_rec,destination)
                data["frozen_contexts"].append(backbone)
                for lane in LANES:
                    for arm in ARMS:
                        f=destination/f"{lane}-h{h}-s{seed}-{arm}";f.mkdir()
                        data["fits"].append(fit(z,mod,domain,h,seed,arm,lane,original,backbone,f))
                        save(path/"results.json",output)
    output["selection"]=summarize(output["domains"])
    output["complete"]=True
    output["new_test_outcomes_opened"]=0
    save(path/"results.json",output)
    print(json.dumps({"event":"ME045_ALL_48_AUTHENTIC_FITS_FINISHED",
                      "gates":output["selection"]["gates"],
                      "fresh_holdout":False}),flush=True)


@torch.no_grad()
def replay(path):
    path=Path(path)
    report=json.loads((path/"results.json").read_text())
    assert report["complete"] and report["study"]==SOURCE
    assert report["new_test_outcomes_opened"]==0
    total=0;models=0
    for domain,data in report["domains"].items():
        mod=DOMAINS[domain];dst=path/domain
        z,_=mod.get_source(dst)
        for row in data["fits"]:
            folder=dst/f'{row["lane"]}-h{row["horizon"]}-s{row["seed"]}-{row["arm"]}'
            checkpoint=folder/row["checkpoint"]
            assert sha(checkpoint)==row["checkpoint_sha256"]
            ck=torch.load(checkpoint,weights_only=True,map_location="cpu")
            assert ck["frozen_backbone_sha256"]==next(
                r["checkpoint_sha256"] for r in data["frozen_contexts"]
                if r["horizon"]==row["horizon"] and r["seed"]==row["seed"])
            old=prior_model("original_context",row["horizon"],row["seed"],period(domain))
            old.load_state_dict(torch.load(dst/next(
                r["checkpoint"] for r in data["frozen_contexts"]
                if r["horizon"]==row["horizon"] and r["seed"]==row["seed"]),
                weights_only=True)["state_dict"],strict=True)
            net=make_model(row["arm"],row["horizon"],row["seed"],period(domain),old)
            net.load_state_dict(ck["state_dict"],strict=True)
            assert net.active_parameters()==row["active_trainable_parameters"]
            ids=[j for g in ck["optimizer"]["param_groups"] for j in g["params"]]
            assert len(ids)==len(ck["optimizer"]["state"]) and all(
                int(ck["optimizer"]["state"][k]["step"])==row["steps"] for k in ids)
            for mode in MODES:
                dest=folder/f"DEV-{mode}.npz"
                assert sha(dest)==row["metrics"][mode]["artifact_sha256"]
                with np.load(dest) as saved:
                    origin=saved["origin"]
                    pp=[];yy=[];qq=[];cc=[];flags=[]
                    for j in range(0,len(origin),16):
                        x,y=mod.samples(z,origin[j:j+16],row["horizon"])
                        pred,cheap,gate,used=net.routed(x,mode)
                        pp.append(pred.numpy());yy.append(y.numpy())
                        qq.append(gate.numpy());cc.append(cheap.numpy())
                        flags.append(used.numpy())
                    for name,values in [("pred",pp),("truth",yy),("router",qq),
                                         ("cheap",cc),("executed",flags)]:
                        if not np.array_equal(np.concatenate(values),saved[name]):
                            raise AssertionError(f"replay mismatch {row['arm']} {mode} {name}")
                    total+=saved["pred"].size
            models+=1
    print(json.dumps({"status":"ME045_PINNED_PROFILE_ALL_48_CHECKPOINT_REPLAY_PASS",
                      "original_model_checkpoints":models,
                      "actual_forecast_scalars_replayed_in_three_modes":total,
                      "new_final_test":False}),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--out",default="results/me045")
    parser.add_argument("--replay",action="store_true")
    args=parser.parse_args()
    if args.replay:replay(args.out)
    else:run(args.out)
