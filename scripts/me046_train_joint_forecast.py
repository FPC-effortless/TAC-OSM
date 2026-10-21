#!/usr/bin/env python3
"""ME046 genuine FIT-trained six-model multiscale architecture tournament.

Original publisher ETTh1 and Tetouan measurements only. Historical already-
exposed DEV outcomes scored; all original final-test outcome values unopened.
No synthetic evaluation, artificial trajectories, agent/planner claims.
"""
from __future__ import annotations
import argparse,hashlib,json,platform,resource,subprocess,time
from pathlib import Path
import numpy as np
import torch
from scripts.me043_train_adaptive import DOMAINS,save,sha
from scripts.me045_train_conditional import pretrain_context,period
from tac_osm.me046_joint_multiscale import ARMS,LOOP_ARMS,NATIVE_ARMS,build_model,objective,clip_trainable,preflight

SEEDS=(46001,46002)
HORIZONS=(96,720)
LANES=("exposure","cpu")
UPDATES=800
CPU_SECONDS=6.
CPU_LIMIT=10000
BATCH=4
STUDY="ME046_REAL_JOINT_MULTISCALE_TIED_REFINEMENT"
torch.set_num_threads(1)


@torch.no_grad()
def evaluate(net,z,mod,h,path=None):
    net.eval()
    origins=np.arange(mod.TR,mod.VA-h+1,48,dtype=np.int64)
    predictions=[];truths=[];cheap=[]
    for j in range(0,len(origins),16):
        x,y=mod.samples(z,origins[j:j+16],h)
        baseline,p,*_=net.full_parts(x)
        if not torch.isfinite(p).all():raise ValueError("nonfinite real forecast")
        predictions.append(p.detach().cpu().numpy())
        truths.append(y.detach().cpu().numpy())
        cheap.append(baseline.detach().cpu().numpy())
    p=np.concatenate(predictions);y=np.concatenate(truths);c=np.concatenate(cheap)
    d=np.ascontiguousarray(p,dtype=np.float64)-np.ascontiguousarray(y,dtype=np.float64)
    d0=np.ascontiguousarray(c,dtype=np.float64)-np.ascontiguousarray(y,dtype=np.float64)
    mse=(d*d).mean((0,1));mae=np.abs(d).mean((0,1))
    low=-3 if z.size(1)==8 else -1
    metric={"primary_MSE":float(mse[low:].mean()),
        "primary_MAE":float(mae[low:].mean()),
        "all_channel_MSE":float(mse.mean()),
        "all_channel_MAE":float(mae.mean()),
        "cheap_primary_MSE":float((d0*d0).mean((0,1))[low:].mean()),
        "channel_MSE":mse.tolist(),"channel_MAE":mae.tolist(),
        "n_real_windows":int(len(origins)),"forecast_scalar_targets":int(p.size)}
    if path is not None:
        np.savez_compressed(path,pred=p,truth=y,cheap=c,origins=origins)
        metric["original_predictions_sha256"]=sha(path)
    return metric


@torch.no_grad()
def latency(net,z,mod,h):
    net.eval();x,_=mod.samples(z,[mod.TR],h)
    for _ in range(5):net(x)
    samples=[]
    for _ in range(5):
        t0=time.process_time()
        for _ in range(20):net(x)
        samples.append((time.process_time()-t0)*1000/20)
    return {"median_full_forecast_CPU_ms":float(np.median(samples)),
            "repetitions_CPU_ms":samples,
            "batch":1,"channels":z.size(1),"horizon":h,
            "complete_normalization_AR_context_and_residual":True}


def fit(z,mod,domain,h,seed,arm,lane,cheap,context_rec,out):
    folder=out/f"{lane}-h{h}-s{seed}-{arm}"
    folder.mkdir(parents=True,exist_ok=False)
    net=build_model(arm,h,seed,period(domain),cheap)
    net.eval();cheap.eval()
    with torch.no_grad():
        probe,_=mod.samples(z,[mod.TR],h)
        if not torch.equal(net(probe),cheap(probe)):
            raise ArithmeticError("common authentic context not exactly preserved")
    opt=torch.optim.AdamW([
        {"params":list(net.ar.parameters()),"lr":.0001},
        {"params":[p for n,p in net.named_parameters() if not n.startswith("ar.")],
         "lr":.001}],weight_decay=1e-4)
    rng=torch.Generator().manual_seed(seed+2000)
    maxsteps=UPDATES if lane=="exposure" else CPU_LIMIT
    trace=[];cpu=0.;wall=time.perf_counter();total_start=time.process_time()
    for step in range(1,maxsteps+1):
        anchors=torch.randint(96,mod.TR-h+1,(BATCH,),generator=rng)
        x,y=mod.samples(z,anchors,h)
        t=time.process_time()
        net.train();opt.zero_grad(set_to_none=True)
        loss,lf,lc=objective(net,x,y)
        if not bool(torch.isfinite(loss)):raise ArithmeticError("nonfinite genuine FIT objective")
        loss.backward()
        grad=clip_trainable(net)
        if not bool(torch.isfinite(grad)):raise ArithmeticError("nonfinite genuine FIT gradient")
        opt.step()
        cpu+=time.process_time()-t
        if step==1 or step%100==0 or lane=="cpu" and cpu>=CPU_SECONDS:
            trace.append({"step":step,"FIT_full_MSE":float(lf.detach()),
                          "FIT_cheap_MSE":float(lc.detach()),
                          "FIT_total_loss":float(loss.detach()),
                          "optimizer_CPU_seconds":cpu})
        if lane=="cpu" and cpu>=CPU_SECONDS:break
    fit_cpu=time.process_time()-total_start
    fit_wall=time.perf_counter()-wall
    metric=evaluate(net,z,mod,h,folder/"DEV-original.npz")
    inference=latency(net,z,mod,h)
    cp=folder/"model.pt"
    torch.save({"study":STUDY,"arm":arm,"horizon":h,"seed":seed,"domain":domain,
         "lane":lane,"updates":step,"state_dict":net.state_dict(),
         "optimizer":opt.state_dict(),"sampler_rng":rng.get_state(),
         "torch_rng":torch.get_rng_state(),
         "common_context_sha256":context_rec["checkpoint_sha256"],
         "source_original_csv_sha256":mod.CSV_SHA,
         "active_parameters":sum(p.numel() for p in net.parameters())},cp)
    save(folder/"train-trace.json",trace)
    row={"arm":arm,"horizon":h,"seed":seed,"domain":domain,"lane":lane,
         "real_FIT_updates":step,"real_FIT_histories_seen":step*BATCH,
         "real_FIT_scalar_targets_seen":step*BATCH*h*z.size(1),
         "active_parameters":sum(p.numel() for p in net.parameters()),
         "optimizer_CPU_seconds":cpu,"fit_process_CPU_seconds":fit_cpu,
         "fit_wall_seconds":fit_wall,
         "shared_AR_plus_context_pretraining_CPU_seconds":
             context_rec["parent_total_CPU"]+context_rec["full_process_CPU"],
         "max_CPU_steps_reached":lane=="cpu" and step==CPU_LIMIT and cpu<CPU_SECONDS,
         "peak_process_RSS_KiB":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
         "metric":metric,"batch_one_latency":inference,
         "saved_checkpoint":str(cp.relative_to(out)),
         "checkpoint_sha256":sha(cp),
         "forecast_path":str((folder/"DEV-original.npz").relative_to(out)),
         "frozen_cheap_context":False,"original_test_numeric_rows_read":0}
    save(folder/"summary.json",row)
    print(json.dumps({"event":"ME046_FIT","domain":domain,"h":h,"seed":seed,
          "arm":arm,"lane":lane,"MSE":round(metric["primary_MSE"],6),
          "batch1_ms":round(inference["median_full_forecast_CPU_ms"],4),
          "updates":step,"CPU":round(cpu,3)}),flush=True)
    return row


def summarize(domains):
    out={}
    for domain,data in domains.items():
        bylane={}
        for lane in LANES:
            arms={}
            for arm in ARMS:
                rows=[r for r in data["fits"] if r["lane"]==lane and r["arm"]==arm]
                if len(rows)!=4:raise ValueError("incomplete domain-lane-arm training grid")
                arms[arm]={
                    "primary_MSE":float(np.mean([r["metric"]["primary_MSE"] for r in rows])),
                    "all_channel_MSE":float(np.mean([r["metric"]["all_channel_MSE"] for r in rows])),
                    "median_batch1_CPU_ms":float(np.median([r["batch_one_latency"]["median_full_forecast_CPU_ms"] for r in rows])),
                    "mean_optimization_CPU":float(np.mean([r["optimizer_CPU_seconds"] for r in rows])),
                    "mean_updates":float(np.mean([r["real_FIT_updates"] for r in rows])),
                    "parameter_counts":sorted(set(r["active_parameters"] for r in rows)),
                    "by_horizon":{str(h):float(np.mean([
                         r["metric"]["primary_MSE"] for r in rows if r["horizon"]==h]))
                         for h in HORIZONS},
                    "by_seed":{str(s):float(np.mean([
                         r["metric"]["primary_MSE"] for r in rows if r["seed"]==s]))
                         for s in SEEDS}}
            ordered=sorted(ARMS,key=lambda k:(arms[k]["primary_MSE"],k))
            pareto=[a for a in ARMS if not any(
                arms[b]["primary_MSE"]<=arms[a]["primary_MSE"]
                and arms[b]["median_batch1_CPU_ms"]<=arms[a]["median_batch1_CPU_ms"]
                and (arms[b]["primary_MSE"]<arms[a]["primary_MSE"]
                     or arms[b]["median_batch1_CPU_ms"]<arms[a]["median_batch1_CPU_ms"])
                for b in ARMS if a!=b)]
            bylane[lane]={"arms":arms,"ranking":ordered,
                            "inference_accuracy_latency_pareto":pareto}
        out[domain]=bylane
    et=out["etth1"];te=out["tetouan"]
    candidates=NATIVE_ARMS
    g={
        "ETTh1_H720_any_native_below_0p190500":
             any(et["exposure"]["arms"][a]["by_horizon"]["720"]<.1905 for a in candidates),
        "Tetouan_fixed6s_any_native_below_0p101500":
             any(te["cpu"]["arms"][a]["primary_MSE"]<.1015 for a in candidates),
        "native_better_than_context_and_GRU40_both_domains_equal_updates":
             any(all(out[d]["exposure"]["arms"][a]["primary_MSE"]<
                     min(out[d]["exposure"]["arms"][b]["primary_MSE"]
                     for b in ("joint_context","joint_gru40"))
                     for d in ("etth1","tetouan")) for a in candidates),
        "native_full_CPU_latency_below_0p45ms":
             any(all(out[d]["exposure"]["arms"][a]["median_batch1_CPU_ms"]<.45
                     for d in ("etth1","tetouan")) for a in candidates)}
    g["all_four_engineering_gates_pass"]=all(g.values())
    return {"by_domain":out,"registered_engineering_gates":g,
            "all_evidence_from_reused_development":True,
            "generalization_and_resource_parity_not_qualified":True,
            "no_model_promoted_or_merged":True,
            "sealed_new_task_native_stop_not_tested":True}


def run(destination):
    out=Path(destination);out.mkdir(parents=True,exist_ok=False)
    report={"study":STUDY,
            "source_commit":subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),
            "completed":False,"domains":{},
            "training_not_synthetic":True,
            "original_AVX512_cross_ISA_exactness_failure_retained":True,
            "untouched_final_test_numeric_rows_read":0,
            "experimental_context_datasets_not_fresh":True,
            "environment":{"torch":torch.__version__,"numpy":np.__version__,
               "python":platform.python_version(),"cpu_profile":torch.backends.cpu.get_cpu_capability(),
               "threads":torch.get_num_threads()}}
    save(out/"results.json",report)
    for domain,mod in DOMAINS.items():
        dest=out/domain;dest.mkdir()
        z,provenance=mod.get_source(dest)
        domain_record={"provenance":provenance,"parents":[],"common_contexts":[],"fits":[]}
        report["domains"][domain]=domain_record
        for h in HORIZONS:
            for seed in SEEDS:
                parent,p_rec=mod.fit_parent(z,h,seed,dest)
                domain_record["parents"].append(p_rec)
                old,c_rec=pretrain_context(z,mod,domain,h,seed,parent,p_rec,dest)
                domain_record["common_contexts"].append(c_rec)
                for lane in LANES:
                    for arm in ARMS:
                        row=fit(z,mod,domain,h,seed,arm,lane,old,c_rec,dest)
                        domain_record["fits"].append(row)
                        save(out/"results.json",report)
    report["selection"]=summarize(report["domains"])
    report["completed"]=True
    save(out/"results.json",report)
    print(json.dumps({"event":"ME046_COMPLETE","models":96,
       "gates":report["selection"]["registered_engineering_gates"],
       "held_out_test_opened":False},sort_keys=True),flush=True)


@torch.no_grad()
def replay(destination):
    out=Path(destination)
    report=json.loads((out/"results.json").read_text())
    assert report["completed"] and report["study"]==STUDY
    checked=0;scalars=0
    for domain,data in report["domains"].items():
        z,_=DOMAINS[domain].get_source(out/domain);mod=DOMAINS[domain]
        for row in data["fits"]:
            cp=out/domain/row["saved_checkpoint"]
            assert sha(cp)==row["checkpoint_sha256"]
            original=torch.load(cp,weights_only=True,map_location="cpu")
            assert original["arm"]==row["arm"] and original["updates"]==row["real_FIT_updates"]
            net=build_model(row["arm"],row["horizon"],row["seed"],period(domain))
            net.load_state_dict(original["state_dict"],strict=True)
            assert sum(p.numel() for p in net.parameters())==row["active_parameters"]
            ids=[k for group in original["optimizer"]["param_groups"] for k in group["params"]]
            assert len(ids)==len(original["optimizer"]["state"])
            assert all(int(original["optimizer"]["state"][i]["step"])==row["real_FIT_updates"] for i in ids)
            pp=out/domain/row["forecast_path"]
            assert sha(pp)==row["metric"]["original_predictions_sha256"]
            with np.load(pp) as saved:
                origin=saved["origins"];preds=[];truths=[];cheap=[]
                for j in range(0,len(origin),16):
                    x,y=mod.samples(z,origin[j:j+16],row["horizon"])
                    c,p,*_=net.full_parts(x)
                    preds.append(p.numpy());truths.append(y.numpy());cheap.append(c.numpy())
                for tag,got in (("pred",preds),("truth",truths),("cheap",cheap)):
                    if not np.array_equal(np.concatenate(got),saved[tag]):
                        raise ArithmeticError(f"ME046 original replay FAILED: {row['arm']} {tag}")
                scalars+=saved["pred"].size
            checked+=1
    print(json.dumps({"status":"ME046_PINNED_96_AUTHENTIC_MODEL_REPLAY_PASS",
                      "all_96_checkpoints_and_optimizer_states_verified":checked,
                      "real_forecast_scalars_bit_exact":scalars,
                      "new_independent_outcomes_read":0}),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--out",default="results/me046")
    p.add_argument("--self-test",action="store_true")
    p.add_argument("--replay",action="store_true")
    a=p.parse_args()
    if a.self_test:
        print(json.dumps(preflight(),sort_keys=True),flush=True)
    elif a.replay:
        replay(a.out)
    else:
        run(a.out)
