#!/usr/bin/env python3
"""ME040 genuinely observed Tetouan history: dual-rate coupled TAC against ME038.

No final held-out period read as numeric observations; the original final20%
is ALREADY exposed to prior ME038 research, NEVER described as new independent.
All fit/validation targets are real, no synthetic performance evaluation.
"""
from __future__ import annotations
import argparse,csv,datetime,hashlib,io,json,math,platform,resource,time,urllib.request,zipfile
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from tac_osm.me036_patch_forecast import build_model as original_ar
from tac_osm.me040_dualrate_innovation import build_model, preflight

ZIP_URL="https://archive.ics.uci.edu/static/public/849/power+consumption+of+tetouan+city.zip"
ZIP_SHA="3c4bf684161180937043a9fb65701a83d44b740b0d42f0492b6e7aec57ddbbfe"
CSV_MEMBER="Tetuan City power consumption.csv"
CSV_SHA="ca8b2f30cf07dd872cb7f2aa45dac110ee38dbe5fc5910a6ff4bcc530dae3a40"
N=52416;TR=36691;VA=41933
SEEDS=(40001,40002);HORIZONS=(96,720)
ARMS=("innovation_tac","paired_innovation_tac","context_cross")
LANES=("exposure","cpu")
SOURCE_URLS=(ZIP_URL,)
UPDATES=250; PARENT_UPDATES=500; BATCH=4
CPU_LANE_SECONDS=5.;CPU_LANE_MAX=4000
STUDY="ME040_GENUINE_TETOUAN_PAIRRATE_COUPLED_NATIVE"
GATES={"min_speed":1.35,"max_power_loss_rel":1.02,"max_horizon_loss_rel":1.05}
torch.set_num_threads(1)


def digest(b):return hashlib.sha256(b).hexdigest()
def sha(p):return digest(Path(p).read_bytes())
def save(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+".tmp")
    tmp.write_text(json.dumps(obj,sort_keys=True,indent=2)+"\n")
    tmp.replace(path)


def get_source(out):
    p=out/"source"/"tetouan-original.zip"
    p.parent.mkdir(parents=True,exist_ok=True)
    if not p.exists():
        with urllib.request.urlopen(urllib.request.Request(
             ZIP_URL,headers={"User-Agent":"tac-osm-research-real-source/1.0"}),timeout=120) as response:
            contents=response.read()
        if digest(contents)!=ZIP_SHA:raise RuntimeError("official source ZIP SHA mismatch")
        p.write_bytes(contents)
    blob=p.read_bytes()
    if digest(blob)!=ZIP_SHA:raise RuntimeError("official source ZIP changed")
    with zipfile.ZipFile(io.BytesIO(blob)) as zf:
        members=zf.namelist()
        matches=[v for v in members if v.endswith(CSV_MEMBER)]
        if len(matches)!=1:raise ValueError("cannot identify original authentic CSV")
        source=zf.read(matches[0])
    if digest(source)!=CSV_SHA:raise RuntimeError("official authentic CSV SHA mismatch")
    rows=list(csv.DictReader(io.StringIO(source.decode("utf-8-sig"))))
    if len(rows)!=N:raise ValueError("Tetouan source count changed")
    columns=list(rows[0])
    if len(columns)!=9 or columns[0]!="DateTime":
        raise ValueError("original Tetouan schema changed")
    dts=[datetime.datetime.strptime(r["DateTime"],"%m/%d/%Y %H:%M") for r in rows]
    if any((b-a).total_seconds()!=600 for a,b in zip(dts,dts[1:])):
        raise ValueError("real temporal cadence is not uniform 10 minutes")
    # Numeric values from original test [41933,52416) are intentionally NEVER parsed.
    data=np.asarray([[float(r[c]) for c in columns[1:]] for r in rows[:VA]],np.float32)
    if data.shape!=(VA,8) or not np.isfinite(data).all():
        raise ValueError("nonfinite/split error before final20")
    mean=data[:TR].mean(0,dtype=np.float64).astype(np.float32)
    std=np.maximum(data[:TR].std(0,dtype=np.float64),1e-5).astype(np.float32)
    z=torch.from_numpy(((data-mean)/std).copy())
    info={"official_zip_sha256":ZIP_SHA,"official_csv_sha256":CSV_SHA,
          "rows_in_original_source":N,"source_timestamp_interval_seconds":600,
          "numeric_source_rows_parsed":VA,"numeric_future_rows_parsed":0,
          "train_numeric_rows":TR,"historical_DEV_end_row":VA,
          "previous_ME038_final20_was_already_exposed":True,
          "data_columns":columns[1:],
          "mean_training_only":mean.tolist(),"std_training_only":std.tolist(),
          "publisher_timezone":"unspecified",
          "new_independent_test_scored":False}
    save(out/"source-provenance.json",info)
    return z,info


def samples(z,anchors,h):
    idx=torch.as_tensor(anchors,dtype=torch.long)
    hist=z[idx[:,None]+torch.arange(-96,0)]
    target=z[idx[:,None]+torch.arange(h)]
    return hist,target


@torch.no_grad()
def forecast(net,z,h,save_predictions=None):
    net.eval()
    ix=np.arange(TR,VA-h+1,48,dtype=np.int64)
    allpred=[];alltrue=[];ss=np.zeros(8,dtype=np.float64)
    aa=np.zeros(8,dtype=np.float64);count=0
    t0=time.process_time()
    for j in range(0,len(ix),16):
        x,y=samples(z,ix[j:j+16],h)
        p=net(x)
        if not bool(torch.isfinite(p).all()):raise ArithmeticError("nonfinite real forecast")
        diff=(p-y).to(torch.float64)
        ss+=diff.square().sum((0,1)).numpy()
        aa+=diff.abs().sum((0,1)).numpy()
        count+=p.size(0)*h
        if save_predictions is not None:
            allpred.append(p.cpu().numpy());alltrue.append(y.cpu().numpy())
    mse=ss/count;mae=aa/count
    report={"full8_MSE":float(np.mean(mse)),"power3_MSE":float(np.mean(mse[-3:])),
            "full8_MAE":float(np.mean(mae)),"power3_MAE":float(np.mean(mae[-3:])),
            "channel_MSE":mse.tolist(),"channel_MAE":mae.tolist(),
            "windows":len(ix),"forecast_horizon":h,"forecast_scalar_targets":count*8,
            "DEV_origin_stride":48,"scoring_CPU_seconds":time.process_time()-t0}
    if save_predictions is not None:
        assert sum(p.shape[0] for p in allpred)==len(ix)
        np.savez_compressed(save_predictions,
             pred=np.concatenate(allpred),truth=np.concatenate(alltrue),
             origin=ix)
        report["saved_DEV_predictions"]=str(save_predictions.name)
        report["saved_DEV_predictions_sha256"]=sha(save_predictions)
    return report


def fit_parent(z,h,seed,out):
    path=out/f"parent-h{h}-s{seed}";path.mkdir(parents=True,exist_ok=True)
    torch.manual_seed(seed)
    parent=original_ar("ar",h,seed)
    opt=torch.optim.AdamW(parent.parameters(),lr=.001,weight_decay=1e-4)
    generator=torch.Generator().manual_seed(seed+1000)
    trace=[];ends={};cpu=0.;t0=time.process_time()
    for step in range(1,PARENT_UPDATES+1):
        anchors=torch.randint(96,TR-h+1,(BATCH,),generator=generator)
        x,y=samples(z,anchors,h)
        t=time.process_time()
        parent.train();opt.zero_grad(set_to_none=True)
        loss=(parent(x)-y).square().mean()
        if not bool(torch.isfinite(loss)):raise ArithmeticError("nonfinite FIT parent MSE")
        loss.backward()
        grad=torch.nn.utils.clip_grad_norm_(parent.parameters(),1.)
        if not bool(torch.isfinite(grad)):raise ArithmeticError("nonfinite FIT parent gradient")
        opt.step();cpu+=time.process_time()-t
        if step in (100,250,500):
            metric=forecast(parent,z,h)
            cp=path/f"parent-u{step}.pt"
            torch.save({"state_dict":parent.state_dict(),"optimizer":opt.state_dict(),
                        "rng":torch.get_rng_state(),"sampler_rng":generator.get_state(),
                        "seed":seed,"horizon":h,"step":step,"FIT_only":True},cp)
            ends[str(step)]={"MSE":metric["full8_MSE"],
                "DEV_power_MSE":metric["power3_MSE"],
                "checkpoint":str(cp.relative_to(out)),
                "checkpoint_sha256":sha(cp)}
        trace.append({"step":step,"real_train_MSE":float(loss.detach()),"optimizer_CPU_seconds":cpu})
    chosen=min(ends,key=lambda k:(ends[k]["MSE"],int(k)))
    state=torch.load(out/ends[chosen]["checkpoint"],weights_only=True)["state_dict"]
    receipt={"horizon":h,"seed":seed,"optimizer_steps":PARENT_UPDATES,
             "training_FIT_histories":PARENT_UPDATES*BATCH,
             "optimizer_CPU_seconds":cpu,"total_CPU_including_DEV":time.process_time()-t0,
             "selected_DEV_parent_step":int(chosen),"selection_candidates":[100,250,500],
             "endpoints":ends,"trace_file":f"{path.relative_to(out)}/trace.json",
             "original_ar_state_sha256":ends[chosen]["checkpoint_sha256"],
             "original_final20_scored":False}
    save(path/"trace.json",trace);save(path/"summary.json",receipt)
    return state,receipt


def fit_arm(z,h,seed,arm,lane,parent_state,parent_receipt,out):
    folder=out/f"{lane}-h{h}-s{seed}-{arm}"
    folder.mkdir(parents=True,exist_ok=True)
    net=build_model(arm,h,seed)
    net.initialize_ar(parent_state)
    parent=original_ar("ar",h,seed)
    parent.load_state_dict(parent_state,strict=True)
    probe,_=samples(z,[TR,TR+48],h)
    net.eval();parent.eval()
    with torch.no_grad():
        if not torch.equal(net(probe),parent(probe)):
            raise AssertionError("same source parent AR exact initial forecast failed")
    params=sum(p.numel() for p in net.parameters())
    if arm=="paired_innovation_tac":
        original=build_model("innovation_tac",h,seed)
        original.initialize_ar(parent_state)
        for key,val in original.state_dict().items():
            if not torch.equal(val,net.state_dict()[key]):
                raise AssertionError("original vs paired native initial weights altered: "+key)
    torch.manual_seed(seed+40040)
    gen=torch.Generator().manual_seed(seed+2000)
    base=list(net.ar.parameters())
    extra=[p for n,p in net.named_parameters() if not n.startswith("ar.")]
    opt=torch.optim.AdamW([{"params":base,"lr":.0001},
                           {"params":extra,"lr":.001}],weight_decay=1e-4)
    trace=[];cpu=0.;start=time.process_time();step=0
    cap=UPDATES if lane=="exposure" else CPU_LANE_MAX
    for step in range(1,cap+1):
        anchors=torch.randint(96,TR-h+1,(BATCH,),generator=gen)
        x,y=samples(z,anchors,h)
        t=time.process_time()
        net.train();opt.zero_grad(set_to_none=True)
        pred,delta,_=net.parts(x)
        mse=(pred-y).square().mean()
        loss=mse+10*net.anchor_loss()+.01*delta.square().mean()
        if not bool(torch.isfinite(loss)):raise ArithmeticError("invalid genuine downstream objective")
        loss.backward()
        grad=torch.nn.utils.clip_grad_norm_(net.parameters(),1.)
        if not bool(torch.isfinite(grad)):raise ArithmeticError("invalid gradient")
        opt.step()
        cpu+=time.process_time()-t
        trace.append({"step":step,"fit_MSE":float(mse.detach()),
                      "fit_loss":float(loss.detach()),"CPU_seconds":cpu})
        if lane=="exposure" and step in (100,UPDATES):
            metric=forecast(net,z,h)
            save(folder/f"u{step}-DEV.json",metric)
        if lane=="cpu" and cpu>=CPU_LANE_SECONDS:break
    if lane=="cpu" and step==CPU_LANE_MAX and cpu<CPU_LANE_SECONDS:
        limit_hit=True
    else:limit_hit=False
    cp=folder/"final.pt";pred_path=folder/"DEV-predictions.npz"
    metric=forecast(net,z,h,pred_path)
    torch.save({"step":step,"horizon":h,"seed":seed,"arm":arm,"lane":lane,
                "state_dict":net.state_dict(),"optimizer":opt.state_dict(),
                "torch_rng":torch.get_rng_state(),"sampler_rng":gen.get_state(),
                "parent_selected_step":parent_receipt["selected_DEV_parent_step"],
                "original_zip_sha256":ZIP_SHA,
                "parameter_count":params},cp)
    save(folder/"train-trace.json",trace)
    row={"horizon":h,"seed":seed,"arm":arm,"lane":lane,"training_steps":step,
         "real_FIT_histories_presented":step*BATCH,
         "real_future_scalar_targets_presented":step*BATCH*h*8,
         "active_weights":params,"optimizer_CPU_seconds":cpu,
         "training_plus_validation_CPU_seconds":time.process_time()-start,
         "parent_charged_CPU_seconds":parent_receipt["total_CPU_including_DEV"],
         "peak_process_RSS_KiB":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
         "max_CPU_updates_reached_before_budget":limit_hit,
         "metric":metric,"checkpoint":str(cp.relative_to(out)),
         "checkpoint_sha256":sha(cp),
         "trace":str((folder/"train-trace.json").relative_to(out)),
         "original_final20_scored":False}
    save(folder/"summary.json",row)
    print(json.dumps({"status":"ME040_FIT","h":h,"seed":seed,
                      "arm":arm,"lane":lane,"steps":step,
                      "power_MSE":round(metric["power3_MSE"],6),
                      "optimizer_CPU":round(cpu,3)}),flush=True)
    return row


def summarize(rows):
    groups={(h,l,a):[r for r in rows if r["horizon"]==h and r["lane"]==l and r["arm"]==a]
            for h in HORIZONS for l in LANES for a in ARMS}
    if any(len(v)!=2 for v in groups.values()):raise ValueError("all three real arms/two seeds needed")
    dev={f"{h}-{l}-{a}":float(np.mean([r["metric"]["power3_MSE"] for r in groups[(h,l,a)]]))
         for h in HORIZONS for l in LANES for a in ARMS}
    mean=lambda lane,arm:float(np.mean([r["metric"]["power3_MSE"] for r in rows
                                        if r["lane"]==lane and r["arm"]==arm]))
    speeds=[next(r["optimizer_CPU_seconds"] for r in rows
                 if r["horizon"]==h and r["lane"]=="exposure" and r["seed"]==s and
                    r["arm"]=="innovation_tac")/
            next(r["optimizer_CPU_seconds"] for r in rows
                 if r["horizon"]==h and r["lane"]=="exposure" and r["seed"]==s and
                    r["arm"]=="paired_innovation_tac")
            for h in HORIZONS for s in SEEDS]
    geo=float(np.exp(np.mean(np.log(speeds))))
    paired=mean("exposure","paired_innovation_tac")
    original=mean("exposure","innovation_tac")
    context=mean("exposure","context_cross")
    horizon_ok=all(dev[f"{h}-exposure-paired_innovation_tac"] <=
        GATES["max_horizon_loss_rel"]*dev[f"{h}-exposure-innovation_tac"] for h in HORIZONS)
    speed_ok=geo>=GATES["min_speed"]
    acc_ok=(paired<=GATES["max_power_loss_rel"]*original and
            paired<=GATES["max_power_loss_rel"]*context and horizon_ok)
    cpu_original=mean("cpu","innovation_tac")
    cpu_paired=mean("cpu","paired_innovation_tac")
    cpu_context=mean("cpu","context_cross")
    cpu_ok=cpu_paired<=cpu_original and cpu_paired<=cpu_context
    return {"group_power_MSE":dev,"mean_exposure_power_MSE":{
            "paired":paired,"original":original,"context":context},
            "mean_CPU_lane_power_MSE":{
            "paired":cpu_paired,"original":cpu_original,"context":cpu_context},
            "paired_speedup_by_H_seed":speeds,
            "paired_native_geometric_training_speedup":geo,
            "speed_gate":speed_ok,"forecast_accuracy_gate":acc_ok,
            "CPU_budget_competitiveness_gate":cpu_ok,
            "all_three_development_gates_pass":speed_ok and acc_ok and cpu_ok,
            "original_test_reused_as_independent":False}


def run(out):
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    report={"study":STUDY,"phase":"fitting","parameters_within_one_percent":False,
            "original_source_sha256":ZIP_SHA,"original_csv_sha256":CSV_SHA,
            "reserved_future_numeric_rows_parsed":0,"independent_new_test":False,
            "all_original_models_complete":False,"parent_results":[],
            "fits":[],"fully_matched_architecture_compute":False}
    save(out/"results.json",report)
    z,meta=get_source(out)
    parents={}
    for h in HORIZONS:
        for s in SEEDS:
            state,rec=fit_parent(z,h,s,out)
            parents[h,s]=state,rec
            report["parent_results"].append(rec)
            save(out/"results.json",report)
    for h in HORIZONS:
        for s in SEEDS:
            parent,prec=parents[h,s]
            for lane in LANES:
                for arm in ARMS:
                    row=fit_arm(z,h,s,arm,lane,parent,prec,out)
                    report["fits"].append(row)
                    save(out/"results.json",report)
    active=[r["active_weights"] for r in report["fits"] if r["horizon"]==96 and r["lane"]=="exposure"]
    if max(active)/min(active)>1.01:
        raise ValueError("near equal active-parameter requirement broken")
    report["parameters_within_one_percent"]=True
    report["evaluation"]=summarize(report["fits"])
    report["phase"]="completed";report["all_original_models_complete"]=len(report["fits"])==24
    report["provenance"]=meta
    report["reserved_future_numeric_rows_parsed"]=0
    report["no_promotion_or_merge"]=True
    save(out/"results.json",report)
    print(json.dumps({"status":"ME040_HISTORICAL_DEV_ONLY_COMPLETE",
                      "evaluation":report["evaluation"],
                      "independent_test_qualified":False},sort_keys=True),flush=True)


def replay(out):
    out=Path(out)
    report=json.loads((out/"results.json").read_text())
    assert report["study"]==STUDY and report["all_original_models_complete"]
    assert report["reserved_future_numeric_rows_parsed"]==0
    z,meta=get_source(out)
    for row in report["fits"]:
        cp=out/row["checkpoint"]
        assert sha(cp)==row["checkpoint_sha256"]
        ck=torch.load(cp,map_location="cpu",weights_only=True)
        assert ck["original_zip_sha256"]==ZIP_SHA
        assert ck["horizon"]==row["horizon"] and ck["seed"]==row["seed"]
        model=build_model(row["arm"],row["horizon"],row["seed"])
        model.load_state_dict(ck["state_dict"],strict=True)
        pp=out/Path(row["checkpoint"]).parent/"DEV-predictions.npz"
        assert sha(pp)==row["metric"]["saved_DEV_predictions_sha256"]
        with np.load(pp) as original:
            p=original["pred"];y=original["truth"]
            metric=forecast(model,z,row["horizon"])
            assert abs(metric["power3_MSE"]-row["metric"]["power3_MSE"])<1e-9
            ix=np.arange(TR,VA-row["horizon"]+1,48,dtype=np.int64)
            preds=[];ys=[]
            model.eval()
            with torch.no_grad():
                for i in range(0,len(ix),16):
                    x,target=samples(z,ix[i:i+16],row["horizon"])
                    preds.append(model(x).cpu().numpy());ys.append(target.cpu().numpy())
            actual=np.concatenate(preds)
            if not np.array_equal(actual,p):
                raise ArithmeticError("saved real DEV forecasts do not replay bit exactly")
            if not np.array_equal(y,np.concatenate(ys)):
                raise ArithmeticError("original authentic real DEV labels altered")
    print(json.dumps({"status":"ME040_ALL24_ORIGINAL_REAL_CHECKPOINTS_REPLAY",
                      "new_hidden_test_opened":False},sort_keys=True))


if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--out",default="results/me040")
    ap.add_argument("--self-test",action="store_true")
    ap.add_argument("--replay",action="store_true")
    a=ap.parse_args()
    if a.self_test:print(json.dumps(preflight(),sort_keys=True))
    elif a.replay:replay(a.out)
    else:run(a.out)
