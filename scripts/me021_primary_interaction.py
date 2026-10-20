#!/usr/bin/env python3
"""ME021: preregistered paired user-clustered H128 interaction and strong simple baselines.

Original trained 18 models, SHA-checked BEFORE genuine source.
Only original user%16==0 early FIT & inspected DEV for baseline/calibration FIT;
first future user%16==5 after all those recipes are fixed. Zero original weight updates.
One primary user-cluster bootstrap interaction, not 3 independent seed tests.
"""
from __future__ import annotations
import argparse, csv, hashlib, io, json, math, time, zipfile
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from scripts.me014b_exact_action_memory import (
    FIT_CUT, DEV_CUT, END_CUT, PIN_MD5, REQUIRED, VOCAB,
    publisher_bytes, member_for_zip, read_early, batch, binary_scores
)
from scripts.me019_authentic_history import (
    original_128_windows, view, FIT_MAX, DEV_MAX,
    ARMS, LENGTHS, SEEDS, PARAMS, SOURCE_SHA, make_model
)
from scripts.me019f_frozen_user4 import restore_models, MANIFEST_SHA
from scripts.me020_frozen_128_dynamics import (
    capture, ridge_fit_predict
)

RUN=39443325909
FUTURE_MOD=5
MAX_FUTURE=50000
BATCH=384
HALFLIVES=(2,4,8,16,32,64,128)
MLP_UPDATES=300
MLP_BATCH=128
BOOT=2500
BOOT_SEED=21021

def source_future_user5():
    """Read ONLY user%16==5 record responses; no user4 retesting or fit peeking."""
    raw,source=publisher_bytes()
    if source["zip_sha256"]!=SOURCE_SHA or source["zip_md5"]!=PIN_MD5:
        raise ValueError("publisher genuine source hash mismatch")
    people={}
    future=0
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        member=member_for_zip(z)
        with z.open(member) as f:
            reader=csv.DictReader(io.TextIOWrapper(f,encoding="utf-8-sig",newline=""))
            if not set(REQUIRED).issubset(reader.fieldnames or []):
                raise ValueError("authentic original source schema changed")
            for r in reader:
                try:
                    uid=int(r["user_id"])
                    if uid<0 or uid%16!=FUTURE_MOD:
                        continue
                    ts=float(r["timestamp"])
                    if not math.isfinite(ts) or ts<=1e9 or ts>=END_CUT:
                        continue
                    item=int(r["video_id"])
                    duration=float(r["video_duration"])
                    ratio=float(r["watch_ratio"])
                except (TypeError,ValueError):
                    continue
                if (not 0<=item<VOCAB or not math.isfinite(duration) or duration<=0
                        or not math.isfinite(ratio) or ratio<0):
                    continue
                people.setdefault(uid,[]).append((ts,item,math.log1p(duration)/12.0,int(ratio>=1)))
                future+=int(ts>=DEV_CUT)
    for rows in people.values():
        rows.sort(key=lambda r:r[0])
    if future<1500:
        raise ValueError("cannot use inadequate genuinely future user5 response cohort")
    source["future5_rows_in_original_csv"]=future
    return people,source

def score_auc(p, y):
    """Exact mid-rank ROC AUC, including tied predictions, no sklearn dependency."""
    p=np.asarray(p,np.float64).ravel(); y=np.asarray(y,np.float64).ravel()
    pos=int(y.sum()); neg=len(y)-pos
    if min(pos,neg)<=0:
        raise ValueError("need both factual outcomes for ROC AUC")
    order=np.argsort(p,kind="mergesort")
    sorted_p=p[order]
    ranks=np.empty(len(p),dtype=np.float64)
    start=0
    while start<len(p):
        end=start+1
        while end<len(p) and sorted_p[end]==sorted_p[start]:
            end+=1
        ranks[order[start:end]]=(start+end+1)/2
        start=end
    return float((ranks[y==1].sum()-pos*(pos+1)/2)/(pos*neg))

def metrics(p,y):
    p=np.asarray(p,np.float64).ravel().clip(1e-7,1-1e-7)
    y=np.asarray(y,np.float64).ravel()
    if len(p)!=len(y) or not np.isfinite(p).all():
        raise ValueError("nonfinite predictions or target alignment")
    return {"Brier":float(np.mean((p-y)**2)),
            "log_loss":float(np.mean(-y*np.log(p)-(1-y)*np.log(1-p))),
            "ROC_AUC":score_auc(p,y),"N":len(p)}

def model_probs(net,data):
    net.eval()
    out=[]
    with torch.no_grad():
        for i in range(0,len(data["y"]),BATCH):
            sl=slice(i,min(i+BATCH,len(data["y"])))
            out.append(torch.sigmoid(net(*batch(data,sl))).flatten().cpu())
    return torch.cat(out).numpy().astype("float64")

def pin_train_and_calibration():
    # All original hash checks BEFORE opening any publisher response source.
    original,manifest = restore_models("results/me019"),None
    if len(original)!=18:
        raise ValueError("original 18 SHA-frozen models missing")
    torch.set_num_threads(1)
    users,src=read_early()
    if src["zip_sha256"] != SOURCE_SHA or src["zip_md5"] != PIN_MD5 or src["future_test_labels_parsed"]:
        raise ValueError("changed early authentic source")
    fit=original_128_windows(users,0,FIT_CUT,FIT_MAX)
    dev=original_128_windows(users,FIT_CUT,DEV_CUT,DEV_MAX)
    if len(fit["y"])!=10776 or len(dev["y"])!=5952:
        raise ValueError("not original matched real train/development windows")
    return original,fit,dev,src

def features(data, item_rate):
    item=data["items"].numpy()
    watched=data["metadata"][:,:,1].numpy().astype("float64")
    if watched.shape[1]!=128:
        raise ValueError("never supply padded or noncausal history")
    def smooth(n):
        h=watched[:,-n:]
        return (h.sum(1)+1)/(n+2)
    cols=[smooth(32),smooth(128)]
    ages=np.arange(127,-1,-1)
    for half in HALFLIVES:
        weights=np.exp2(-ages/float(half))
        cols.append((watched*weights).sum(1)/weights.sum())
    current=data["action"].numpy()
    cols.append(item_rate[current])
    cols.append(data["duration"].numpy().reshape(-1).astype("float64"))
    cols.append(data["metadata"][:,-32:,2].numpy().mean(1).astype("float64"))
    return np.stack(cols,axis=-1)

def item_rates(train):
    actions=train["action"].numpy()
    targets=train["y"].numpy().flatten()
    c=np.bincount(actions,minlength=VOCAB)
    v=np.bincount(actions,weights=targets,minlength=VOCAB)
    pop=float(targets.mean())
    return (v+16*pop)/(c+16),pop

def calibrate(logits,y,regularizer=1e-3, niter=80):
    """Fixed Platt intercept/slope on ORIGINAL historical DEV only."""
    logits=np.asarray(logits,np.float64).reshape(-1)
    target=np.asarray(y,np.float64).reshape(-1)
    if len(logits)!=len(target):
        raise ValueError("calibrator pairing mismatch")
    x=torch.as_tensor(logits,dtype=torch.float64).reshape(-1,1)
    y=torch.as_tensor(target,dtype=torch.float64).reshape(-1,1)
    w=torch.tensor([[1.0]],dtype=torch.float64,requires_grad=True)
    b=torch.tensor([0.0],dtype=torch.float64,requires_grad=True)
    opt=torch.optim.LBFGS([w,b],lr=.6,max_iter=niter,line_search_fn="strong_wolfe")
    def closure():
        opt.zero_grad()
        loss=F.binary_cross_entropy_with_logits(x@w+b,y)+regularizer*(w-1).square().sum()
        loss.backward()
        return loss
    opt.step(closure)
    with torch.no_grad():
        return float(w.item()),float(b.item())

def train_logistic_or_mlp(train_features, train_y, seed, mlp):
    """All 300 baseline optimization steps run BEFORE future5 source opens."""
    X=np.asarray(train_features,dtype="float64")
    y=np.asarray(train_y,dtype="float64").reshape(-1)
    mu=X.mean(0);scale=np.maximum(X.std(0),1e-5)
    train=torch.as_tensor((X-mu)/scale,dtype=torch.float32)
    torch.manual_seed(seed+(600 if mlp else 500))
    net=(nn.Sequential(nn.Linear(X.shape[1],16),nn.Tanh(),nn.Linear(16,1))
         if mlp else nn.Linear(X.shape[1],1))
    gen=torch.Generator().manual_seed(seed+300)
    opt=torch.optim.AdamW(net.parameters(),lr=.006 if mlp else .015,weight_decay=.001)
    start=time.perf_counter()
    labels=torch.as_tensor(y,dtype=torch.float32)
    for step in range(MLP_UPDATES):
        ids=torch.randint(len(labels),(MLP_BATCH,),generator=gen)
        logits=net(train[ids]).view(-1)
        loss=F.binary_cross_entropy_with_logits(logits,labels[ids])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    elapsed=time.perf_counter()-start
    net.eval()
    return {"network":net,"mean":mu,"std":scale,
       "parameters":sum(q.numel() for q in net.parameters()),
       "train_seconds":round(elapsed,3)}

def predict_simple(fitted, features):
    x=np.asarray(features,dtype="float64")
    query=torch.as_tensor((x-fitted["mean"])/fitted["std"],dtype=torch.float32)
    with torch.no_grad():
        return torch.sigmoid(fitted["network"](query).view(-1)).numpy().astype("float64")


def user_cluster_ci(event_by_seed, users):
    """One event-wise mean over seeds, one common resampled user draw."""
    values=np.asarray(event_by_seed,dtype="float64")
    if values.ndim!=2 or values.shape[0]!=3:
        raise ValueError("3 seeds are not bootstrap clusters")
    d=values.mean(0)
    u,idx=np.unique(np.asarray(users,dtype=np.int64),return_inverse=True)
    sums=np.bincount(idx,weights=d,minlength=len(u))
    counts=np.bincount(idx,minlength=len(u))
    rng=np.random.default_rng(BOOT_SEED)
    simulated=np.empty(BOOT,dtype=np.float64)
    for j in range(BOOT):
        sample=rng.integers(0,len(u),size=len(u))
        simulated[j]=sums[sample].sum()/counts[sample].sum()
    return {"point_mean":float(d.mean()),
        "CI_95_user_clustered":[float(z) for z in np.quantile(simulated,[.025,.975])],
        "unique_users":len(u),"N_events":len(d),"bootstrap_seed":BOOT_SEED,
        "bootstrap_replicates":BOOT,
        "per_seed_interactions":[float(z) for z in values.mean(1)],
        "joint_user_draw_shared_by_all_seed_arms":True,
        "seeds_not_independent_bootstrap_units":True}

def frozen_probe_control(fit, originals):
    """Random TAC128 vs trained TAC128 and trained GRU128 on identical REAL FIT probes."""
    from scripts.me020_frozen_128_dynamics import PROBE_LIMIT, DEPTHS
    ids=torch.linspace(0,len(fit["y"])-1,PROBE_LIMIT,dtype=torch.long).unique()
    if len(ids)!=PROBE_LIMIT:
        raise ValueError("original FIT probe source changed")
    d={"items":fit["items"][ids],"metadata":fit["metadata"][ids],
       "action":fit["action"][ids],"duration":fit["duration"][ids],
       "y":fit["y"][ids]}
    uid=np.asarray([fit["users"][int(i)] for i in ids],dtype=np.int64)
    tr=(uid%5!=4); te=~tr
    if tr.sum()<400 or te.sum()<90 or len(np.unique(uid[te]))<6:
        raise ValueError("user-disjoint probe split changed")
    early=d["metadata"][:,:8,1].numpy().mean(1)
    rows=[]
    for seed in SEEDS:
        for kind in ("native_trained","gru_trained","native_random"):
            if kind=="native_random":
                torch.manual_seed(seed)
                net=make_model("native_chrono_anchor",128,script_native=False).eval()
            else:
                arm="native_chrono_anchor" if kind=="native_trained" else "gru_original"
                net=next(net for e,net in originals
                         if e["arm"]==arm and e["history"]==128 and e["seed"]==seed)
            features=[]
            with torch.no_grad():
                for first in range(0,len(ids),128):
                    sl=slice(first,min(len(ids),first+128))
                    features.append(capture(net,batch(d,sl))["states"][128].cpu())
            representation=torch.cat(features).numpy()
            pred=ridge_fit_predict(representation[tr],early[tr],representation[te])
            mse=float(np.mean((pred-early[te])**2))
            rows.append({"seed":seed,"model":kind,"oldest8_decoding_MSE":mse,
                "N_probe_train":int(tr.sum()),"N_probe_test":int(te.sum()),
                "N_probe_test_users":len(np.unique(uid[te])),
                "fixed_ridge_lambda":1,"training_updates_of_probe_source_net":
                    0 if kind=="native_random" else 300})
    return rows

def run(output):
    torch.set_num_threads(1)
    originals,fit,dev,src=pin_train_and_calibration()
    rates,global_prior=item_rates(fit)
    fit_f=features(fit,rates)
    dev_f=features(dev,rates)
    # Freeze the calibrator using previously inspected, nonfuture original DEV.
    dev_predictions={}
    calibration={}
    for row,net in originals:
        key=(row["arm"],row["history"],row["seed"])
        tested=view(dev,32) if row["history"]==32 else dev
        probs=model_probs(net,tested)
        dev_predictions[key]=probs
        rawlogit=np.log(np.clip(probs,1e-7,1-1e-7)/np.clip(1-probs,1e-7,1-1e-7))
        calibration[key]=calibrate(rawlogit,dev["y"].numpy())
    # Pre-fit ALL small baselines on the original real FIT BEFORE source5 exposure.
    fitted_simple={}
    for name,fitted in fitted_simple.items():
        baselines[name]=predict_simple(fitted,ffeatures)
    preds={}
    corrected={}
    model_scores=[]
    future_short=view(future,32)
    for row,net in originals:
        key=(row["arm"],row["history"],row["seed"])
        real=future_short if row["history"]==32 else future
        p=model_probs(net,real)
        preds[key]=p
        slope,intercept=calibration[key]
        rawlogit=np.log(np.clip(p,1e-7,1-1e-7)/np.clip(1-p,1e-7,1-1e-7))
        cal=1/(1+np.exp(-np.clip(slope*rawlogit+intercept,-70,70)))
        corrected[key]=cal
        # descriptive reliability fit on future5 ONLY for reporting, never scoring.
        test_slope,test_int=calibrate(rawlogit,truth)
        model_scores.append({"arm":row["arm"],"seed":row["seed"],
            "history":row["history"],"original_weights_sha256":row["checkpoint_sha256"],
            "unrecalibrated":metrics(p,truth),
            "calibrated_historical_DEV_only":metrics(cal,truth),
            "historical_DEV_platt_slope":slope,
            "historical_DEV_platt_intercept":intercept,
            "future_only_descriptive_reliability_slope_not_applied":test_slope,
            "future_only_descriptive_reliability_intercept_not_applied":test_int,
            "original_model_optimizer_updates_on_future":0,
            "original_training_CPU_seconds":row["train_seconds"],
            "original_sparse_training_trace":row["train_trace"]})
        print(json.dumps({"arm":row["arm"],"seed":row["seed"],
            "history":row["history"],"Brier":model_scores[-1]["unrecalibrated"]["Brier"]},
            sort_keys=True),flush=True)

    def interaction(dictionary):
        lines=[]
        for seed in SEEDS:
            def error(arm,h):
                p=dictionary[(arm,h,seed)]
                return (p-truth)**2
            lines.append(
                (error("native_chrono_anchor",128)-error("native_chrono_anchor",32))
                - (error("gru_original",128)-error("gru_original",32))
            )
        return user_cluster_ci(lines,future["users"])
    main=interaction(preds)
    second=interaction(corrected)
    baseline_scores={k:metrics(p,truth) for k,p in baselines.items()}
    label_ratings={
      "random_initialization_untrained_probe":probes,
      "primary_paired_unrecalibrated":main,
      "secondary_paired_historical_DEV_calibrated":second,
      "fitonly_simple_baselines":baseline_scores,
      "models":model_scores,
      "trained_simple_model_costs":baseline_train,
    }
    result={
       "study":"ME021_FIRST_NEW_USER5_PAIRED_INTERACTION_STAT_CALIBRATION_BASELINES",
       "original_me019_train_run":RUN,
       "original_me019_train_manifest_sha256":MANIFEST_SHA,
       "all_18_original_checkpoint_sha256_verified_before_source":True,
       "publisher_zip_sha256":SOURCE_SHA,
       "calibrator_trained_only_original_opened_DEV":True,
       "strong_baselines_trained_only_original_FIT":True,
       "new_future_user_modulo_16":FUTURE_MOD,
       "real_future_unique_users":len(set(future["users"])),
       "real_future_interaction_rows":len(truth),
       "original_neural_model_weights_updated":False,
       "full_architecture_training_CPU_or_FLOP_fair":False,
       "test_user_responses_used_in_model_training":False,
       "original_future_user4_reopened":False,
       "single_primary_interaction_registered_before_future5":True,
       "seeds_not_independent":True,
       "training_trace_is_sparse_steps_1_150_300":True,
       "fixed_chrono_fast_initial_time_constant":8,
       "fixed_chrono_slow_initial_time_constant":24,
       "uses_true_T128_scaled_chrono_initialization":False,
       "zero_init_delta_guarantees_post_training_nonregression":False,
       **label_ratings
    }
    out=Path(output);out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(result,sort_keys=True,indent=2)+"\n")
    # Persist source-hash-grounded original output probabilities to audit paired
    # interactions without ever reopening a further new test cohort.
    arrays={"observed_y":truth.astype("float32"),
            "user_ids":np.asarray(future["users"],dtype="int64")}
    for key,p in preds.items():
        arrays[f"raw_{key[0]}_T{key[1]}_s{key[2]}"]=p.astype("float32")
        arrays[f"cal_{key[0]}_T{key[1]}_s{key[2]}"]=corrected[key].astype("float32")
    for key,p in baselines.items():
        arrays["baseline_"+key]=np.asarray(p,dtype="float32")
    np.savez_compressed(out.parent/"frozen_future5_event_scores.npz",**arrays)
    print(json.dumps({"study":result["study"],
       "frozen_models":len(model_scores),
       "users":result["real_future_unique_users"],
       "events":len(truth),
       "primary_mean":main["point_mean"],
       "primary_CI":main["CI_95_user_clustered"],
       "user4_reopened":False},sort_keys=True),flush=True)
    return result

def preflight():
    torch.set_num_threads(1)
    fake_y=np.asarray([0.,1.,0.,1.,1.,0.])
    fake_p=np.asarray([.25,.8,.1,.55,.6,.4])
    assert .5<score_auc(fake_p,fake_y)<=1
    m=metrics(fake_p,fake_y)
    assert math.isfinite(m["Brier"]) and math.isfinite(m["ROC_AUC"])
    fit_x=np.arange(48,dtype=float).reshape(12,4)/48
    fitted=train_logistic_or_mlp(fit_x,np.arange(12)%2,14001,False)
    pp=predict_simple(fitted,fit_x)
    assert fitted["parameters"]==5 and pp.shape==(12,) and np.isfinite(pp).all()
    ci=user_cluster_ci(np.tile(np.array([-.01,.02,-.03,.04]),(3,1)),[10,10,20,20])
    assert ci["unique_users"]==2 and len(ci["CI_95_user_clustered"])==2
    sl,b=calibrate(np.array([-.5,.7,-.3,1.2]),np.array([0,1,0,1]))
    assert np.isfinite(sl) and np.isfinite(b)
    print(json.dumps({"status":"ME021_SOFTWARE_ONLY_PREFLIGHT_PASS",
        "source_or_future_labels_opened":False,
        "new_research_performance_claim":False},sort_keys=True))

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--out",default="results/me021/frozen_new_user5_interaction.json")
    p.add_argument("--self-test",action="store_true")
    a=p.parse_args()
    preflight() if a.self_test else run(a.out)
