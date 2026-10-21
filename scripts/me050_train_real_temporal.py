#!/usr/bin/env python3
"""ME050: train and replay authentic user-linked chronological click predictors.

Critical: future calendar-date outcome fields are NOT read until every
model checkpoint is finalized and hash-frozen. Test histories are prequential:
only prior real observed user interactions are supplied at inference time.
"""
from __future__ import annotations
import argparse,csv,hashlib,json,math,time,urllib.request
from pathlib import Path
from collections import defaultdict
import numpy as np
import torch
from torch.nn import functional as F
from tac_osm.me050_real_temporal_memory import (
    ARMS,HISTORY,BUCKETS,RealHistoryPredictor,real_observed_click_loss)

REV="b86438794b71ac3ebb95d96e167f52d22edccda7"
SOURCE_URL=f"https://huggingface.co/datasets/numberbeat6/kuairand/resolve/{REV}/log_random_4_22_to_5_08_1k.csv"
SOURCE_NAME="log_random_4_22_to_5_08_1k.csv"
SEEDS=(50001,50002)
UPDATES=500
BATCH=64
FIT_END=20220502
DEV_END=20220505
torch.set_num_threads(1)


def digest(path):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for block in iter(lambda:f.read(1<<20),b""):h.update(block)
    return h.hexdigest()


def save(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(data,indent=2,sort_keys=True)+"\n")


def fetch_source(root):
    folder=Path(root)/"source-private-NOT-ARTIFACT";folder.mkdir(parents=True,exist_ok=True)
    source=folder/SOURCE_NAME
    if source.is_file():return source
    temp=folder/"download.part"
    req=urllib.request.Request(SOURCE_URL,headers={"User-Agent":"TAC-OSM-ME050-authentic-temporal-memory/1.0"})
    try:
        with urllib.request.urlopen(req,timeout=120) as resp,temp.open("wb") as out:
            while True:
                block=resp.read(1<<20)
                if not block:break
                out.write(block)
        if not 2_000_000<=temp.stat().st_size<=5_000_000:
            raise RuntimeError("publisher mirror bytes outside expected original small-data bounds")
        temp.replace(source)
    finally:temp.unlink(missing_ok=True)
    return source


def read_real_events(source, fit_only):
    """Only reads FIT-row feature/action/outcome data before model freeze."""
    rows=[];required={"user_id","video_id","time_ms","date","is_click","is_rand"}
    with open(source,"r",newline="",encoding="utf-8") as f:
        reader=csv.DictReader(f)
        if not required.issubset(set(reader.fieldnames or [])):
            raise RuntimeError("wrong published real KuaiRand record schema")
        for index,row in enumerate(reader):
            date=int(row["date"])
            if fit_only and date>=FIT_END:continue
            if not 20220422<=date<=20220508:
                raise RuntimeError("unexpected actual recorded event date")
            userid=int(row["user_id"])
            vid=int(row["video_id"])
            stamp=int(row["time_ms"])
            clicked=int(row["is_click"])
            isrand=int(row["is_rand"])
            if userid<0 or vid<0 or stamp<=0 or clicked not in (0,1) or isrand!=1:
                raise RuntimeError("source event action, time or actual outcome malformed")
            rows.append((userid,vid,stamp,date,clicked,index))
    return rows


def assemble_histories(rows):
    users=defaultdict(list)
    for row in rows:users[row[0]].append(row)
    partitions={p:{"h_vid":[],"h_num":[],"v":[],"hour":[],"y":[],"user":[]} for p in ("FIT","DEV","TEST")}
    for user,events in sorted(users.items()):
        events.sort(key=lambda e:(e[2],e[5]))
        past=[]
        pos=0
        while pos<len(events):
            end=pos+1
            while end<len(events) and events[end][2]==events[pos][2]:end+=1
            # Equal timestamp actions are simultaneous, never part of one another's history.
            for current in events[pos:end]:
                if len(past)<HISTORY:continue
                uid,vid,timestamp,date,clicked,_=current
                part="FIT" if date<FIT_END else "DEV" if date<DEV_END else "TEST"
                last=past[-HISTORY:]
                ids=[];features=[]
                for _,history_vid,t_prior,_,history_click,_ in last:
                    if t_prior>=timestamp:raise RuntimeError("temporal future leakage")
                    delta=max(0.,(timestamp-t_prior)/3_600_000.)
                    ids.append(history_vid%BUCKETS)
                    features.append([float(history_click),
                         float(min(math.log1p(delta)/10.,2.)),
                         float((t_prior//3_600_000)%24)/24.])
                entry=partitions[part]
                entry["h_vid"].append(ids)
                entry["h_num"].append(features)
                entry["v"].append(vid%BUCKETS)
                entry["hour"].append([float((timestamp//3_600_000)%24)/24.])
                entry["y"].append(clicked)
                entry["user"].append(uid)
            past.extend(events[pos:end]);pos=end
    final={}
    for name,values in partitions.items():
        if not values["y"]:continue
        final[name]={
            "h_vid":torch.tensor(np.asarray(values["h_vid"],np.int64)),
            "h_num":torch.tensor(np.asarray(values["h_num"],np.float32)),
            "v":torch.tensor(np.asarray(values["v"],np.int64)),
            "hour":torch.tensor(np.asarray(values["hour"],np.float32)),
            "y":torch.tensor(np.asarray(values["y"],np.float32)),
            "user":np.asarray(values["user"],np.int32)}
    return final


def fitted_example_digest(dataset):
    h=hashlib.sha256()
    for key in ("h_vid","h_num","v","hour","y"):
        h.update(dataset[key].numpy().tobytes())
    h.update(dataset["user"].tobytes())
    return h.hexdigest()


def verify_partition(dataset,name):
    n=len(dataset["y"])
    if n<250:raise RuntimeError(f"real {name} has too few genuine 12-event histories: {n}")
    if len(set(dataset["user"]))<15:
        raise RuntimeError(f"real {name} has insufficient distinct repeat users")
    positives=int(dataset["y"].sum())
    if not 0<positives<n:raise RuntimeError(f"real {name} outcome support failed")
    if dataset["h_vid"].shape!=(n,HISTORY) or dataset["h_num"].shape!=(n,HISTORY,3):
        raise RuntimeError("real temporal tensor dimensional contract failed")
    return {"n":n,"user_count":len(set(dataset["user"])),
            "positive_clicks":positives,"actual_click_rate":float(dataset["y"].mean())}


def train(fit,arm,seed,directory,source_hash):
    directory.mkdir(parents=True,exist_ok=False)
    torch.manual_seed(seed)
    model=RealHistoryPredictor(arm)
    optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
    sampler=torch.Generator().manual_seed(seed+1000)
    trace=[];cpu=0.
    for step in range(1,UPDATES+1):
        indices=torch.randint(len(fit["y"]),(BATCH,),generator=sampler)
        started=time.process_time()
        optimizer.zero_grad(set_to_none=True);model.train()
        loss=real_observed_click_loss(model,fit["h_vid"][indices],
             fit["h_num"][indices],fit["v"][indices],
             fit["hour"][indices],fit["y"][indices])
        if not torch.isfinite(loss):raise RuntimeError("real click training loss invalid")
        loss.backward()
        norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        if not torch.isfinite(norm):raise RuntimeError("invalid real click gradients")
        optimizer.step()
        cpu+=time.process_time()-started
        if step%100==0:trace.append({"step":step,"real_observed_click_loss":float(loss.detach()),
                                     "optimizer_CPU_seconds":cpu})
    path=directory/"authentic-user-history-trained.pt"
    torch.save({"arm":arm,"seed":seed,"weights":model.state_dict(),
                "optimizer":optimizer.state_dict(),"sampler_state":sampler.get_state(),
                "torch_rng":torch.get_rng_state(),"original_source_sha256":source_hash,
                "history_length":HISTORY,"updates":UPDATES},path)
    save(directory/"training-trace.json",trace)
    record={"arm":arm,"seed":seed,"active_parameters":model.num_active(),
            "optimizer_CPU_seconds":cpu,"real_training_updates":UPDATES,
            "actual_logged_example_exposures":UPDATES*BATCH,
            "checkpoint":path.name,"checkpoint_sha256":digest(path),
            "later_calendar_outcomes_read_pre_freeze":False}
    print(json.dumps({"event":"ME050_REAL_HISTORY_FIT","arm":arm,"seed":seed,
                      "parameters":model.num_active(),"optimizer_CPU":round(cpu,3)}),flush=True)
    return record


@torch.no_grad()
def predict(model,dataset):
    model.eval()
    preds=[]
    for start in range(0,len(dataset["y"]),256):
        sl=slice(start,start+256)
        logits=model(dataset["h_vid"][sl],dataset["h_num"][sl],
                     dataset["v"][sl],dataset["hour"][sl])
        preds.append(torch.sigmoid(logits).numpy())
    return np.concatenate(preds)


def score(probs,labels):
    p=np.clip(probs,1e-6,1-1e-6)
    return {"Brier":float(np.mean((p-labels)**2)),
            "logloss":float(np.mean(-(labels*np.log(p)+(1-labels)*np.log1p(-p))))}


def cluster_ci(tac,control,y,users,seed):
    """Paired improvement in observed squared loss, resampling whole genuine users."""
    difference=(control-y)**2-(tac-y)**2
    unique,inverse=np.unique(users,return_inverse=True)
    sums=np.bincount(inverse,weights=difference,minlength=len(unique))
    counts=np.bincount(inverse,minlength=len(unique))
    rng=np.random.default_rng(seed+95050)
    resamples=[]
    for _ in range(200):
        which=rng.integers(len(unique),size=len(unique))
        resamples.append(float(sums[which].sum()/counts[which].sum()))
    return [float(np.quantile(resamples,.025)),float(np.quantile(resamples,.975))]


def evaluate(root,cohort,models,part):
    labels=cohort["y"].numpy()
    results={};predictions={}
    for rec in models:
        folder=root/f'{rec["arm"]}-s{rec["seed"]}'
        cp=folder/rec["checkpoint"]
        if digest(cp)!=rec["checkpoint_sha256"]:
            raise RuntimeError("frozen real model changed after outcome revelation")
        state=torch.load(cp,weights_only=True)
        net=RealHistoryPredictor(rec["arm"]);net.load_state_dict(state["weights"],strict=True)
        prediction=predict(net,cohort)
        name=part+"-predictions.npz"
        np.savez_compressed(folder/name,predicted_observed_click=prediction)
        key=f'{rec["arm"]}-s{rec["seed"]}'
        results[key]={**score(prediction,labels),"prediction_file":name,
                      "prediction_sha256":digest(folder/name)}
        predictions[key]=prediction
    return results,predictions


def run(out):
    root=Path(out);root.mkdir(parents=True,exist_ok=False)
    report={"study":"ME050_REAL_LINKED_USER_TEMPORAL_HISTORY",
            "complete":False,"source_revision":REV,
            "source_url":SOURCE_URL,"source_licensing":"CC-BY-SA-4.0 source mirror; attribution required",
            "source_raw_user_records_redistributed":False,
            "continuous_memory_beyond_12_events_claim":False,
            "real_logged_counterfactual_policy_qualified":False,
            "genuine_temporal_history_used":True,
            "future_calendar_outcomes_loaded_before_source_fit":False,
            "prequential_test_uses_only_earlier_observed_clicks":True,
            "models":[],"sealed_test_opened":False,"no_promotion_or_merge":True}
    save(root/"results.json",report)
    source=fetch_source(root);source_hash=digest(source)
    report["original_source_sha256"]=source_hash
    report["original_source_bytes"]=source.stat().st_size
    # Only FIT-record numeric values are exposed at this point.
    fit_rows=read_real_events(source,fit_only=True)
    if len(fit_rows)<1000:raise RuntimeError("too few genuine FIT events")
    fit=assemble_histories(fit_rows)["FIT"]
    report["fit_cohort"]=verify_partition(fit,"FIT")
    report["FIT_tensor_sha256"]=fitted_example_digest(fit)
    save(root/"results.json",report)
    for seed in SEEDS:
        for arm in ARMS:
            rec=train(fit,arm,seed,root/f"{arm}-s{seed}",source_hash)
            report["models"].append(rec)
            save(root/"results.json",report)
    if len(report["models"])!=8:raise RuntimeError("eight source models not frozen")
    for rec in report["models"]:
        path=root/f'{rec["arm"]}-s{rec["seed"]}'/rec["checkpoint"]
        if digest(path)!=rec["checkpoint_sha256"]:
            raise RuntimeError("original training freeze failed")
    report["all_eight_models_frozen_before_later_outcome_access"]=True
    save(root/"results.json",report)
    # First time DEV/TEST numeric features/outcome fields may be examined.
    all_rows=read_real_events(source,fit_only=False)
    all_data=assemble_histories(all_rows)
    if fitted_example_digest(all_data["FIT"])!=report["FIT_tensor_sha256"]:
        raise RuntimeError("frozen FIT real user-history reconstruction mismatched")
    report["source_original_rows"]=len(all_rows)
    report["later_cohorts"]={p:verify_partition(all_data[p],p) for p in ("DEV","TEST")}
    report["sealed_test_opened"]=True
    report["DEV_results"],_=evaluate(root,all_data["DEV"],report["models"],"DEV")
    report["TEST_results"],preds=evaluate(root,all_data["TEST"],report["models"],"TEST")
    labels=all_data["TEST"]["y"].numpy()
    uids=all_data["TEST"]["user"]
    report["FIT_rate_baseline_TEST_Brier"]=float(np.mean((labels-report["fit_cohort"]["actual_click_rate"])**2))
    gates={};ci={}
    for seed in SEEDS:
        a=f"tac_chronological-s{seed}"
        r=f"tac_reversed-s{seed}"
        g=f"gru_chronological-s{seed}"
        c=f"context_history-s{seed}"
        ci[str(seed)]={
            "same_native_order_vs_reversed_cluster_improvement_CI":
                cluster_ci(preds[a],preds[r],labels,uids,seed),
            "vs_gru_cluster_improvement_CI":
                cluster_ci(preds[a],preds[g],labels,uids,seed)}
        gates[str(seed)]={
            "genuine_order_beats_same_native_reversed":
                report["TEST_results"][a]["Brier"]<report["TEST_results"][r]["Brier"]
                and ci[str(seed)]["same_native_order_vs_reversed_cluster_improvement_CI"][0]>0,
            "native_better_than_gru":
                report["TEST_results"][a]["Brier"]<report["TEST_results"][g]["Brier"],
            "native_better_than_context":
                report["TEST_results"][a]["Brier"]<report["TEST_results"][c]["Brier"]}
    report["paired_cluster_intervals"]=ci
    report["registered_gates"]=gates
    report["capability_acceptance_all_gates"]=all(all(v.values()) for v in gates.values())
    report["resource_parity_across_distinct_architectures_qualified"]=False
    report["complete"]=True
    save(root/"results.json",report)
    print(json.dumps({"event":"ME050_REAL_TEMPORAL_RESULTS",
        "test_Brier":{k:v["Brier"] for k,v in report["TEST_results"].items()},
        "split_counts":{"FIT":report["fit_cohort"],**report["later_cohorts"]},
        "gates":gates,"true_real_sequential_memory_capability_qualified":
            report["capability_acceptance_all_gates"]}),flush=True)


@torch.no_grad()
def replay(root):
    root=Path(root);report=json.loads((root/"results.json").read_text())
    if not report["complete"] or not report["all_eight_models_frozen_before_later_outcome_access"]:
        raise RuntimeError("no original complete frozen models")
    source=fetch_source(root)
    if digest(source)!=report["original_source_sha256"]:raise RuntimeError("source changed")
    cohorts=assemble_histories(read_real_events(source,fit_only=False))
    for part in ("DEV","TEST"):
        data=cohorts[part]
        for rec in report["models"]:
            folder=root/f'{rec["arm"]}-s{rec["seed"]}'
            cp=folder/rec["checkpoint"]
            if digest(cp)!=rec["checkpoint_sha256"]:raise RuntimeError("model changed")
            saved=torch.load(cp,weights_only=True)
            model=RealHistoryPredictor(rec["arm"])
            model.load_state_dict(saved["weights"],strict=True)
            p=predict(model,data)
            receipt=report[part+"_results"][f'{rec["arm"]}-s{rec["seed"]}']
            dest=folder/receipt["prediction_file"]
            if digest(dest)!=receipt["prediction_sha256"]:raise RuntimeError("forecast file changed")
            with np.load(dest) as original:
                if not np.array_equal(p,original["predicted_observed_click"]):
                    raise RuntimeError("real source forecast not bit exact")
    print("ME050_ORIGINAL_8_REAL_TEMPORAL_MODELS_REPLAY_EXACTLY",flush=True)


def self_test():
    """Only generated numerical tensors as software assertions, not results."""
    torch.manual_seed(50050)
    h=torch.randint(4096,(3,HISTORY));v=torch.randint(4096,(3,))
    n=torch.rand(3,HISTORY,3);hour=torch.rand(3,1);y=torch.tensor([0.,1.,0.])
    a=RealHistoryPredictor("tac_chronological")
    b=RealHistoryPredictor("tac_reversed")
    assert a.num_active()==b.num_active()
    b.load_state_dict(a.state_dict(),strict=True)
    for net in (a,b,RealHistoryPredictor("gru_chronological"),RealHistoryPredictor("context_history")):
        loss=real_observed_click_loss(net,h,n,v,hour,y)
        loss.backward()
        assert torch.isfinite(loss)
        assert all(param.grad is not None and torch.isfinite(param.grad).all()
                   for param in net.parameters() if param.requires_grad)
    assert torch.allclose(a(h,n,v,hour),b(h.flip(1),n.flip(1),v,hour))
    print("ME050_SOFTWARE_NUMERICAL_ONLY_PASS",flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--out",default="results/me050")
    p.add_argument("--self-test",action="store_true")
    p.add_argument("--replay",action="store_true")
    flags=p.parse_args()
    if flags.self_test:self_test()
    elif flags.replay:replay(flags.out)
    else:run(flags.out)
