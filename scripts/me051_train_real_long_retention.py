#!/usr/bin/env python3
"""ME051 original MovieLens human rating histories, retention training and replay.

Two-phase outcome access: only source user/movie/timestamp metadata for all
rows before fitting; original DEV/TEST RATING fields are first parsed after
ALL ten real FIT-trained weights/optimizer/RNG are checksum-frozen.
"""
from __future__ import annotations
import argparse,bisect,hashlib,json,math,re,time,urllib.request,zipfile
from pathlib import Path
from collections import defaultdict
import numpy as np
import torch
from tac_osm.me051_long_retention import (
    ARMS,HORIZONS,RealRetentionPredictor,real_quarter_target,genuine_rating_loss)

SOURCE="https://files.grouplens.org/datasets/movielens/ml-1m.zip"
SOURCE_MD5_URL=SOURCE+".md5"
SOURCE_MD5="c4d9eecfca2ab87c1945afe126590906"
DATA_MD5="a89aa3591bc97d6d4e0c89459ff39362"
RECORDS=1_000_209
USERS=6_040
SEEDS=(51001,51002)
STEPS=500
BATCH=64
TEST_LIMIT=8000
DEV_LIMIT=3000
MIN_FIT_ROWS=500_000
torch.set_num_threads(1)


def checksum(path,algo="sha256"):
    h=hashlib.new(algo)
    with open(path,"rb") as f:
        for chunk in iter(lambda:f.read(1<<20),b""):h.update(chunk)
    return h.hexdigest()


def save(path,obj):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(obj,sort_keys=True,indent=2)+"\n")


def fetch_source(out):
    private=Path(out)/"source-private-NOT-ARTIFACT"
    private.mkdir(parents=True,exist_ok=True)
    archive=private/"ml-1m.zip"
    if not archive.is_file() or checksum(archive,"md5")!=SOURCE_MD5:
        tmp=archive.with_suffix(".part")
        tmp.unlink(missing_ok=True)
        req=urllib.request.Request(SOURCE,headers={"User-Agent":"TACOSM-real-memory-research-ME051"})
        try:
            with urllib.request.urlopen(req,timeout=150) as input_stream,tmp.open("wb") as output:
                while True:
                    chunk=input_stream.read(1<<20)
                    if not chunk:break
                    output.write(chunk)
            if checksum(tmp,"md5")!=SOURCE_MD5:raise RuntimeError("immutable MovieLens ZIP MD5 failure")
            tmp.replace(archive)
        finally:tmp.unlink(missing_ok=True)
    with zipfile.ZipFile(archive) as z:
        if "ml-1m/ratings.dat" not in z.namelist():raise RuntimeError("original ratings.dat missing")
        raw=z.read("ml-1m/ratings.dat")
    if hashlib.md5(raw).hexdigest()!=DATA_MD5:
        raise RuntimeError("MovieLens original ratings.dat integrity failure")
    # Publisher checksum document is not a training label and may be read freely.
    with urllib.request.urlopen(SOURCE_MD5_URL,timeout=45) as resp:
        receipt=resp.read(300).decode("ascii",errors="ignore").lower()
    if not re.search(r"(?<![0-9a-f])"+SOURCE_MD5+r"(?![0-9a-f])",receipt):
        raise RuntimeError("publisher MD5 checksum no longer matches frozen registered source")
    return archive,hashlib.sha256(raw).hexdigest()


def lines(archive):
    with zipfile.ZipFile(archive) as z:
        with z.open("ml-1m/ratings.dat") as f:
            for i,raw in enumerate(f):
                parts=raw.rstrip(b"\r\n").split(b"::")
                if len(parts)!=4:raise RuntimeError("invalid publisher record format")
                yield i,parts


def build_metadata(archive):
    """Do not convert or inspect any outcome/ratings fields in this pass."""
    sequences=defaultdict(list)
    for row,parts in lines(archive):
        user,movie,timestamp=int(parts[0]),int(parts[1]),int(parts[3])
        if not(1<=user<=USERS and 1<=movie<=3952 and timestamp>0):
            raise RuntimeError("invalid authentic rating metadata")
        sequences[user].append((timestamp,row,movie))
    count=sum(len(s) for s in sequences.values())
    if count!=RECORDS or len(sequences)!=USERS:
        raise RuntimeError("source rating or user counts mismatch")
    sequences={user:sorted(rows,key=lambda e:(e[0],e[1]))
               for user,rows in sequences.items()}
    assignment=[None]*count
    state={}
    for user,rows in sequences.items():
        n=len(rows)
        if n<20:raise RuntimeError("original per-user history count invalid")
        fit=int(.70*n);dev=int(.85*n)
        stars=np.full(n,-1.,dtype=np.float32)
        timestamps=np.array([r[0] for r in rows],dtype=np.int64)
        movies=np.array([r[2] for r in rows],dtype=np.int64)
        eligible=np.searchsorted(timestamps,timestamps,side="left")
        state[user]={"movie":movies,"time":timestamps,"rating":stars,
                     "prior":eligible,"fit":fit,"dev":dev}
        for index,(_,original,_) in enumerate(rows):
            assignment[original]=(user,index)
    return state,assignment


def apply_ratings(archive,state,assignment,stage):
    """Stage FIT ignores every DEV and TEST rating field; FINAL reveals them."""
    nread=0
    for row,parts in lines(archive):
        user,index=assignment[row]
        if stage=="FIT" and index>=state[user]["fit"]:continue
        value=float(parts[2])
        if value not in (1.,2.,3.,4.,5.):raise RuntimeError("real rating label invalid")
        old=state[user]["rating"][index]
        if old!=-1. and old!=value:raise RuntimeError("FIT factual ratings changed")
        state[user]["rating"][index]=value
        nread+=1
    if stage=="FIT" and nread<MIN_FIT_ROWS:
        raise RuntimeError("unexpectedly few genuine FIT rating observations")
    if stage=="FINAL" and nread!=RECORDS:
        raise RuntimeError("missing original future ratings")
    return nread


def eligible(state,split,length):
    result=[]
    for user,entry in state.items():
        lower=0 if split=="FIT" else entry["fit"] if split=="DEV" else entry["dev"]
        upper=entry["fit"] if split=="FIT" else entry["dev"] if split=="DEV" else len(entry["movie"])
        for i in range(lower,upper):
            if int(entry["prior"][i])>=length:
                result.append((user,i))
    return result


def sample_pairs(pairs,limit,seed):
    if len(pairs)<=limit:return sorted(pairs)
    rng=np.random.default_rng(seed)
    chosen=rng.choice(len(pairs),size=limit,replace=False)
    return sorted(pairs[int(i)] for i in chosen)


def make_batch(state,pairs,horizon):
    """Construct strictly prior, real H-event inputs; left-pad other 256 slots."""
    b=len(pairs)
    movie=np.zeros((b,256),np.int64)
    rating=np.zeros((b,256),np.float32)
    gap=np.zeros((b,256),np.float32)
    valid=np.zeros((b,256),np.float32)
    candidate=np.zeros(b,np.int64)
    target=np.zeros(b,np.float32)
    for k,(uid,index) in enumerate(pairs):
        entry=state[uid];until=int(entry["prior"][index])
        if until<horizon:raise RuntimeError("missing strictly prior genuine events")
        start=until-horizon
        older=entry["rating"][start:until]
        y=float(entry["rating"][index])
        if (older<=0).any() or y<=0:
            raise RuntimeError("forbidden future or unknown actual historical rating")
        movie[k,-horizon:]=entry["movie"][start:until]
        rating[k,-horizon:]=(older-3.)/2.
        gap_seconds=np.maximum(0,(entry["time"][index]-entry["time"][start:until])/3600.)
        gap[k,-horizon:]=np.minimum(np.log1p(gap_seconds)/12.,2.)
        valid[k,-horizon:]=1.
        candidate[k]=int(entry["movie"][index]);target[k]=y
    return {"movie":torch.from_numpy(movie),"rating":torch.from_numpy(rating),
            "gap":torch.from_numpy(gap),"valid":torch.from_numpy(valid),
            "candidate":torch.from_numpy(candidate),"target":torch.from_numpy(target)}


def fit(state,pools,arm,seed,path,source_sha):
    path.mkdir(parents=True,exist_ok=False)
    torch.manual_seed(seed)
    model=RealRetentionPredictor(arm)
    optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
    generator=np.random.default_rng(seed+2000)
    old=arm.endswith("_old")
    sampler_digest=hashlib.sha256()
    t_cpu=0.;trace=[]
    for step in range(1,STEPS+1):
        H=HORIZONS[(step-1)%len(HORIZONS)]
        pool=pools[H]
        ids=generator.integers(0,len(pool),size=BATCH)
        sampler_digest.update(np.asarray(ids,dtype=np.int64).tobytes())
        sample=[pool[int(i)] for i in ids]
        batch=make_batch(state,sample,H)
        t=time.process_time();model.train();optimizer.zero_grad(set_to_none=True)
        loss,primary,retrieval=genuine_rating_loss(model,batch,oldest=old)
        if not torch.isfinite(loss):raise RuntimeError("invalid genuine human preference loss")
        loss.backward()
        norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        if not torch.isfinite(norm):raise RuntimeError("nonfinite real long-history gradients")
        optimizer.step()
        t_cpu+=time.process_time()-t
        if step%100==0:
            trace.append({"step":step,"real_rating_primary_loss":float(primary),
                          "factual_past_retrieval_loss":float(retrieval),
                          "optimizer_CPU_seconds":t_cpu})
    cp=path/"original-human-memory-model.pt"
    torch.save({"arm":arm,"seed":seed,"weights":model.state_dict(),
                "optimizer":optimizer.state_dict(),"torch_rng":torch.get_rng_state(),
                "sample_generator_state":generator.bit_generator.state,
                "source_SHA256":source_sha,"steps":STEPS},cp)
    save(path/"training-trace.json",trace)
    record={"arm":arm,"seed":seed,"parameters":model.num_active(),
            "optimizer_CPU_seconds":t_cpu,"real_observed_examples_presented":STEPS*BATCH,
            "real_training_updates":STEPS,"sampling_schedule_SHA256":sampler_digest.hexdigest(),
            "checkpoint":cp.name,"checkpoint_SHA256":checksum(cp),
            "test_rating_parsed_before_fit":False}
    print(json.dumps({"event":"ME051_AUTHENTIC_PREFERENCE_MODEL_FIT",
                      "arm":arm,"seed":seed,"CPU":round(t_cpu,2),
                      "parameters":record["parameters"]}),flush=True)
    return record


@torch.no_grad()
def outputs(model,state,pairs,length):
    model.eval()
    predictions=[];retrievals=[];actual=[];older=[];recent=[];users=[]
    for start in range(0,len(pairs),128):
        selected=pairs[start:start+128]
        batch=make_batch(state,selected,length)
        p,r=model(batch["movie"],batch["rating"],batch["gap"],
                  batch["valid"],batch["candidate"])
        predictions.append(p.numpy());retrievals.append(r.numpy())
        actual.append(batch["target"].numpy())
        older.append(real_quarter_target(batch["rating"],batch["valid"],True).numpy())
        recent.append(real_quarter_target(batch["rating"],batch["valid"],False).numpy())
        users.extend(uid for uid,_ in selected)
    return (np.concatenate(predictions),np.concatenate(retrievals),
            np.concatenate(actual),np.concatenate(older),
            np.concatenate(recent),np.asarray(users,np.int32))


def metric(pred,target):
    return {"MAE":float(np.mean(np.abs(pred-target))),
            "RMSE":float(np.sqrt(np.mean((pred-target)**2)))}


def score(root,state,pairs,rows,part):
    report={};predictions={}
    for rec in rows:
        folder=root/f'{rec["arm"]}-s{rec["seed"]}'
        cp=folder/rec["checkpoint"]
        if checksum(cp)!=rec["checkpoint_SHA256"]:raise RuntimeError("trained model SHA changed")
        saved=torch.load(cp,weights_only=True)
        net=RealRetentionPredictor(rec["arm"])
        net.load_state_dict(saved["weights"],strict=True)
        results={}
        for length in HORIZONS:
            pred,recalled,y,old,recent,users=outputs(net,state,pairs,length)
            # Save predictions ONLY, not individual observed ratings, user IDs or histories.
            name=f"{part}-H{length}-original-predictions.npz"
            np.savez_compressed(folder/name,prediction=pred,retrieval=recalled)
            results[str(length)]={**metric(pred,y),
                "retrieval_old_MAE":metric(recalled,old)["MAE"],
                "retrieval_recent_MAE":metric(recalled,recent)["MAE"],
                "original_prediction_file":name,
                "original_prediction_SHA256":checksum(folder/name)}
            if part=="TEST" and length==256:
                predictions[f'{rec["arm"]}-s{rec["seed"]}']=pred
        report[f'{rec["arm"]}-s{rec["seed"]}']=results
    return report,predictions


def paired_user_interval(new,other,labels,users,seed):
    unique,inverse=np.unique(users,return_inverse=True)
    improvement=np.abs(other-labels)-np.abs(new-labels)
    sums=np.bincount(inverse,weights=improvement,minlength=len(unique))
    counts=np.bincount(inverse,minlength=len(unique))
    rng=np.random.default_rng(seed+8001)
    stats=[]
    for _ in range(200):
        ids=rng.integers(len(unique),size=len(unique))
        stats.append(float(sums[ids].sum()/counts[ids].sum()))
    return [float(np.quantile(stats,.025)),float(np.quantile(stats,.975))]


def run(dest):
    root=Path(dest);root.mkdir(parents=True,exist_ok=False)
    receipt={"experiment":"ME051_REAL_LONG_HORIZON_DECISION_RELEVANT_RETENTION",
      "complete":False,"source_url":SOURCE,"registered_publisher_zip_MD5":SOURCE_MD5,
      "research_only_no_redistribution":True,
      "all_future_rating_labels_frozen_until_after_fit":True,
      "retrospective_metadata_based_per_user_split":True,
      "historical_256_inputs_compressed_to_32_updates":True,
      "action_outcomes_causal_claim":False,"continuous_256_step_state_claim":False,
      "no_promotion_or_merge":True,"models":[]}
    save(root/"results.json",receipt)
    source,ratings_sha=fetch_source(root)
    receipt["source_archive_SHA256"]=checksum(source)
    receipt["ratings_dat_SHA256"]=ratings_sha
    state,assignment=build_metadata(source)
    pool={H:eligible(state,"FIT",H) for H in HORIZONS}
    test=sample_pairs(eligible(state,"TEST",256),TEST_LIMIT,51051)
    dev=sample_pairs(eligible(state,"DEV",256),DEV_LIMIT,51052)
    counts={"FIT_eligible":{str(H):len(pool[H]) for H in HORIZONS},
            "DEV_H256":len(dev),"TEST_H256":len(test),
            "TEST_user_count":len({u for u,_ in test})}
    if min(counts["FIT_eligible"].values())<1500 or len(test)<500 or counts["TEST_user_count"]<20:
        raise RuntimeError("too few original human long-history samples")
    receipt["metadata_only_selection_before_outcomes"]=counts
    # Predeclared cohorts assigned without ever opening DEV/TEST rating.
    receipt["test_selection_SHA256"]=hashlib.sha256(np.asarray(test,np.int64).tobytes()).hexdigest()
    receipt["dev_selection_SHA256"]=hashlib.sha256(np.asarray(dev,np.int64).tobytes()).hexdigest()
    save(root/"results.json",receipt)
    receipt["FIT_rating_rows_parsed"]=apply_ratings(source,state,assignment,"FIT")
    save(root/"results.json",receipt)
    for seed in SEEDS:
        for arm in ARMS:
            record=fit(state,pool,arm,seed,root/f"{arm}-s{seed}",receipt["source_archive_SHA256"])
            receipt["models"].append(record)
            save(root/"results.json",receipt)
    if len(receipt["models"])!=len(ARMS)*len(SEEDS):
        raise RuntimeError("original fitted checkpoints incomplete")
    for row in receipt["models"]:
        if checksum(root/f'{row["arm"]}-s{row["seed"]}'/row["checkpoint"])!=row["checkpoint_SHA256"]:
            raise RuntimeError("pre-revelation checkpoint tamper")
    receipt["all_ten_original_checkpoints_frozen_before_future_outcomes"]=True
    save(root/"results.json",receipt)
    # First conversion of future outcome values into numerical labels.
    receipt["all_rating_rows_parsed_after_freeze"]=apply_ratings(source,state,assignment,"FINAL")
    receipt["sealed_test_opened"]=True
    receipt["DEV_results"],_=score(root,state,dev,receipt["models"],"DEV")
    receipt["TEST_results"],pred=score(root,state,test,receipt["models"],"TEST")
    y=np.asarray([state[u]["rating"][i] for u,i in test],dtype=np.float32)
    users=np.asarray([u for u,_ in test],dtype=np.int32)
    receipt["TEST_global_rating_mean_reference_MAE"]=float(np.mean(
        np.abs(y-np.mean([float(r) for entry in state.values()
           for r in entry["rating"][:entry["fit"]]]))))
    gate={};ci={}
    for seed in SEEDS:
        candidate=f"tac_old-s{seed}";recent=f"tac_recent-s{seed}"
        peers=[f"gru_old-s{seed}",f"gru_recent-s{seed}",f"context_recent-s{seed}"]
        ci[str(seed)]={"native_old_vs_recent_MAE_improvement_CI":
            paired_user_interval(pred[candidate],pred[recent],y,users,seed)}
        p=receipt["TEST_results"]
        native_mae=p[candidate]["256"]["MAE"]
        gate[str(seed)]={
            "old_retention_beats_identical_active_native_recent":
                native_mae<p[recent]["256"]["MAE"] and
                ci[str(seed)]["native_old_vs_recent_MAE_improvement_CI"][0]>0,
            "native_old_beats_gru_old_and_context":
                all(native_mae<p[other]["256"]["MAE"] for other in peers)}
    receipt["paired_user_cluster_intervals"]=ci
    receipt["registered_research_gates"]=gate
    receipt["intelligence_capability_gate_pass"]=all(all(v.values()) for v in gate.values())
    receipt["cross_architecture_full_parameter_and_CPU_parity"]=False
    receipt["complete"]=True
    save(root/"results.json",receipt)
    print(json.dumps({"event":"ME051_REAL_LONG_MEMORY_RESULTS",
        "h256_test_MAE":{k:r["256"]["MAE"] for k,r in receipt["TEST_results"].items()},
        "horizon_tests":counts,"gates":gate,
        "capability_qualified":receipt["intelligence_capability_gate_pass"]}),flush=True)


@torch.no_grad()
def replay(root):
    root=Path(root);r=json.loads((root/"results.json").read_text())
    if not r["complete"] or not r["all_ten_original_checkpoints_frozen_before_future_outcomes"]:
        raise RuntimeError("not a completed independent original fitting run")
    source,_=fetch_source(root)
    if checksum(source)!=r["source_archive_SHA256"]:raise RuntimeError("source SHA changed")
    state,assignment=build_metadata(source)
    apply_ratings(source,state,assignment,"FINAL")
    for part,limit,seed in (("DEV",DEV_LIMIT,51052),("TEST",TEST_LIMIT,51051)):
        sample=sample_pairs(eligible(state,part,256),limit,seed)
        expected=r["dev_selection_SHA256"] if part=="DEV" else r["test_selection_SHA256"]
        if hashlib.sha256(np.asarray(sample,np.int64).tobytes()).hexdigest()!=expected:
            raise RuntimeError("original sealed evaluation cohort changed")
        for rec in r["models"]:
            folder=root/f'{rec["arm"]}-s{rec["seed"]}'
            cp=folder/rec["checkpoint"]
            if checksum(cp)!=rec["checkpoint_SHA256"]:raise RuntimeError("original weights changed")
            model=RealRetentionPredictor(rec["arm"])
            model.load_state_dict(torch.load(cp,weights_only=True)["weights"],strict=True)
            for H in HORIZONS:
                prediction,old_head,_,_,_,_=outputs(model,state,sample,H)
                details=r[part+"_results"][f'{rec["arm"]}-s{rec["seed"]}'][str(H)]
                file=folder/details["original_prediction_file"]
                if checksum(file)!=details["original_prediction_SHA256"]:
                    raise RuntimeError("original forecast archive changed")
                with np.load(file) as preserved:
                    if not (np.array_equal(prediction,preserved["prediction"]) and
                            np.array_equal(old_head,preserved["retrieval"])):
                        raise RuntimeError("actual genuine rating forecast replay mismatch")
    print("ME051_ALL_TEN_ORIGINAL_HUMAN_MEMORY_MODELS_REPLAY_BIT_EXACT",flush=True)


if __name__=="__main__":
    cli=argparse.ArgumentParser()
    cli.add_argument("--out",default="results/me051")
    cli.add_argument("--replay",action="store_true")
    opt=cli.parse_args()
    if opt.replay:replay(opt.out)
    else:run(opt.out)
