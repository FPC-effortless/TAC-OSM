#!/usr/bin/env python3
"""ME014B: EXACT trainable-parameter native TAC/GRU logged-action replication.

Strict original early-time parsing; original later TEST never interpreted here.
Real logged outcomes only, no simulated task/causal treatment uplift.
"""
from __future__ import annotations
import csv,hashlib,io,json,math,time,urllib.request,zipfile
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

URL="https://zenodo.org/records/18164998/files/KuaiRec.zip"
PIN_MD5="261550d472c48eff4990fb13c0e5bcf7"
FIT_CUT=1596240000.0  # 2020-08-01T00:00:00Z
DEV_CUT=1597363200.0  # 2020-08-14T00:00:00Z
END_CUT=1599350400.0  # 2020-09-06T00:00:00Z; later TEST only
HISTORY=32;VOCAB=16000;EMB=8
SEEDS=(14001,14002,14003)
ARMS=("native_tac","gru")
UPDATES=300;BATCH=128
REQUIRED=("user_id","video_id","video_duration","watch_ratio","timestamp")

def publisher_bytes():
    sha=hashlib.sha256();md5=hashlib.md5();parts=[];total=0
    request=urllib.request.Request(URL,headers={"User-Agent":"TAC-OSM-ME014-real-log/1.0"})
    with urllib.request.urlopen(request,timeout=180) as f:
        if getattr(f,"status",200)!=200:raise ValueError("not official downloaded source")
        while (part:=f.read(2**20)):
            sha.update(part);md5.update(part);total+=len(part)
            if total>550_000_000:raise ValueError("KuaiRec unexpectedly oversized")
            parts.append(part)
    if md5.hexdigest()!=PIN_MD5:raise ValueError("published original ZIP MD5 mismatch")
    return b"".join(parts),{"url":URL,"zip_md5":md5.hexdigest(),
          "zip_sha256":sha.hexdigest(),"zip_bytes":total}

def member_for_zip(z):
    matches=[n for n in z.namelist() if Path(n).name=="big_matrix.csv"]
    if len(matches)!=1:raise ValueError("ambiguous KuaiRec big_matrix source")
    return matches[0]

def read_early():
    archive,meta=publisher_bytes()
    users={};stats={"all_csv_rows":0,"cohort_rows":0,"invalid_observation":0,
                    "missing_timestamp":0,"later_rows_skipped_before_label_parsing":0,
                    "cohort_users":0}
    with zipfile.ZipFile(io.BytesIO(archive)) as z:
        member=member_for_zip(z)
        with z.open(member) as raw:
            reader=csv.DictReader(io.TextIOWrapper(raw,encoding="utf-8-sig",newline=""))
            if not set(REQUIRED).issubset(set(reader.fieldnames or [])):
                raise ValueError("publisher reaction log schema changed")
            for r in reader:
                stats["all_csv_rows"]+=1
                try:uid=int(r["user_id"])
                except (ValueError,TypeError):continue
                if uid<0 or uid%16!=0:continue
                try:timestamp=float(r["timestamp"])
                except (ValueError,TypeError):stats["missing_timestamp"]+=1;continue
                if not math.isfinite(timestamp) or timestamp<=1e9:
                    stats["missing_timestamp"]+=1;continue
                if timestamp>=DEV_CUT:
                    stats["later_rows_skipped_before_label_parsing"]+=1
                    continue
                try:
                    item=int(r["video_id"])
                    duration=float(r["video_duration"])
                    ratio=float(r["watch_ratio"])
                except (ValueError,TypeError):
                    stats["invalid_observation"]+=1;continue
                if not (0<=item<VOCAB and math.isfinite(duration) and duration>0
                        and math.isfinite(ratio) and ratio>=0):
                    stats["invalid_observation"]+=1;continue
                users.setdefault(uid,[]).append((timestamp,item,math.log1p(duration)/12.0,
                                                 int(ratio>=1.0)))
                stats["cohort_rows"]+=1
        stats["source_member"]=member
    for v in users.values():v.sort(key=lambda x:x[0])
    stats["cohort_users"]=len(users)
    meta.update(stats)
    meta["max_exposed_timestamp_parsed"]=str(datetime.fromtimestamp(
        max(t for rows in users.values() for t,_,_,_ in rows),tz=timezone.utc))
    meta["future_test_labels_parsed"]=False
    meta["cohort_definition"]="user_id % 16 == 0; no outcome conditioning"
    return users,meta

def build_examples(users,begin,end,max_windows=110000):
    seq=[]
    for uid,records in sorted(users.items()):
        for i in range(HISTORY,len(records)):
            ts,item,duration,y=records[i]
            if not begin<=ts<end:continue
            prior=records[i-HISTORY:i]
            # No current y appears in inputs: only the 32 preceding item+feedbacks.
            items=[row[1] for row in prior]
            prior_y=[row[3] for row in prior]
            meta=[[row[2],row[3],min(8.0,max(0.,math.log1p(max(
                  0.0,(row[0]-prior[j-1][0]) if j else 0.0)/3600)))/8.] for j,row in enumerate(prior)]
            seq.append((uid,ts,items,meta,item,duration,y,float(sum(prior_y))/HISTORY))
    if not seq:raise ValueError("no legitimate causal windows")
    # Deterministic source-only striding; no future score/model selection.
    stride=max(1,math.ceil(len(seq)/max_windows))
    seq=seq[::stride]
    hist_items=torch.tensor([p[2] for p in seq],dtype=torch.long)
    hist_meta=torch.tensor([p[3] for p in seq],dtype=torch.float32)
    current=torch.tensor([p[4] for p in seq],dtype=torch.long)
    duration=torch.tensor([[p[5]] for p in seq],dtype=torch.float32)
    target=torch.tensor([[p[6]] for p in seq],dtype=torch.float32)
    running=torch.tensor([[p[7]] for p in seq],dtype=torch.float32)
    ids=[p[0] for p in seq]
    return {"items":hist_items,"metadata":hist_meta,"action":current,
        "duration":duration,"y":target,"running":running,"users":ids,
        "timestamps":[p[1] for p in seq],"rows":len(seq),"stride":stride}

class ActionMemory(nn.Module):
    def __init__(self,arm):
        super().__init__()
        if arm not in ARMS:raise ValueError("unknown recurrent model")
        self.arm=arm
        self.emb=nn.Embedding(VOCAB,EMB)
        self.dim=42 if arm=="gru" else 32
        if arm=="gru":
            self.recur=nn.GRU(EMB+3,42,batch_first=True)
            self.head=nn.Linear(42+EMB+1,1)
            # 16->4 action-conditioned nonlinear residual = 68 ACTIVE params.
            # Matching the native TAC 135050 trainables exactly, not dead padding.
            self.action_refine=nn.Linear(EMB+8,4)
            nn.init.zeros_(self.action_refine.weight)
            nn.init.zeros_(self.action_refine.bias)
        else:
            self.enc=nn.Linear(EMB+3,32)
            self.transition=nn.Linear(32,32,bias=False)
            self.gate=nn.Linear(EMB+3+32,32)
            nn.init.constant_(self.gate.bias,-2.5)
            self.slow=nn.Linear(64,32)
            self.slow_gate=nn.Linear(64,32)
            nn.init.constant_(self.slow_gate.bias,-4)
            self.head=nn.Linear(64+EMB+1,1)
        nn.init.zeros_(self.head.weight);nn.init.zeros_(self.head.bias)

    def forward(self,items,metadata,action,duration,ignore_history=False):
        if items.ndim!=2 or items.shape[1]!=HISTORY or metadata.shape!=(len(items),HISTORY,3):
            raise ValueError("expected actual 32-event historical records")
        if action.shape!=(len(items),) or duration.shape!=(len(items),1):
            raise ValueError("current action shape")
        if int(items.min())<0 or int(items.max())>=VOCAB or int(action.min())<0 or int(action.max())>=VOCAB:
            raise ValueError("unregistered action item")
        x=torch.cat([self.emb(items),metadata],dim=-1)
        if ignore_history:x=torch.zeros_like(x)
        if self.arm=="gru":
            state=self.recur(x)[0][:,-1]
        else:
            z=x.new_zeros(len(x),32);s=z.clone()
            for t in range(HISTORY):
                u=x[:,t];proposal=torch.tanh(self.enc(u)+self.transition(z))
                g=torch.sigmoid(self.gate(torch.cat((u,z),dim=-1)))
                z=(1-g)*z+g*proposal
                context=torch.cat((z,s),dim=-1)
                gs=torch.sigmoid(self.slow_gate(context))
                s=(1-gs)*s+gs*torch.tanh(self.slow(context))
            state=torch.cat([z,s],dim=-1)
        logit=self.head(torch.cat((state,self.emb(action),duration),dim=-1))
        if self.arm=="gru":
            # Different but active nonlinear post-readout capability; never a
            # simulation-only parameter or unused dummy capacity.
            refine_in=torch.cat((state[:,:8],self.emb(action)),dim=-1)
            logit=logit+torch.tanh(self.action_refine(refine_in)).mean(-1,keepdim=True)
        return logit

def parameters(model):return sum(p.numel() for p in model.parameters())
def batch(dataset,indices):
    return (dataset["items"][indices],dataset["metadata"][indices],
            dataset["action"][indices],dataset["duration"][indices])

def empirical_baselines(train,hold):
    total=float(train["y"].mean())
    # FIT-only Laplace smoothing and no dev responses in item means.
    counts=np.bincount(train["action"].numpy(),minlength=VOCAB)
    positives=np.bincount(train["action"].numpy(),
                          weights=train["y"].reshape(-1).numpy(),minlength=VOCAB)
    item=(positives+8*total)/(counts+8)
    return {"global_event_rate":total,"video_fit_counts":int((counts>0).sum()),
      "global_predictions":np.full(len(hold["y"]),total,dtype=np.float32),
      "item_predictions":item[hold["action"].numpy()].astype("float32"),
      "history_mean_predictions":hold["running"].reshape(-1).numpy()}

def binary_scores(prob,truth):
    p=np.asarray(prob,dtype=np.float64).reshape(-1)
    y=np.asarray(truth,dtype=np.float64).reshape(-1)
    if p.shape!=y.shape or not np.isfinite(p).all():raise ValueError("bad probability")
    p=p.clip(1e-6,1-1e-6)
    return {"n":len(p),"full_watch_prevalence":float(y.mean()),
       "brier":float(np.mean((p-y)**2)),
       "binary_logloss":float(-(y*np.log(p)+(1-y)*np.log(1-p)).mean()),
       "predicted_frequency":float(p.mean())}

def evaluate(model,hold,limit=None,ignore_history=False):
    model.eval()
    scores=[]
    with torch.inference_mode():
        n=len(hold["y"]) if limit is None else min(len(hold["y"]),limit)
        for first in range(0,n,512):
            idx=slice(first,min(first+512,n))
            logits=model(*batch(hold,idx),ignore_history=ignore_history)
            scores.append(torch.sigmoid(logits).reshape(-1))
    return torch.cat(scores).numpy()

def fit(train,dev,out,src):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    manifest={"study":"ME014B_EXACT_PARAMETER_REAL_LOGGED_ACTION_MEMORY",
              "source":src,
              "model_provenance":"genuine actual video exposures and full-watch outcomes",
              "causal_effect_estimated":False,"independent_reward_verification":False,
              "train_examples":len(train["y"]),"dev_examples":len(dev["y"]),
              "original_fit_steps":UPDATES,"fits":[],"future_test_parsed":False,
              "provisional_only":True}
    empirical=empirical_baselines(train,dev)
    manifest["train_global_event_rate"]=empirical["global_event_rate"]
    manifest["historic_dev_baselines"]={
      name:binary_scores(empirical[k],dev["y"].numpy()) for name,k in
      [("global_rate","global_predictions"),("item_rate","item_predictions"),
       ("prior32_rate","history_mean_predictions")]}
    manifest["baseline_item_statistics"]=None # never read later test item labels here
    torch.set_num_threads(1)
    for seed in SEEDS:
        for arm in ARMS:
            torch.manual_seed(seed)
            model=ActionMemory(arm)
            p=parameters(model)
            opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=1e-4)
            gen=torch.Generator().manual_seed(seed+300)
            clock=time.perf_counter();trace=[]
            for step in range(1,UPDATES+1):
                ids=torch.randint(len(train["y"]),(BATCH,),generator=gen)
                logits=model(*batch(train,ids))
                loss=F.binary_cross_entropy_with_logits(logits,train["y"][ids])
                if not torch.isfinite(loss):raise ArithmeticError("nonfinite loss")
                opt.zero_grad(set_to_none=True);loss.backward()
                norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1)
                if not torch.isfinite(norm):raise ArithmeticError("nonfinite grad")
                opt.step()
                if step in (1,UPDATES//2,UPDATES):
                    trace.append({"step":step,"train_bce":float(loss.detach())})
            historic=binary_scores(evaluate(model,dev),dev["y"].numpy())
            ablated=binary_scores(evaluate(model,dev,ignore_history=True),dev["y"].numpy())
            path=out/"checkpoints"/f"{arm}-{seed}.pt";path.parent.mkdir(parents=True,exist_ok=True)
            torch.save({"arm":arm,"seed":seed,"steps":UPDATES,
                        "data_sha256":src["zip_sha256"],"state_dict":model.state_dict()},path)
            record={"arm":arm,"seed":seed,"parameters":p,
              "train_wall_seconds":round(time.perf_counter()-clock,3),
              "historic_dev":historic,"history_blanked_dev":ablated,
              "loss_curve":trace,"checkpoint":str(path.relative_to(out)),
              "sha256":hashlib.sha256(path.read_bytes()).hexdigest()}
            manifest["fits"].append(record)
            print(json.dumps({"arm":arm,"seed":seed,"dev_brier":historic["brier"],
             "blank_history_brier":ablated["brier"],"params":p},sort_keys=True),flush=True)
    pairs={s:{r["arm"]:r["parameters"] for r in manifest["fits"] if r["seed"]==s} for s in SEEDS}
    delta=max(abs(v["native_tac"]-v["gru"])/min(v.values()) for v in pairs.values())
    manifest["parameter_mismatch_fraction_max"]=delta
    if delta!=0:raise ValueError("compulsory fairness gate: parameter counts not EXACT")
    (out/"results.json").write_text(json.dumps(manifest,sort_keys=True,indent=2)+"\n")
    print(json.dumps({"status":"ME014B_EXACT_REAL_EARLY_HISTORY_FITS_DONE",
      "fits":len(manifest["fits"]),"future_labels_read":False,
      "parameters_mismatch_fraction":delta}),flush=True)

def run(out):
    users,src=read_early()
    train=build_examples(users,0,FIT_CUT)
    dev=build_examples(users,FIT_CUT,DEV_CUT)
    fit(train,dev,out,src)

def preflight():
    torch.set_num_threads(1);torch.manual_seed(20)
    history=torch.randint(VOCAB,(3,HISTORY))
    observed=torch.rand(3,HISTORY,3)
    action=torch.randint(VOCAB,(3,))
    dur=torch.rand(3,1)
    counts={}
    for arm in ARMS:
        model=ActionMemory(arm)
        counts[arm]=parameters(model)
        logits=model(history,observed,action,dur)
        assert logits.shape==(3,1) and bool(torch.isfinite(logits).all())
        # Simulated tensors here are ONLY software probes, not evidence of model value.
        assert torch.all(logits==0)
        assert torch.all(model(history,observed,action,dur,ignore_history=True)==0)
        F.binary_cross_entropy_with_logits(logits,torch.rand(3,1)).backward()
        assert model.head.weight.grad.abs().sum()>0
    mismatch=abs(counts["native_tac"]-counts["gru"])/min(counts.values())
    assert mismatch==0.0,(counts,mismatch)
    # Constructed chronological history excludes current response by schema.
    assert len(REQUIRED)==5 and HISTORY==32
    print(json.dumps({"status":"ME014B_EXACT_SOFTWARE_ONLY_PASS","parameter_counts":counts,
                      "mismatch_fraction":mismatch}))

if __name__=="__main__":
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument("--out",default="results/me014b")
    parser.add_argument("--self-test",action="store_true")
    a=parser.parse_args()
    preflight() if a.self_test else run(a.out)
