#!/usr/bin/env python3
"""BENCH-FM-002-B: real mixed tabular UCI Bank outcomes, TAC adapter vs TabFM."""
from __future__ import annotations
import argparse,csv,hashlib,io,json,math,time,urllib.request,zipfile
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score,average_precision_score,log_loss,brier_score_loss

URL="https://archive.ics.uci.edu/static/public/222/bank+marketing.zip"
OUTER_SHA="e0bf5f5de5b846e2f18e9d90606637267d46dfa260e0f17bb12e605db5efbeb4"
LABEL="y";NUMERIC=("age","balance","day","campaign","pdays","previous")
# Drop call duration: this is known only after a call, hence not a pre-contact action feature.
NAMES=("age","job","marital","education","default","balance","housing","loan","contact","day",
       "month","campaign","pdays","previous","poutcome")
CATS=tuple(n for n in NAMES if n not in NUMERIC)
N_TRAIN=256;N_TEST=96;SEEDS=(31001,31002);UPDATES=400

def fetch_source():
    req=urllib.request.Request(URL,headers={"User-Agent":"TAC-OSM-BENCH-FM-002"})
    with urllib.request.urlopen(req,timeout=120) as stream:raw=stream.read(5_000_001)
    sha=hashlib.sha256(raw).hexdigest()
    if sha!=OUTER_SHA:raise ValueError("Original UCI official Bank Marketing ZIP SHA mismatch "+sha)
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        files=[v for v in z.namelist() if v.endswith("bank.zip")]
        if len(files)!=1:raise ValueError("Ambiguous official inner bank.zip")
        with zipfile.ZipFile(io.BytesIO(z.read(files[0]))) as zz:
            raw_csv=zz.read("bank-full.csv")
    df=pd.read_csv(io.BytesIO(raw_csv),sep=";",keep_default_na=False)
    if not (len(df)==45211 and set(NAMES+(LABEL,"duration")).issubset(df.columns)):
        raise ValueError("Bank Marketing source contract invalid")
    return df,{"official_url":URL,"official_zip_sha256":sha,
      "inner_bank_full_csv_sha256":hashlib.sha256(raw_csv).hexdigest(),
      "source_rows":len(df),"usable_columns":list(NAMES),"excluded_post_contact_duration":True}

def splits():
    frame,source=fetch_source()
    rng=np.random.default_rng(20261021)
    trix=np.sort(rng.choice(np.arange(int(.7*len(frame))),size=N_TRAIN,replace=False))
    teix=np.sort(rng.choice(np.arange(int(.8*len(frame)),len(frame)),size=N_TEST,replace=False))
    Xtr=frame.iloc[trix][list(NAMES)].copy().reset_index(drop=True)
    Xte=frame.iloc[teix][list(NAMES)].copy().reset_index(drop=True)
    ytr=(frame.iloc[trix].y=="yes").astype("int64").to_numpy()
    yte=(frame.iloc[teix].y=="yes").astype("int64").to_numpy()
    if len(np.unique(ytr))!=2 or len(np.unique(yte))!=2:raise ValueError("Need both authentic outcome classes")
    source.update({"train_n":N_TRAIN,"test_n":N_TEST,
      "train_positive":int(ytr.sum()),"test_positive":int(yte.sum()),
      "train_index_digest":hashlib.sha256(trix.tobytes()).hexdigest(),
      "test_index_digest":hashlib.sha256(teix.tobytes()).hexdigest(),
      "chronology":"rows 0:70percent for TRAIN; 80:100percent for later TEST; no user IDs available"})
    for c in NUMERIC:
        Xtr[c]=pd.to_numeric(Xtr[c],errors="raise")
        Xte[c]=pd.to_numeric(Xte[c],errors="raise")
    return Xtr,ytr,Xte,yte,source

def metrics(prob,y):
    p=np.asarray(prob,dtype=np.float64).reshape(-1)
    if len(p)!=len(y) or not np.isfinite(p).all():raise ValueError("wrong prediction probabilities")
    p=np.clip(p,1e-6,1-1e-6)
    return {"n":int(len(p)),"Brier":float(brier_score_loss(y,p)),
       "logloss":float(log_loss(y,p,labels=[0,1])),
       "ROC_AUC":float(roc_auc_score(y,p)),
       "PR_AUC":float(average_precision_score(y,p)),
       "predict_positive_rate":float(np.mean(p)),
       "observed_positive_rate":float(np.mean(y))}

def representation(tr,te):
    # Data-only encoding, fit exclusively on the original 256 training rows.
    vals=np.zeros((len(tr),len(NAMES)),np.float32)
    testvals=np.zeros((len(te),len(NAMES)),np.float32)
    ids=np.zeros((len(tr),len(NAMES)),np.int64)
    testids=np.zeros((len(te),len(NAMES)),np.int64)
    offset=1
    for j,key in enumerate(NAMES):
        if key in NUMERIC:
            x=tr[key].to_numpy(dtype=np.float64);t=te[key].to_numpy(dtype=np.float64)
            mean=x.mean();std=max(x.std(),1e-3)
            vals[:,j]=np.clip((x-mean)/std,-10,10)
            testvals[:,j]=np.clip((t-mean)/std,-10,10)
        else:
            categories={value:k+offset for k,value in enumerate(sorted(tr[key].unique()))}
            ids[:,j]=tr[key].map(categories).to_numpy(dtype=np.int64)
            testids[:,j]=te[key].map(categories).fillna(0).to_numpy(dtype=np.int64)
            offset+=len(categories)
    return tuple(torch.from_numpy(a) for a in (vals,ids,testvals,testids)),offset

class FeatureRecurrentTAC(nn.Module):
    """Two-timescale gated TAC-style memory over fixed feature sequence, NOT a temporal TAC pretrain."""
    def __init__(self,n_categories):
        super().__init__()
        self.feat_embed=nn.Embedding(len(NAMES),32)
        self.cat_embed=nn.Embedding(n_categories+1,32)
        self.num_factor=nn.Parameter(torch.randn(len(NAMES),32)*.03)
        self.fast=nn.Linear(64,32);self.fast_gate=nn.Linear(64,32)
        self.slow=nn.Linear(64,32);self.slow_gate=nn.Linear(64,32)
        self.head=nn.Sequential(nn.Linear(64,32),nn.SiLU(),nn.Linear(32,1))
        nn.init.constant_(self.fast_gate.bias,-1.)
        nn.init.constant_(self.slow_gate.bias,-2.)
    def forward(self,vals,ids):
        b=vals.shape[0];f=vals.new_zeros((b,32));s=f.clone()
        for j in range(len(NAMES)):
            emb=self.cat_embed(ids[:,j])+self.feat_embed.weight[j]+vals[:,j:j+1]*self.num_factor[j]
            d=torch.cat((emb,f),-1)
            f=f+torch.sigmoid(self.fast_gate(d))*(torch.tanh(self.fast(d))-f)
            ds=torch.cat((f,s),-1)
            s=s+torch.sigmoid(self.slow_gate(ds))*(torch.tanh(self.slow(ds))-s)
        return self.head(torch.cat((f,s),-1)).flatten()

def write(path,obj):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    Path(path).write_text(json.dumps(obj,indent=2,sort_keys=True)+"\n")

def train(out):
    torch.set_num_threads(1)
    tr,y,trtest,yy,meta=splits()
    (v,idv,vtest,idtest),ncat=representation(tr,trtest)
    result={"study":"BENCH-FM-002-B","status":"real_labeled_task_train","source":meta,
       "models":{},"native_tac_pretrained":False,"tac_is_feature_recurrence_adapter":True,
       "tabfm_weights_not_loaded_in_this_phase":True}
    result["models"]["constant"]=metrics(np.full(len(yy),y.mean()),yy)
    ct=ColumnTransformer([("numeric",StandardScaler(),list(NUMERIC)),
                          ("category",OneHotEncoder(handle_unknown="ignore",sparse_output=False),list(CATS))])
    base_x=ct.fit_transform(tr);test_x=ct.transform(trtest)
    logistic=LogisticRegression(max_iter=800,C=1.,random_state=42)
    logistic.fit(base_x,y)
    result["models"]["logistic"]=metrics(logistic.predict_proba(test_x)[:,1],yy)
    boost=HistGradientBoostingClassifier(max_iter=100,learning_rate=.08,max_leaf_nodes=8,
                                         l2_regularization=1.,random_state=42)
    boost.fit(base_x,y)
    result["models"]["hist_boost"]=metrics(boost.predict_proba(test_x)[:,1],yy)
    for seed in SEEDS:
        torch.manual_seed(seed)
        net=FeatureRecurrentTAC(ncat)
        optimizer=torch.optim.AdamW(net.parameters(),lr=.003,weight_decay=1e-4)
        gen=torch.Generator().manual_seed(seed+999)
        truth=torch.from_numpy(y.astype(np.float32))
        started=time.process_time()
        for step in range(UPDATES):
            ix=torch.randint(len(y),(64,),generator=gen)
            outlog=net(v[ix],idv[ix]);loss=F.binary_cross_entropy_with_logits(outlog,truth[ix])
            if not torch.isfinite(loss):raise ArithmeticError("Nonfinite genuine supervised classification")
            optimizer.zero_grad(set_to_none=True);loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(),1.);optimizer.step()
        net.eval()
        with torch.inference_mode():p=net(vtest,idtest).sigmoid().numpy()
        key=f"TAC_feature_recurrence_seed{seed}"
        ck=out/(key+".pt")
        torch.save({"weights":net.state_dict(),"seed":seed,"total_categories":ncat,"steps":UPDATES},ck)
        result["models"][key]={**metrics(p,yy),"train_updates":UPDATES,
             "train_cpu_seconds":time.process_time()-started,
             "n_parameters":sum(x.numel() for x in net.parameters()),
             "checkpoint_sha256":hashlib.sha256(ck.read_bytes()).hexdigest()}
        print(json.dumps({"model":key,"metrics":result["models"][key]}),flush=True)
    write(out/"tabular-baselines.json",result)
    print(json.dumps({"status":"TABULAR_REAL_BASELINES_COMPLETE",
      "train":len(y),"test":len(yy),"positive_train":int(y.sum()),"positive_test":int(yy.sum())}),flush=True)

def google(out):
    import torch
    from tabfm import TabFMClassifier,tabfm_v1_0_0_pytorch
    torch.set_num_threads(2)
    tr,y,te,yy,meta=splits()
    reference=json.loads((out/"tabular-baselines.json").read_text())
    if reference["source"]!=meta:raise ValueError("Original training/test selection or data changed")
    started=time.perf_counter()
    model=tabfm_v1_0_0_pytorch.load(model_type="classification")
    loaded_s=time.perf_counter()-started
    started=time.perf_counter()
    clf=TabFMClassifier(model=model)
    clf.fit(tr,y)
    probabilities=np.asarray(clf.predict_proba(te),dtype=np.float64)
    if probabilities.shape!=(len(yy),2):raise ValueError("TabFM official classifier wrong probabilities")
    result={"model":"google/tabfm-1.0.0-pytorch","pretrained_weights_frozen":True,
      "official_tabfm_v1.1_not_claimed":True,"in_context_rows":N_TRAIN,
      "same_test_rows_as_tac":True,"loader_wall_seconds":loaded_s,
      "predict_wall_seconds":time.perf_counter()-started,
      "source_sha256":meta["official_zip_sha256"],**metrics(probabilities[:,1],yy)}
    write(out/"tabfm10-results.json",result);print(json.dumps(result),flush=True)

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--stage",choices=["train","tabfm"],required=True)
    p.add_argument("--out",type=Path,default=Path("results/bench-fm-002/tabular"))
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    if a.stage=="train":train(a.out)
    else:google(a.out)
