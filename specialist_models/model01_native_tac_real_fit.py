#!/usr/bin/env python3
"""Real UCI374 exposed-FIT native TAC capability candidate; no protected TEST.

100k-parameter actual native fast/slow/associative model, partial parallel
encoder, sequential state execution. DEV only, no architecture superiority.
Original fit snapshots include optimizer, model, RNG and sampler hashes.
"""
from __future__ import annotations
import argparse,base64,gzip,hashlib,io,json,math,os,random,sys
from pathlib import Path
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import uci_appliances_real_dev as uci
import uci_appliances_ridge_dev as ridge

STUDY="MODEL01-TAC-REAL-FIT-001"
SOURCE=ridge.PINNED_UCI374_SHA
HISTORY=48
SEEDS=(1701,1702,1703)
UPDATES=160
BATCH=64
LR=0.001

class StudyError(ValueError):pass

class NativeTAC(nn.Module):
    """Parallel input FFN; bounded recurrent fast32/slow32/delta-matrix8x8."""
    def __init__(self):
        super().__init__()
        self.emb=nn.Linear(3,96)
        self.ff=nn.Sequential(nn.Linear(96,288),nn.GELU(),nn.Linear(288,96))
        self.norm=nn.LayerNorm(96)
        self.fast=nn.GRUCell(96,32)
        self.sgate=nn.Linear(160,32)
        self.sstep=nn.Linear(160,32)
        self.key=nn.Linear(128,8)
        self.value=nn.Linear(128,8)
        self.write=nn.Linear(128,1)
        self.retain=nn.Linear(128,1)
        self.readgate=nn.Linear(128,1)
        self.head=nn.Sequential(nn.Linear(168,128),nn.GELU(),nn.Linear(128,1))

    def forward(self,x):
        if x.ndim!=3 or x.shape[1:]!=(HISTORY,3):
            raise StudyError("expected causal fixed 48-step series")
        z=self.emb(x)
        z=self.norm(z+self.ff(z)) # all positions encoded in parallel
        b=len(x);f=x.new_zeros((b,32));s=x.new_zeros((b,32))
        M=x.new_zeros((b,8,8));k=None
        for t in range(HISTORY):
            h=z[:,t]
            f=self.fast(h,f)
            inp=torch.cat([h,f,s],-1)
            g=torch.sigmoid(self.sgate(inp))
            s=g*s+(1-g)*torch.tanh(self.sstep(inp))
            r=torch.cat([h,f],-1)
            k=F.normalize(self.key(r),p=2,dim=-1,eps=1e-6)
            v=torch.tanh(self.value(r))
            beta=.5*torch.sigmoid(self.write(r))
            alpha=.9+.099*torch.sigmoid(self.retain(r))
            old=torch.bmm(M,k.unsqueeze(-1)).squeeze(-1)
            M=(alpha.unsqueeze(-1)*M+
               beta.unsqueeze(-1)*(v-old).unsqueeze(-1)*k.unsqueeze(1)).clamp(-5,5)
        gate=torch.sigmoid(self.readgate(torch.cat([z[:,-1],f],-1)))
        read=gate*torch.bmm(M,k.unsqueeze(-1)).squeeze(-1)
        return x[:,-1,0]+self.head(torch.cat([z[:,-1],f,s,read],-1)).squeeze(-1)

def params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def windows(y,indices,mean,std):
    ii=np.asarray(list(indices),dtype=np.int32)
    t=ii[:,None]-np.arange(HISTORY-1,-1,-1)[None,:]
    x=np.stack([((y[t]-mean)/std).astype("float32"),
                np.sin(2*np.pi*(t%144)/144).astype("float32"),
                np.cos(2*np.pi*(t%144)/144).astype("float32")],-1)
    return torch.from_numpy(x),torch.from_numpy(((y[ii+uci.H]-mean)/std).astype("float32")),ii

def score(net,data,mean,std,q=None):
    net.eval()
    x,y,ii=data
    with torch.no_grad():
        preds=torch.cat([net(x[a:a+128]).cpu() for a in range(0,len(x),128)])
    actual=y.numpy()*std+mean
    pred=preds.numpy()*std+mean
    err=np.abs(pred-actual)
    if not np.all(np.isfinite(err)):raise StudyError("nonfinite source scores")
    out={"n":len(ii),"mae_wh":float(np.mean(err)),
         "rmse_wh":float(np.sqrt(np.mean((pred-actual)**2)))}
    if q is not None:
        out.update({"nominal90_observed_coverage":float(np.mean(err<=q)),
                    "width90_wh":2*q,
                    "interval_score90":float(np.mean(2*q+20*np.maximum(err-q,0)))})
    return out,err

def fit_seed(seed,fit,cal,dev,mean,std):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    net=NativeTAC();n=params(net)
    if not 95000<=n<=105000:raise StudyError("actual learned parameter budget failure")
    opt=torch.optim.AdamW(net.parameters(),lr=LR,weight_decay=.01)
    x,y,_=fit
    sampler=np.random.default_rng(seed);draws=hashlib.sha256()
    net.train()
    for step in range(UPDATES):
        ids=sampler.integers(0,len(x),size=BATCH)
        draws.update(ids.tobytes())
        prediction=net(x[ids])
        loss=F.smooth_l1_loss(prediction,y[ids],beta=.5)
        if not torch.isfinite(loss):raise StudyError("nonfinite genuine FIT loss")
        opt.zero_grad(set_to_none=True);loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(),1.)
        opt.step()
    cal_score,cal_err=score(net,cal,mean,std)
    q=float(np.sort(cal_err)[math.ceil((len(cal_err)+1)*.9)-1])
    dev_score,_=score(net,dev,mean,std,q)
    buf=io.BytesIO()
    torch.save({"model":net.state_dict(),"optimizer":opt.state_dict(),
                "torch_rng":torch.get_rng_state(),
                "numpy_rng":sampler.bit_generator.state,
                "python_rng":random.getstate(),
                "seed":seed,"steps":UPDATES,
                "sample_order_sha256":draws.hexdigest()},buf)
    raw=buf.getvalue()
    return {"seed":seed,"cal":cal_score,"dev":dev_score,
            "q90_wh":q,"steps":UPDATES,"sample_order_sha256":draws.hexdigest(),
            "checkpoint_sha256":hashlib.sha256(raw).hexdigest(),
            "checkpoint_b64_gzip":base64.b64encode(gzip.compress(raw,mtime=0)).decode()}

def replay_seed(bundle,cal,dev,mean,std):
    raw=gzip.decompress(base64.b64decode(bundle["checkpoint_b64_gzip"],validate=True))
    if hashlib.sha256(raw).hexdigest()!=bundle["checkpoint_sha256"]:
        raise StudyError("original checkpoint SHA mismatch")
    saved=torch.load(io.BytesIO(raw),map_location="cpu",weights_only=True)
    if saved["sample_order_sha256"]!=bundle["sample_order_sha256"] or saved["seed"]!=bundle["seed"]:
        raise StudyError("original sampler/provenance mismatch")
    net=NativeTAC()
    if params(net)!=103252:raise StudyError("native model altered")
    net.load_state_dict(saved["model"],strict=True)
    opt=torch.optim.AdamW(net.parameters(),lr=LR,weight_decay=.01)
    opt.load_state_dict(saved["optimizer"])
    cal_score,cal_err=score(net,cal,mean,std)
    q=float(np.sort(cal_err)[math.ceil((len(cal_err)+1)*.9)-1])
    dev_score,_=score(net,dev,mean,std,q)
    for name,a,b in (("CAL",cal_score,bundle["cal"]),("DEV",dev_score,bundle["dev"])):
        for k in a:
            if not math.isclose(a[k],b[k],rel_tol=1e-6,abs_tol=1e-6):
                raise StudyError("original "+name+" score mismatch "+k)
    if not math.isclose(q,bundle["q90_wh"],rel_tol=1e-6,abs_tol=1e-6):
        raise StudyError("original calibration mismatch")
    return {"seed":bundle["seed"],"original_checkpoint_sha256":bundle["checkpoint_sha256"],
            "CAL_DEV_replayed_within_1e6":True,"fitted_during_replay":False}

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--replay-from",type=Path)
    a=p.parse_args(argv)
    try:
        if a.out.exists():raise StudyError("cannot overwrite source/replay record")
        if not a.replay_from and os.environ.get("MODEL01_REAL_FIT_ENABLE")!="EXPLICIT_CPU_DEV_FIT":
            raise StudyError("explicit CPU real-development FIT gate not enabled")
        source_bytes=uci.fetch_official_zip()
        if uci.digest(source_bytes)!=SOURCE:raise StudyError("original publisher ZIP changed")
        y=np.asarray(uci.original_rows(source_bytes),dtype=np.float64)
        mean=float(y[:uci.FIT_END].mean());std=float(y[:uci.FIT_END].std())
        if std<=0:raise StudyError("invalid FIT source scale")
        fit=windows(y,range(HISTORY-1,uci.FIT_END-uci.H),mean,std)
        cal=windows(y,uci.eligible_indices(uci.FIT_END,uci.CAL_END),mean,std)
        dev=windows(y,uci.eligible_indices(uci.CAL_END,uci.DEV_END),mean,std)
        torch.set_num_threads(2)
        n=params(NativeTAC())
        if n!=103252:raise StudyError("live parameter count mismatch")
        if a.replay_from:
            saved=json.loads(a.replay_from.read_bytes())
            if saved.get("source_zip_sha256")!=SOURCE or saved.get("trainable_params")!=n:
                raise StudyError("original source/model identity not replayable")
            res={"study":STUDY,"mode":"GIT_ORIGINAL_NO_REFIT_REPLAY",
                 "results":[replay_seed(b,cal,dev,mean,std) for b in saved["runs"]],
                 "scientific_status":"REPLAY_VERIFIED_EXPOSED_DEV_ONLY"}
        else:
            runs=[fit_seed(seed,fit,cal,dev,mean,std) for seed in SEEDS]
            res={"study":STUDY,"mode":"ORIGINAL_AUTHENTIC_FIT_CAL_DEV",
                 "source_zip_sha256":SOURCE,"source_doi":uci.DOI,
                 "source_time_available_at_proven":False,
                 "trainable_params":n,"deployed_params":n,
                 "train_updates_per_seed":UPDATES,"batch_size":BATCH,
                 "seeds":list(SEEDS),"input_history_steps":HISTORY,
                 "horizon_steps":uci.H,"fit_end_exclusive":uci.FIT_END,
                 "cal_end_exclusive":uci.CAL_END,"dev_end_exclusive":uci.DEV_END,
                 "original_protected_tail_parsed":False,
                 "fit_mean":mean,"fit_std":std,
                 "runs":runs,"scientific_status":"TRAINED_EXPOSED_DEV_ONLY_NOT_INDEPENDENT",
                 "warnings":["no original source available-at time proof",
                             "same-household already-exposed DEV cannot confirm generalization",
                             "48-step memory does not prove 1024-step retention",
                             "associative delta prior art; no novelty claim",
                             "no matched budget architecture comparator fitted",
                             "original binary checkpoints must be owned by Git for replay claim"]}
        a.out.parent.mkdir(parents=True,exist_ok=True)
        a.out.write_text(json.dumps(res,sort_keys=True,indent=2,allow_nan=False)+"\n")
        summary={"study":STUDY,"mode":res["mode"],"scientific_status":res["scientific_status"]}
        if a.replay_from:summary["results"]=res["results"]
        else:summary.update({"trainable_params":n,"scores":[
            {"seed":r["seed"],"cal":r["cal"],"dev":r["dev"],
             "checkpoint_sha256":r["checkpoint_sha256"]} for r in runs]})
        print(json.dumps(summary,sort_keys=True,allow_nan=False))
        return 0
    except (StudyError,uci.SourceError,OSError,ValueError,TypeError,RuntimeError) as e:
        print(json.dumps({"status":"BLOCKED_OR_FAILED","reason":str(e)}),file=sys.stderr)
        return 2

if __name__=="__main__":
    sys.exit(main())
