#!/usr/bin/env python3
"""New real FIT multi-client TAC: hourly 1h/24h load, not UCI374 transfer.

Train full native fast/slow/matrix original 103k-scale architecture with two
forecast heads on 176 genuine Monash customer identities, original source
ZIP SHA and all per-seed source/optimizer/RNG original checkpoints. CAL 34,
DEV 36 distinct customers, 75 sealed original test clients NOT float-parsed.
No confirmed source observation available_at, deployment, or architecture win.
"""
from __future__ import annotations
import argparse,base64,datetime as dt,gzip,hashlib,io,json,math,os,random,sys
from pathlib import Path
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import model01_monash_multiclient_baselines as source
import model01_native_tac_real_fit as ancestral

ROOT=Path(__file__).resolve().parents[1]
ORIGINAL_SOURCE=ROOT/"research/source-evidence/MODEL01-MONASH321-ORIGINAL-CAL-DEV-SOURCE-BASELINES.json"
STUDY="MODEL01-MONASH-NATIVE-MULTIHORIZON-001"
PARAMETERS=103381
SEEDS=(1801,1802,1803)
UPDATES=160
BATCH=64
HISTORY=ancestral.HISTORY
HORIZONS=source.HORIZONS
MODEL_LIMIT_LOG=17.0

class CapabilityError(ValueError):pass

class MultiHorizonTAC(ancestral.NativeTAC):
    """Exactly ancestral 103252p TAC plus one extra output head neuron."""
    def __init__(self):
        super().__init__()
        self.head[-1]=nn.Linear(128,len(HORIZONS))
    def forward(self,x):
        # Preserve original native recurrence exactly; only expand persistence
        # residual for the explicitly two-output source-time forecasting task.
        if x.ndim!=3 or x.shape[1:]!=(HISTORY,3):
            raise CapabilityError("expected causal 48-hour source history")
        z=self.norm(self.emb(x)+self.ff(self.emb(x)))
        n=len(x);fast=x.new_zeros((n,32));slow=x.new_zeros((n,32))
        memory=x.new_zeros((n,8,8));key=None
        for t in range(HISTORY):
            h=z[:,t]
            fast=self.fast(h,fast)
            inp=torch.cat([h,fast,slow],-1)
            gate=torch.sigmoid(self.sgate(inp))
            slow=gate*slow+(1-gate)*torch.tanh(self.sstep(inp))
            r=torch.cat([h,fast],-1)
            key=F.normalize(self.key(r),p=2,dim=-1,eps=1e-6)
            value=torch.tanh(self.value(r))
            beta=.5*torch.sigmoid(self.write(r))
            alpha=.9+.099*torch.sigmoid(self.retain(r))
            old_read=torch.bmm(memory,key.unsqueeze(-1)).squeeze(-1)
            memory=(alpha.unsqueeze(-1)*memory+
                    beta.unsqueeze(-1)*(value-old_read).unsqueeze(-1)*key.unsqueeze(1)).clamp(-5,5)
        read_gate=torch.sigmoid(self.readgate(torch.cat([z[:,-1],fast],-1)))
        read=read_gate*torch.bmm(memory,key.unsqueeze(-1)).squeeze(-1)
        result=x[:,-1,0].unsqueeze(-1)+self.head(torch.cat([z[:,-1],fast,slow,read],-1))
        if result.shape!=(n,len(HORIZONS)):
            raise CapabilityError("native two-horizon forecasts misaligned")
        return result

def source_data():
    frozen=json.loads(ORIGINAL_SOURCE.read_bytes())
    if (frozen.get("schema")!="model01-monash-hourly-independent-client-source-baseline-v1"
        or frozen.get("source_zip_md5")!=source.ZIP_MD5
        or frozen.get("TEST_customer_numeric_targets_parsed") is not False
        or frozen.get("CAL_DEV_are_exposed_from_this_source_audit") is not True):
        raise CapabilityError("original real source receipt invalid")
    zipraw=source.acquire()
    if hashlib.sha256(zipraw).hexdigest()!=frozen["source_zip_sha256"]:
        raise CapabilityError("publisher content does not match original registered source")
    clients={"FIT":{},"CAL":{},"DEV":{}}
    counts={v:0 for v in ("FIT","CAL","DEV","SEALED_TEST")}
    for arm,data in source.iter_source(zipraw):
        if arm=="HEADER":continue
        name,t0,series=data
        counts[arm]+=1
        if arm!="SEALED_TEST":
            clients[arm][name]=(t0,series)
    if counts!=frozen["source_arm_counts"]:
        raise CapabilityError("physical client original split changed")
    for group in ("CAL","DEV"):
        original_ids={r["client_public_hash"] for r in frozen["per_client_CAL_DEV"][group]}
        live_ids={hashlib.sha256(n.encode()).hexdigest()[:16] for n in clients[group]}
        if original_ids!=live_ids:
            raise CapabilityError("exposed customer identities changed")
    return clients,frozen

def fit_scaling(clients):
    # Only FIT client values; avoid leakage from 34 CAL / 36 DEV / 75 TEST.
    xs=np.concatenate([np.log1p(np.asarray(values,dtype=np.float64))
                       for _,values in clients["FIT"].values()])
    mean=float(xs.mean());std=float(xs.std())
    if not(math.isfinite(mean) and math.isfinite(std) and .01<std<1e4):
        raise CapabilityError("original FIT-only source log scaler invalid")
    return mean,std

def input_window(raw,anchor,mean,std):
    # Past 48 source records only. Hourly deterministic position is known.
    indices=np.arange(anchor-HISTORY+1,anchor+1,dtype=np.int32)
    y=np.log1p(np.asarray(raw[anchor-HISTORY+1:anchor+1],dtype=np.float64))
    if len(y)!=HISTORY:
        raise CapabilityError("no complete observed causal history")
    data=np.stack([(y-mean)/std,
                    np.sin(2*np.pi*(indices%24)/24),
                    np.cos(2*np.pi*(indices%24)/24)],axis=-1).astype(np.float32)
    return data

def eligible(t0,series,anchor,horizon):
    if anchor<source.HISTORY or anchor+max(HORIZONS)>=len(series):
        return False
    last=series[anchor-source.HISTORY+1:anchor+1]
    if sum(z>0 for z in last)<24:
        return False
    now=t0+dt.timedelta(hours=anchor)
    if source.dst_ambiguous(now):
        return False
    future=t0+dt.timedelta(hours=anchor+horizon)
    return not source.dst_ambiguous(future)

def training_batch(clients,gen,mean,std):
    all_names=sorted(clients["FIT"])
    if len(all_names)<15:raise CapabilityError("insufficient real FIT identities")
    x=[];y=[];sample_receipts=[]
    for _ in range(BATCH):
        for trial in range(300):
            name=all_names[int(gen.integers(0,len(all_names)))]
            t0,series=clients["FIT"][name]
            anchor=int(gen.integers(source.HISTORY,len(series)-max(HORIZONS)))
            if all(eligible(t0,series,anchor,h) for h in HORIZONS):
                break
        else:raise CapabilityError("no causal valid real FIT sample")
        x.append(input_window(series,anchor,mean,std))
        y.append([(math.log1p(series[anchor+h])-mean)/std for h in HORIZONS])
        sample_receipts.append(f"{name}:{anchor}")
    return torch.from_numpy(np.stack(x)),torch.tensor(y,dtype=torch.float32),sample_receipts

def eval_batches(clients,group,mean,std,original):
    data={}
    for name,(t0,series) in clients[group].items():
        orig_client=next(r for r in original["per_client_CAL_DEV"][group]
                         if r["client_public_hash"]==hashlib.sha256(name.encode()).hexdigest()[:16])
        length=len(series)
        start=length-max(HORIZONS)-source.EVAL_FINAL_HOURS
        features=[];target=[];mask=[]
        for anchor in range(start,start+source.EVAL_FINAL_HOURS):
            features.append(input_window(series,anchor,mean,std))
            target.append([series[anchor+h] for h in HORIZONS])
            mask.append([eligible(t0,series,anchor,h) for h in HORIZONS])
        a=torch.from_numpy(np.stack(features))
        target=np.asarray(target,dtype=np.float64)
        mask=np.asarray(mask,dtype=bool)
        for i,h in enumerate(HORIZONS):
            expected=orig_client["source_baselines"][str(h)]["last_observation"]["n"]
            if int(mask[:,i].sum())!=expected:
                raise CapabilityError("new native model eligible case count differs from original baseline")
        data[name]={"x":a,"target":target,"mask":mask,
                    "public_source_hash":orig_client["client_public_hash"],
                    "source_baselines":orig_client["source_baselines"]}
    return data

def collect(net,dataset,mean,std,offset=None):
    # Applies identical point predictions; interval offsets are optional
    # CAL-only global symmetric absolute-error quantiles.
    net.eval()
    out={str(h):[] for h in HORIZONS}
    per_client={}
    with torch.no_grad():
        for name,item in sorted(dataset.items()):
            x=item["x"]
            pred=torch.cat([net(x[i:i+128]).cpu() for i in range(0,len(x),128)]).numpy()
            load=np.expm1(np.clip(pred*std+mean,0,MODEL_LIMIT_LOG))
            target=item["target"]
            mask=item["mask"]
            client={}
            for j,h in enumerate(HORIZONS):
                errors=load[:,j][mask[:,j]]-target[:,j][mask[:,j]]
                if not np.all(np.isfinite(errors)) or len(errors)<200:
                    raise CapabilityError("invalid source predictions/horizon")
                ref=errors.astype(np.float64)
                summary={"n":int(len(ref)),"mae_kw":float(np.mean(abs(ref))),
                         "rmse_kw":float(np.sqrt(np.mean(ref**2)))}
                if offset is not None:
                    q=offset[str(h)]
                    summary.update({"observed_coverage90":float(np.mean(abs(ref)<=q)),
                                    "mean_width90_kw":float(2*q),
                                    "interval_score90_kw":float(np.mean(2*q+20*np.maximum(abs(ref)-q,0)))})
                client[str(h)]=summary
                out[str(h)].append(summary)
            per_client[item["public_source_hash"]]=client
    macro={}
    for h,items in out.items():
        macro[h]={"customers":len(items),
                  "total_decisions":sum(v["n"] for v in items),
                  "mean_client_mae_kw":sum(v["mae_kw"] for v in items)/len(items),
                  "mean_client_rmse_kw":sum(v["rmse_kw"] for v in items)/len(items)}
        if offset is not None:
            macro[h].update({
                "mean_client_coverage90":sum(v["observed_coverage90"] for v in items)/len(items),
                "mean_client_interval_score90_kw":sum(v["interval_score90_kw"] for v in items)/len(items),
                "interval_width90_kw":2*offset[h]})
    return macro,per_client

def fit(seed,clients,orig,mean,std):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    model=MultiHorizonTAC()
    if ancestral.params(model)!=PARAMETERS:
        raise CapabilityError("new native multihorizon learned parameter target changed")
    optim=torch.optim.AdamW(model.parameters(),lr=ancestral.LR,weight_decay=.01)
    gen=np.random.default_rng(seed);sampler=hashlib.sha256()
    for _ in range(UPDATES):
        x,y,receipt=training_batch(clients,gen,mean,std)
        sampler.update(("\n".join(receipt)+"\n").encode())
        pred=model(x)
        loss=F.smooth_l1_loss(pred,y,beta=.5)
        if not torch.isfinite(loss):raise CapabilityError("nonfinite source FIT error")
        optim.zero_grad(set_to_none=True);loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        optim.step()
    cal=eval_batches(clients,"CAL",mean,std,orig)
    dev=eval_batches(clients,"DEV",mean,std,orig)
    _,cal_rows=collect(model,cal,mean,std)
    # Compute a single source CAL-only empirical global absolute residual quantile per horizon.
    # This is weighted by real eligible source decisions, not independent customer coverage.
    # Needed for interval diagnostic, not a conformal guarantee across customers.
    # We compute directly from the same frozen model predictions once more.
    errors={str(h):[] for h in HORIZONS}
    with torch.no_grad():
        model.eval()
        for item in cal.values():
            preds=torch.cat([model(item["x"][i:i+128]).cpu()
                             for i in range(0,len(item["x"]),128)]).numpy()
            pv=np.expm1(np.clip(preds*std+mean,0,MODEL_LIMIT_LOG))
            for j,h in enumerate(HORIZONS):
                mask=item["mask"][:,j]
                errors[str(h)].extend(np.abs(pv[mask,j]-item["target"][mask,j]).tolist())
    offsets={}
    for h in map(str,HORIZONS):
        vals=np.sort(np.asarray(errors[h],dtype=np.float64))
        i=min(math.ceil((len(vals)+1)*.9)-1,len(vals)-1)
        offsets[h]=float(vals[i])
    cal_macro,_=collect(model,cal,mean,std,offsets)
    dev_macro,dev_clients=collect(model,dev,mean,std,offsets)
    buffer=io.BytesIO()
    torch.save({"model":model.state_dict(),"optimizer":optim.state_dict(),
                "torch_rng":torch.get_rng_state(),"numpy_rng":gen.bit_generator.state,
                "python_rng":random.getstate(),
                "seed":seed,"steps":UPDATES,"sampler_sha256":sampler.hexdigest()},buffer)
    raw=buffer.getvalue()
    return {"seed":seed,"train_steps":UPDATES,"sampler_sha256":sampler.hexdigest(),
            "cal_offsets90_kw":offsets,"cal_macro":cal_macro,"dev_macro":dev_macro,
            "dev_per_customer":dev_clients,"checkpoint_sha256":hashlib.sha256(raw).hexdigest(),
            "checkpoint_b64_gzip":base64.b64encode(gzip.compress(raw,mtime=0)).decode()}

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args(argv)
    try:
        if a.out.exists():raise CapabilityError("refusing original source fit overwrite")
        if os.environ.get("MODEL01_MONASH_REAL_FIT_ENABLE")!="EXPLICIT_CPU_CUSTOMER_FIT":
            raise CapabilityError("real customer source neural fit needs explicit study gating")
        torch.set_num_threads(2)
        clients,orig=source_data()
        mean,std=fit_scaling(clients)
        runs=[fit(seed,clients,orig,mean,std) for seed in SEEDS]
        result={
            "study":STUDY,"source_zip_sha256":orig["source_zip_sha256"],
            "original_source_manifest":"research/source-evidence/MODEL01-MONASH321-ORIGINAL-CAL-DEV-SOURCE-BASELINES.json",
            "training_clients":len(clients["FIT"]),"cal_clients":len(clients["CAL"]),
            "dev_clients":len(clients["DEV"]),"sealed_test_clients":orig["source_arm_counts"]["SEALED_TEST"],
            "sealed_test_target_values_parsed":False,
            "native_learned_deployed_parameters":PARAMETERS,
            "input_history_hours":HISTORY,"prediction_horizons_hours":list(HORIZONS),
            "seeds":list(SEEDS),"optimizer_updates_per_seed":UPDATES,"batch_size":BATCH,
            "FIT_only_log1p_mean":mean,"FIT_only_log1p_std":std,
            "test_source_asof_verified":False,
            "source_is_same_target_as_UCI374_appliances":False,
            "CAL_DEV_previously_exposed_to_simple_baseline_study":True,
            "independent_confirmatory_test_performed":False,
            "scientific_stage":"MULTICLIENT_SOURCE_DEV_ONLY",
            "runs":runs,
            "warnings":[
                "no original site as-of availability and Portuguese DST artifacts were controlled heuristically only",
                "client-disjoint DEV is genuine but its labels exposed in earlier baseline study, not blind independent confirmation",
                "no original customer TEST values numerically parsed or scored",
                "source FIT has many client series; no vendor/household user-provided consented production dataset",
                "not a fair architecture superiority study; old TAC vs GRU results are from different source and target",
                "some source values padded as zero before service activation; eligibility uses observed past-only positives",
                "CAL global conformal-style absolute q is not a valid exchangeable coverage guarantee",
                "three fitting seeds are development only, not independent customer-group confirmation"
            ]}
        a.out.parent.mkdir(parents=True,exist_ok=True)
        a.out.write_text(json.dumps(result,sort_keys=True,indent=2,allow_nan=False)+"\n")
        print(json.dumps({"study":STUDY,"counts":[len(clients[a]) for a in ("FIT","CAL","DEV")],
                          "nparams":PARAMETERS,
                          "real_dev":[{"seed":r["seed"],"horizons":r["dev_macro"],
                                       "checkpoint_sha256":r["checkpoint_sha256"]}
                                      for r in runs],"scientific_stage":"MULTICLIENT_SOURCE_DEV_ONLY"},
                         sort_keys=True,allow_nan=False))
        return 0
    except (CapabilityError,source.SourceError,ancestral.StudyError,
            OSError,RuntimeError,ValueError,TypeError,KeyError) as exc:
        print(json.dumps({"status":"BLOCKED_OR_FAILED","reason":str(exc)}),file=sys.stderr)
        return 2

if __name__=="__main__":
    sys.exit(main())
