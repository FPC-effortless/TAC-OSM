"""LATFUSE013: REAL observed PM2.5 linear decodability from frozen TAC states.

Oracle historical WRITE keys / original PM source truth are PRIVILEGED offline
research variables and NEVER part of native online historical read inputs.
External fitted ridge sidecars are new diagnostics, not part of the 99,390-
parameter deployed model, and these FIT-era cases are not unseen model tests.
"""
from __future__ import annotations
import argparse
import datetime as dt
import hashlib
import json
import math
from pathlib import Path
import torch
import torch.nn.functional as F

from scripts import latfuse002_real_fit as source
from scripts import latfuse008_paired_real_training as fit
from scripts import latfuse011_real_memory_falsifiers as readpath
from scripts import latfuse012_original_source_oracle as oracle
from scripts.validate_research_parity import check_manifest

ROOT=Path(__file__).resolve().parents[1]
REG=ROOT/"research/mathematical-engineering/latfuse013/REGISTERED_REAL_CONTENT_PROBE.json"
OUT=ROOT/"research/mathematical-engineering/latfuse013/RESULTS.json"
LAGS=(16,64,128,512)
TRAIN_STOP=source.hour(dt.datetime(2014,3,30))
NBOOT=4000
SEED=13013
KINDS=("source_only","nonassoc","native_read","oracle_read",
       "wrong_key_read","random_key_read","oracle_only","wrong_only","random_only")

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def input_features(m,d,pairs,lag,*,split):
    all_features={name:[] for name in KINDS}
    true=[]
    pairsites=[]
    persistence=[]
    actual_hour=[]
    for a in range(0,len(pairs),16):
        rows=pairs[a:a+16]
        if len(rows)%2:raise RuntimeError("corrupted physical source matched pair")
        b=len(rows)
        site=torch.tensor([x[0] for x in rows],dtype=torch.long)
        anchor=torch.tensor([x[1] for x in rows],dtype=torch.long)
        state=m.initial(b)
        hist_key=None
        for h in range(-23,lag+1):
            t=anchor+h
            if h==0:
                u=m.encode(d["values"][site,t],d["valid"][site,t],
                           d["age"][site,t],d["last"][site,t])
                hist_key=F.normalize(m.key_map(u).reshape(b,2,8),dim=-1,eps=1e-8)
            state=m.step(state,d["values"][site,t],d["valid"][site,t],
                        d["age"][site,t],d["last"][site,t],site)
        if hist_key is None:raise RuntimeError("missing original source WRITE key")
        # The CURRENT source and requested old clock are legal; neither
        # event/PM value at anchor nor old oracle write key may enter native q.
        t=anchor+lag
        clock=source.calendar([int(v) for v in anchor])
        current=d["values"][site,t]
        observed=d["valid"][site,t]
        last_at_query=d["last"][site,t+1]
        age=d["age"][site,t]
        nuisance=torch.cat([current,observed.float(),last_at_query,
                           age,clock,F.one_hot(site,num_classes=12).float()],dim=-1)
        idx=site[:,None,None].expand(-1,1,80)
        entity=state.entities.gather(1,idx).squeeze(1)
        entity=entity*state.active.gather(1,site[:,None]).float()
        noassoc=torch.cat([nuisance,state.fast,state.slow,entity],dim=-1)
        q,route=readpath.read_query(m,state,anchor)
        # Each output is in EXACT same native 16D memory value coordinates.
        native=oracle.read_value(state.associative,q,route)
        right=oracle.read_value(state.associative,hist_key,route)
        wrong_key=hist_key.reshape(b//2,2,2,8).flip(1).reshape(b,2,8)
        wrong=oracle.read_value(state.associative,wrong_key,route)
        # Label independent random query: one seeded pseudo-random source
        # stream keyed only from legal (site,anchor,split), not label/target.
        rng=[]
        for s,ts in zip(site.tolist(),anchor.tolist()):
            g=torch.Generator().manual_seed((SEED*1000003+split*221+s*100001+ts)%((1<<63)-1))
            rng.append(F.normalize(torch.randn(2,8,generator=g),dim=-1))
        random=oracle.read_value(state.associative,torch.stack(rng),route)
        feats=dict(source_only=nuisance,nonassoc=noassoc,
                   native_read=torch.cat([nuisance,native],dim=-1),
                   oracle_read=torch.cat([nuisance,right],dim=-1),
                   wrong_key_read=torch.cat([nuisance,wrong],dim=-1),
                   random_key_read=torch.cat([nuisance,random],dim=-1),
                   oracle_only=right,wrong_only=wrong,random_only=random)
        for k in KINDS:
            if not torch.isfinite(feats[k]).all():raise RuntimeError("nonfinite genuine source feature: "+k)
            all_features[k].append(feats[k])
        # Only after all reads have been computed may privileged original
        # historical source PM targets be looked up for external regressors.
        y=d["raw"][site,anchor,0]
        if not torch.isfinite(y).all():raise RuntimeError("historical original source truth absent")
        for v,row in zip(y.tolist(),rows):
            if not (v>=150 if row[2] else v<=35):
                raise RuntimeError("original source high/ordinary matching invalid")
        true.append(torch.log1p(y.clamp_min(0)))
        # Actual physically observed most recent query-time PM2.5, not
        # unobserved oracle anchor value, serves as practical persistence.
        restored=torch.expm1((last_at_query[:,0]*d["spread"][0]+d["center"][0]).clamp(-12,12)).clamp_min(0)
        persistence.append(torch.log1p(restored))
        pairsites.extend([row[0] for row in rows[::2]])
        actual_hour.extend(anchor.tolist())
    return dict(x={k:torch.cat(v) for k,v in all_features.items()},
        y=torch.cat(true),persist=torch.cat(persistence),
        sites=torch.tensor(pairsites,dtype=torch.long),anchors=actual_hour)

def sidecar(train_x,train_y,test_x):
    if train_x.shape[1]!=test_x.shape[1] or len(train_x)!=len(train_y):
        raise RuntimeError("original sidecar feature/source target shape mismatch")
    center=train_x.mean(0)
    spread=train_x.std(0,unbiased=False).clamp_min(1e-4)
    tx=((train_x-center)/spread).clamp(-6,6)/math.sqrt(train_x.shape[1])
    vx=((test_x-center)/spread).clamp(-6,6)/math.sqrt(train_x.shape[1])
    yc=train_y.mean()
    ys=train_y.std(unbiased=False).clamp_min(.001)
    yy=(train_y-yc)/ys
    coeff=torch.linalg.solve(tx@tx.T+torch.eye(len(tx)),yy[:,None])
    out=(vx@tx.T@coeff).squeeze(-1)*ys+yc
    if not torch.isfinite(out).all():raise RuntimeError("nonfinite physical source ridge prediction")
    return out.clamp(0,9),int(train_x.shape[1])

def describe(p,y,train_y):
    z=(p-y).abs()
    baseline=torch.ones_like(y)*train_y.mean()
    var=((y-baseline).square().sum()).clamp_min(1e-8)
    paired=(p[::2]>p[1::2]).float()+.5*(p[::2]==p[1::2]).float()
    pred=torch.expm1(p.clamp(0,9))
    truth=torch.expm1(y)
    return dict(log_PM25_MAE=float(z.mean()),
                real_PM25_raw_ugm3_MAE=float((pred-truth).abs().mean()),
                real_PM25_log_MSE=float((p-y).square().mean()),
                original_train_mean_R2=float(1-((p-y).square().sum()/var)),
                source_high_vs_ordinary_paired_rank=float(paired.mean()),
                target_abs_log_error=z)

def station_bootstrap(eval_sites,a,b):
    # Contrast each same physical original episode once per seed; do not
    # claim independent bootstrap units or an unexposed source-model holdout.
    groups=[torch.where(eval_sites==s)[0] for s in sorted(set(eval_sites.tolist()))]
    if len(groups)<5:raise RuntimeError("too few source station groups")
    g=torch.Generator().manual_seed(SEED)
    differences=(a-b)
    draws=[]
    for i in range(NBOOT):
        choose=torch.randint(0,len(groups),(len(groups),),generator=g)
        idx=torch.cat([groups[int(k)] for k in choose])
        draws.append(float(differences[idx].mean()))
    ordered=sorted(draws)
    return [ordered[int(.025*(NBOOT-1))],ordered[int(.975*(NBOOT-1))]]

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--source",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    arg=parser.parse_args()
    if arg.output.exists():raise RuntimeError("original preregistered real source result exists; NO OVERWRITE")
    torch.set_num_threads(1)
    manifest=json.loads(REG.read_text())
    check_manifest(manifest,"preflight")
    if manifest["study_id"]!="LATFUSE013-FROZEN-ORACLE-REAL-CONTENT-PROBE" or manifest["read_lags"]!=list(LAGS):
        raise RuntimeError("real source decodability prospective registration changed")
    if sha(arg.source)!=source.SOURCE_SHA:
        raise RuntimeError("physical original source archive SHA mismatch")
    d=fit.source_observations(arg.source)
    fitrows=fit.select_pairs(d,fit.TRAIN_START,TRAIN_STOP,8)
    testrows=fit.select_pairs(d,fit.EVAL_START,fit.EVAL_STOP,2)
    original=json.loads((fit.OUT/"RESULTS.json").read_text())
    testsha=hashlib.sha256(json.dumps(testrows,separators=(",",":")).encode()).hexdigest()
    if len(testrows)!=48 or testsha!=original["source_eval_cohort_sha256"]:
        raise RuntimeError("original source eval cohort differs from frozen LATFUSE008 receipts")
    if len(fitrows)<24 or max(t for _,t,_ in fitrows)+512>=min(t for _,t,_ in testrows):
        raise RuntimeError("original March event training read histories overlap original May evaluation")
    original_shas=[]
    allout=[]
    with torch.no_grad():
        for seed in fit.SEEDS:
            arm="assoc_historic_query"
            rec=next((x for x in original["model_results"] if x["seed"]==seed and x["arm"]==arm),None)
            if rec is None:raise RuntimeError("original seed missing")
            cp=fit.OUT/f"seed-{seed}"/arm/"final.pt"
            digest=sha(cp)
            if digest!=rec["original_checkpoint_sha256"]:
                raise RuntimeError("original source checkpoint changed")
            m=fit.create_model(seed,arm)
            m.load_state_dict(torch.load(cp,map_location="cpu",weights_only=False)["original_model"],strict=True)
            m.eval()
            model_before=[p.detach().clone() for p in m.parameters()]
            one={}
            for lag in LAGS:
                tr=input_features(m,d,fitrows,lag,split=0)
                te=input_features(m,d,testrows,lag,split=1)
                metrics={}
                predictions={}
                for name in KINDS:
                    p,ncoeff=sidecar(tr["x"][name],tr["y"],te["x"][name])
                    predictions[name]=p
                    z=describe(p,te["y"],tr["y"])
                    metrics[name]={k:v for k,v in z.items() if k!="target_abs_log_error"}
                    metrics[name]["external_fitted_ridge_input_coefficients"]=ncoeff
                    metrics[name]["external_sidecar_weights_NOT_native_architecture"]=True
                for name,p in [("source_last_observed_persistence",te["persist"]),
                  ("original_train_prior",torch.full_like(te["y"],float(tr["y"].mean())))]:
                    z=describe(p,te["y"],tr["y"])
                    metrics[name]={k:v for k,v in z.items() if k!="target_abs_log_error"}
                # Positive improvement means oracle actual physical log MAE
                # LOWER than counterpart. Predeclared H64 go-gate requires
                # >5% separately vs all 3 and source-station boot CI >0
                comparators=("wrong_key_read","random_key_read","nonassoc")
                gate={}
                for n in comparators:
                    oracle_error=(predictions["oracle_read"]-te["y"]).abs()
                    other_error=(predictions[n]-te["y"]).abs()
                    baseline=float(other_error.mean())
                    delta=other_error-oracle_error
                    ci=station_bootstrap(te["sites"],other_error[::2].add(other_error[1::2]).div(2),
                           oracle_error[::2].add(oracle_error[1::2]).div(2))
                    gate[n]=dict(
                        original_source_log_mae_relative_improvement=(baseline-float(oracle_error.mean()))/max(baseline,1e-7),
                        original_site_bootstrap_95_mean_abs_log_error_gain=ci,
                        per_seed_pass=(baseline-float(oracle_error.mean()))/max(baseline,1e-7)>=.05 and ci[0]>0)
                if any(not torch.isfinite(p).all() for p in predictions.values()):
                    raise RuntimeError("probe numeric invalid")
                one[str(lag)]=dict(
                  train_real_source_physical_pairs=len(fitrows)//2,
                  eval_exposed_real_source_physical_pairs=len(testrows)//2,
                  source_read_time_requested_lag=lag,
                  sidecar_metrics=metrics,
                  oracle_vs_comparators=gate,
                  H64_oracle_content_gate_on_original_source=bool(all(x["per_seed_pass"] for x in gate.values())) if lag==64 else None)
                print("LATFUSE013_REAL_SOURCE_RIDGE="+json.dumps(
                    dict(seed=seed,lag=lag,oracle_logMAE=metrics["oracle_read"]["log_PM25_MAE"],
                         wrong_logMAE=metrics["wrong_key_read"]["log_PM25_MAE"],
                         random_logMAE=metrics["random_key_read"]["log_PM25_MAE"],
                         fastslow_logMAE=metrics["nonassoc"]["log_PM25_MAE"],
                         original_B_gate=one[str(lag)]["H64_oracle_content_gate_on_original_source"]),
                         sort_keys=True),flush=True)
            if digest!=sha(cp) or any(not torch.equal(x,p) for x,p in zip(model_before,m.parameters())):
                raise RuntimeError("frozen trained source model mutated during external probe")
            allout.append(dict(seed=seed,original_model_checkpoint_sha256=digest,delays=one))
            original_shas.append(digest)
    gate=all(z["delays"]["64"]["H64_oracle_content_gate_on_original_source"] for z in allout)
    out=dict(study_id=manifest["study_id"],real_source_zip_sha256=source.SOURCE_SHA,
         original_fixed_checkpoint_sha256=original_shas,
         source_train_pairs=len(fitrows)//2,source_eval_pairs=len(testrows)//2,
         source_train_cohort_sha256=hashlib.sha256(json.dumps(fitrows,separators=(",",":")).encode()).hexdigest(),
         source_eval_cohort_sha256=testsha,
         study_metrics_are_FIT_era_exposed_and_NOT_independent_native_model_test=True,
         original_99390_param_6800byte_models_unchanged=True,
         external_linear_ridge_probes_fit_on_real_source_only=True,
         external_sidecars_not_part_of_trained_TAC_OSM_weights=True,
         oracle_keys_not_native_online_query=True,
         original_event_labels_or_values_never_enter_native_historic_clock_query=True,
         no_original_neural_optimizer_updates=True,
         protected_original_2016_2017_numeric_unopened=True,
         no_architecture_superiority_claim=True,
         registered_H64_all_2_seed_and_3_comparator_gate_pass=gate,
         scientific_interpretation="A fail of LINEAR oracle content decodability cannot rule out nonlinear stored information, independent new data evaluation or future writer-side learning",
         models=allout)
    arg.output.parent.mkdir(parents=True,exist_ok=True)
    arg.output.write_text(json.dumps(out,indent=2,sort_keys=True,allow_nan=False)+"\n")
    print("LATFUSE013_REAL_SOURCE_ORACLE_CONTENT_RIDGE_PASS="+json.dumps(
        {"n_original_models":len(allout),"source_train_pairs":len(fitrows)//2,
         "source_eval_pairs":len(testrows)//2,"H64_gate_pass":gate,
         "original_neural_optimizer_updates":0},sort_keys=True),flush=True)
if __name__=="__main__":main()
