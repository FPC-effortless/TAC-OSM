#!/usr/bin/env python3
"""Frozen Model01 TAC original checkpoint CAL-only signed/phase interval screen.

Fits NO neural weights. First 60% chronological CAL computes three fixed
empirical interval schemes, next 20% CAL selects candidate once, last 20%
CAL audits choice without tuning; only THEN score already-exposed DEV once.
Original protected 20% source tail remains unparsed. Formal independent
coverage guarantee NOT established under correlated time-series data.
"""
from __future__ import annotations
import argparse,base64,gzip,hashlib,io,json,math,sys
from pathlib import Path
import numpy as np
import torch
import model01_native_tac_real_fit as native

ROOT=Path(__file__).resolve().parents[1]
ORIGINAL=ROOT/"research/replay-bundles/MODEL01-TAC-REAL-FIT-001/ORIGINAL_FIT.json"
RESULTS_SHA={
  1701:"2d33226a652304ad69770a2509be33b221cacfc424b5ad6b6c79dce3570b4d4f",
  1702:"9a37e4afb7cd9cc59912e437a4d706faecb393cb177dce0da5f6ea4837c5e0b9",
  1703:"86fbb16d42c697582a885891bc80274f2af26ae1e9c9f4cd2cf531f6e4e544d4",
}
ALPHA=.10
SELECTION_MIN_COVERAGE=.88
MIN_SELECT_SCORE_IMPROVEMENT=.01

class IntegrityError(ValueError):pass

def quantile(values,p):
    v=np.sort(np.asarray(values,dtype=np.float64))
    if len(v)<20 or not np.all(np.isfinite(v)) or not 0<p<1:
        raise IntegrityError("insufficient/invalid CAL quantile samples")
    k=min(math.ceil((len(v)+1)*p)-1,len(v)-1)
    return float(v[k])

def partition(n):
    if n<1000:raise IntegrityError("insufficient chronological CAL records")
    a=(3*n)//5
    b=(4*n)//5
    if min(a,b-a,n-b)<150:raise IntegrityError("insufficient CAL subperiod")
    return slice(0,a),slice(a,b),slice(b,n)

def phase(indices):
    return (np.asarray(indices,dtype=np.int32)%144)//36

def candidates(signed,indices):
    if len(signed)!=len(indices):raise IntegrityError("CAL source mismatch")
    err=np.asarray(signed,dtype=np.float64)
    if not np.all(np.isfinite(err)):raise IntegrityError("nonfinite residual")
    q=quantile(abs(err),.9)
    lower=quantile(err,.05);upper=quantile(err,.95)
    if not lower<upper:raise IntegrityError("invalid signed residual interval")
    parts=phase(indices)
    bins={}
    for i in range(4):
        r=err[parts==i]
        if len(r)<200:raise IntegrityError("empty/nonrepresentative phase bucket")
        l=quantile(r,.05);u=quantile(r,.95)
        if not l<u:raise IntegrityError("invalid phase interval")
        bins[str(i)]=[l,u]
    return {
        "symmetric_CAL_60":[-q,q],
        "asymmetric_CAL_60":[lower,upper],
        "phase_4_signed_CAL_60":bins
    }

def bounds(config,indices):
    idx=np.asarray(indices,dtype=np.int32)
    if isinstance(config,list):
        lo,hi=config
        return np.full(len(idx),lo),np.full(len(idx),hi)
    if isinstance(config,dict) and set(config)==set(map(str,range(4))):
        p=phase(idx)
        lo=np.asarray([config[str(int(x))][0] for x in p])
        hi=np.asarray([config[str(int(x))][1] for x in p])
        return lo,hi
    raise IntegrityError("unrecognized interval scheme")

def score(config,pred,y,indices):
    pred=np.asarray(pred,dtype=np.float64)
    y=np.asarray(y,dtype=np.float64)
    if not(len(pred)==len(y)==len(indices) and len(y)):
        raise IntegrityError("score source cardinality mismatch")
    lo,hi=bounds(config,indices)
    if not (np.all(np.isfinite(pred)) and np.all(np.isfinite(y))
            and np.all(np.isfinite(lo)) and np.all(np.isfinite(hi))
            and np.all(hi>lo)):
        raise IntegrityError("invalid finite interval")
    left=pred+lo
    right=pred+hi
    penalty=(2/ALPHA)*(np.maximum(left-y,0)+np.maximum(y-right,0))
    return {"n":len(y),
            "interval_score90":float(np.mean((right-left)+penalty)),
            "coverage90":float(np.mean((y>=left)&(y<=right))),
            "mean_width_wh":float(np.mean(right-left)),
            "undercount":int(np.sum(y<left)),
            "overcount":int(np.sum(y>right))}

def select(fit_pred,fit_y,fit_idx,select_pred,select_y,select_idx):
    options=candidates(np.asarray(fit_y)-np.asarray(fit_pred),fit_idx)
    ss={name:score(config,select_pred,select_y,select_idx)
        for name,config in options.items()}
    baseline=ss["symmetric_CAL_60"]["interval_score90"]
    allowed=[name for name in ("asymmetric_CAL_60","phase_4_signed_CAL_60")
             if ss[name]["coverage90"]>=SELECTION_MIN_COVERAGE
             and ss[name]["interval_score90"] <= baseline*(1-MIN_SELECT_SCORE_IMPROVEMENT)]
    chosen=min(allowed,key=lambda x:(ss[x]["interval_score90"],x)) if allowed else "symmetric_CAL_60"
    return chosen,options,ss

def immutable_checkpoints():
    data=json.loads(ORIGINAL.read_bytes())
    if (data.get("study")!=native.STUDY or
        data.get("source_zip_sha256")!=native.SOURCE or
        data.get("trainable_params")!=103252 or
        data.get("deployed_params")!=103252 or
        data.get("seeds")!=[1701,1702,1703] or
        len(data.get("runs",[]))!=3 or
        data.get("original_protected_tail_parsed") is not False):
        raise IntegrityError("original model identity/source gate failed")
    for run in data["runs"]:
        if run["checkpoint_sha256"]!=RESULTS_SHA.get(run["seed"]):
            raise IntegrityError("training SHA does not match original registered model")
    return data

def restore(run):
    raw=gzip.decompress(base64.b64decode(run["checkpoint_b64_gzip"],validate=True))
    if hashlib.sha256(raw).hexdigest()!=run["checkpoint_sha256"]:
        raise IntegrityError("checkpoint original SHA changed")
    ck=torch.load(io.BytesIO(raw),map_location="cpu",weights_only=True)
    if ck.get("seed")!=run["seed"] or ck.get("sample_order_sha256")!=run["sample_order_sha256"]:
        raise IntegrityError("original optimizer/sampler identity changed")
    net=native.NativeTAC()
    if native.params(net)!=103252:raise IntegrityError("model architecture no longer frozen")
    net.load_state_dict(ck["model"],strict=True)
    return net.eval()

def predict(net,data,mean,std):
    x,y,idx=data
    with torch.no_grad():
        guesses=np.concatenate([
            net(x[i:i+128]).numpy() for i in range(0,len(x),128)])
    return guesses.astype(np.float64)*std+mean,y.numpy().astype(np.float64)*std+mean,idx

def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out",type=Path,required=True)
    args=parser.parse_args(argv)
    try:
        if args.out.exists():raise IntegrityError("original CAL selection result may not be overwritten")
        original=immutable_checkpoints()
        raw=native.uci.fetch_official_zip()
        if native.uci.digest(raw)!=native.SOURCE:
            raise IntegrityError("source zip changed")
        y=np.asarray(native.uci.original_rows(raw),dtype=np.float64)
        mean=float(y[:native.uci.FIT_END].mean())
        std=float(y[:native.uci.FIT_END].std())
        torch.set_num_threads(2)
        allcal=native.windows(y,native.uci.eligible_indices(native.uci.FIT_END,native.uci.CAL_END),mean,std)
        cal0,cal1,cal2=partition(len(allcal[2]))
        def take(sl):
            return tuple(a[sl] for a in allcal)
        summary=[]
        for run in original["runs"]:
            net=restore(run)
            fp,fy,fi=predict(net,take(cal0),mean,std)
            sp,sy,si=predict(net,take(cal1),mean,std)
            chosen,schemes,selection=select(fp,fy,fi,sp,sy,si)
            ap,ay,ai=predict(net,take(cal2),mean,std)
            audit=score(schemes[chosen],ap,ay,ai)
            audit_symmetric=score(schemes["symmetric_CAL_60"],ap,ay,ai)
            # Decision is frozen using CAL selection before opening DEV in this run.
            dev=native.windows(y,native.uci.eligible_indices(native.uci.CAL_END,native.uci.DEV_END),mean,std)
            dp,dy,di=predict(net,dev,mean,std)
            old_q=run["q90_wh"]
            prior=score([-old_q,old_q],dp,dy,di)
            candidate=score(schemes[chosen],dp,dy,di)
            saved_old=run["dev"]
            if not (math.isclose(prior["interval_score90"],saved_old["interval_score90"],rel_tol=5e-6,abs_tol=1e-3)
                    and math.isclose(prior["coverage90"],saved_old["nominal90_observed_coverage"],abs_tol=1e-10)):
                raise IntegrityError("original DEV scores don't match source checkpoint")
            summary.append({
                "seed":run["seed"],"original_checkpoint_sha256":run["checkpoint_sha256"],
                "CAL_partition_index_counts":[len(fi),len(si),len(ai)],
                "chosen_without_DEV":chosen,
                "scheme_offsets_wh":schemes,
                "CAL_SELECT_all_candidate_metrics":selection,
                "CAL_AUDIT_chosen":audit,
                "CAL_AUDIT_symmetric":audit_symmetric,
                "DEV_original_full_CAL_symmetric":prior,
                "DEV_selected_CAL_only_candidate":candidate,
                "DEV_interval_score_relative_change":candidate["interval_score90"]/prior["interval_score90"]-1,
                "weights_unchanged":True
            })
        result={
            "schema":"MODEL01-TAC-CAL-ONLY-UNCERTAINTY-001-v1",
            "study":"MODEL01-TAC-CAL-ONLY-UNCERTAINTY-001",
            "source_zip_sha256":native.SOURCE,
            "original_git_checkpoint_path":str(ORIGINAL.relative_to(ROOT)),
            "model_training_performed":False,
            "original_neural_trainable_parameters_unchanged":103252,
            "CAL_split_ratios_chronological":[.6,.2,.2],
            "CAL_candidate_selection_min_interval_score_relative_gain":MIN_SELECT_SCORE_IMPROVEMENT,
            "CAL_selection_min_coverage":SELECTION_MIN_COVERAGE,
            "DEV_exposed_before_study":True,
            "protected_future_source_parsed":False,
            "independent_scientific_qualification":False,
            "runs":summary,
            "warnings":[
                "time-series residuals autocorrelated; finite-sample iid conformal coverage claim invalid",
                "CAL candidate selection and audit subblocks within one historical household",
                "DEV was previously exposed in old training, so this is development comparison",
                "three seeds do not establish independent site/outcome replication",
                "original ridge and persistence baselines remain unaltered",
                "reliability cannot be promoted when interval score improves but coverage worsens",
                "no 1024-step associative retention or architecture superiority has been shown"
            ]}
        args.out.parent.mkdir(parents=True,exist_ok=True)
        args.out.write_text(json.dumps(result,indent=2,sort_keys=True,allow_nan=False)+"\n")
        print(json.dumps({"schema":result["schema"],"runs":[{
            "seed":r["seed"],"chosen":r["chosen_without_DEV"],
            "CAL_AUDIT":r["CAL_AUDIT_chosen"],
            "DEV_original":r["DEV_original_full_CAL_symmetric"],
            "DEV_candidate":r["DEV_selected_CAL_only_candidate"],
            "DEV_score_delta_fraction":r["DEV_interval_score_relative_change"]
        } for r in summary]},sort_keys=True,allow_nan=False))
        return 0
    except (IntegrityError,native.StudyError,native.uci.SourceError,OSError,ValueError,TypeError,RuntimeError) as ex:
        print(json.dumps({"status":"BLOCKED_OR_FAILED","reason":str(ex)}),file=sys.stderr)
        return 2

if __name__=="__main__":
    raise SystemExit(main())
