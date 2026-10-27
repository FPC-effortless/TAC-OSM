"""LATFUSE014: label-INDEPENDENT within-window real-source retrieval controls.

Frozen original two LATFUSE008 source-trained memory models, no optimizer.
Eight actual writes from offsets[0,8,..56], query after actual64h. All
candidate targets evaluated, not just selected extreme source anchor.
Writer keys/values are privileged OFFLINE reference only; real deployed
historical query is constructed ONLY from known past calendar+current state.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
import torch
import torch.nn.functional as F
from scripts import latfuse002_real_fit as source
from scripts import latfuse008_paired_real_training as fit
from scripts import latfuse012_original_source_oracle as prev
from scripts.validate_research_parity import check_manifest
from tac_osm.latent_fusion_004_factorial import head_weights

ROOT=Path(__file__).resolve().parents[1]
REG=ROOT/"research/mathematical-engineering/latfuse014/REGISTERED_WINDOW_CONTENT.json"
OFFSETS=tuple(range(0,57,8))
SEED=14014
BOOT=2000
KINDS=("native","oracle","next8_wrong","random","nonassoc","source_only")
assert len(OFFSETS)==8

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def extract(m,d,rows,*,split):
    # Original source high/ordinary cohort selection is historical; negatives
    # are now seven *unconditioned* observed writes at deterministic +8h
    # source offsets, with NO source target high/low selection.
    pack={k:[] for k in KINDS}
    allkeys=[];allvalues=[];ys=[];sites=[];eligible=[]
    persistence=[];original_site=[]
    for start in range(0,len(rows),12):
        batch=rows[start:start+12]
        b=len(batch)
        src=torch.tensor([a[0] for a in batch],dtype=torch.long)
        ts=torch.tensor([a[1] for a in batch],dtype=torch.long)
        s=m.initial(b)
        keys=[None]*8;vals=[None]*8;is_source=[None]*8
        for relative in range(-23,65):
            at=ts+relative
            if relative in OFFSETS:
                j=OFFSETS.index(relative)
                x=d["values"][src,at]
                obs=d["valid"][src,at]
                u=m.encode(x,obs,d["age"][src,at],d["last"][src,at])
                keys[j]=F.normalize(m.key_map(u).reshape(b,2,8),dim=-1,eps=1e-8)
                vals[j]=torch.tanh(m.val_map(u))
                is_source[j]=obs.any(-1)
            s=m.step(s,d["values"][src,at],d["valid"][src,at],
                     d["age"][src,at],d["last"][src,at],src)
        if any(v is None for v in keys+vals+is_source):
            raise RuntimeError("physical source write-window missing candidate")
        key=torch.stack(keys,dim=1)       # B,8,2,8
        value=torch.stack(vals,dim=1)     # B,8,16
        has_write=torch.stack(is_source,dim=1)
        # Keep every original source case. Source masks establish eligibility.
        # No written candidate id drawn using true PM2.5 targets.
        if not bool(has_write.all()):
            raise RuntimeError("original registered site has no observed sensor for genuine candidate write")
        clock=torch.stack([source.calendar([int(x+off) for x in ts])
                           for off in OFFSETS],dim=1)
        joint=torch.cat((s.fast,s.slow),dim=-1)
        query=F.normalize((torch.tanh(m.historic_clock_query(clock))+
               m.belief_query(joint)[:,None,:]).reshape(b,8,2,8),dim=-1,eps=1e-8)
        route,_=head_weights(m.read_router(joint),m.log_head_temperature,
                            head_count=2,soft=True)
        random=[]
        for sid,at in zip(src.tolist(),ts.tolist()):
            many=[]
            for j in range(8):
                g=torch.Generator().manual_seed((SEED*9199+split*3907+
                    sid*13001+at*79+j*97)%((1<<63)-1))
                many.append(F.normalize(torch.randn(2,8,generator=g),dim=-1))
            random.append(torch.stack(many))
        random=torch.stack(random)
        mismatched=key.roll(-1,dims=1)
        # Same original matrix and HEAD ROUTER weights for ALL query arms,
        # 16D output in identical value basis and scaling.
        heads=prev.headview(s.associative)
        def read(k):
            each=torch.einsum("bhvk,bt hk->bt hv",heads,k)
            return (each*route[:,None,:,None]).sum(2)
        # No target physical sensor value is supplied to native query.
        for mode,k in (("native",query),("oracle",key),
                       ("next8_wrong",mismatched),("random",random)):
            val=read(k)
            if val.shape!=(b,8,16):raise RuntimeError("bad read basis")
            # Save base nuisance/read in features after source legal input.
            pack[mode].append(val)
        curr=ts+64
        current=d["values"][src,curr]
        mask=d["valid"][src,curr].float()
        last=d["last"][src,curr+1]  # original legal previous/current most recent
        age=d["age"][src,curr]
        base=torch.cat((current,mask,last,age,
            F.one_hot(src,num_classes=12).float()),dim=-1)
        entity=s.entities.gather(1,src[:,None,None].expand(-1,1,80)).squeeze(1)
        entity=entity*s.active.gather(1,src[:,None]).float()
        nonassoc=torch.cat((s.fast,s.slow,entity),dim=-1)
        source_feats=torch.cat((base[:,None,:].expand(-1,8,-1),clock),dim=-1)
        pack["source_only"].append(source_feats)
        pack["nonassoc"].append(torch.cat((
            source_feats,nonassoc[:,None,:].expand(-1,8,-1)),dim=-1))
        # Continuous recorded original source target consulted ONLY after
        # original model s and ALL read/clock outputs are constructed.
        target=torch.stack([d["raw"][src,ts+off,0] for off in OFFSETS],dim=1)
        valid_y=torch.isfinite(target)
        y=torch.log1p(target.clamp_min(0))
        # Legal original last-known PM2.5 at query point, independent target.
        raw_last=torch.expm1((last[:,0]*d["spread"][0]+d["center"][0]).clamp(-12,12)).clamp_min(0)
        allkeys.append(key);allvalues.append(value)
        ys.append(y);eligible.append(valid_y)
        persistence.append(raw_last[:,None].expand(-1,8))
        sites.extend(src.tolist());original_site.extend([i for i in src.tolist() for _ in range(8)])
    pack={k:torch.cat(v) for k,v in pack.items()}
    if any(not torch.isfinite(z).all() for z in pack.values()):
        raise RuntimeError("source model non-finite frozen feature")
    return dict(reads={k:pack[k] for k in ("native","oracle","next8_wrong","random")},
        source=pack["source_only"],nonassoc=pack["nonassoc"],
        values=torch.cat(allvalues),keys=torch.cat(allkeys),
        true_y=torch.cat(ys),eligible=torch.cat(eligible),
        persistence=torch.cat(persistence),cases=len(rows),
        station=torch.tensor(sites),site_target=torch.tensor(original_site))

def fit_probe(tx,ty,mask,vx):
    # Fully predeclared fixed lambda1 source-only dual ridge, no held-out tuning.
    train=tx[mask]
    y=ty[mask]
    if len(y)<80:raise RuntimeError("too few observed true training source targets")
    center=train.mean(0)
    spread=train.std(0,unbiased=False).clamp_min(1e-4)
    x=((train-center)/spread).clamp(-6,6)/math.sqrt(train.shape[-1])
    z=((vx-center)/spread).clamp(-6,6)/math.sqrt(train.shape[-1])
    mu=y.mean();sd=y.std(unbiased=False).clamp_min(1e-4)
    out=z@x.T@torch.linalg.solve(x@x.T+torch.eye(len(x)),((y-mu)/sd)[:,None])
    result=(out[:,0]*sd+mu).clamp(0,9)
    if not bool(torch.isfinite(result).all()):raise RuntimeError("ridge source numeric invalid")
    return result,len(y),len(tx[0])

def metrics(pred,truth,eligible):
    pred=pred[eligible]
    true=truth[eligible]
    return dict(eligible_original_PM25_count=len(true),
       log1p_PM25_MAE=float((pred-true).abs().mean()),
       original_raw_ugm3_MAE=float((torch.expm1(pred)-torch.expm1(true)).abs().mean()),
       log1p_PM25_MSE=float((pred-true).square().mean()))

def grouped_ci(gains,stations,seed):
    # Site bootstrap of paired episodes, NOT independent across nearby sites.
    ids=sorted(set(stations.tolist()))
    if len(ids)<5:raise RuntimeError("insufficient physical station blocks")
    groups=[torch.where(stations==s)[0] for s in ids]
    g=torch.Generator().manual_seed(seed)
    draws=[]
    for i in range(BOOT):
        choose=torch.randint(0,len(groups),(len(groups),),generator=g)
        idx=torch.cat([groups[int(j)] for j in choose])
        draws.append(float(gains[idx].mean()))
    values=sorted(draws)
    return [values[int(.025*(BOOT-1))],values[int(.975*(BOOT-1))]]

def content_top1(reads,values,mu,write_valid):
    # Center ALL writer VALUES using only original genuine March training
    # cohort, never May evaluation values. Same 16D coordinates for read.
    # Each of the 8 *original true* temporal target slots occurs equally.
    rr=F.normalize(reads-mu,dim=-1,eps=1e-7)
    vv=F.normalize(values-mu,dim=-1,eps=1e-7)
    matrix=torch.einsum("btv,bsv->bts",rr,vv)
    matrix=matrix.masked_fill(~write_valid[:,None,:],-1e9)
    maxes=matrix.max(-1).values
    wins=(matrix>=maxes[:,:,None]-1e-7).float()
    hit=(wins.diagonal(dim1=1,dim2=2)/wins.sum(-1).clamp_min(1)).mean(-1)
    # balanced over all positions; untrained nearest-value identity can be
    # artificially high with strongly shared latent value means.
    return hit,dict(top1=float(hit.mean()),
        requested_position_top1=[float(hit_position) for hit_position in (
          wins.diagonal(dim1=1,dim2=2)/wins.sum(-1).clamp_min(1)).mean(0)],
        centered_candidate_offdiagonal_abs_cos=float(
          (torch.einsum("btv,bsv->bts",vv,vv).abs().sum((-1,-2))-8).mean()/56),
        value_train_center_only=True)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--source",type=Path,required=True)
    ap.add_argument("--output",type=Path,required=True)
    a=ap.parse_args()
    if a.output.exists():raise RuntimeError("source measured LATFUSE014 original result is immutable")
    torch.set_num_threads(1)
    mfest=json.loads(REG.read_text())
    check_manifest(mfest,"preflight")
    if (mfest["study_id"]!="LATFUSE014-SOURCE-NEUTRAL-EIGHT-WINDOW-CONTENT"
        or mfest["requested_past_eight_timestamps_offsets"]!=list(OFFSETS)):
        raise RuntimeError("registered source same-window keys drift")
    if sha(a.source)!=source.SOURCE_SHA:raise RuntimeError("unauthorized physical source zip")
    d=fit.source_observations(a.source)
    train=fit.select_pairs(d,fit.TRAIN_START,source.hour(__import__("datetime").datetime(2014,3,30)),8)
    test=fit.select_pairs(d,fit.EVAL_START,fit.EVAL_STOP,2)
    receipts=json.loads((fit.OUT/"RESULTS.json").read_text())
    testsha=hashlib.sha256(json.dumps(test,separators=(",",":")).encode()).hexdigest()
    if len(test)!=48 or testsha!=receipts["source_eval_cohort_sha256"] or len(train)<24:
        raise RuntimeError("original source cohort no longer matched")
    if max(t for _,t,_ in train)+64>=min(t for _,t,_ in test):
        raise RuntimeError("physical source split not strictly chronologically separated")
    results=[]
    with torch.no_grad():
        for seed in fit.SEEDS:
            row=next((p for p in receipts["model_results"] if p["seed"]==seed and
                      p["arm"]=="assoc_historic_query"),None)
            if row is None:raise RuntimeError("original reference missing")
            cp=fit.OUT/f"seed-{seed}"/"assoc_historic_query"/"final.pt"
            digest=sha(cp)
            if digest!=row["original_checkpoint_sha256"]:
                raise RuntimeError("original fitted model checkpoint SHA violation")
            model=fit.create_model(seed,"assoc_historic_query")
            model.load_state_dict(torch.load(cp,map_location="cpu",weights_only=False)["original_model"],strict=True)
            model.eval()
            original_params=[p.detach().clone() for p in model.parameters()]
            tr=extract(model,d,train,split=0)
            ev=extract(model,d,test,split=1)
            # Original training-source mean across all eight writer values.
            mu=tr["values"].reshape(-1,16).mean(0)
            if not torch.isfinite(mu).all():raise RuntimeError("source writer center non-finite")
            write_valid=torch.ones_like(ev["eligible"])
            top1={}
            scored={}
            for mode in ("native","oracle","next8_wrong","random"):
                hit,detail=content_top1(ev["reads"][mode],ev["values"],mu,write_valid)
                detail["station_block_bootstrap_95"]=grouped_ci(hit-.125,ev["station"],SEED+seed)
                detail["site_window_count"]=len(hit)
                top1[mode]=detail
                xx=torch.cat((tr["source"],tr["reads"][mode]),dim=-1)
                zz=torch.cat((ev["source"],ev["reads"][mode]),dim=-1)
                yp,n,dim=fit_probe(xx.reshape(-1,xx.shape[-1]),tr["true_y"].reshape(-1),
                     tr["eligible"].reshape(-1),zz.reshape(-1,zz.shape[-1]))
                yp=yp.reshape(ev["true_y"].shape)
                scored[mode]=metrics(yp,ev["true_y"],ev["eligible"])
                scored[mode]["source_train_used_true_observed_targets"]=n
                scored[mode]["external_source_ridge_input_dim"]=dim
                scored[mode]["ridge_is_NOT_native_model"]=True
                scored[mode]["site_error"]=((yp-ev["true_y"]).abs()).where(ev["eligible"],0).sum(-1)/ev["eligible"].sum(-1).clamp_min(1)
                scored[mode]["prediction"]=yp
            for mode in ("source_only","nonassoc"):
                xx=tr["source"] if mode=="source_only" else tr["nonassoc"]
                zz=ev["source"] if mode=="source_only" else ev["nonassoc"]
                pred,n,dim=fit_probe(xx.reshape(-1,xx.shape[-1]),tr["true_y"].reshape(-1),
                      tr["eligible"].reshape(-1),zz.reshape(-1,zz.shape[-1]))
                pred=pred.reshape(ev["true_y"].shape)
                scored[mode]=metrics(pred,ev["true_y"],ev["eligible"])
                scored[mode]["source_train_used_true_observed_targets"]=n
                scored[mode]["external_source_ridge_input_dim"]=dim
                scored[mode]["site_error"]=((pred-ev["true_y"]).abs()).where(ev["eligible"],0).sum(-1)/ev["eligible"].sum(-1).clamp_min(1)
            # Physical prior/current persistence is legal present stream,
            # when target time unknown: constant same last observed PM.
            observed=ev["eligible"]
            loglast=torch.log1p(ev["persistence"].clamp_min(0))
            scored["persistence"]=metrics(loglast,ev["true_y"],observed)
            gate={}
            for mode in ("next8_wrong","random","nonassoc"):
                baseline=scored[mode]["log1p_PM25_MAE"]
                oracle_mae=scored["oracle"]["log1p_PM25_MAE"]
                improvement=(baseline-oracle_mae)/max(baseline,1e-8)
                interval=grouped_ci(scored[mode]["site_error"]-
                     scored["oracle"]["site_error"],ev["station"],SEED+seed)
                gate[mode]=dict(log_PM25_MAE_relative_reduction=improvement,
                                source_station_bootstrap_CI_error_gain=interval,
                                pass_5pct_and_positive_CI=improvement>=.05 and interval[0]>0)
            top_ci=top1["oracle"]["station_block_bootstrap_95"]
            primary=all(v["pass_5pct_and_positive_CI"] for v in gate.values()) and (
                  top1["oracle"]["top1"]>=.225 and top_ci[0]>0)
            # Remove raw prediction/intermediate arrays from results.
            for name in scored:
                scored[name].pop("site_error",None)
                scored[name].pop("prediction",None)
            if sha(cp)!=digest or any(not torch.equal(x,p) for x,p in zip(original_params,model.parameters())):
                raise RuntimeError("original TAC-OSM fitted model weights mutated")
            results.append(dict(seed=seed,original_checkpoint_sha256=digest,
                   original_train_eight_window_count=tr["cases"],
                   original_eval_eight_window_count=ev["cases"],
                   requested_source_offsets=list(OFFSETS),centered_identity_top1=top1,
                   actual_source_content= scored,
                   privileged_oracle_v_unbiased_intervening_gate=gate,
                   original_source_H64_per_seed_gate_pass=primary))
            print("LATFUSE014_ORIGINAL_SOURCE="+json.dumps({
                "seed":seed,"source_train_windows":tr["cases"],
                "native_top1":top1["native"]["top1"],
                "oracle_top1":top1["oracle"]["top1"],
                "neutral_wrong_top1":top1["next8_wrong"]["top1"],
                "oracle_logged_PM_MAE":scored["oracle"]["log1p_PM25_MAE"],
                "source_wrong_logged_PM_MAE":scored["next8_wrong"]["log1p_PM25_MAE"],
                "source_nonassoc_logged_PM_MAE":scored["nonassoc"]["log1p_PM25_MAE"],
                "predeclared_gate_pass":primary},sort_keys=True),flush=True)
    report=dict(study_id=mfest["study_id"],original_real_source_zip_sha256=source.SOURCE_SHA,
          original_source_train_windows=len(train),original_exposed_eval_windows=len(test),
          native_model_count=2,model_original_6800byte_state_and_99390_deployed_parameters=True,
          source_neutral_negative_offsets=list(OFFSETS),
          query_delay=64,
          within_window_candidate_target_positions_all_evaluated_equally=True,
          original_source_eval_SHA=testsha,
          no_original_Tac_model_optimizer_updates=True,
          oracle_write_keys_are_privileged_offline_diagnostic=True,
          sidecar_ridge_is_extra_NOT_deployed=True,
          source_train_only_value_center=True,
          not_independent_neural_model_evaluation_original_FIT_exposed=True,
          original_protected_2015_2017_sensor_numerics_not_read=True,
          registered_both_seed_gate_pass=all(r["original_source_H64_per_seed_gate_pass"] for r in results),
          learned_models=results)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(report,sort_keys=True,indent=2,allow_nan=False)+"\n")
    print("LATFUSE014_SOURCE_NEUTRAL_RETRIEVAL_PASS="+json.dumps({
       "models":2,"source_train_windows":len(train),"source_eval_windows":len(test),
       "gate_pass":report["registered_both_seed_gate_pass"],"no_model_optimizer":True},sort_keys=True))
if __name__=="__main__":main()
