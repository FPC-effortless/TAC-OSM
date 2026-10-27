"""LATFUSE019 frozen authentic-source Gaussian NLL attribution (NO TRAINING).

All original LATFUSE018 checkpoint hashes checked. Query source+6 future on
original H64 station-owned real sensor histories, review actual masked
standardized physical observations per station/channel. No model recalibration,
label-based changes, protected data or synthetic source claims.
"""
from __future__ import annotations
import argparse,datetime as dt,hashlib,json,math
from collections import defaultdict
from pathlib import Path
import torch
from scripts import latfuse002_real_fit as source
from scripts import latfuse008_paired_real_training as model_factory
from scripts.latfuse018_real_content_training import (
    PLAN,OUT,source_rows,EVAL_START,EVAL_STOP,BATCH)
from scripts.validate_research_parity import check_manifest

ROOT=Path(__file__).resolve().parents[1]
REG=ROOT/"research/mathematical-engineering/latfuse019/REGISTERED_UNCERTAINTY_AUDIT.json"
CHANNELS=tuple(source.CHANNELS)+("WIND_SIN","WIND_COS")
assert len(CHANNELS)==13
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def run(original_source,output):
    if output.exists():raise RuntimeError("original source NLL forensic already recorded")
    j=json.loads(REG.read_text())
    check_manifest(j,"preflight")
    if j["study_id"]!="LATFUSE019-FROZEN-REAL-FORECAST-CALIBRATION-FORENSICS":
        raise RuntimeError("forensic source prereg drift")
    if sha(original_source)!=source.SOURCE_SHA:
        raise RuntimeError("original authentic source SHA mismatch")
    torch.set_num_threads(1)
    d=model_factory.source_observations(original_source)
    plan=json.loads(PLAN.read_text())
    cases=source_rows(d,EVAL_START,EVAL_STOP,96)
    case_sha=hashlib.sha256(json.dumps(cases,separators=(",",":")).encode()).hexdigest()
    if len(cases)!=117 or plan["exposed_source_development_eval_case_SHA256"]!=case_sha:
        raise RuntimeError("source station-hour episode plan changed after trained models")
    trained=json.loads((OUT/"RESULTS.json").read_text())
    result=[]
    with torch.no_grad():
        for receipt in trained["models"]:
            seed=receipt["seed"];arch=receipt["architecture"]
            file=OUT/f"seed-{seed}"/arch/"final.pt"
            if sha(file)!=receipt["original_checkpoint_SHA256"]:
                raise RuntimeError("original independently replayed FIT model SHA changed")
            st=torch.load(file,map_location="cpu",weights_only=False)
            model=model_factory.create_model(seed,arch)
            model.load_state_dict(st["model"],strict=True)
            model.eval()
            weights=[p.detach().clone() for p in model.parameters()]
            obs=[]
            with torch.no_grad():
                for start in range(0,len(cases),BATCH):
                    group=cases[start:start+BATCH]
                    site=torch.tensor([x[0] for x in group])
                    anchor=torch.tensor([x[1] for x in group])
                    s=model.initial(len(group))
                    for off in range(-23,65):
                        when=anchor+off
                        s=model.step(s,d["values"][site,when],d["valid"][site,when],
                                    d["age"][site,when],d["last"][site,when],site)
                    known_future=source.calendar([int(x+70) for x in anchor])[:,None,:]
                    mu,sd=model.forecast(s,site,known_future)
                    mu=mu[:,0];sd=sd[:,0]
                    # Recorded original physical source future values only after
                    # the query, no outcome access while producing forecasts.
                    targets=d["values"][site,anchor+70]
                    masks=d["valid"][site,anchor+70]
                    for i,src in enumerate(site.tolist()):
                        for c in range(13):
                            if not bool(masks[i,c]):continue
                            truth=float(targets[i,c]);mean=float(mu[i,c]);scale=float(sd[i,c])
                            nll=.5*(math.log(2*math.pi)+2*math.log(scale)+((truth-mean)/scale)**2)
                            obs.append(dict(site_id=src,site_name=source.SITES[src],
                                 channel_id=c,channel=CHANNELS[c],
                                 forecast_at_hour=int(anchor[i]+70),
                                 original_UTC_datetime=(source.START+dt.timedelta(hours=int(anchor[i]+70))).isoformat(),
                                 source_observed_standardized=truth,
                                 predicted_mean_standardized=mean,
                                 predicted_positive_std=scale,
                                 absolute_z_residual=abs(truth-mean)/scale,
                                 physical_source_mask_observed=True,
                                 log_gaussian_nll=nll,
                                 within_nominal_90=abs(truth-mean)<=1.6448536269514722*scale))
            original_nll=receipt["original_actual_source_development_eval"]["64"]["original_source_future_gaussian_NLL"]
            actual_nll=sum(x["log_gaussian_nll"] for x in obs)/len(obs)
            if abs(actual_nll-original_nll)>3e-5:
                raise RuntimeError("replayed per-source original channel NLL not equal stored old true-source evaluation")
            def aggregate(records):
                n=len(records)
                return dict(observed_count=n,
                    original_mean_log_gaussian_nll=sum(x["log_gaussian_nll"] for x in records)/n,
                    original_90_interval_coverage=sum(x["within_nominal_90"] for x in records)/n,
                    mean_abs_standardized_residual=sum(abs(x["source_observed_standardized"]-
                                  x["predicted_mean_standardized"]) for x in records)/n,
                    source_mean_scale=sum(x["predicted_positive_std"] for x in records)/n,
                    max_abs_residual_over_std=max(x["absolute_z_residual"] for x in records))
            bychannel={}
            for c in range(13):
                group=[z for z in obs if z["channel_id"]==c]
                bychannel[CHANNELS[c]]=aggregate(group)
            bysite={}
            for site in range(12):
                group=[z for z in obs if z["site_id"]==site]
                if group:bysite[source.SITES[site]]=aggregate(group)
            positives=sorted((x for x in obs if x["log_gaussian_nll"]>0),
                             key=lambda x:x["log_gaussian_nll"],reverse=True)
            positive_sum=sum(x["log_gaussian_nll"] for x in positives)
            top_sorted=sorted(obs,key=lambda x:x["log_gaussian_nll"],reverse=True)
            ratio=sorted(x["absolute_z_residual"] for x in obs)
            index=lambda quant:ratio[min(len(ratio)-1,int((len(ratio)-1)*quant))]
            topk=max(1,math.ceil(.01*len(obs)))
            trimmed=top_sorted[topk:]
            row=dict(seed=seed,arm=arch,
               original_checkpoint_SHA256=receipt["original_checkpoint_SHA256"],
               original_2014_true_physical_observed_future_elements=len(obs),
               full_original_real_source_metrics=aggregate(obs),
               channel_level_original_source_metrics=bychannel,
               station_level_original_source_metrics=bysite,
               top_positive_nll_mass_fraction_top1=(positives[0]["log_gaussian_nll"]/positive_sum if positives else 0),
               top_positive_nll_mass_fraction_top10=(sum(x["log_gaussian_nll"] for x in positives[:10])/positive_sum if positives else 0),
               absolute_standardized_residual_over_scale_quantiles={"p90":index(.9),"p99":index(.99),
                                                                      "max":ratio[-1]},
               largest_observed_real_NLL_source_rows=top_sorted[:10],
               top1pct_dropped_for_DIAGNOSTIC_ONLY_not_valid_metric=dict(
                   removed_largest_NLL_element_count=topk,
                   observed_mean_NLL_after_ignoring_largest=float(sum(z["log_gaussian_nll"] for z in trimmed)/len(trimmed))),
               H64_NLL_original_trained_checkpoint_exact_replay=True,
               no_tuning_retraining_or_source_label_imputation=True)
            if any(not torch.equal(x,y) for x,y in zip(weights,model.parameters())):
                raise RuntimeError("no-training forensic changed trained original model parameters")
            result.append(row)
            print("LATFUSE019_ORIGINAL_SOURCE_NLL="+json.dumps({
               "seed":seed,"arm":arch,"source_count":len(obs),
               "mean_NLL":actual_nll,
               "coverage":row["full_original_real_source_metrics"]["original_90_interval_coverage"],
               "top10_positive_nll_mass":row["top_positive_nll_mass_fraction_top10"],
               "worst_channel":max(bychannel.items(),key=lambda x:x[1]["original_mean_log_gaussian_nll"])[0],
               "worst_source":max(bysite.items(),key=lambda x:x[1]["original_mean_log_gaussian_nll"])[0]},
               sort_keys=True),flush=True)
    saved=dict(study_id=j["study_id"],source_sha256=source.SOURCE_SHA,
         real_source_2014_eval_SHA256=case_sha,
         original_recorded_source_eval_cases=len(cases),
         original_model_checkpoint_shas_and_all_NLL_exact_replayed=True,
         genuine_original_source_development_ALREADY_EXPOSED=True,
         no_neural_model_retraining_or_calibration_fit=True,
         no_numeric_original_2015_or_protected_2016_17_read=True,
         original_calibration_or_world_model_superiority_NOT_established=True,
         models=result)
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(saved,indent=2,sort_keys=True,allow_nan=False)+"\n")
    print("LATFUSE019_REAL_SOURCE_UNCERTAINTY_DIAGNOSTIC_PASS")

def main():
    a=argparse.ArgumentParser()
    a.add_argument("--source",type=Path,required=True)
    a.add_argument("--output",type=Path,required=True)
    args=a.parse_args()
    run(args.source,args.output)
if __name__=="__main__":main()
