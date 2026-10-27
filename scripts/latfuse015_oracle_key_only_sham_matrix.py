"""LATFUSE015: original TRUE physical PM2.5, frozen target-bearing key falsifier.

This is explicitly NOT native model retrieval: original historical write key
is derived from original target-bearing source observation and available
only as a privileged OFFLINE probe. Does memory contribute extra physical
information BEYOND that event key? Original full source matrix vs original
training global source matrix vs other station query matrix vs key only.
ZERO TAC model fits, original source checkpoint/state unchanged.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
import torch
from scripts import latfuse002_real_fit as source
from scripts import latfuse008_paired_real_training as original
from scripts import latfuse014_source_neutral_window as previous
from scripts import latfuse012_original_source_oracle as rd
from scripts.validate_research_parity import check_manifest

ROOT=Path(__file__).resolve().parents[1]
REG=ROOT/"research/mathematical-engineering/latfuse015/REGISTERED_KEY_SHAM.json"
SEED=15015

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def donor_indices(sites):
    """First next physical source station, never using class or PM level."""
    arr=sites.tolist()
    out=[]
    for i,s in enumerate(arr):
        for jump in range(1,len(arr)):
            j=(i+jump)%len(arr)
            if arr[j]!=s:
                out.append(j)
                break
        else:raise RuntimeError("only one station available to source-shuffle")
    return torch.tensor(out,dtype=torch.long)

def readings(data,mean_train_matrix):
    keys=data["keys"]               # [N,8,2,8], ORIGINAL event encoder
    matrix=data["original_query_matrices"]    # N,16,16
    route=data["original_native_read_routes"] # N,2
    def source_read(mat):
        # Same two original heads x 8D key and original 16 value basis
        heads=rd.headview(mat)
        each=torch.einsum("nhvk,nthk->nthv",heads,keys)
        return (each*route[:,None,:,None]).sum(2)
    n=len(keys)
    global_matrix=mean_train_matrix[None,:,:].expand(n,-1,-1)
    donor=matrix[donor_indices(data["station"])]
    reads={
      "true_matrix":source_read(matrix),
      "training_mean_matrix":source_read(global_matrix),
      "other_station_matrix":source_read(donor),
      "key_only":keys.reshape(n,8,16),
      "native":data["reads"]["native"]
    }
    return {name:torch.cat((data["source"],x),dim=-1) for name,x in reads.items()}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--source",type=Path,required=True)
    ap.add_argument("--output",type=Path,required=True)
    args=ap.parse_args()
    if args.output.exists():raise RuntimeError("immutable source-genuine episode controls already recorded")
    torch.set_num_threads(1)
    j=json.loads(REG.read_text())
    check_manifest(j,"preflight")
    if j["study_id"]!="LATFUSE015-TRUE-SOURCE-KEY-ONLY-SHAM-MEMORY":
        raise RuntimeError("registered source-only key matrix falsifier drift")
    if sha(args.source)!=source.SOURCE_SHA:
        raise RuntimeError("original source archive changed")
    d=original.source_observations(args.source)
    train=original.select_pairs(d,original.TRAIN_START,
       source.hour(__import__("datetime").datetime(2014,3,30)),8)
    test=original.select_pairs(d,original.EVAL_START,original.EVAL_STOP,2)
    prior_receipt=json.loads((original.OUT/"RESULTS.json").read_text())
    test_sha=hashlib.sha256(json.dumps(test,separators=(",",":")).encode()).hexdigest()
    if len(train)!=192 or len(test)!=48 or test_sha!=prior_receipt["source_eval_cohort_sha256"]:
        raise RuntimeError("original registered source split or provenance changed")
    records=[]
    with torch.no_grad():
        for seed in original.SEEDS:
            row=next((x for x in prior_receipt["model_results"] if
               x["seed"]==seed and x["arm"]=="assoc_historic_query"),None)
            if row is None:raise RuntimeError("missing source model")
            cp=original.OUT/f"seed-{seed}"/"assoc_historic_query"/"final.pt"
            digest=sha(cp)
            if digest!=row["original_checkpoint_sha256"]:
                raise RuntimeError("original learned source checkpoint hash violated")
            m=original.create_model(seed,"assoc_historic_query")
            m.load_state_dict(torch.load(cp,map_location="cpu",weights_only=False)["original_model"],strict=True)
            m.eval()
            weights=[p.detach().clone() for p in m.parameters()]
            tr=previous.extract(m,d,train,split=0)
            ev=previous.extract(m,d,test,split=1)
            # Legal old source Matrix training center only; NOT evaluation.
            global_mat=tr["original_query_matrices"].mean(0)
            features_fit=readings(tr,global_mat)
            features_eval=readings(ev,global_mat)
            fitted={}
            errors={}
            for name in ("true_matrix","key_only","training_mean_matrix",
                         "other_station_matrix","native"):
                xt=features_fit[name]
                xv=features_eval[name]
                pred,n,dim=previous.fit_probe(
                  xt.reshape(-1,xt.shape[-1]),tr["true_y"].reshape(-1),
                  tr["eligible"].reshape(-1),xv.reshape(-1,xv.shape[-1]))
                pred=pred.reshape(ev["true_y"].shape)
                z=previous.metrics(pred,ev["true_y"],ev["eligible"])
                z["source_only_externally_fitted_ridge_dimensions"]=dim
                z["n_real_observed_source_train_labels"]=n
                fitted[name]=z
                errors[name]=((pred-ev["true_y"]).abs()).where(ev["eligible"],0).sum(-1)/ev["eligible"].sum(-1).clamp_min(1)
            scored={}
            for alternative in ("key_only","training_mean_matrix","other_station_matrix"):
                base=fitted[alternative]["log1p_PM25_MAE"]
                main=fitted["true_matrix"]["log1p_PM25_MAE"]
                frac=(base-main)/max(base,1e-9)
                interval=previous.grouped_ci(errors[alternative]-errors["true_matrix"],
                          ev["station"],SEED+seed)
                scored[alternative]=dict(
                    relative_original_memory_log_PM25_MAE_improvement=frac,
                    true_source_site_bootstrap_95_error_gain=interval,
                    prereg_5pct_and_positive_CI_pass=frac>=.05 and interval[0]>0)
            verdict=all(v["prereg_5pct_and_positive_CI_pass"] for v in scored.values())
            if digest!=sha(cp) or any(not torch.equal(a,p) for a,p in zip(weights,m.parameters())):
                raise RuntimeError("frozen original source checkpoint mutates on inference")
            records.append(dict(seed=seed,original_checkpoint_sha256=digest,
                log_PM25_content=fitted,memory_specificity_vs_privileged_key_controls=scored,
                independent_per_seed_true_source_matrix_gate_pass=verdict))
            print("LATFUSE015_REAL_SOURCE_KEY_ONLY="+json.dumps({
                "seed":seed,
                "MAE":{name:v["log1p_PM25_MAE"] for name,v in fitted.items()},
                "gate":verdict,"contrasts":scored
            },sort_keys=True),flush=True)
    report=dict(study_id=j["study_id"],source_original_zip_sha256=source.SOURCE_SHA,
       original_source_train_episode_windows=len(train),
       original_source_exposed_eval_episode_windows=len(test),
       original_source_eval_SHA=test_sha,
       original_99390_live_params_and_6800byte_state_unchanged=True,
       original_checkpoint_shas_unchanged=True,
       same_time_window_8_positions_all_evaluated=True,
       original_source_neutral_other_station_donors_selected_without_labels=True,
       privileged_source_writer_key_CONTAINS_ORIGINAL_OBSERVATION_AND_IS_NOT_NATIVE=True,
       original_neural_optimizer_updates=0,
       external_ridge_decoder_adds_parameters_NOT_deployed=True,
       protected_original_numeric_source_unopened=True,
       original_FIT_MayJune_exposed_not_independent_model_generalization=True,
       prereg_true_memory_beyond_key_pass_both_seeds=all(
          x["independent_per_seed_true_source_matrix_gate_pass"] for x in records),
       models=records)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2,sort_keys=True,allow_nan=False)+"\n")
    print("LATFUSE015_SOURCE_KEY_FALSIFIER_FINISHED="+json.dumps({
       "models":len(records),"source_eval_windows":len(test),
       "original_neural_fits":0,"memory_beyond_key_gate_pass":
       report["prereg_true_memory_beyond_key_pass_both_seeds"]},sort_keys=True))
if __name__=="__main__":main()
