"""LATFUSE005b: source-authentic FP32 affine memory erase/write separation.

Run AFTER the preregistered genuine-event long-delay source experiment.
Original 8 trained LATFUSE004 models unchanged, original FIT cohort exact.
Decomposes learned input-conditioned right-affine transfer of pre-anchor
matrix differences into per-head scalar retention alpha and key-directed
delta erase (I - beta k kT). Mathematically scoped: does not prove factual
historical event memory, correct forecast ability, or counterfactual actions.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import torch
import torch.nn.functional as F

from scripts import latfuse002_real_fit as original
from scripts import latfuse004_factorial_training as trained
from scripts import latfuse005_source_event_delay as first
from scripts.validate_research_parity import check_manifest
from tac_osm.latent_fusion_004_factorial import FactorialFusion,ARMS,head_weights

ROOT=Path(__file__).resolve().parents[1]
REG=ROOT/"research/mathematical-engineering/latfuse005/REGISTERED_ERASURE_DECOMPOSITION.json"
PRIOR=ROOT/"research/mathematical-engineering/latfuse005/RESULTS.json"

def hash_file(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def headview(m,mat):
    return mat.reshape(mat.shape[0],16,m.head_count,m.key_width).permute(0,2,1,3).contiguous()

def ratio(mat,old):
    # Per-example normalized matrix Frobenius survival, never a retrieval score.
    return mat.flatten(1).norm(dim=1)/old.flatten(1).norm(dim=1).clamp_min(1e-9)

def summarize(m,base,delta,retain,overwrite,labels,mean_alpha,mean_beta):
    result={}
    for label in ("observed_pm25_high","same_station_clock_ordinary"):
        selection=torch.tensor([x==label for x in labels],dtype=torch.bool)
        b=base[selection];d=delta[selection]
        a=retain[selection];e=overwrite[selection]
        r=ratio(d,b)
        ar=ratio(a,b)
        er=ratio(e,b)
        per_head=[]
        for h in range(m.head_count):
            bh=b[:,h,:,:];dh=d[:,h,:,:]
            ah=a[:,h,:,:];eh=e[:,h,:,:]
            per_head.append(dict(
                head=h,
                original_head_M_frobenius=float(bh.norm(dim=(-2,-1)).mean()),
                actual_source_affine_transport_ratio=float(ratio(dh,bh).mean()),
                retention_alpha_only_ratio=float(ratio(ah,bh).mean()),
                normalized_key_overwrite_only_ratio=float(ratio(eh,bh).mean()),
                source_measured_mean_alpha=float(mean_alpha[selection,h].mean()),
                source_measured_mean_beta=float(mean_beta[selection,h].mean())))
        result[label]=dict(n_anchors=int(selection.sum()),
            original_prefix_memory_norm_mean=float(b.flatten(1).norm(dim=1).mean()),
            actual_memory_difference_norm_ratio_mean=float(r.mean()),
            retention_only_norm_ratio_mean=float(ar.mean()),
            key_directed_erase_only_norm_ratio_mean=float(er.mean()),
            per_head=per_head)
    return result

def audit_one(m,d,cohort):
    b=len(cohort);h=m.head_count;k=m.key_width
    station=torch.tensor([row[0] for row in cohort],dtype=torch.long)
    anchor=torch.tensor([row[1] for row in cohort],dtype=torch.long)
    labels=[row[2] for row in cohort]
    state=m.initial(b)
    for offset in range(first.HISTORY):
        time=anchor-first.HISTORY+1+offset
        state=first.source_step(m,state,d,station,time)
    initial=headview(m,state.associative)
    live=state
    blank=state._replace(associative=torch.zeros_like(state.associative))
    eye=torch.eye(k,dtype=torch.float32).view(1,1,k,k)
    full=eye.expand(b,h,-1,-1).clone()
    directed=eye.expand(b,h,-1,-1).clone()
    scalar=torch.ones((b,h),dtype=torch.float32)
    alpha_total=torch.zeros((b,h),dtype=torch.float32)
    beta_total=torch.zeros((b,h),dtype=torch.float32)
    snapshots={}
    for offset in range(1,first.DELAY_MAX+1):
        t=anchor+offset
        x=d["values"][station,t]
        obs=d["valid"][station,t]
        elapsed=d["age"][station,t];last=d["last"][station,t]
        u=m.encode(x,obs,elapsed,last)
        key=F.normalize(m.key_map(u).reshape(b,h,k),dim=-1,eps=1e-8)
        write_w,selected=head_weights(m.write_router(u),m.log_head_temperature,
                        head_count=h,soft=m.soft_routing)
        permitted=obs.any(-1)[:,None]&selected
        retention=torch.sigmoid(m.retention_map(u))
        strength=torch.sigmoid(m.write_map(u))
        alpha=torch.where(permitted,retention.expand(-1,h),
                          torch.ones((b,h),dtype=torch.float32))
        beta=strength*write_w*permitted.to(write_w.dtype)
        projector=eye-beta[:,:,None,None]*key[:,:,:,None]*key[:,:,None,:]
        P=alpha[:,:,None,None]*projector
        full=full@P
        directed=directed@projector
        scalar=scalar*alpha
        alpha_total=alpha_total+alpha
        beta_total=beta_total+beta
        live=first.source_step(m,live,d,station,t)
        blank=first.source_step(m,blank,d,station,t)
        if offset not in first.DELAYS:continue
        after=headview(m,live.associative-blank.associative)
        expected=initial@full
        # Arithmetic checks original fitted Torch step, shape/heads and
        # original source gates (absolute+relative tolerances for long FP32).
        torch.testing.assert_close(after,expected,atol=3e-5,rtol=2e-4)
        retention_only=initial*scalar[:,:,None,None]
        overwrite_only=initial@directed
        composed=retention_only@directed
        torch.testing.assert_close(expected,composed,atol=3e-5,rtol=2e-4)
        # This test guards *associative only* perturbation: the source
        # recurrent fast, slow, entity fields must remain byte-identical
        # under the two histories to validate conditional transfer.
        if any(not torch.equal(a,c) for a,c in zip(
           (live.fast,live.slow,live.entities,live.active),
           (blank.fast,blank.slow,blank.entities,blank.active))):
            raise RuntimeError("other source recurrent paths differ; cannot isolate matrix linear operator")
        diag=summarize(m,initial,after,retention_only,overwrite_only,
                       labels,alpha_total/offset,beta_total/offset)
        snapshots[str(offset)]=dict(source_delay_hours=offset,
            original_check_abs_max=float((expected-after).abs().max()),
            source_decomposition_abs_max=float((expected-composed).abs().max()),
            actual_full_prefix_matrix_retention_ratio=float(ratio(after,initial).mean()),
            retention_alpha_only_ratio=float(ratio(retention_only,initial).mean()),
            key_directed_beta_erase_only_ratio=float(ratio(overwrite_only,initial).mean()),
            source_mean_effective_alpha=float((alpha_total/offset).mean()),
            source_mean_effective_beta=float((beta_total/offset).mean()),
            conditional_original_memory_operator_correct=True,
            identical_nonassociative_source_state=True,
            source_conditioned_groups=diag)
    return snapshots

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--source",type=Path,required=True)
    ap.add_argument("--output",type=Path,required=True)
    args=ap.parse_args()
    if args.output.exists():
        raise RuntimeError("no original memory-source-forensic overwrite")
    torch.set_num_threads(1)
    j=json.loads(REG.read_text())
    check_manifest(j,"preflight")
    if j["study_id"]!="LATFUSE005-GENUINE-SOURCE-ALPHA-BETA-SEPARATION":
        raise RuntimeError("unexpected real source-affine study")
    if hash_file(args.source)!=original.SOURCE_SHA:
        raise RuntimeError("illegal original source archive")
    first_result=json.loads(PRIOR.read_text())
    d=original.source_frames(args.source,phase="fit")
    cohort=first.source_only_cohort(d)
    if (first_result["source_cohort_count"]!=len(cohort) or
        hashlib.sha256(json.dumps(cohort,separators=(",",":")).encode()).hexdigest()!=
        first_result["cohort_sha256"]):
        raise RuntimeError("cohort changed after authentic event audit; no posthoc matching")
    fit=json.loads((trained.OUT/"RESULTS.json").read_text())
    if len(fit["models"])!=8 or fit["total_original_optimizer_updates"]!=1024:
        raise RuntimeError("original source-trained model family incomplete")
    models=[]
    with torch.no_grad():
        for seed in trained.SEEDS:
            for arm in ARMS:
                row=next((r for r in fit["models"] if r["seed"]==seed and
                          r["arm"]==arm),None)
                prior=next((r for r in first_result["models"]
                        if r["seed"]==seed and r["arm"]==arm),None)
                if row is None or prior is None:
                    raise RuntimeError("missing original source model reference")
                path=trained.OUT/f"seed-{seed}"/arm/"final.pt"
                sha=hash_file(path)
                if sha!=row["original_checkpoint_sha256"] or sha!=prior["original_checkpoint_sha256"]:
                    raise RuntimeError("original old checkpoint SHA modified")
                c=torch.load(path,map_location="cpu",weights_only=False)
                m=FactorialFusion(arm)
                m.load_state_dict(c["model"],strict=True)
                m.eval()
                old=[x.detach().clone() for x in m.parameters()]
                result=audit_one(m,d,cohort)
                if sha!=hash_file(path) or any(not torch.equal(p,z) for p,z in zip(m.parameters(),old)):
                    raise RuntimeError("original learned weights changed during source operator proof")
                output=dict(seed=seed,arm=arm,checkpoint_sha256=sha,delay_operator=result)
                models.append(output)
                print("LATFUSE005_SOURCE_OPERATOR_ARM="+json.dumps({
                    "seed":seed,"arm":arm,
                    "delays":{h:{"actual":d["actual_full_prefix_matrix_retention_ratio"],
                          "alpha_only":d["retention_alpha_only_ratio"],
                          "beta_only":d["key_directed_beta_erase_only_ratio"],
                          "numeric_abs_error_max":d["original_check_abs_max"]}
                          for h,d in result.items()}},sort_keys=True),flush=True)
    report=dict(study_id="LATFUSE005-GENUINE-SOURCE-ALPHA-BETA-SEPARATION",
        true_original_source_sha256=original.SOURCE_SHA,
        model_original_fp32_state_bytes=6800,
        original_SHA_frozen_model_count=len(models),source_cohort_count=len(cohort),
        true_source_event_pair_count=len(cohort)//2,
        original_cohort_sha256=first_result["cohort_sha256"],
        numeric_scope="original FIT observed sources only. No target numeric labels parsed after prediction, no 2015 DEV/2016-17 TEST data.",
        operator_identity="delta memory evolved by original headwise alpha(I-beta kk^T) right multiplication; scalar retention factor and directional projection commute only within a head at the same source observation",
        metric="mean per-sample prefix memory Frobenius norm ratios; not independent operator performance or retrieval",
        no_training=True,no_2015_dev_or_protected_numeric=True,
        no_predicted_outcome_utility_claim=True,no_architecture_superiority_claim=True,
        models=models)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,sort_keys=True,indent=2,allow_nan=False)+"\n")
    print("LATFUSE005_SOURCE_OPERATOR_PASS="+json.dumps({
        "original_models":len(models),"no_optimizer":True,
        "max_delta_hours":first.DELAY_MAX},sort_keys=True),flush=True)

if __name__=="__main__":
    main()
