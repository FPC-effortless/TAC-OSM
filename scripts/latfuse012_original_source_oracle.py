"""LATFUSE012 source-authentic oracle-write/read capacity forensic, NO MODEL TRAINING.

Privilege oracle old write-key ONLY for read-only diagnosis. Never native query.
Original UCI501 FIT May-June2014 exactly 24 source-matched event/control pairs.
Frozen original LATFUSE008 assoc-trained weights x2 and ORIGINAL UNTRAINED
initializations (4 seeds). Preserves 6800B state, protected data unopened.
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
from scripts import latfuse008_paired_real_training as fitted
from scripts import latfuse011_real_memory_falsifiers as eleventh
from scripts.validate_research_parity import check_manifest
from tac_osm.latent_fusion_004_factorial import head_weights

ROOT=Path(__file__).resolve().parents[1]
REG=ROOT/"research/mathematical-engineering/latfuse012/REGISTERED_ORACLE.json"
DELAYS=(16,64,128,512)
UNTRAINED=(80701,80702,81901,81902)
BOOTSTRAPS=4000
BOOT_SEED=12012

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def headview(matrix):
    return matrix.reshape(matrix.shape[0],16,2,8).permute(0,2,1,3)

def read_value(matrix,key,route):
    heads=headview(matrix)
    return (torch.einsum("bhvk,bhk->bhv",heads,key)*route[:,:,None]).sum(1)

def directional_write(m,state,d,site,time):
    """Exactly source-conditioned original delta-write residual, NOT alpha decay."""
    x=d["values"][site,time]
    valid=d["valid"][site,time]
    u=m.encode(x,valid,d["age"][site,time],d["last"][site,time])
    key=F.normalize(m.key_map(u).reshape(len(site),2,8),dim=-1,eps=1e-8)
    value=torch.tanh(m.val_map(u))
    w,selected=head_weights(m.write_router(u),m.log_head_temperature,
                            head_count=2,soft=True)
    good=valid.any(-1)[:,None]&selected
    alpha=torch.where(good,
        torch.sigmoid(m.retention_map(u)).expand(-1,2),
        torch.ones_like(w))
    beta=torch.sigmoid(m.write_map(u))*w*good.to(w.dtype)
    prev=read_value_head(state.associative,key)
    write_resid=beta[:,:,None]*(value[:,None,:]-alpha[:,:,None]*prev)
    norm=write_resid.flatten(1).norm(dim=1)
    return dict(key=key,value=value,alpha=alpha,beta=beta,
                residual=write_resid,norm=norm)

def read_value_head(matrix,key):
    return torch.einsum("bhvk,bhk->bhv",headview(matrix),key)

def normalized_cos(x,y):
    return F.cosine_similarity(x,y,dim=-1,eps=1e-7)

def station_block_ci(case_station,pair_wins):
    # Paired labels share physical source, episodes may also be correlated
    # across stations via regional weather; this is descriptive only.
    unique=sorted(set(case_station.tolist()))
    if len(unique)<3:raise RuntimeError("missing source station blocks")
    groups=[torch.where(case_station==s)[0] for s in unique]
    gen=torch.Generator().manual_seed(BOOT_SEED)
    simulations=[]
    for _ in range(BOOTSTRAPS):
        idx=torch.randint(0,len(groups),(len(groups),),generator=gen)
        selected=torch.cat([groups[int(i)] for i in idx])
        simulations.append(float(pair_wins[selected].mean()))
    t=torch.tensor(simulations).sort().values
    return [float(t[int(.025*(BOOTSTRAPS-1))]),
            float(t[int(.975*(BOOTSTRAPS-1))])]

def simple_covariate_fit(d,site,anchor,prob):
    # Descriptive fitted regression to see whether chosen source pair days and
    # prehistory correlate with the historic classifier's score inversion.
    cov=[]
    for s,t in zip(site.tolist(),anchor.tolist()):
        sample=d["raw"][s,t-24:t,0]
        obs=sample[torch.isfinite(sample)]
        if not len(obs):raise RuntimeError("unobservable source-real preanchor PM2.5")
        cov.append(float(obs.mean()))
    pm=torch.tensor(cov)
    day=(anchor[::2]-anchor[1::2]).float()/24
    pre=(pm[::2]-pm[1::2])
    current=d["raw"][site,anchor+64,0]
    if not bool(torch.isfinite(current).all()):
        # Do not leak/impute target with anchor. Recorded source last is legal
        # but for descriptive confound regression use observed flag + zero
        # imputation after centering on genuinely available readout means.
        seen=torch.isfinite(current)
        fill=torch.where(seen,current,torch.zeros_like(current))
        now=fill[::2]-fill[1::2]
    else:
        now=current[::2]-current[1::2]
    X=torch.stack((pre,day,now),dim=-1)
    X=(X-X.mean(0))/X.std(0,unbiased=False).clamp_min(1e-4)
    X=torch.cat((torch.ones((len(day),1)),X),dim=-1)
    target=prob[::2]-prob[1::2]
    weights=torch.linalg.solve(X.T@X+torch.eye(4)*.01,X.T@target)
    centered=target-target.mean()
    pred=X@weights
    return dict(source_clock_hour_difference_is_zero=True,
       source_date_gap_days_abs_median=float(day.abs().median()),
       mean_preanchor_pm25_raw_difference=float(pre.mean()),
       centered_regularized_descriptive_coefficients=[float(v) for v in weights],
       descriptive_fit_R2=float(1-((target-pred).square().sum()/centered.square().sum().clamp_min(1e-8))),
       n_matched_pairs=len(target),
       not_causal_not_independent_evaluation=True)

def frozen_source_oracle(m,d,pairs):
    b=len(pairs)
    source_id=torch.tensor([a[0] for a in pairs],dtype=torch.long)
    anchor=torch.tensor([a[1] for a in pairs],dtype=torch.long)
    state=m.initial(b)
    for offset in range(-23,0):
        ts=anchor+offset
        state=m.step(state,d["values"][source_id,ts],d["valid"][source_id,ts],
             d["age"][source_id,ts],d["last"][source_id,ts],source_id)
    before=state
    write=directional_write(m,state,d,source_id,anchor)
    normal=m.step(state,d["values"][source_id,anchor],d["valid"][source_id,anchor],
                  d["age"][source_id,anchor],d["last"][source_id,anchor],source_id)
    lesioned=normal._replace(associative=before.associative)
    oracle_key=write["key"]
    oracle_value=write["value"]
    wrong_key=oracle_key.reshape(b//2,2,2,8).flip(1).reshape(b,2,8)
    wrong_value=oracle_value.reshape(b//2,2,16).flip(1).reshape(b,16)
    event_written=write["norm"]
    write_norms=[]
    result={}
    for lag in range(1,513):
        ts=anchor+lag
        otherwrite=directional_write(m,normal,d,source_id,ts)
        write_norms.append(otherwrite["norm"])
        normal=m.step(normal,d["values"][source_id,ts],d["valid"][source_id,ts],
            d["age"][source_id,ts],d["last"][source_id,ts],source_id)
        lesioned=m.step(lesioned,d["values"][source_id,ts],d["valid"][source_id,ts],
            d["age"][source_id,ts],d["last"][source_id,ts],source_id)
        if lag not in DELAYS:continue
        if any(not torch.equal(u,v) for u,v in zip(
          (normal.fast,normal.slow,normal.entities,normal.active),
          (lesioned.fast,lesioned.slow,lesioned.entities,lesioned.active))):
            raise RuntimeError("original source writer depends on prior matrix via another state path; fixed transfer invalid")
        delta=normal.associative-lesioned.associative
        key,route=eleventh.read_query(m,normal,anchor)
        true_oracle=read_value(normal.associative,oracle_key,route)
        wrong_oracle=read_value(normal.associative,wrong_key,route)
        actual_read=read_value(normal.associative,key,route)
        true_effect=read_value(delta,key,route)
        wrong_q=key.reshape(b//2,2,2,8).roll(1,0).reshape(b,2,8)
        scrambled_effect=read_value(delta,wrong_q,route)
        nuisance=read_value(lesioned.associative,key,route)
        with torch.no_grad():
            clock=source.calendar([int(t) for t in anchor])[:,None,:]
            logit=m.historic_event_logits(normal,source_id,clock)[:,0]
            p=torch.sigmoid(logit)
        valuesim=normalized_cos(oracle_value,wrong_value)
        coscorrect=normalized_cos(true_oracle,oracle_value)
        coswrongkey=normalized_cos(wrong_oracle,oracle_value)
        coswrongval=normalized_cos(true_oracle,wrong_value)
        cosactual=normalized_cos(actual_read,oracle_value)
        ratio=true_effect.norm(dim=-1)/(nuisance.norm(dim=-1)+1e-8)
        shuffled=scrambled_effect.norm(dim=-1)/(read_value(
            lesioned.associative,wrong_q,route).norm(dim=-1)+1e-8)
        rank=eleventh.paired_summary(p)
        rank["station_episode_bootstrap_95_interval"]=station_block_ci(
            source_id[::2],(p[::2]>p[1::2]).float()+
            .5*(p[::2]==p[1::2]).float())
        median_other=torch.stack(write_norms).median(dim=0).values
        if lag==64:
            regcov=simple_covariate_fit(d,source_id,anchor,p)
        else:regcov=None
        # Anti-leak: original event value/key comes from recorded input only
        # in diagnostics, not the learned historic query.
        result[str(lag)]=dict(
            original_physical_delay_hours=lag,
            source_true_event_pair_rank=rank,
            oracle_read_cos_source_event_value=float(coscorrect.mean()),
            matched_wrong_key_read_cos_source_event_value=float(coswrongkey.mean()),
            oracle_read_cos_wrong_event_value=float(coswrongval.mean()),
            native_historic_read_cos_source_event_value=float(cosactual.mean()),
            original_event_value_cos_matched_source_other_value=float(valuesim.mean()),
            oracle_key_reconstruction_specificity_correct_minus_wrong_key=float(
                 (coscorrect-coswrongkey).mean()),
            oracle_target_value_specificity_correct_minus_wrong_value=float(
                 (coscorrect-coswrongval).mean()),
            current_learned_query_event_delta_signal_to_other_read=float(ratio.mean()),
            physically_shuffled_query_event_delta_signal_to_other_read=float(shuffled.mean()),
            anchor_true_event_residual_delta_write_norm=float(event_written.mean()),
            original_intervening_genuine_write_residual_norm_median_case_mean=float(
                median_other.mean()),
            original_anchor_residual_to_intervening_write_ratio_mean=float(
                (event_written/(median_other+1e-8)).mean()),
            empirical_pre_event_source_covariate_regression=regcov,
            source_truth_never_passed_as_query=True,
            oracle_access_IS_PRIVILEGED_DIAGNOSTIC_ONLY=True)
    return result

def untrained_source_rank(m,d,pairs):
    answer={}
    for lag in DELAYS:
        with torch.no_grad():
            predicted=[]
            for start in range(0,len(pairs),8):
                part=pairs[start:start+8]
                logits,_,_,_=fitted.forward_case(m,d,part,lag,with_truth=False)
                predicted.append(torch.sigmoid(logits))
            p=torch.cat(predicted)
            res=eleventh.paired_summary(p)
            site=torch.tensor([row[0] for row in pairs[::2]])
            res["station_episode_bootstrap_95_interval"]=station_block_ci(
                site,(p[::2]>p[1::2]).float()+.5*(p[::2]==p[1::2]).float())
            answer[str(lag)]=res
    return answer

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--source",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    a=parser.parse_args()
    if a.output.exists():raise RuntimeError("source oracle result immutable, no overwrite")
    torch.set_num_threads(1)
    manifest=json.loads(REG.read_text())
    check_manifest(manifest,"preflight")
    if manifest["study_id"]!="LATFUSE012-SOURCE-ORACLE-READ-AND-WRITE-CAPACITY":
        raise RuntimeError("source oracle prereg drift")
    if sha(a.source)!=source.SOURCE_SHA:raise RuntimeError("original physical archive mismatch")
    d=fitted.source_observations(a.source)
    pairs=fitted.select_pairs(d,fitted.EVAL_START,fitted.EVAL_STOP,2)
    receipt=json.loads((fitted.OUT/"RESULTS.json").read_text())
    cohort=hashlib.sha256(json.dumps(pairs,separators=(",",":")).encode()).hexdigest()
    if len(pairs)!=48 or cohort!=receipt["source_eval_cohort_sha256"]:
        raise RuntimeError("original source matched episodes altered")
    evidence=[]
    with torch.no_grad():
        for seed in fitted.SEEDS:
            arm="assoc_historic_query"
            cp=fitted.OUT/f"seed-{seed}"/arm/"final.pt"
            saved=next((x for x in receipt["model_results"] if
                        x["seed"]==seed and x["arm"]==arm),None)
            if saved is None or sha(cp)!=saved["original_checkpoint_sha256"]:
                raise RuntimeError("original pretrained source checkpoint tampered")
            m=fitted.create_model(seed,arm)
            data_cp=torch.load(cp,map_location="cpu",weights_only=False)
            m.load_state_dict(data_cp["original_model"],strict=True)
            m.eval()
            weights=[x.detach().clone() for x in m.parameters()]
            detail=frozen_source_oracle(m,d,pairs)
            if sha(cp)!=saved["original_checkpoint_sha256"] or any(
                not torch.equal(x,y) for x,y in zip(weights,m.parameters())):
                raise RuntimeError("frozen source learned model changed")
            evidence.append(dict(seed=seed,model_training="original_source_trained",
                source_original_checkpoint_sha256=sha(cp),delays=detail))
            print("LATFUSE012_TRAINED_SOURCE_ORACLE="+json.dumps({
                "seed":seed,"h64":detail["64"]},sort_keys=True),flush=True)
        untrained=[]
        for seed in UNTRAINED:
            m=fitted.create_model(seed,"assoc_historic_query")
            m.eval()
            rank=untrained_source_rank(m,d,pairs)
            untrained.append(dict(seed=seed,model_training="RANDOM_INITIALIZATION_NO_FIT",
                                  delays=rank))
            print("LATFUSE012_NOFIT_INITIAL_SOURCE="+json.dumps({
                "seed":seed,"rank":{h:z["rank"] for h,z in rank.items()},
                "no_train":True},sort_keys=True),flush=True)
    out=dict(study_id=manifest["study_id"],source_sha256=source.SOURCE_SHA,
        original_24_matched_source_pairs=len(pairs)//2,
        source_cohort_sha256=cohort,
        trained_original_frozen_model_count=len(evidence),
        random_untrained_initial_seed_count=len(untrained),
        data_exposure="already exposed May-June2014 FIT diagnostics, not virgin",
        oracle_old_write_key_NOT_native_query=True,
        oracle_source_write_value_NOT_in_native_query=True,
        no_optimizer_updates=True,
        read_keys_and_value_oracle_are_diagnostic_privileged=True,
        null_query_physical_shuffled_not_independent_random_isotropic=True,
        95pct_bootstrap_source_station_blocks_not_full_independent_episode_population=True,
        no_original_2015DEV_or_protected_2016_17_numeric=True,
        full_original_weight_SHAs_unchanged=True,
        models=evidence,untrained_controls=untrained)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(out,sort_keys=True,indent=2,allow_nan=False)+"\n")
    print("LATFUSE012_SOURCE_ORACLE_PASS="+json.dumps({
        "source_pairs":len(pairs)//2,"trained_originals":len(evidence),
        "untrained_controls":len(untrained),"optimizer_updates":0},sort_keys=True))
if __name__=="__main__":main()
