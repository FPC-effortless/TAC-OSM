#!/usr/bin/env python3
"""ME048 actual randomized treatment/reward model training, heldout OPE.

No synthetic success, no observed individual counterfactual outcome. Primary
offline trial is authentic corrected Criteo Uplift v2.1 Visit; conversion extra.
Not a learned temporal closed-loop controller. Licensed raw CSV excluded from
research artifacts.
"""
from __future__ import annotations
import argparse,csv,gzip,hashlib,json,math,os,platform,resource,time,urllib.request
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from tac_osm.me048_real_action_outcomes import (ARMS,ActionOutcomeModel,
    true_observed_action_loss,preflight)

ROOT_URL="https://huggingface.co/datasets/criteo/criteo-uplift/resolve/2424920019e49d52d72c13ac1143ec5d53af276b/criteo-research-uplift-v2.1.csv.gz"
BACKUP_URL="https://go.criteo.net/criteo-research-uplift-v2.1.csv.gz"
ORIGINAL_SHA="2716e1bf0fd157a93b5bf86924d9088419dfbac2022c6cd90030220634f616dc"
BYTES=311422618
N_ALL=13979592
FIT_END=180000;DEV_END=240000;TEST_END=300000
ARMS_SEEDS=(48001,48002)
STEPS=800;BATCH=256
STUDY="ME048_AUTHENTIC_RANDOMIZED_ACTION_OUTCOMES"
torch.set_num_threads(1)


def digest(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda:f.read(1<<20),b""):h.update(chunk)
    return h.hexdigest()


def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,sort_keys=True)+"\n")


def fetch_original(folder):
    folder=Path(folder);folder.mkdir(exist_ok=True)
    path=folder/"criteo-uplift-v2.1.csv.gz"
    if path.exists() and path.stat().st_size==BYTES and digest(path)==ORIGINAL_SHA:
        return path
    for uri in (ROOT_URL,BACKUP_URL):
        temp=path.with_suffix(".part")
        temp.unlink(missing_ok=True)
        h=hashlib.sha256();size=0
        try:
            req=urllib.request.Request(uri,
                headers={"User-Agent":"TAC-OSM-ME048-randomized-uplift-research/1.0"})
            with urllib.request.urlopen(req,timeout=180) as stream, temp.open("wb") as dest:
                while True:
                    chunk=stream.read(1<<20)
                    if not chunk:break
                    dest.write(chunk);h.update(chunk);size+=len(chunk)
            if size==BYTES and h.hexdigest()==ORIGINAL_SHA:
                temp.replace(path)
                return path
        except Exception as err:
            print("ME048 authentic source fallback attempted:",type(err).__name__,flush=True)
        temp.unlink(missing_ok=True)
    raise RuntimeError("publisher original corrected real randomized archive unavailable or checksum mismatch")


def load_range(path,start,end):
    """Actually parse ONLY allowed original source rows for this phase."""
    features=[];treatment=[];visit=[];conversion=[];exposure=[]
    with gzip.open(path,"rt",newline="") as f:
        reader=csv.DictReader(f)
        keys=set(reader.fieldnames or ())
        must=set([f"f{i}" for i in range(12)]+["treatment","visit","conversion","exposure"])
        if not must.issubset(keys):raise RuntimeError("unrecognized corrected real trial schema")
        for i,row in enumerate(reader):
            if i>=end:break
            if i<start:continue
            features.append([float(row[f"f{j}"]) for j in range(12)])
            treatment.append(int(row["treatment"]))
            visit.append(float(row["visit"]))
            conversion.append(float(row["conversion"]))
            exposure.append(int(row["exposure"]))
    x=np.asarray(features,np.float32);a=np.asarray(treatment,np.int64)
    yv=np.asarray(visit,np.float32);yc=np.asarray(conversion,np.float32)
    ex=np.asarray(exposure,np.int64)
    if x.shape!=(end-start,12) or not np.isfinite(x).all():
        raise RuntimeError("source numeric feature integrity failed")
    if not np.isin(a,[0,1]).all() or not np.isin(ex,[0,1]).all() or \
       not np.isin(yv,[0,1]).all() or not np.isin(yc,[0,1]).all():
        raise RuntimeError("publisher real treatment/outcome integrity failed")
    return {"x":torch.tensor(x),"a":torch.tensor(a),
            "visit":torch.tensor(yv),"conversion":torch.tensor(yc),
            "exposure":torch.tensor(ex)}


def fit(train,arm,seed,folder):
    folder.mkdir(parents=True,exist_ok=False)
    torch.manual_seed(seed)
    net=ActionOutcomeModel(arm)
    mean=train["x"].mean(0)
    std=train["x"].std(0).clamp_min(.0001)
    net.set_training_normalization(mean,std)
    opt=torch.optim.AdamW(net.parameters(),lr=.001,weight_decay=.0001)
    rng=torch.Generator().manual_seed(seed+2000)
    cpu=0.;start=time.process_time();trace=[]
    for step in range(1,STEPS+1):
        indices=torch.randint(FIT_END,(BATCH,),generator=rng)
        t=time.process_time()
        net.train();opt.zero_grad(set_to_none=True)
        loss=true_observed_action_loss(net,train["x"][indices],
          train["a"][indices],train["visit"][indices],train["conversion"][indices])
        if not torch.isfinite(loss):raise ArithmeticError("real visit/conversion loss nonfinite")
        loss.backward()
        grad=torch.nn.utils.clip_grad_norm_(net.parameters(),1.)
        if not torch.isfinite(grad):raise ArithmeticError("nonfinite real randomized experiment grads")
        opt.step();cpu+=time.process_time()-t
        if step%100==0:trace.append({"step":step,
                 "observed_true_treatment_outcome_loss":float(loss.detach()),
                 "CPU":cpu})
    cp=folder/"original-trained-action-model.pt"
    torch.save({"arm":arm,"seed":seed,"weights":net.state_dict(),
          "optimizer":opt.state_dict(),"rng":rng.get_state(),
          "torch_rng":torch.get_rng_state(),"source_SHA256":ORIGINAL_SHA,
          "fit_rows_parsed":FIT_END,"steps":STEPS},cp)
    save(folder/"FIT-trace.json",trace)
    record={"arm":arm,"seed":seed,"trained_on_real_randomized_treatment":True,
       "real_FIT_user_rows":FIT_END,"real_record_exposures":STEPS*BATCH,
       "active_trainable_parameters":net.num_active(),
       "optimizer_CPU_seconds":cpu,"full_FIT_CPU_seconds":time.process_time()-start,
       "peak_RSS_KiB":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
       "original_checkpoint":cp.name,"checkpoint_SHA256":digest(cp),
       "TEST_labels_loaded_during_training":False}
    save(folder/"summary.json",record)
    print(json.dumps({"event":"ME048_SOURCE_MODEL_FIT","arm":arm,
                      "seed":seed,"CPU":round(cpu,3)}),flush=True)
    return record


@torch.no_grad()
def outcomes(net,cohort):
    net.eval();v=[];c=[]
    for i in range(0,len(cohort["x"]),2048):
        features=cohort["x"][i:i+2048]
        v.append(net.probability(features,torch.zeros(len(features),dtype=torch.long)).numpy())
        c.append(net.probability(features,torch.ones(len(features),dtype=torch.long)).numpy())
    return np.concatenate(v),np.concatenate(c)


def ips_value(actions,actually_treated,observed_reward,p):
    allowed=(actions==actually_treated)
    weights=allowed/np.where(actually_treated==1,p,1-p)
    return float(np.mean(weights*observed_reward)),float(weights.sum()**2/(weights*weights).sum())


def dr_value(actions,actually_treated,reward,q,p):
    idx=np.arange(len(actions))
    q_policy=q[idx,actions]
    q_logged=q[idx,actually_treated]
    pi=np.where(actually_treated==1,p,1-p)
    return float(np.mean(q_policy+(actually_treated==actions)*(reward-q_logged)/pi))


def paired_bootstrap_delta(policy,reference,a,y,q,p,seed):
    """One fixed 200-resample iid interval; no user IDs => not cluster robust."""
    rng=np.random.default_rng(seed+10200)
    n=len(y);dis=[]
    idx=np.arange(n)
    delta=(q[idx,policy]-q[idx,reference]+
      (a==policy)*(y-q[idx,a])/np.where(a==1,p,1-p)-
      (a==reference)*(y-q[idx,a])/np.where(a==1,p,1-p))
    for _ in range(200):
        take=rng.integers(n,size=n)
        dis.append(delta[take].mean())
    return [float(np.quantile(dis,.025)),float(np.quantile(dis,.975))]


def score_models(root,cohort,models,frequency,name):
    summary={}
    a=cohort["a"].numpy();v=cohort["visit"].numpy()
    conv=cohort["conversion"].numpy()
    for rec in models:
        folder=root/f'{rec["arm"]}-s{rec["seed"]}'
        cp=torch.load(folder/rec["original_checkpoint"],weights_only=True)
        net=ActionOutcomeModel(rec["arm"]);net.load_state_dict(cp["weights"],strict=True)
        q,qs=outcomes(net,cohort)
        observed=q[np.arange(len(a)),a]
        observed_conversion=qs[np.arange(len(a)),a]
        candidate=(q[:,1]>q[:,0]).astype(np.int64)
        treatall=np.ones_like(a);never=np.zeros_like(a)
        ips={};dr={};ess={}
        for key,policy in [("learned_action_policy",candidate),
                           ("always_treat",treatall),("never_treat",never)]:
            ips[key],ess[key]=ips_value(policy,a,v,frequency)
            dr[key]=dr_value(policy,a,v,q,frequency)
        # Compare against the stronger frozen fixed action, neither benchmark
        # label is used to train the candidate or select a hyperparameter.
        baseline="always_treat" if dr["always_treat"]>=dr["never_treat"] else "never_treat"
        fixed=treatall if baseline=="always_treat" else never
        band=paired_bootstrap_delta(candidate,fixed,a,v,q,frequency,rec["seed"])
        error=(observed-v)
        clip=np.clip(observed,1e-6,1-1e-6)
        cclip=np.clip(observed_conversion,1e-6,1-1e-6)
        row={"dataset":name,"observed_user_records":len(a),
             "observed_visit_base_rate":float(v.mean()),
             "observed_conversion_base_rate":float(conv.mean()),
             "observed_treatment_ratio":float(a.mean()),
             "visit_Brier":float(np.mean(error**2)),
             "visit_log_loss":float(np.mean(-(v*np.log(clip)+(1-v)*np.log1p(-clip)))),
             "conversion_Brier":float(np.mean((observed_conversion-conv)**2)),
             "conversion_log_loss":float(np.mean(-(conv*np.log(cclip)+(1-conv)*np.log1p(-cclip)))),
             "learned_treat_fraction":float(candidate.mean()),
             "ips_visit_values":ips,"doubly_robust_exploratory_values":dr,
             "ips_effective_sample_sizes":ess,
             "assumed_global_logging_treatment_probability":frequency,
             "paired_iid_DR_diff_95pct_CI_vs_better_fixed_policy":band,
             "fixed_reference_selected_for_report_only":baseline,
             "can_claim_individually_identified_counterfactual":False,
             "can_claim_deployed_closed_loop":False,
             "cannot_assume_exact_conditional_propensity":True}
        # Original learned logits; not copying licensed user or reward records.
        np.savez_compressed(folder/f"{name}-prediction-arrays.npz",
                            pred_visit=q,pred_conversion=qs)
        row["prediction_file"]=f"{name}-prediction-arrays.npz"
        row["prediction_SHA256"]=digest(folder/row["prediction_file"])
        summary[f'{rec["arm"]}-s{rec["seed"]}']=row
        rec.setdefault("evaluation",{})[name]=row
    return summary


def run(destination):
    output=Path(destination);output.mkdir(parents=True,exist_ok=False)
    report={"study":STUDY,"complete":False,"raw_archive_redistributed":False,
            "original_corrected_archive_sha256":ORIGINAL_SHA,
            "preregistered_SPLIT":[FIT_END,DEV_END,TEST_END],
            "original_source_rows":N_ALL,"licensed_research_only":True,
            "actual_time_order_or_per_user_ids_present":False,
            "seeds":ARMS_SEEDS,"models":[],"sealed_test_parsed":False}
    save(output/"results.json",report)
    source=fetch_original(output/"source-private-NOT-ARTIFACT")
    train=load_range(source,0,FIT_END)
    frequency=float(train["a"].float().mean())
    report["fit_treatment_ratio"]=frequency
    report["fit_observed_visit_rate"]=float(train["visit"].mean())
    for seed in ARMS_SEEDS:
        for arm in ARMS:
            row=fit(train,arm,seed,output/f'{arm}-s{seed}')
            report["models"].append(row)
            save(output/"results.json",report)
    assert len(report["models"])==6
    for rec in report["models"]:
        if digest(output/f'{rec["arm"]}-s{rec["seed"]}'/
                  rec["original_checkpoint"])!=rec["checkpoint_SHA256"]:
            raise ArithmeticError("cannot expose test with changed source-trained model")
    report["all_six_source_model_checkpoints_frozen_before_evaluation"]=True
    save(output/"results.json",report)
    dev=load_range(source,FIT_END,DEV_END)
    test=load_range(source,DEV_END,TEST_END)
    report["sealed_test_parsed"]=True
    report["dev_outcomes"]=score_models(output,dev,report["models"],frequency,"DEV")
    report["heldout_test_outcomes"]=score_models(output,test,report["models"],frequency,"HELDOUT")
    results=report["heldout_test_outcomes"]
    sums={}
    for arm in ARMS:
        r=[results[f"{arm}-s{s}"] for s in ARMS_SEEDS]
        sums[arm]={"visit_Brier":float(np.mean([v["visit_Brier"] for v in r])),
                  "visit_log_loss":float(np.mean([v["visit_log_loss"] for v in r])),
                  "conversion_log_loss":float(np.mean([v["conversion_log_loss"] for v in r])),
                  "mean_ips_policy_value":float(np.mean([v["ips_visit_values"]["learned_action_policy"] for v in r])),
                  "mean_DR_policy_value":float(np.mean([v["doubly_robust_exploratory_values"]["learned_action_policy"] for v in r])),
                  "all_iid_DR_CIs_exclude_nonpositive":all(v["paired_iid_DR_diff_95pct_CI_vs_better_fixed_policy"][0]>0 for v in r)}
    tac=sums["action_tac"];report["selection"]={
        "mean_HOLDOUT":sums,
        "native_better_than_context_and_gru_on_real_observed_visit_prediction":
            tac["visit_Brier"]<min(sums["action_context"]["visit_Brier"],
                                   sums["action_gru"]["visit_Brier"]),
        "native_policy_improves_over_better_fixed_action_with_iid_CI":
            tac["all_iid_DR_CIs_exclude_nonpositive"],
        "conditional_propensity_not_verified":True,
        "temporal_world_state_not_qualified":True,
        "strict_action_superiority_qualified":False,
        "no_promotion_or_merge":True}
    report["complete"]=True
    save(output/"results.json",report)
    print(json.dumps({"event":"ME048_REAL_RANDOMIZED_ACTION_COMPLETE",
                      "scores":sums,"real_decision_claim":False,
                      "historical_sequential_state_qualified":False}),flush=True)


@torch.no_grad()
def replay(root):
    root=Path(root)
    rec=json.loads((root/"results.json").read_text())
    assert rec["complete"] and rec["all_six_source_model_checkpoints_frozen_before_evaluation"]
    original=fetch_original(root/"source-private-NOT-ARTIFACT")
    cohorts={"DEV":load_range(original,FIT_END,DEV_END),
             "HELDOUT":load_range(original,DEV_END,TEST_END)}
    for row in rec["models"]:
        folder=root/f'{row["arm"]}-s{row["seed"]}'
        cp=folder/row["original_checkpoint"]
        assert digest(cp)==row["checkpoint_SHA256"]
        saved=torch.load(cp,weights_only=True)
        model=ActionOutcomeModel(row["arm"])
        model.load_state_dict(saved["weights"],strict=True)
        ids=[k for g in saved["optimizer"]["param_groups"] for k in g["params"]]
        assert len(ids)==len(saved["optimizer"]["state"])
        for cohort,records in cohorts.items():
            q,c=outcomes(model,records)
            path=folder/f"{cohort}-prediction-arrays.npz"
            assert digest(path)==row["evaluation"][cohort]["prediction_SHA256"]
            with np.load(path) as reference:
                assert np.array_equal(q,reference["pred_visit"])
                assert np.array_equal(c,reference["pred_conversion"])
    print(json.dumps({"status":"ME048_ALL_SIX_RANDOMIZED_ACTION_MODELS_REPLAY_EXACTLY",
          "actual_test_outcomes_only_observed_logged_actions":True}),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--out",default="results/me048")
    p.add_argument("--self-test",action="store_true")
    p.add_argument("--replay",action="store_true")
    a=p.parse_args()
    if a.self_test:print(json.dumps(preflight(),sort_keys=True),flush=True)
    elif a.replay:replay(a.out)
    else:run(a.out)
