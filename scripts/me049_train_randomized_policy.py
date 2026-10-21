#!/usr/bin/env python3
"""ME049: three-action genuine randomized trial, valid positivity and OPE.

Uses original publisher SHA-pinned 64,000-customer Hillstrom randomised test,
not old Criteo's first contiguous 100% treated block. ALL FIT models and RNG
receipts are frozen BEFORE outcome variables in DEV/TEST are parsed.
"""
from __future__ import annotations
import argparse,csv,hashlib,json,math,platform,resource,time,urllib.request
from pathlib import Path
import numpy as np
import torch
from torch import nn
from tac_osm.me049_rct_action_models import (
    ARMS,ACTION_NAMES,ThreeActionWorldModel,actual_logged_action_loss,
    test_math_preflight)

SOURCE_URL=("https://raw.githubusercontent.com/AdityaDabrase/"
    "ab-testing-email-marketing/5866d7e8b50f46239c80d0ffa543fda501939ecc/"
    "data/raw/hillstrom.csv")
SOURCE_SHA="0e5893329d8b93cefecc571777672028290ab69865718020c78c7284f291aece"
SOURCE_BLOB="083522a43100d6c8a15ae960877a5520a1860bd7"
SOURCE_LENGTH=3964977
STUDY="ME049_HILLSTROM_RANDOMIZED_THREE_ACTION_GOAL_MODEL"
SEEDS=(49001,49002)
UPDATES=800;BATCH=256
N=64000;PI=1.0/3
torch.set_num_threads(1)
SOURCE_COLS=("recency","history","mens","womens","zip_code","newbie","channel","segment","visit","conversion","spend")


def hash_file(path):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for chunk in iter(lambda:f.read(1024*1024),b""):h.update(chunk)
    return h.hexdigest()


def save(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(obj,sort_keys=True,indent=2)+"\n")


def original_source(root):
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    path=root/"hillstrom-authentic.csv"
    if path.is_file() and path.stat().st_size==SOURCE_LENGTH and hash_file(path)==SOURCE_SHA:
        return path
    request=urllib.request.Request(SOURCE_URL,headers={"User-Agent":"TAC-OSM-ME049-research/1.0"})
    with urllib.request.urlopen(request,timeout=120) as stream:
        original=stream.read()
    if len(original)!=SOURCE_LENGTH or hashlib.sha256(original).hexdigest()!=SOURCE_SHA:
        raise ValueError("Publisher Hillstrom integrity mismatch; refuse all results")
    path.write_bytes(original)
    return path


def split(index):
    mod=index%20
    return "FIT" if mod<14 else "DEV" if mod<17 else "TEST"


def row_features(row):
    z=row["zip_code"].strip().lower()
    channel=row["channel"].strip().lower()
    zipvalues=("urban","surburban","rural")
    channels=("phone","web","multichannel")
    if z not in zipvalues or channel not in channels:
        raise ValueError("unexpected authentic covariate category")
    # Historical (PRE treatment) features only; fixed no-op last coordinate.
    return [float(row["recency"])/12.,math.log1p(float(row["history"])),
           float(row["mens"]),float(row["womens"]),float(row["newbie"]),
           *[float(z==x) for x in zipvalues],
           *[float(channel==x) for x in channels],0.]


def read_cohort(source,partition):
    """Only decode specified split's actual outcomes in given stage."""
    x=[];a=[];visit=[];conversion=[];spend=[];ids=[]
    with open(source,newline="") as stream:
        rd=csv.DictReader(stream)
        if not set(SOURCE_COLS).issubset(set(rd.fieldnames or [])):
            raise ValueError("original Hillstrom source schema unexpected")
        n=0
        for i,row in enumerate(rd):
            n+=1
            if split(i)!=partition:continue
            treatment=ACTION_NAMES.index(row["segment"])
            x.append(row_features(row))
            a.append(treatment)
            visit.append(float(row["visit"]))
            conversion.append(float(row["conversion"]))
            spend.append(float(row["spend"]))
            ids.append(i)
        if n!=N:raise ValueError("original cohort has fewer or more than 64000 customers")
    x=torch.tensor(x,dtype=torch.float32)
    y={"a":torch.tensor(a,dtype=torch.long),
       "visit":torch.tensor(visit,dtype=torch.float32),
       "conversion":torch.tensor(conversion,dtype=torch.float32),
       "spend":torch.tensor(spend,dtype=torch.float32)}
    if x.ndim!=2 or x.size(1)!=12 or not torch.isfinite(x).all():
        raise ValueError("nonfinite original pre-treatment features")
    if not all(torch.isfinite(v).all() for v in y.values()):raise ValueError("invalid actual outcome")
    if (y["spend"]<0).any():raise ValueError("negative actual purchase spend")
    counts=np.bincount(np.array(a),minlength=3).tolist()
    min_support=5000 if partition=="FIT" else 2500
    if min(counts)<min_support:raise ValueError("RCT arm SUPPORT FAILED "+str(counts))
    if any(abs(counts[k]/len(a)-PI)>.035 for k in range(3)):
        raise ValueError("registered randomized 1/3 propensity balance FAILED")
    expected={"FIT":44800,"DEV":9600,"TEST":9600}[partition]
    if len(x)!=expected:raise ValueError("deterministic split row count violated")
    return {"x":x,**y,"original_index":np.asarray(ids,dtype=np.int64),
         "action_counts":counts}


def training_stats(fit):
    mean=fit["x"].mean(0);std=fit["x"].std(0)
    std=torch.where(std<.02,torch.ones_like(std),std)
    return mean,std


def train(fit,arm,seed,root):
    folder=root/f'{arm}-s{seed}';folder.mkdir(parents=True,exist_ok=False)
    torch.manual_seed(seed)
    model=ThreeActionWorldModel(arm)
    model.set_fit_normalization(*training_stats(fit))
    optim=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
    gen=torch.Generator().manual_seed(seed+2000)
    cpu=0.;t0=time.process_time();trace=[]
    for step in range(1,UPDATES+1):
        idx=torch.randint(len(fit["x"]),(BATCH,),generator=gen)
        tick=time.process_time()
        model.train();optim.zero_grad(set_to_none=True)
        loss=actual_logged_action_loss(model,fit["x"][idx],fit["a"][idx],
                      fit["visit"][idx],fit["conversion"][idx],fit["spend"][idx])
        if not torch.isfinite(loss):raise ArithmeticError("ME049 nonfinite genuine FIT")
        loss.backward();g=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        if not torch.isfinite(g):raise ArithmeticError("nonfinite action training grad")
        optim.step();cpu+=time.process_time()-tick
        if step%100==0:trace.append({"step":step,"actual_randomized_outcome_loss":float(loss.detach()),"optimizer_CPU":cpu})
    cp=folder/"trained-real-three-action.pt"
    torch.save({"study":STUDY,"arm":arm,"seed":seed,"updates":UPDATES,
        "weights":model.state_dict(),"optimizer":optim.state_dict(),
        "rng":gen.get_state(),"torch_rng":torch.get_rng_state(),
        "original_csv_SHA256":SOURCE_SHA,"FIT_original_rows":len(fit["x"]),
        "heldout_outcomes_loaded_during_training":False},cp)
    receipt={"arm":arm,"seed":seed,"parameter_count":model.active_parameters(),
        "real_fit_rows":len(fit["x"]),"FIT_actual_arm_counts":fit["action_counts"],
        "real_randomized_training_exposures":UPDATES*BATCH,
        "optimizer_CPU_seconds":cpu,"complete_fit_CPU_seconds":time.process_time()-t0,
        "peak_RSS_KiB":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "original_checkpoint":cp.name,"checkpoint_SHA256":hash_file(cp),
        "known_true_outcomes_only":True,"TEST_outcomes_unread":True}
    save(folder/"training_trace.json",trace)
    save(folder/"receipt.json",receipt)
    print(json.dumps({"event":"ME049_GENUINE_THREE_ARM_FIT","arm":arm,"seed":seed,
       "CPU":round(cpu,2),"parameters":receipt["parameter_count"]}),flush=True)
    return receipt


@torch.no_grad()
def all_outcomes(net,x):
    outputs=[]
    for i in range(0,len(x),512):
        batch=x[i:i+512]
        output=[]
        for goal in range(3):
            pred=net(batch,torch.full((len(batch),),goal,dtype=torch.long))
            if goal<2:pred=torch.sigmoid(pred)
            else:pred=pred.clamp(-10,10)
            output.append(pred.numpy())
        outputs.append(np.stack(output,axis=1))
    return np.concatenate(outputs,axis=0)


def offline_value(target_policy,assigned,actual_visits,q):
    n=len(assigned)
    idx=np.arange(n)
    # Known randomized three-arm assignment probability, not fitted/guessed
    # from heldout reward statistics. Positivity already verified.
    propensity=np.full(n,PI)
    active=assigned==target_policy
    weights=active/propensity
    ips=float(np.mean(weights*actual_visits))
    ess=float((weights.sum()**2)/np.sum(weights**2))
    observed_q=q[idx,assigned]
    policy_q=q[idx,target_policy]
    dr_point=policy_q+active*(actual_visits-observed_q)/propensity
    return {"IPS":ips,"DR":float(dr_point.mean()),
            "effective_sample_size":ess},dr_point


def paired_ci(policy_points,baseline_points,seed):
    diff=policy_points-baseline_points
    gen=np.random.default_rng(seed+3011)
    means=[]
    for _ in range(400):
        idx=gen.integers(len(diff),size=len(diff))
        means.append(float(diff[idx].mean()))
    return [float(np.quantile(means,.025)),float(np.quantile(means,.975))]


def score_models(root,records,cohort,split_name,preselected_fixed):
    out={}
    a=cohort["a"].numpy()
    visit=cohort["visit"].numpy()
    conv=cohort["conversion"].numpy()
    spend=np.log1p(cohort["spend"].numpy())
    assert min(cohort["action_counts"])>=2500
    for receipt in records:
        folder=root/f'{receipt["arm"]}-s{receipt["seed"]}'
        saved=torch.load(folder/receipt["original_checkpoint"],weights_only=True,map_location="cpu")
        m=ThreeActionWorldModel(receipt["arm"])
        m.load_state_dict(saved["weights"],strict=True)
        m.eval()
        output=all_outcomes(m,cohort["x"]) # shape: N,3 goals,3 actual actions
        visit_q=output[:,0,:].astype(np.float64)
        conv_q=output[:,1,:].astype(np.float64)
        spend_q=output[:,2,:].astype(np.float64)
        ix=np.arange(len(a))
        obs=visit_q[ix,a]
        obsconv=conv_q[ix,a]
        obsspend=spend_q[ix,a]
        chosen=visit_q.argmax(axis=1)
        fixed=np.full(len(a),preselected_fixed,dtype=np.int64)
        policies={"learned":chosen,**{ACTION_NAMES[j]:np.full(len(a),j,dtype=np.int64) for j in range(3)}}
        policy_scores={};dr_arrays={}
        for name,policy in policies.items():
            policy_scores[name],dr_arrays[name]=offline_value(policy,a,visit,visit_q)
        _,ref_array=offline_value(fixed,a,visit,visit_q)
        ci=paired_ci(dr_arrays["learned"],ref_array,receipt["seed"])
        npz=folder/f"{split_name}-original-action-predictions.npz"
        # Raw licensed user features/outcome rows NOT included in shared artifact.
        np.savez_compressed(npz,visit=visit_q,conversion=conv_q,logspend=spend_q,
                           chosen_action=chosen)
        q=np.clip(obs,1e-7,1-1e-7);qc=np.clip(obsconv,1e-7,1-1e-7)
        row={"split":split_name,"original_randomized_customers":len(a),
             "real_action_counts":cohort["action_counts"],
             "empirical_arm_fractions":[v/len(a) for v in cohort["action_counts"]],
             "known_random_assignment_probability":PI,
             "observed_visit_rate":float(np.mean(visit)),
             "observed_conversion_rate":float(np.mean(conv)),
             "observed_mean_dollars":float(np.mean(cohort["spend"].numpy())),
             "visit_Brier":float(np.mean((obs-visit)**2)),
             "visit_log_loss":float(np.mean(-visit*np.log(q)-(1-visit)*np.log1p(-q))),
             "conversion_Brier":float(np.mean((obsconv-conv)**2)),
             "conversion_log_loss":float(np.mean(-conv*np.log(qc)-(1-conv)*np.log1p(-qc))),
             "logspend_MAE":float(np.mean(abs(obsspend-spend))),
             "learned_policy_treatment_fractions":[float(np.mean(chosen==j)) for j in range(3)],
             "policy_values":policy_scores,
             "baseline_action_chosen_using_only_FIT":ACTION_NAMES[preselected_fixed],
             "paired_DR_policy_minus_training_best_fixed_95CI":ci,
             "original_array":npz.name,"original_array_sha256":hash_file(npz),
             "RCT_support_verified_before_evaluation":True,
             "individual_counterfactual_outcomes_observed":False,
             "persistent_closed_loop_world_state_proven":False}
        receipt.setdefault("scores",{})[split_name]=row
        save(folder/"receipt.json",receipt)
        out[f'{receipt["arm"]}-s{receipt["seed"]}']=row
    return out


def run(directory):
    root=Path(directory);root.mkdir(parents=True,exist_ok=False)
    result={"study":STUDY,"complete":False,"models":[],
      "source_original_SHA256":SOURCE_SHA,
      "original_source_git_blob":SOURCE_BLOB,
      "registered_three_randomized_actions":ACTION_NAMES,
      "raw_source_excluded_from_actions_artifact":True,
      "no_per_user_sequential_action_histories":True,
      "heldout_labels_parsed_before_all_FIT_checkpoint_frozen":False}
    save(root/"results.json",result)
    src=original_source(root/"source-NOT-ARTIFACT")
    fit=read_cohort(src,"FIT")
    result["FIT_support"]={"customer_count":len(fit["x"]),"action_counts":fit["action_counts"],
         "true_visit_rate_by_action":[float(fit["visit"][fit["a"]==j].mean()) for j in range(3)]}
    best_fixed=int(np.argmax(result["FIT_support"]["true_visit_rate_by_action"]))
    result["FIT_only_preselected_best_fixed_action"]=ACTION_NAMES[best_fixed]
    save(root/"results.json",result)
    for seed in SEEDS:
        for arm in ARMS:
            result["models"].append(train(fit,arm,seed,root))
            save(root/"results.json",result)
    if len(result["models"])!=6:raise AssertionError("cannot access heldout before all fits")
    counts=[row["parameter_count"] for row in result["models"]]
    if max(counts)/min(counts)>1.01:
        raise AssertionError("matching active parameters failed; scores not admissible")
    for row in result["models"]:
        cp=root/f'{row["arm"]}-s{row["seed"]}'/row["original_checkpoint"]
        if hash_file(cp)!=row["checkpoint_SHA256"]:
            raise AssertionError("changed trained model before test access")
    result["all_original_FIT_checkpoints_SHA_frozen_before_test"]=True
    save(root/"results.json",result)
    dev=read_cohort(src,"DEV");test=read_cohort(src,"TEST")
    result["DEV_support"]=dev["action_counts"]
    result["TEST_support"]=test["action_counts"]
    result["DEV"]=score_models(root,result["models"],dev,"DEV",best_fixed)
    result["TEST"]=score_models(root,result["models"],test,"TEST",best_fixed)
    hold=result["TEST"];summary={}
    for arm in ARMS:
        rows=[hold[f"{arm}-s{s}"] for s in SEEDS]
        summary[arm]={"mean_visit_Brier":float(np.mean([r["visit_Brier"] for r in rows])),
          "mean_visit_log_loss":float(np.mean([r["visit_log_loss"] for r in rows])),
          "mean_learned_DR_value":float(np.mean([r["policy_values"]["learned"]["DR"] for r in rows])),
          "both_seed_policy_CI_lower_positive":all(r["paired_DR_policy_minus_training_best_fixed_95CI"][0]>0 for r in rows)}
    tac=summary["action_tac"]
    native_qualified=(tac["mean_visit_Brier"]<min(summary["action_context"]["mean_visit_Brier"],
             summary["action_gru"]["mean_visit_Brier"])
          and tac["both_seed_policy_CI_lower_positive"])
    result["selection"]={"scores":summary,
        "native_vs_matched_models_and_FIT_best_fixed_reward_qualified":bool(native_qualified),
        "genuine_random_action_support_in_all_cohorts":True,
        "two_potential_outcomes_per_customer_unobserved":True,
        "persistent_sequential_action_intelligence_not_qualified":True,
        "semantic_language_instruction_ability_not_tested":True,
        "no_model_merged_or_promoted":True,
        "parameter_count_ratio":max(counts)/min(counts),
        "training_exposure_matched_but_CPU_cost_unequal":True}
    result["complete"]=True
    save(root/"results.json",result)
    print(json.dumps({"event":"ME049_THREE_ACTION_RCT_FINISHED",
         "support":[result["FIT_support"]["action_counts"],result["DEV_support"],result["TEST_support"]],
         "scores":summary,"gates":result["selection"]},sort_keys=True),flush=True)


@torch.no_grad()
def replay(root):
    root=Path(root);report=json.loads((root/"results.json").read_text())
    assert report["complete"] and report["selection"]["genuine_random_action_support_in_all_cohorts"]
    src=original_source(root/"source-NOT-ARTIFACT")
    for label in ("DEV","TEST"):
        cohort=read_cohort(src,label)
        for row in report["models"]:
            folder=root/f'{row["arm"]}-s{row["seed"]}'
            checkpoint=folder/row["original_checkpoint"]
            assert hash_file(checkpoint)==row["checkpoint_SHA256"]
            original=torch.load(checkpoint,weights_only=True,map_location="cpu")
            model=ThreeActionWorldModel(row["arm"])
            model.load_state_dict(original["weights"],strict=True)
            ids=[i for param in original["optimizer"]["param_groups"] for i in param["params"]]
            assert len(original["optimizer"]["state"])==len(ids)
            assert all(int(original["optimizer"]["state"][i]["step"])==UPDATES for i in ids)
            pred=all_outcomes(model,cohort["x"])
            oldfile=folder/row["scores"][label]["original_array"]
            assert hash_file(oldfile)==row["scores"][label]["original_array_sha256"]
            with np.load(oldfile) as old:
                assert np.array_equal(pred[:,0,:],old["visit"])
                assert np.array_equal(pred[:,1,:],old["conversion"])
                assert np.array_equal(pred[:,2,:],old["logspend"])
                assert np.array_equal(pred[:,0,:].argmax(1),old["chosen_action"])
    print(json.dumps({"status":"ME049_ORIGINAL_ALL_REAL_TREATMENT_CHECKPOINTS_AND_SCORES_REPLAY_PASS",
         "models":len(report["models"]),"supported_real_actions":3}),flush=True)


if __name__=="__main__":
    a=argparse.ArgumentParser()
    a.add_argument("--out",default="results/me049")
    a.add_argument("--self-test",action="store_true")
    a.add_argument("--replay",action="store_true")
    o=a.parse_args()
    if o.self_test:print(json.dumps(test_math_preflight(),sort_keys=True),flush=True)
    elif o.replay:replay(o.out)
    else:run(o.out)
