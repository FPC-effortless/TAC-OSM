#!/usr/bin/env python3
"""ME049: real randomized-action support + action-injected native recurrence.

No synthetic results or fabricated counterfactual outcomes. Raw noncommercial
publisher records remain local to the training runner, never GitHub artifacts.
"""
from __future__ import annotations
import argparse,csv,gzip,hashlib,json,time,resource
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from scripts.me048_train_real_action import (fetch_original,digest,save,outcomes,
    ORIGINAL_SHA,N_ALL)
from tac_osm.me049_action_state import ARMS,NEW_ARM,make_model,authentic_logged_action_loss

STUDY="ME049_AUTHENTIC_TWO_ARM_ACTION_DEPENDENT_STATE"
CANDIDATES=1_500_000
PER_ACTION={"FIT":80_000,"DEV":20_000,"TEST":20_000}
SEEDS=(49001,49002)
STEPS=800
BATCH=256
torch.set_num_threads(1)


def header_columns(line):
    keys=next(csv.reader([line]))
    mandatory={f"f{i}" for i in range(12)}|{"treatment","visit","conversion","exposure"}
    if not mandatory.issubset(keys) or len(set(keys))!=len(keys):
        raise RuntimeError("authentic v2.1 schema missing/duplicated")
    return keys


def plan_supported_indices(source):
    """Only inspect treatment flags; no FIT/DEV/TEST features or outcomes opened."""
    rng=np.random.default_rng(49049)
    sample=np.sort(rng.choice(N_ALL,size=CANDIDATES,replace=False))
    found={0:[],1:[]};ptr=0;count=0
    with gzip.open(source,"rt",newline="") as f:
        keys=header_columns(next(f));action_col=keys.index("treatment")
        for idx,line in enumerate(f):
            count+=1
            if ptr>=len(sample) or idx!=sample[ptr]:continue
            row=next(csv.reader([line]))
            if len(row)!=len(keys):raise RuntimeError("nonflat publisher numeric row")
            action=int(row[action_col])
            if action not in (0,1):raise RuntimeError("nonbinary treatment")
            found[action].append(idx)
            ptr+=1
    if count!=N_ALL or ptr!=len(sample):
        raise RuntimeError("Criteo original population row count mismatch")
    n=sum(PER_ACTION.values())
    if min(len(found[0]),len(found[1]))<n:
        raise RuntimeError("ACTION_SUPPORT_FAILURE: cannot form sealed both-arm cohorts")
    rng=np.random.default_rng(49050)
    allocations={"FIT":[],"DEV":[],"TEST":[]}
    for arm in (0,1):
        perm=np.asarray(found[arm],dtype=np.int64)
        rng.shuffle(perm)
        start=0
        for part,number in PER_ACTION.items():
            allocations[part].extend(perm[start:start+number].tolist())
            start+=number
    all_ids=[i for ids in allocations.values() for i in ids]
    if len(all_ids)!=len(set(all_ids)):
        raise RuntimeError("cohort row IDs overlap")
    evidence={"population":N_ALL,"source_SHA256":ORIGINAL_SHA,
              "candidate_draw_without_replacement":CANDIDATES,
              "candidate_treatment_counts":{"0":len(found[0]),"1":len(found[1])},
              "cohort_treatment_counts":{k:{"0":PER_ACTION[k],"1":PER_ACTION[k]}
                                         for k in PER_ACTION},
              "candidate_indices_digest":hashlib.sha256(sample.tobytes()).hexdigest(),
              "cohort_index_digests":{k:hashlib.sha256(np.sort(
                  np.asarray(v,dtype=np.int64)).tobytes()).hexdigest()
                  for k,v in allocations.items()},
              "split_selection_saw_features_or_outcomes":False,
              "source_file_order_used_as_temporal_order":False}
    return allocations,evidence


def load_partition(source,indices,expected):
    """Only parse numeric outcome rows for the explicitly allowed partition."""
    selected=set(indices)
    if len(selected)!=expected:raise RuntimeError("incorrect selected cohort")
    x=[];a=[];v=[];c=[];ex=[]
    with gzip.open(source,"rt",newline="") as f:
        keys=header_columns(next(f));location={k:keys.index(k) for k in keys}
        for idx,line in enumerate(f):
            if idx not in selected:continue
            row=next(csv.reader([line]))
            if len(row)!=len(keys):raise RuntimeError("malformed numeric record")
            x.append([float(row[location[f"f{j}"]]) for j in range(12)])
            a.append(int(row[location["treatment"]]))
            v.append(float(row[location["visit"]]))
            c.append(float(row[location["conversion"]]))
            ex.append(int(row[location["exposure"]]))
            if len(a)==expected:break
    matrix=np.asarray(x,dtype=np.float32)
    treatment=np.asarray(a,dtype=np.int64)
    visit=np.asarray(v,dtype=np.float32)
    conversion=np.asarray(c,dtype=np.float32)
    exposure=np.asarray(ex,dtype=np.int64)
    if matrix.shape!=(expected,12) or not np.isfinite(matrix).all():
        raise RuntimeError("bad original real covariate shape or value")
    if not np.isin(treatment,[0,1]).all() or not np.isin(exposure,[0,1]).all():
        raise RuntimeError("bad real action/exposure")
    if not np.isin(visit,[0,1]).all() or not np.isin(conversion,[0,1]).all():
        raise RuntimeError("bad observed outcome")
    if np.bincount(treatment,minlength=2).tolist()!=[expected//2]*2:
        raise RuntimeError("ACTION_SUPPORT_FAILURE: cohort treatment frequencies invalid")
    for group in (0,1):
        select=(treatment==group)
        if len(np.unique(visit[select]))<2 or len(np.unique(conversion[select]))<2:
            raise RuntimeError("OUTCOME_SUPPORT_FAILURE: genuine observed reward class missing")
    return {"x":torch.from_numpy(matrix),"a":torch.from_numpy(treatment),
            "visit":torch.from_numpy(visit),"conversion":torch.from_numpy(conversion),
            "exposure":torch.from_numpy(exposure)}


def cohort_summary(cohort):
    a=cohort["a"].numpy();v=cohort["visit"].numpy();c=cohort["conversion"].numpy()
    return {str(k):{"n":int((a==k).sum()),"visit_positive":int(v[a==k].sum()),
                    "conversion_positive":int(c[a==k].sum()),
                    "visit_rate":float(v[a==k].mean()),
                    "conversion_rate":float(c[a==k].mean())} for k in (0,1)}


def train(source_fit,arm,seed,folder):
    folder.mkdir(parents=True,exist_ok=False)
    torch.manual_seed(seed)
    model=make_model(arm)
    mu=source_fit["x"].mean(0)
    std=source_fit["x"].std(0).clamp_min(0.0001)
    model.set_training_normalization(mu,std)
    optimizer=torch.optim.AdamW(model.parameters(),lr=0.001,weight_decay=0.0001)
    stream=torch.Generator().manual_seed(seed+2000)
    start=time.process_time();optimizer_cpu=0.
    trace=[]
    for step in range(1,STEPS+1):
        sample=torch.randint(len(source_fit["a"]),(BATCH,),generator=stream)
        t=time.process_time();model.train();optimizer.zero_grad(set_to_none=True)
        loss=authentic_logged_action_loss(model,source_fit["x"][sample],
            source_fit["a"][sample],source_fit["visit"][sample],source_fit["conversion"][sample])
        if not torch.isfinite(loss):raise ArithmeticError("bad observed treatment loss")
        loss.backward()
        norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
        if not torch.isfinite(norm):raise ArithmeticError("bad observed treatment gradient")
        optimizer.step()
        optimizer_cpu+=time.process_time()-t
        if step%100==0:
            trace.append({"step":step,"genuine_observed_outcome_loss":float(loss.detach()),
                          "optimizer_CPU_seconds":optimizer_cpu})
    checkpoint=folder/"original-action-state-weights.pt"
    torch.save({"arm":arm,"seed":seed,"weights":model.state_dict(),
                "optimizer":optimizer.state_dict(),"sample_rng":stream.get_state(),
                "torch_rng":torch.get_rng_state(),"source_SHA256":ORIGINAL_SHA,
                "steps":STEPS,"batch_size":BATCH},checkpoint)
    save(folder/"FIT-trace.json",trace)
    receipt={"arm":arm,"seed":seed,"parameter_count":model.num_active(),
             "optimizer_CPU_seconds":optimizer_cpu,
             "total_process_CPU_seconds":time.process_time()-start,
             "total_authentic_sample_exposures":STEPS*BATCH,
             "training_rows":len(source_fit["a"]),
             "checkpoint":checkpoint.name,"checkpoint_SHA256":digest(checkpoint),
             "sealed_test_outcomes_loaded_before_freeze":False}
    save(folder/"fit-receipt.json",receipt)
    print(json.dumps({"event":"ME049_GENUINE_FIT","arm":arm,"seed":seed,
                      "optimizer_CPU":round(optimizer_cpu,2)}),flush=True)
    return receipt


def metrics_for_action(pred,y,a,chosen):
    mask=(a==chosen);p=np.clip(pred[mask],1e-6,1-1e-6);target=y[mask]
    return {"n":int(mask.sum()),"Brier":float(np.mean((p-target)**2)),
            "logloss":float(np.mean(-(target*np.log(p)+(1-target)*np.log1p(-p)))),
            "base_rate":float(target.mean())}


def policy_exploratory(q,a,y,seed):
    """Balanced sampling OPE is exploratory, NOT identified publisher propensity."""
    rng=np.random.default_rng(seed+49149)
    n=len(a);idx=np.arange(n)
    learned=(q[:,1]>q[:,0]).astype(np.int64)
    strategies={"learned":learned,"always_treat":np.ones_like(a),
                "never_treat":np.zeros_like(a)}
    values={}
    residual=y-q[idx,a]
    def per_record(policy):
        return q[idx,policy]+2*(a==policy)*residual
    for key,action in strategies.items():
        indicator=(a==action)
        weights=2*indicator
        ess=float(weights.sum()**2/np.square(weights).sum()) if weights.sum()>0 else 0.
        values[key]={"exploratory_doubly_robust_value":float(per_record(action).mean()),
                     "sample_action_support_ESS":ess,"action_fraction":float(action.mean())}
    for fixed in ("always_treat","never_treat"):
        delta=per_record(learned)-per_record(strategies[fixed])
        intervals=[float(np.mean(delta[rng.integers(n,size=n)])) for _ in range(200)]
        values["learned"]["paired_iid_95pct_CI_vs_"+fixed]=[
            float(np.quantile(intervals,.025)),float(np.quantile(intervals,.975))]
    return values


@torch.no_grad()
def evaluate(folder,records,model_rows,name):
    results={}
    for rec in model_rows:
        sub=folder/f'{rec["arm"]}-s{rec["seed"]}'
        if digest(sub/rec["checkpoint"])!=rec["checkpoint_SHA256"]:
            raise RuntimeError("original frozen model changed")
        checkpoint=torch.load(sub/rec["checkpoint"],weights_only=True)
        model=make_model(rec["arm"])
        model.load_state_dict(checkpoint["weights"],strict=True)
        model.eval()
        qv,qc=outcomes(model,records)
        action=records["a"].numpy();y=records["visit"].numpy()
        conv=records["conversion"].numpy()
        observed_v=qv[np.arange(len(y)),action]
        observed_c=qc[np.arange(len(y)),action]
        m={"observed_visit":metrics_for_action(observed_v,y,action,0),
           "observed_visit_treated":metrics_for_action(observed_v,y,action,1),
           "observed_conversion":metrics_for_action(observed_c,conv,action,0),
           "observed_conversion_treated":metrics_for_action(observed_c,conv,action,1),
           "policy_exploratory_UNQUALIFIED":policy_exploratory(qv,action,y,rec["seed"]),
           "original_individual_counterfactual_labels_observed":False,
           "publisher_conditional_propensity_verified":False}
        for key,probs,labels in (("visit",observed_v,y),("conversion",observed_c,conv)):
            p=np.clip(probs,1e-6,1-1e-6)
            m[key+"_Brier"]=float(np.mean((p-labels)**2))
            m[key+"_logloss"]=float(np.mean(-(labels*np.log(p)+(1-labels)*np.log1p(-p))))
        name_file=name+"-real-predictions.npz"
        np.savez_compressed(sub/name_file,visit=qv,conversion=qc)
        m["prediction_file"]=name_file
        m["prediction_SHA256"]=digest(sub/name_file)
        results[f'{rec["arm"]}-s{rec["seed"]}']=m
    return results


def research_gates(test):
    gates={}
    for seed in SEEDS:
        new=test[f'{NEW_ARM}-s{seed}']
        native=test[f'action_tac-s{seed}']
        context=test[f'action_context-s{seed}']
        gru=test[f'action_gru-s{seed}']
        gates[str(seed)]={
            "native_action_state_better_than_original_native_pooled":
                new["visit_Brier"]<native["visit_Brier"],
            "native_action_state_better_than_original_native_untreated":
                new["observed_visit"]["Brier"]<native["observed_visit"]["Brier"],
            "native_action_state_better_than_original_native_treated":
                new["observed_visit_treated"]["Brier"]<native["observed_visit_treated"]["Brier"],
            "native_action_state_beats_context_and_gru_pooled":
                new["visit_Brier"]<min(context["visit_Brier"],gru["visit_Brier"])}
    return gates,all(all(row.values()) for row in gates.values())


def run(out):
    root=Path(out);root.mkdir(parents=True,exist_ok=False)
    report={"study":STUDY,"complete":False,"authentic_source_SHA256":ORIGINAL_SHA,
            "license":"CC-BY-NC-SA-4.0 research-only; no raw redistribution",
            "temporal_entity_memory_qualified":False,
            "original_logged_conditional_propensities_known":False,
            "original_treatment_outcome_for_unassigned_action_observed":False,
            "trained_models":[],"sealed_outcomes_opened":False,
            "no_merge_no_commercial_promotion":True}
    save(root/"results.json",report)
    source=fetch_original(root/"source-private-NOT-ARTIFACT")
    partitions,manifest=plan_supported_indices(source)
    report["support_design"]=manifest
    save(root/"results.json",report)
    train_data=load_partition(source,partitions["FIT"],160000)
    report["FIT_actual_outcomes_by_treatment"]=cohort_summary(train_data)
    save(root/"results.json",report)
    for seed in SEEDS:
        for arm in ARMS:
            record=train(train_data,arm,seed,root/f"{arm}-s{seed}")
            report["trained_models"].append(record)
            save(root/"results.json",report)
    if len(report["trained_models"])!=8:raise RuntimeError("source fits not complete")
    for rec in report["trained_models"]:
        cp=root/f'{rec["arm"]}-s{rec["seed"]}'/rec["checkpoint"]
        if digest(cp)!=rec["checkpoint_SHA256"]:raise RuntimeError("model freeze changed")
    report["all_original_models_SHA_frozen_before_DEV_TEST"]=True
    save(root/"results.json",report)
    # This is the FIRST numerical parsing of either non-FIT outcome cohort.
    dev=load_partition(source,partitions["DEV"],40000)
    test=load_partition(source,partitions["TEST"],40000)
    report["sealed_outcomes_opened"]=True
    report["DEV_actual_outcomes_by_treatment"]=cohort_summary(dev)
    report["TEST_actual_outcomes_by_treatment"]=cohort_summary(test)
    report["DEV_results"]=evaluate(root,dev,report["trained_models"],"DEV")
    report["TEST_results"]=evaluate(root,test,report["trained_models"],"TEST")
    gates,pass_accuracy=research_gates(report["TEST_results"])
    counts={a:[r["parameter_count"] for r in report["trained_models"] if r["arm"]==a]
            for a in ARMS}
    report["parameter_counts"]=counts
    report["research_accuracy_gates_by_seed"]=gates
    report["all_preregistered_accuracy_gates_met"]=pass_accuracy
    report["full_parameter_and_CPU_parity_established"]=False
    report["strict_causal_policy_value_qualified"]=False
    report["general_world_state_intelligence_qualified"]=False
    report["complete"]=True
    save(root/"results.json",report)
    compact={a:{str(seed):report["TEST_results"][f"{a}-s{seed}"]["visit_Brier"]
                for seed in SEEDS} for a in ARMS}
    print(json.dumps({"event":"ME049_SUPPORTED_REAL_ACTION_RESULTS",
                      "test_visit_Brier":compact,"gates":gates,
                      "causal_claim":False,"real_sequential_state_claim":False}),flush=True)


@torch.no_grad()
def replay(root):
    root=Path(root);report=json.loads((root/"results.json").read_text())
    if not report["complete"] or not report["all_original_models_SHA_frozen_before_DEV_TEST"]:
        raise RuntimeError("no complete frozen original run")
    source=fetch_original(root/"source-private-NOT-ARTIFACT")
    splits,manifest=plan_supported_indices(source)
    if manifest!=report["support_design"]:
        raise RuntimeError("sealed cohort selection altered")
    cohorts={"DEV":load_partition(source,splits["DEV"],40000),
             "TEST":load_partition(source,splits["TEST"],40000)}
    for rec in report["trained_models"]:
        sub=root/f'{rec["arm"]}-s{rec["seed"]}'
        cp=sub/rec["checkpoint"]
        if digest(cp)!=rec["checkpoint_SHA256"]:raise RuntimeError("model changed")
        state=torch.load(cp,weights_only=True)
        model=make_model(rec["arm"]);model.load_state_dict(state["weights"],strict=True)
        model.eval()
        for part,cohort in cohorts.items():
            v,c=outcomes(model,cohort)
            details=report[part+"_results"][f'{rec["arm"]}-s{rec["seed"]}']
            f=sub/details["prediction_file"]
            if digest(f)!=details["prediction_SHA256"]:raise RuntimeError("prediction bytes changed")
            with np.load(f) as saved:
                if not np.array_equal(v,saved["visit"]) or not np.array_equal(c,saved["conversion"]):
                    raise RuntimeError("real original trained-prediction replay mismatch")
    print("ME049_EIGHT_ORIGINAL_REAL_TRAINED_MODELS_REPLAY_EXACTLY",flush=True)


def self_test():
    """Software-only numerical contract, never synthetic performance."""
    torch.manual_seed(49049)
    for arm in ARMS:
        model=make_model(arm)
        x=torch.randn(4,12);action=torch.tensor([0,1,0,1])
        goal=torch.zeros(4,dtype=torch.long)
        logits=model.forward(x,goal)
        assert logits.shape==(4,2) and torch.isfinite(logits).all()
        loss=authentic_logged_action_loss(model,x,action,torch.ones(4),torch.zeros(4))
        loss.backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all()
                   for p in model.parameters() if p.requires_grad)
    print("ME049_SOFTWARE_NUMERICAL_ONLY_PASS",flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--out",default="results/me049")
    parser.add_argument("--self-test",action="store_true")
    parser.add_argument("--replay",action="store_true")
    flags=parser.parse_args()
    if flags.self_test:self_test()
    elif flags.replay:replay(flags.out)
    else:run(flags.out)
