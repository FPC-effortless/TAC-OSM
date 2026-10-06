#!/usr/bin/env python3
from __future__ import annotations
import argparse, hashlib, json, platform
from pathlib import Path
import numpy as np
from tac_osm.contract import load_contract
from tac_osm.measurement.results import Design,Gate,MeasurementRecord,Provenance,contract_fingerprint,now,report_smoke,results_dir_for,write_record
from tac_osm.adaptive_timescale import ARMS,BENCHMARK_HASH,PAIR_TYPES_TEST,PAIR_TYPES_TRAIN,TRAIN_FILTER_ALPHAS,TEST_FILTER_ALPHAS,make_pairs,make_policy,_features,evaluate,parameter_count,benchmark_hash
from tac_osm.timescale_transfer import TransferMLPPolicy

EXPERIMENT_ID="TACOSM-PLM-TDBU-ADAPTIVE-TIMESCALE-001"
TRAIN_PAIRS=300
TEST_PAIRS=100
POLICY_LR=0.05
ALPHA_LR=0.02
MATERIALITY=0.10
STEPS_DEFAULT=500
EVAL_STEPS_DEFAULT=200
ARMS=("no_state","one_fixed","three_fixed","three_adaptive")
TRAINED_ARMS=ARMS

def _seed_examples(seed:int,H:int,split:str):
    base=(50_000 if split=="train" else 60_000)+seed*1009+H*17
    alphas=TRAIN_FILTER_ALPHAS if split=="train" else TEST_FILTER_ALPHAS
    pairs=PAIR_TYPES_TRAIN if split=="train" else PAIR_TYPES_TEST
    return make_pairs(base,H,TRAIN_PAIRS if split=="train" else TEST_PAIRS,alphas=alphas,pair_types=pairs)

def _pairs_ok(examples):
    by={}
    for e in examples: by.setdefault(e.pair_id,[]).append(e)
    assert by and all(len(v)==2 for v in by.values())
    for v in by.values():
        assert v[0].current_observation==v[1].current_observation==0.0
        assert v[0].label!=v[1].label and np.array_equal(v[1].history,-v[0].history)

def _hash(policy):
    h=hashlib.sha256()
    for name in ("W1","b1","W2","b2"):
        h.update(np.ascontiguousarray(getattr(policy,name)).tobytes())
    if hasattr(policy,"alpha_logits"):
        h.update(np.ascontiguousarray(policy.alpha_logits).tobytes())
    return h.hexdigest()

def _bootstrap(a,b,seed=2991,rounds=20000):
    diff=np.asarray(a)-np.asarray(b)
    rng=np.random.default_rng(seed)
    draws=rng.choice(diff,size=(rounds,len(diff)),replace=True).mean(axis=1)
    return float(diff.mean()),float(np.quantile(draws,0.025)),float(np.quantile(draws,0.975))

def run(args):
    contract=load_contract(EXPERIMENT_ID)
    contract.require_levels(args.h_levels); contract.require_seeds(args.seeds); contract.require_steps(args.steps); contract.require_eval_steps(args.eval_steps); contract.require_arms([a.name for a in contract.arms])
    assert benchmark_hash()==BENCHMARK_HASH and len(set(np.round(TRAIN_FILTER_ALPHAS,12))&set(np.round(TEST_FILTER_ALPHAS,12)))==0
    rows=[]; initials={}
    for H in contract.h_levels:
        for seed in contract.seeds:
            train=_seed_examples(seed,H,"train"); test=_seed_examples(seed,H,"test"); _pairs_ok(train); _pairs_ok(test)
            y=np.asarray([e.label for e in train],dtype=np.int64)
            for arm in TRAINED_ARMS:
                if arm=="three_adaptive":
                    policy=make_policy(seed+7,arm)
                    initials[(H,seed,arm)]=_hash(policy)
                    policy.fit(train,contract.steps,POLICY_LR,ALPHA_LR)
                else:
                    policy=TransferMLPPolicy(seed=seed+7,input_dim=4,hidden=12)
                    initials[(H,seed,arm)]=_hash(policy)
                    X=_features(train,arm)
                    y=np.asarray([e.label for e in train],dtype=np.int64)
                    policy.fit(X,y,contract.steps,POLICY_LR)
                base_eval=evaluate(policy,test,arm)
                reset=evaluate(policy,test,"three_adaptive",intervention="reset") if arm=="three_adaptive" else None
                shuffle=evaluate(policy,test,"three_adaptive",intervention="shuffle") if arm=="three_adaptive" else None
                rows.append({"H":H,"seed":seed,"arm":arm,"success":base_eval["success"],"action_gap":base_eval["action_gap"],"parameter_count":parameter_count(),"checkpoint_hash":_hash(policy),"initial_checkpoint_hash":initials[(H,seed,arm)],"learned_alphas":policy.alphas.tolist() if arm=="three_adaptive" else None,"reset_success":reset["success"] if reset else None,"shuffle_success":shuffle["success"] if shuffle else None})
    e=[r["success"] for r in rows if r["H"]==256 and r["arm"]=="three_adaptive"]; f=[r["success"] for r in rows if r["H"]==256 and r["arm"]=="three_fixed"]; n=[r["success"] for r in rows if r["H"]==256 and r["arm"]=="no_state"]
    delta,lo,hi=_bootstrap(e,f); reset=[r["reset_success"] for r in rows if r["H"]==256 and r["arm"]=="three_adaptive"]; shuffle=[r["shuffle_success"] for r in rows if r["H"]==256 and r["arm"]=="three_adaptive"]
    return MeasurementRecord(provenance=Provenance(experiment_id=EXPERIMENT_ID,contract_source=f"contracts/{EXPERIMENT_ID}.json",contract_sha256=contract_fingerprint(Path(__file__).resolve().parent.parent/"contracts"/f"{EXPERIMENT_ID}.json"),git_commit=args.commit,script=Path(__file__).name,python=platform.python_version(),recorded_at=now()),design=Design(steps=contract.steps,eval_steps=contract.eval_steps,seeds=tuple(contract.seeds),h_levels=tuple(contract.h_levels),k_levels=tuple(contract.k_levels),arms=tuple(a.name for a in contract.arms),smoke=False,contract_checked=True),gate=Gate(name="TDBU-ADAPTIVE-TIMESCALE-001-preconditions",tolerance="exact contract, train/test separation and paired-history invariants",passed=True,cells=()),endpoints={"primary":{"H":256,"three_adaptive_success_by_seed":e,"three_fixed_success_by_seed":f,"no_state_success_by_seed":n,"adaptive_minus_fixed_mean":delta,"seed_bootstrap_95ci":[lo,hi],"adaptive_minus_no_state_mean":float(np.mean(e)-np.mean(n)),"materiality_threshold":MATERIALITY},"interventions":{"adaptive_reset_success_by_seed":reset,"adaptive_shuffle_success_by_seed":shuffle},"learned_alphas_by_seed":[r["learned_alphas"] for r in rows if r["H"]==256 and r["arm"]=="three_adaptive"],"parameter_count":parameter_count(),"generator_hash":BENCHMARK_HASH},decision_rule=tuple({"condition":b.condition,"licenses":b.licenses,"does_not_license":b.does_not_license} for b in contract.decision_rule),audit={"benchmark":{"generator_hash":BENCHMARK_HASH,"train_filter_alphas":TRAIN_FILTER_ALPHAS.tolist(),"test_filter_alphas":TEST_FILTER_ALPHAS.tolist(),"filter_sets_disjoint":True,"paired_histories":True,"current_observation_constant":True,"prior_transfer_test_constants_not_reused":[0.40,0.58,0.74,0.86,0.93,0.989,0.993]},"capacity":{"parameter_count":86,"all_arms_match":True},"adaptive":{"initial_alphas":[0.50,0.90,0.98],"policy_lr":POLICY_LR,"alpha_lr":ALPHA_LR}},per_seed={"rows":rows,"bootstrap":{"seed":2991,"rounds":20000,"adaptive_minus_fixed":[delta,lo,hi]}})

def main():
    p=argparse.ArgumentParser(); p.add_argument("--steps",type=int,default=STEPS_DEFAULT); p.add_argument("--eval-steps",type=int,default=EVAL_STEPS_DEFAULT); p.add_argument("--h-levels",type=int,nargs="+",default=[64,256,512]); p.add_argument("--seeds",type=int,nargs="+",default=[0,1,2,3,4]); p.add_argument("--commit",default="UNKNOWN"); p.add_argument("--output",default=f"results/{EXPERIMENT_ID}.json"); p.add_argument("--smoke",action="store_true"); args=p.parse_args()
    c=load_contract(EXPERIMENT_ID)
    if args.smoke: report_smoke(c,EXPERIMENT_ID,steps=args.steps,eval_steps=args.eval_steps,h_levels=args.h_levels,seeds=args.seeds,arms=[a.name for a in c.arms]); return 0
    rec=run(args); path=Path(args.output)
    if path==Path(f"results/{EXPERIMENT_ID}.json"): path=results_dir_for(__file__)/path.name
    write_record(rec,path); print(json.dumps(rec.to_dict()["endpoints"]["primary"],indent=2,sort_keys=True)); return 0
if __name__=="__main__": raise SystemExit(main())
