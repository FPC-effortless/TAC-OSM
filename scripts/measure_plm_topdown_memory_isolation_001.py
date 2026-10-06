#!/usr/bin/env python3
from __future__ import annotations
import argparse,hashlib,json,platform
from pathlib import Path
import numpy as np
from tac_osm.contract import load_contract
from tac_osm.measurement.results import Design,Gate,MeasurementRecord,Provenance,contract_fingerprint,now,report_smoke,results_dir_for,write_record
from tac_osm.mtsk_topdown import MLPPolicy
from tac_osm.memory_isolation import (
    BENCHMARK_HASH,
    benchmark_hash,
    dependency_hash,
    evaluate,
    featurize,
    make_pairs,
    representability_sign_accuracy,
)

EXPERIMENT_ID="TACOSM-PLM-TDBU-MEMORY-ISOLATION-001"
TRAIN_PAIRS=300
TEST_PAIRS=100
LR=0.05
MATERIALITY=0.10
STEPS_DEFAULT=500
EVAL_STEPS_DEFAULT=200
TRAINED_ARMS=ARMS

def seed_examples(seed,H,split):
    base=(70_000 if split=="train" else 80_000)+seed*1009+H*17
    return make_pairs(base,H,TRAIN_PAIRS if split=="train" else TEST_PAIRS)

def pairs_ok(ex):
    by={}
    for e in ex: by.setdefault(e.pair_id,[]).append(e)
    assert by and all(len(v)==2 for v in by.values())
    for v in by.values():
        assert v[0].current_observation==v[1].current_observation==0.0
        assert v[0].label!=v[1].label and np.array_equal(v[1].history,-v[0].history)

def ph(p):
    h=hashlib.sha256()
    for a in (p.W1,p.b1,p.W2,p.b2):
        h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()

def boot(a,b,seed=3991,rounds=20000):
    d=np.asarray(a)-np.asarray(b)
    rng=np.random.default_rng(seed)
    z=rng.choice(d,size=(rounds,len(d)),replace=True).mean(axis=1)
    return float(d.mean()),float(np.quantile(z,.025)),float(np.quantile(z,.975))

def run(args):
    c=load_contract(EXPERIMENT_ID)
    c.require_levels(args.h_levels)
    c.require_seeds(args.seeds)
    c.require_steps(args.steps)
    c.require_eval_steps(args.eval_steps)
    c.require_arms(ARMS)
    c.require_k_levels(c.k_levels)
    assert benchmark_hash()==BENCHMARK_HASH
    representation=representability_sign_accuracy()
    if representation<0.90:
        raise RuntimeError(f"representability gate failed: {representation}")
    rows=[]; initials={}
    for H in c.h_levels:
        for seed in c.seeds:
            tr=seed_examples(seed,H,"train"); te=seed_examples(seed,H,"test")
            pairs_ok(tr); pairs_ok(te)
            y=np.asarray([e.label for e in tr],dtype=np.int64)
            for arm in TRAINED_ARMS:
                p=MLPPolicy(seed=seed+7,hidden=12)
                initials[(H,seed,arm)]=ph(p)
                X=featurize(tr,arm)
                assert X.shape==(len(tr),4)
                p.fit(X,y,c.steps,LR)
                ev=evaluate(p,te,arm)
                reset=evaluate(p,te,"mtsk","reset") if arm=="mtsk" else None
                shuffle=evaluate(p,te,"mtsk","shuffle") if arm=="mtsk" else None
                rows.append({
                    "H":H,"seed":seed,"arm":arm,
                    "success":ev["success"],"action_gap":ev["action_gap"],
                    "evaluation_work_per_episode":ev["work"],
                    "parameter_count":p.parameter_count,
                    "checkpoint_hash":ph(p),
                    "initial_checkpoint_hash":initials[(H,seed,arm)],
                    "reset_success":reset["success"] if reset else None,
                    "shuffle_success":shuffle["success"] if shuffle else None,
                })
    m=[r["success"] for r in rows if r["H"]==256 and r["arm"]=="mtsk"]
    s=[r["success"] for r in rows if r["H"]==256 and r["arm"]=="single_timescale"]
    n=[r["success"] for r in rows if r["H"]==256 and r["arm"]=="no_state"]
    d,lo,hi=boot(m,s)
    reset=[r["reset_success"] for r in rows if r["H"]==256 and r["arm"]=="mtsk"]
    shuffle=[r["shuffle_success"] for r in rows if r["H"]==256 and r["arm"]=="mtsk"]
    return MeasurementRecord(
      provenance=Provenance(experiment_id=EXPERIMENT_ID,contract_source=f"contracts/{EXPERIMENT_ID}.json",contract_sha256=contract_fingerprint(Path(__file__).resolve().parent.parent/"contracts"/f"{EXPERIMENT_ID}.json"),git_commit=args.commit,script=Path(__file__).name,python=platform.python_version(),recorded_at=now()),
      design=Design(steps=c.steps,eval_steps=c.eval_steps,seeds=tuple(c.seeds),h_levels=tuple(c.h_levels),k_levels=tuple(c.k_levels),arms=ARMS,smoke=False,contract_checked=True),
      gate=Gate(name="TDBU-MEMORY-ISOLATION-001-preconditions",tolerance="exact contract and benchmark invariants",passed=True,cells=()),
      endpoints={
        "primary":{"H":256,"mtsk_success_by_seed":m,"single_timescale_success_by_seed":s,"no_state_success_by_seed":n,"mtsk_minus_single_mean":d,"seed_bootstrap_95ci":[lo,hi],"mtsk_minus_no_state_mean":float(np.mean(m)-np.mean(n)),"materiality_threshold":MATERIALITY},
        "interventions":{"mtsk_reset_success_by_seed":reset,"mtsk_shuffle_success_by_seed":shuffle},
        "representation_sign_accuracy":representation,
        "generator_hash":BENCHMARK_HASH,
        "dependency_hash":dependency_hash(),
        "parameter_count":86,
      },
      decision_rule=tuple({"condition":b.condition,"licenses":b.licenses,"does_not_license":b.does_not_license} for b in c.decision_rule),
      audit={"benchmark":{"generator_hash":BENCHMARK_HASH,"ground_truth_alphas":[0.45,0.985],"paired_histories":True,"current_observation_constant":True,"train_test_seed_streams_disjoint":True,"hidden_relation_family":False,"new_test_stream":True},"model_state":{"parameter_count":86,"initial_checkpoint_hashes":{f"{H}:{seed}:{arm}":v for (H,seed,arm),v in initials.items()}}},
      per_seed={"rows":rows,"bootstrap":{"seed":3991,"rounds":20000,"mtsk_minus_single":[d,lo,hi]}}
    )

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--steps",type=int,default=STEPS_DEFAULT); p.add_argument("--eval-steps",type=int,default=EVAL_STEPS_DEFAULT)
    p.add_argument("--h-levels",type=int,nargs="+",default=[64,256,1024]); p.add_argument("--seeds",type=int,nargs="+",default=[0,1,2,3,4])
    p.add_argument("--commit",default="UNKNOWN"); p.add_argument("--output",default=f"results/{EXPERIMENT_ID}.json"); p.add_argument("--smoke",action="store_true")
    args=p.parse_args(); c=load_contract(EXPERIMENT_ID)
    if args.smoke:
        report_smoke(c,EXPERIMENT_ID,steps=args.steps,eval_steps=args.eval_steps,h_levels=args.h_levels,seeds=args.seeds,arms=list(ARMS)); return 0
    rec=run(args); path=Path(args.output)
    if path==Path(f"results/{EXPERIMENT_ID}.json"): path=results_dir_for(__file__)/path.name
    write_record(rec,path); print(json.dumps(rec.to_dict()["endpoints"]["primary"],indent=2,sort_keys=True)); print(json.dumps(rec.to_dict()["endpoints"]["interventions"],indent=2,sort_keys=True)); return 0
if __name__=="__main__": raise SystemExit(main())
