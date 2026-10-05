#!/usr/bin/env python3
"""PLM-018A: amortized one-step active-probe policy."""
from __future__ import annotations
import argparse, hashlib, json, math, os, random, resource, statistics, sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence
import torch
from torch import Tensor
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src")); sys.path.insert(0,str(ROOT/"scripts"))
import run_graph_casm_structured_action_probe_015 as g15
from tac_osm.amortized_probe_policy import AmortizedProbePolicy, budget_capped_utility, choose_greedy_budget_action, make_action_sketch_matrix
EXPERIMENT_ID="TACOSM-PLM-AMORTIZED-PROBE-POLICY-018A"
GENERATOR_COMMIT=g15.GENERATOR_COMMIT
TRAIN_SEEDS=tuple(range(20)); VAL_SEEDS=tuple(range(20,25)); TEST_SEEDS=tuple(range(25,30))
M_LEVELS=(32,64,128,256,512); TASKS_PER_M=32; TRAIN_SUBSETS_PER_M=8; BUDGET=8
HIDDEN=64; EPOCHS=300; LR=1e-3; WEIGHT_DECAY=1e-5; BATCH_SIZE=32; TRAIN_SEED=18018
INPUT_ROWS=g15.INPUT_ROWS; TRACE_WIDTH=g15.TRACE_READ_UNITS; ACTIONS=tuple(range(len(INPUT_ROWS)))

@dataclass(frozen=True)
class LibraryBundle:
    library: tuple
    cache: object

def split_library(seed:int)->LibraryBundle:
    training=g15.g010.generate(seed+100,g15.TRAIN_PROGRAMS)
    ts={g15.g010.structure_key(ep) for ep in training}; tt={g15.g010.truth_signature(ep) for ep in training}
    library=g15.g010.generate(seed+5000,g15.LIBRARY_SIZE,exclude_structures=ts,exclude_truths=tt)
    if len(library)!=g15.LIBRARY_SIZE: raise RuntimeError("evaluation library construction failed")
    ls={g15.g010.structure_key(ep) for ep in library}; lt={g15.g010.truth_signature(ep) for ep in library}
    if ts & ls: raise RuntimeError("train/evaluation structural leakage")
    if tt & lt: raise RuntimeError("train/evaluation truth-table leakage")
    cache,_=g15.build_cache(library)
    return LibraryBundle(tuple(library),cache)

def teacher_state(bundle:LibraryBundle, indices:Sequence[int]):
    evidence={}; costs={}
    for row in ACTIONS:
        evidence[row]=tuple(bundle.cache.trace[(idx,row)][0] for idx in indices)
        costs[row]=statistics.fmean(bundle.cache.trace[(idx,row)][1]+TRACE_WIDTH for idx in indices)
    teacher=choose_greedy_budget_action(evidence,budget=BUDGET,acquisition_cost_by_action=costs)
    scores=[]
    for row in ACTIONS:
        ub=budget_capped_utility(evidence[row],BUDGET); base=min(BUDGET,len(indices))/len(indices)
        scores.append((ub-base)/costs[row])
    return teacher,tuple(float(x) for x in scores)

def action_features(bundle:LibraryBundle, indices:Sequence[int]):
    sig={row:[bundle.cache.trace[(idx,row)][0] for idx in indices] for row in ACTIONS}
    rows={row:tuple(float(x) for x in INPUT_ROWS[row]) for row in ACTIONS}
    costs={row:statistics.fmean(bundle.cache.trace[(idx,row)][1]+TRACE_WIDTH for idx in indices) for row in ACTIONS}
    return make_action_sketch_matrix(sig,rows,costs,budget=BUDGET)

def build_training_examples(seeds:Sequence[int],levels:Sequence[int],subsets_per_m:int):
    xs=[]; ys=[]; ss=[]
    for seed in seeds:
        bundle=split_library(seed)
        for m in levels:
            prefix=tuple(range(m))
            for si in range(subsets_per_m):
                rng=random.Random(seed*9011+m*101+si*17+7)
                size=max(2,int(round(m*(0.55+0.40*rng.random()))))
                indices=tuple(sorted(rng.sample(prefix,size)))
                features,actions=action_features(bundle,indices)
                teacher,scores=teacher_state(bundle,indices)
                xs.append(features); ys.append(actions.index(teacher.action)); ss.append(torch.tensor(scores,dtype=torch.float32))
    return torch.cat(xs,dim=0),torch.tensor(ys,dtype=torch.long),torch.stack(ss)

def train_model(x:Tensor,y:Tensor,scores:Tensor,val_x:Tensor,val_y:Tensor,val_scores:Tensor):
    torch.manual_seed(TRAIN_SEED)
    model=AmortizedProbePolicy(int(x.shape[-1]),HIDDEN)
    opt=torch.optim.AdamW(model.parameters(),lr=LR,weight_decay=WEIGHT_DECAY)
    loader=DataLoader(TensorDataset(x,y,scores),batch_size=BATCH_SIZE,shuffle=True,generator=torch.Generator().manual_seed(TRAIN_SEED))
    best_state=None; best_regret=float("inf"); history=[]
    for epoch in range(EPOCHS):
        model.train(); losses=[]
        for xb,yb,sb in loader:
            logits=model(xb)
            with torch.no_grad():
                soft=torch.softmax((sb-sb.max(dim=1,keepdim=True).values)*20.0,dim=1)
            logp=torch.log_softmax(logits,dim=1)
            loss=F.nll_loss(logp,yb)+0.25*F.kl_div(logp,soft,reduction="batchmean")
            opt.zero_grad(); loss.backward(); opt.step(); losses.append(float(loss.detach()))
        model.eval()
        with torch.no_grad(): pred=torch.argmax(model(val_x),dim=1)
        sn=val_scores.numpy(); pn=pred.numpy(); chosen=sn[range(len(pn)),pn]; best=sn.max(axis=1)
        regret=float(((best-chosen)/(best+1e-12)).mean()); agreement=float((pred==val_y).float().mean())
        history.append({"epoch":epoch,"train_loss":statistics.fmean(losses),"val_regret":regret,"val_teacher_agreement":agreement})
        if regret<best_regret:
            best_regret=regret; best_state={k:v.detach().clone() for k,v in model.state_dict().items()}
    if best_state is None: raise RuntimeError("validation never produced a checkpoint")
    model.load_state_dict(best_state); return model,history

def target_index(seed:int,m:int,task_i:int)->int:
    return random.Random(seed*3000001+m*10007+task_i*1009+17).randrange(m)

def exact_verify(target,candidates,shortlist):
    for idx in shortlist:
        cand=candidates[idx]
        for bits in INPUT_ROWS:
            got,_=g15.g010.execute_exact(cand,bits)
            if got!=int(target.truth_table[bits]): break
        else: return 1.0
    return 0.0

def evaluate(model:AmortizedProbePolicy,seeds:Sequence[int],levels:Sequence[int],tasks_per_m:int):
    out={}; model.eval()
    for seed in seeds:
        bundle=split_library(seed); out[seed]={}
        for m in levels:
            indices=tuple(range(m)); features,actions=action_features(bundle,indices); teacher,teacher_scores=teacher_state(bundle,indices)
            entropy_scores=[]
            for row in ACTIONS:
                sigs=tuple(bundle.cache.trace[(idx,row)][0] for idx in indices); n=len(sigs)
                entropy=-sum((c/n)*math.log2(c/n) for c in __import__("collections").Counter(sigs).values())
                cost=statistics.fmean(bundle.cache.trace[(idx,row)][1]+TRACE_WIDTH for idx in indices)
                entropy_scores.append(entropy/cost)
            info_action=max(ACTIONS,key=lambda r:(entropy_scores[r],-r))
            learner_action=int(actions[model.choose(features)])
            rows=[]
            for task_i in range(tasks_per_m):
                ti=target_index(seed,m,task_i)
                rr=random.Random(seed*7777+m*313+task_i*41+19); rand_action=rr.choice(ACTIONS)
                arm_actions={"greedy_budget_teacher":int(teacher.action),"greedy_information_teacher":int(info_action),"amortized_policy":learner_action,"fixed_random":int(rand_action)}
                task={"task_i":task_i,"target_index":ti,"arms":{}}
                for arm,row in arm_actions.items():
                    evidence=tuple(bundle.cache.trace[(idx,row)][0] for idx in indices); target_evidence=bundle.cache.trace[(ti,row)][0]
                    bucket=tuple(i for i,sig in enumerate(evidence) if sig==target_evidence); selected=bucket[:BUDGET]
                    ub=budget_capped_utility(evidence,BUDGET); base=min(BUDGET,m)/m
                    cost=statistics.fmean(bundle.cache.trace[(idx,row)][1]+TRACE_WIDTH for idx in indices); score=(ub-base)/cost
                    success=exact_verify(bundle.library[ti],bundle.library,selected)
                    max_teacher_score=max(teacher_scores); regret=(max_teacher_score-score)/max_teacher_score if max_teacher_score>1e-12 else 0.0
                    task["arms"][arm]={"selected_row":row,"budget_capped_utility":float(ub),"utility_per_work":float(score),"normalized_regret":float(regret),"target_bucket_size":len(bucket),"shortlist_size":len(selected),"verified_success":float(success),"target_in_shortlist":float(ti in selected)}
                rows.append(task)
            out[seed][m]=rows
    return out

def bootstrap(values:Sequence[float],seed:int,rounds:int=4000):
    if not values: return [None,None]
    rng=random.Random(seed); n=len(values); draws=sorted(statistics.fmean(values[rng.randrange(n)] for _ in range(n)) for _ in range(rounds))
    return [float(draws[int(.025*(rounds-1))]),float(draws[int(.975*(rounds-1))])]

def summarize(evaluated,seeds,levels):
    by_m={}
    for m in levels:
        by_m[str(m)]={}
        for arm in ("greedy_budget_teacher","greedy_information_teacher","amortized_policy","fixed_random"):
            vals=[r["arms"][arm] for s in seeds for r in evaluated[s][m]]
            by_m[str(m)][arm]={"U8_mean":statistics.fmean(v["budget_capped_utility"] for v in vals),"verified_success_mean":statistics.fmean(v["verified_success"] for v in vals),"utility_per_work_mean":statistics.fmean(v["utility_per_work"] for v in vals)}
    if 512 not in levels: return {"by_M":by_m,"primary":{"available":False,"reason":"smoke omitted M=512"}}
    regrets=[statistics.fmean(r["arms"]["amortized_policy"]["normalized_regret"] for r in evaluated[s][512]) for s in seeds]
    verified=[statistics.fmean(r["arms"]["amortized_policy"]["verified_success"] for r in evaluated[s][512]) for s in seeds]
    return {"by_M":by_m,"primary":{"M":512,"B":BUDGET,"seed_level_normalized_regret":regrets,"mean_normalized_regret":statistics.fmean(regrets),"seed_bootstrap_95ci":bootstrap(regrets,18019),"seed_level_verified_success":verified,"mean_verified_success":statistics.fmean(verified)}}

def main(smoke=False):
    from tac_osm.contract import load_contract
    c=load_contract(EXPERIMENT_ID)
    if smoke: train_seeds=(0,); val_seeds=(20,); test_seeds=(25,); levels=(32,128); subsets=1; tasks=4
    else:
        train_seeds=TRAIN_SEEDS; val_seeds=VAL_SEEDS; test_seeds=TEST_SEEDS; levels=M_LEVELS; subsets=TRAIN_SUBSETS_PER_M; tasks=TASKS_PER_M
        c.require_levels(M_LEVELS); c.require_seeds(TEST_SEEDS); c.require_steps(1); c.require_eval_steps(TASKS_PER_M); c.require_arms(["greedy_budget_teacher","greedy_information_teacher","amortized_policy","fixed_random"])
    train_x,train_y,train_scores=build_training_examples(train_seeds,levels,subsets)
    val_x,val_y,val_scores=build_training_examples(val_seeds,levels,1)
    model,history=train_model(train_x,train_y,train_scores,val_x,val_y,val_scores)
    evaluated=evaluate(model,test_seeds,levels,tasks); summary=summarize(evaluated,test_seeds,levels)
    result={"experiment_id":EXPERIMENT_ID,"status":"measured","provenance":{"tacosm_commit":os.environ.get("GITHUB_SHA","local"),"run_id":os.environ.get("GITHUB_RUN_ID","local"),"generator_commit":GENERATOR_COMMIT,"python":sys.version,"torch":torch.__version__},"protocol":{"training_seeds":list(train_seeds),"validation_seeds":list(val_seeds),"test_seeds":list(test_seeds),"M_levels":list(levels),"B":BUDGET,"tasks_per_seed_M":tasks,"feature_dim":int(train_x.shape[-1]),"hidden_dim":HIDDEN,"epochs":EPOCHS,"learning_rate":LR,"weight_decay":WEIGHT_DECAY,"one_step_only":True},"split_integrity":{"train_eval_structure_disjoint":True,"train_eval_truth_disjoint":True,"test_teacher_outputs_generated_after_checkpoint_freeze":True,"test_teacher_outputs_used_for_training":False,"target_identity_used_by_policy":False,"realized_target_evidence_used_before_action":False},"training_history":history,"summary":summary,"results":evaluated,"scope":{"finite_domain_amortized_policy":True,"target_blind_action_selection":True,"learned_probe_policy_claim":True,"general_active_learning_claim":False,"submodularity_claim":False,"asymptotic_claim":False,"external_generalization_claim":False,"language_image_audio_claim":False,"hardware_speedup_claim":False},"peak_rss_mb":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024.0}
    Path("artifacts").mkdir(exist_ok=True); Path("artifacts",f"{EXPERIMENT_ID}.json").write_text(json.dumps(result,indent=2,sort_keys=True)); print(json.dumps(summary,indent=2,sort_keys=True))

if __name__=="__main__":
    p=argparse.ArgumentParser(); p.add_argument("--smoke",action="store_true"); main(p.parse_args().smoke)