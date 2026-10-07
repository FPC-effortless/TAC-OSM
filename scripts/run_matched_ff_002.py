#!/usr/bin/env python3
from __future__ import annotations
import argparse,hashlib,json,os,platform,random,statistics
from pathlib import Path
import torch
from torch import nn
ROOT=Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/"src"))
from tac_osm.integrated_e2e_008_benchmark import EVAL_EPISODES,HELDOUT,TRAIN_COMBOS,GENERATOR_VERSION,OPS,generator_hash,episode_fingerprint,episode_key,sample_episode,sample_evaluation_episodes
from tac_osm.integrated_e2e_ff_feedback_baseline import FeedForwardFeedbackMultimodal
EXPERIMENT_ID="TACOSM-PLM-MATCHED-FF-002";CONTRACT_PATH=ROOT/"contracts"/f"{EXPERIMENT_ID}.json"
SEEDS=(0,1,2,3,4);STEPS=300;BATCH_SIZE=96;PLM_PARAMS=74126;PARAM_TOL=.01

def reps(m,ep):
    return [m.encode(t.unsqueeze(0),im.unsqueeze(0),au.unsqueeze(0).unsqueeze(0)) for _,t,im,au in ep[0]]
def train_batch(m,episodes):
    losses=[]
    for ep in episodes:
        r=reps(m,ep);q1=ep[1];q2=ep[2]
        l1=m.query(r[0],r[0],torch.tensor([q1[1]]),torch.tensor([q1[2]]),torch.tensor([OPS.index(q1[3])]),torch.zeros(1)) 
        y1=torch.tensor([q1[4]]);losses.append(nn.functional.cross_entropy(l1,y1))
        outcome=((l1.argmax(-1)==y1).float()).detach()
        l2=m.query(r[0],r[1],torch.tensor([q2[1]]),torch.tensor([q2[2]]),torch.tensor([OPS.index(q2[3])]),outcome)
        losses.append(nn.functional.cross_entropy(l2,torch.tensor([q2[4]])))
    return torch.stack(losses).mean()
@torch.no_grad()
def evaluate(m,episodes):
    m.eval();q1=q2=0;per={}
    for ep in episodes:
        r=reps(m,ep);a=ep[1];b=ep[2]
        l1=m.query(r[0],r[0],torch.tensor([a[1]]),torch.tensor([a[2]]),torch.tensor([OPS.index(a[3])]),torch.zeros(1))
        out=((l1.argmax(-1)==torch.tensor([a[4]])).float()).detach()
        l2=m.query(r[0],r[1],torch.tensor([b[1]]),torch.tensor([b[2]]),torch.tensor([OPS.index(b[3])]),out)
        q1+=int(l1.argmax(-1).item()==a[4]);q2+=int(l2.argmax(-1).item()==b[4]);k=(b[3],b[1],b[2]);per.setdefault(k,[0,0]);per[k][0]+=int(l2.argmax(-1).item()==b[4]);per[k][1]+=1
    n=len(episodes);return {"q1_accuracy":q1/n,"q2_accuracy":q2/n,"per_composition_q2_accuracy":{str(k):a/b for k,(a,b) in per.items()}}
def bootstrap(rows):
    c=sorted(rows[0]["per_composition_q2_accuracy"]);means=[statistics.fmean(r["per_composition_q2_accuracy"][x] for r in rows) for x in c];rng=random.Random(20261008);v=sorted(statistics.fmean(rng.choices(means,k=len(means))) for _ in range(5000));return [v[125],v[4874]]
def run(smoke=False):
    c=json.loads(CONTRACT_PATH.read_text());p=FeedForwardFeedbackMultimodal().parameter_count()
    assert c["status"]=="pre-registered" and c["seeds"]==list(SEEDS) and c["steps"]==STEPS and c["batch_size"]==BATCH_SIZE and c["heldout"]==[list(x) for x in HELDOUT] and c["benchmark_generator_version"]==GENERATOR_VERSION
    assert abs(p-PLM_PARAMS)/PLM_PARAMS<=PARAM_TOL
    seeds=(0,) if smoke else SEEDS;steps=25 if smoke else STEPS;bs=8 if smoke else BATCH_SIZE;en=16 if smoke else EVAL_EPISODES;rows=[]
    for seed in seeds:
        torch.manual_seed(seed);rng=random.Random(seed+180800);m=FeedForwardFeedbackMultimodal();opt=torch.optim.AdamW(m.parameters(),lr=.002,weight_decay=.0001);keys=set()
        for _ in range(steps):
            eps=[sample_episode(rng) for _ in range(bs)];keys.update(episode_key(e) for e in eps);opt.zero_grad(set_to_none=True);loss=train_batch(m,eps);loss.backward();nn.utils.clip_grad_norm_(m.parameters(),1.0);opt.step()
        ev=sample_evaluation_episodes(random.Random(seed+181000),en);assert not(keys & {episode_key(e) for e in ev});rows.append({"seed":seed,"evaluation_episode_fingerprint":episode_fingerprint(ev),**evaluate(m,ev)})
    assert len({r["evaluation_episode_fingerprint"] for r in rows})==len(rows);vals=[r["q2_accuracy"] for r in rows]
    return {"experiment_id":EXPERIMENT_ID,"status":"smoke" if smoke else "measured","provenance":{"git_commit":os.environ.get("GITHUB_SHA","unknown"),"workflow_run_id":os.environ.get("GITHUB_RUN_ID","unknown"),"python_version":platform.python_version(),"torch_version":torch.__version__,"contract_sha256":hashlib.sha256(CONTRACT_PATH.read_bytes()).hexdigest(),"benchmark_sha256":generator_hash()},"protocol":{"seeds":list(seeds),"steps":STEPS,"batch_size":BATCH_SIZE,"evaluation_episodes_per_seed":en,"heldout":[list(x) for x in HELDOUT],"benchmark_generator_version":GENERATOR_VERSION,"plm_parameter_count":PLM_PARAMS,"baseline_parameter_count":p,"parameter_relative_difference":abs(p-PLM_PARAMS)/PLM_PARAMS,"model_selection":"none","persistence":"none","feedback":"detached q1 environment outcome supplied directly"},"seed_results":rows,"summary":{"primary_q2_mean":statistics.fmean(vals),"primary_q2_min":min(vals),"composition_bootstrap_ci95":bootstrap(rows) if not smoke else [None,None],"per_composition_q2_accuracy":({k:statistics.fmean(r["per_composition_q2_accuracy"][k] for r in rows) for k in sorted(rows[0]["per_composition_q2_accuracy"])} if not smoke else {})}}
if __name__=="__main__":
    ap=argparse.ArgumentParser();ap.add_argument("--smoke",action="store_true");a=ap.parse_args();o=run(a.smoke);(ROOT/"artifacts").mkdir(exist_ok=True);(ROOT/"artifacts"/f"{EXPERIMENT_ID}.json").write_text(json.dumps(o,indent=2)+"\n");print(json.dumps(o["summary"],indent=2))
