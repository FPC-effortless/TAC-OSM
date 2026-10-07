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
from tac_osm.integrated_e2e_ff_baseline import FeedForwardMultimodal
EXPERIMENT_ID="TACOSM-PLM-MATCHED-FF-001";CONTRACT_PATH=ROOT/"contracts"/f"{EXPERIMENT_ID}.json"
SEEDS=(0,1,2,3,4);STEPS=300;BATCH_SIZE=96;PARAM_TOL=.01

def encode_episode(model,ep):
    return [model.encode(t.unsqueeze(0),im.unsqueeze(0),au.unsqueeze(0).unsqueeze(0)) for _,t,im,au in ep[0]]
def train_batch(model,episodes):
    losses=[]
    for ep in episodes:
        reps=encode_episode(model,ep)
        for qi,slot in ((1,0),(2,1)):
            q=ep[qi]
            logits=model.query(reps[slot],torch.tensor([q[1]]),torch.tensor([q[2]]),torch.tensor([OPS.index(q[3])]))
            losses.append(nn.functional.cross_entropy(logits,torch.tensor([q[4]])))
    return torch.stack(losses).mean()
def gradient_probe():
    torch.manual_seed(20261008);rng=random.Random(8008);m=FeedForwardMultimodal()
    loss=train_batch(m,[sample_episode(rng) for _ in range(4)]);loss.backward()
    req=("text.emb.weight","text.rnn.weight_ih_l0","image.net.0.weight","audio.net.0.weight","rep.fuse.0.weight","main.0.weight","main.2.weight","query_residual.0.weight")
    p=dict(m.named_parameters());bad=[n for n in req if p[n].grad is None or not torch.isfinite(p[n].grad).all()]
    return {"pass":not bad,"bad":bad}
@torch.no_grad()
def evaluate(m,episodes):
    m.eval();q1=q2=pos=0;per={}
    for ep in episodes:
        reps=encode_episode(m,ep)
        for qi,slot in ((1,0),(2,1)):
            q=ep[qi];pred=int(m.query(reps[slot],torch.tensor([q[1]]),torch.tensor([q[2]]),torch.tensor([OPS.index(q[3])])).argmax(-1).item())
            if qi==1:q1+=pred==q[4]
            else:
                q2+=pred==q[4];pos+=pred;k=(q[3],q[1],q[2]);per.setdefault(k,[0,0]);per[k][0]+=pred==q[4];per[k][1]+=1
    n=len(episodes)
    return {"q1_accuracy":q1/n,"q2_accuracy":q2/n,"q2_positive_fraction":pos/n,"per_composition_q2_accuracy":{str(k):a/b for k,(a,b) in per.items()}}
def bootstrap(rows):
    comps=sorted(rows[0]["per_composition_q2_accuracy"]);means=[statistics.fmean(r["per_composition_q2_accuracy"][c] for r in rows) for c in comps]
    rng=random.Random(20261008);v=sorted(statistics.fmean(rng.choices(means,k=len(means))) for _ in range(5000));return [v[125],v[4874]]
def run(smoke=False):
    c=json.loads(CONTRACT_PATH.read_text());m0=FeedForwardMultimodal();p=m0.parameter_count()
    assert c["status"]=="pre-registered" and c["seeds"]==list(SEEDS) and c["steps"]==STEPS and c["batch_size"]==BATCH_SIZE
    assert c["heldout"]==[list(x) for x in HELDOUT] and c["benchmark_generator_version"]==GENERATOR_VERSION
    assert abs(p-c["plm_parameter_count"])/c["plm_parameter_count"]<=PARAM_TOL
    assert gradient_probe()["pass"]
    seeds=(0,) if smoke else SEEDS;steps=25 if smoke else STEPS;bs=8 if smoke else BATCH_SIZE;en=16 if smoke else EVAL_EPISODES;rows=[]
    for seed in seeds:
        torch.manual_seed(seed);rng=random.Random(seed+180800);m=FeedForwardMultimodal();opt=torch.optim.AdamW(m.parameters(),lr=.002,weight_decay=.0001);keys=set()
        for _ in range(steps):
            eps=[sample_episode(rng) for _ in range(bs)];keys.update(episode_key(e) for e in eps);opt.zero_grad(set_to_none=True);loss=train_batch(m,eps);loss.backward();nn.utils.clip_grad_norm_(m.parameters(),1.0);opt.step()
        ev=sample_evaluation_episodes(random.Random(seed+181000),en);assert not(keys & {episode_key(e) for e in ev})
        rows.append({"seed":seed,"evaluation_episode_fingerprint":episode_fingerprint(ev),**evaluate(m,ev)})
    assert len({r["evaluation_episode_fingerprint"] for r in rows})==len(rows)
    vals=[r["q2_accuracy"] for r in rows]
    out={"experiment_id":EXPERIMENT_ID,"status":"smoke" if smoke else "measured","provenance":{"git_commit":os.environ.get("GITHUB_SHA","unknown"),"workflow_run_id":os.environ.get("GITHUB_RUN_ID","unknown"),"python_version":platform.python_version(),"torch_version":torch.__version__,"contract_sha256":hashlib.sha256(CONTRACT_PATH.read_bytes()).hexdigest(),"benchmark_sha256":generator_hash()},"protocol":{"seeds":list(seeds),"steps":STEPS,"batch_size":BATCH_SIZE,"evaluation_episodes_per_seed":en,"heldout":[list(x) for x in HELDOUT],"benchmark_generator_version":GENERATOR_VERSION,"plm_parameter_count":c["plm_parameter_count"],"baseline_parameter_count":p,"parameter_relative_difference":abs(p-c["plm_parameter_count"])/c["plm_parameter_count"],"model_selection":"none","persistence":"none","query_entity_resolution":"direct addressed entity representation"},"seed_results":rows,"summary":{"primary_q2_mean":statistics.fmean(vals),"primary_q2_min":min(vals),"composition_bootstrap_ci95":bootstrap(rows) if not smoke else [None,None],"per_composition_q2_accuracy":({k:statistics.fmean(r["per_composition_q2_accuracy"][k] for r in rows) for k in sorted(rows[0]["per_composition_q2_accuracy"])} if not smoke else {})}}
    return out
if __name__=="__main__":
    ap=argparse.ArgumentParser();ap.add_argument("--smoke",action="store_true");a=ap.parse_args();o=run(a.smoke);(ROOT/"artifacts").mkdir(exist_ok=True);(ROOT/"artifacts"/f"{EXPERIMENT_ID}.json").write_text(json.dumps(o,indent=2)+"\n");print(json.dumps(o["summary"],indent=2))
