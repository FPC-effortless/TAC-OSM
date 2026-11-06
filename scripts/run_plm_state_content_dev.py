#!/usr/bin/env python3
from __future__ import annotations
import hashlib,json,os,platform,random,statistics
from pathlib import Path
import torch
from torch import nn
ROOT=Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/"src"))
from tac_osm.integrated_e2e_005 import FunctionalConfig,FunctionalMultimodalPLM
from tac_osm.integrated_e2e_005_benchmark import ALL_COMBOS,HELDOUT as SEALED_E2E005,episode_fingerprint,episode_key,sample_episode
from tac_osm.integrated_e2e_006_benchmark import HELDOUT as E2E006_HELDOUT

EXPERIMENT_ID="TACOSM-PLM-STATE-CONTENT-DEV-001"
CONTRACT_PATH=ROOT/"contracts"/f"{EXPERIMENT_ID}.json"
SEEDS=(0,1,2,3,4);STEPS=300;BATCH_SIZE=96;EVAL_EPISODES=200
DEV_COMBOS=(
 ("xor",0,5),("xor",0,6),("xor",0,7),("xor",0,8),("xor",0,9),("xor",0,10),("xor",0,11),("xor",1,4),
 ("and",0,4),("and",0,5),("and",0,6),("and",0,7),("and",0,8),("and",0,9),("and",0,10),("and",0,11),
 ("or",0,4),("or",0,5),("or",0,6),("or",0,7),("or",0,8),("or",0,9),("or",0,10),("or",0,11),
 ("xnor",0,4),("xnor",0,5),("xnor",0,6),("xnor",0,7),("xnor",0,8),("xnor",0,9),("xnor",0,10),("xnor",0,11),
)
TRAIN_COMBOS=tuple(c for c in ALL_COMBOS if c not in set(DEV_COMBOS)|set(SEALED_E2E005)|set(E2E006_HELDOUT))
MODES=("linear","residual_linear","residual_mlp")

def build_batch(eps):
    from scripts.run_integrated_e2e_005 import build_batch
    return build_batch(eps)
def train_batch(model,batch):
    from scripts.run_integrated_e2e_005 import train_batch
    return train_batch(model,batch)

def make_stream(seed):
    rng=random.Random(seed+110700)
    batches=[]
    keys=set()
    for _ in range(STEPS):
        eps=[sample_episode(rng,combo1=rng.choice(TRAIN_COMBOS),combo2=rng.choice(TRAIN_COMBOS)) for _ in range(BATCH_SIZE)]
        batches.append(eps);keys.update(episode_key(e) for e in eps)
    return batches,keys

def dev_stream(seed):
    rng=random.Random(seed+110800);out=[]
    for i in range(EVAL_EPISODES):
        a=DEV_COMBOS[i%len(DEV_COMBOS)];b=DEV_COMBOS[(i*7+3)%len(DEV_COMBOS)]
        if a==b:b=DEV_COMBOS[(i*7+4)%len(DEV_COMBOS)]
        out.append(sample_episode(rng,combo1=a,combo2=b))
    return out

def fit(seed,mode,batches):
    torch.manual_seed(seed)
    model=FunctionalMultimodalPLM(config=FunctionalConfig(hidden_dim=64,state_write_mode=mode)).cpu()
    opt=torch.optim.AdamW(model.parameters(),lr=0.002,weight_decay=0.0001)
    for eps in batches:
        model.train();opt.zero_grad(set_to_none=True)
        loss=train_batch(model,build_batch(eps));loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(),1.0);opt.step()
    return model

@torch.no_grad()
def eval_model(model,episodes):
    model.eval();q2=xor=bits_ok=state_wrong=decision_mismatch=0
    xor_total=0
    for ep in episodes:
        mem=model.state.initial(1,torch.device("cpu"))
        for entity,text,image,audio in ep[0]:
            z=model.encode(text.unsqueeze(0),image.unsqueeze(0),audio.unsqueeze(0).unsqueeze(0))
            mem,_=model.state.write(mem,z,torch.tensor([entity]))
        entity,i,j,op,target=ep[2]
        out=model.query(mem,torch.tensor([entity]),torch.tensor([i]),torch.tensor([j]),torch.tensor([("xor","and","or","xnor").index(op)]))
        pred=int(out["logits"].argmax(-1).item())
        threshold_pred=int((out["action"]>=0.5).long().item())
        decision_mismatch+=int(pred!=threshold_pred)
        q2+=int(pred==target)
        if op=="xor":
            xor_total+=1;xor+=int(pred==target)
            pb=(out["bits"]>=0.5).long().squeeze(0)
            ok=(pb[i].item()==ep[3][entity][i]) and (pb[j].item()==ep[3][entity][j])
            bits_ok+=int(ok);state_wrong+=int(not ok)
    return {
      "q2_accuracy":q2/len(episodes),
      "xor_q2_accuracy":xor/max(xor_total,1),
      "xor_queried_bit_joint_accuracy":bits_ok/max(xor_total,1),
      "xor_state_error_rate":state_wrong/max(xor_total,1),
      "classifier_decision_mismatches":decision_mismatch
    }

def run():
    c=json.loads(CONTRACT_PATH.read_text())
    assert c["status"]=="pre-registered" and c["seeds"]==list(SEEDS) and c["steps"]==STEPS and c["eval_steps"]==EVAL_EPISODES
    rows=[]
    for seed in SEEDS:
        batches,train_keys=make_stream(seed);episodes=dev_stream(seed)
        eval_keys={episode_key(e) for e in episodes}
        overlap=train_keys&eval_keys
        if overlap: raise AssertionError(f"train/dev overlap seed {seed}")
        mode_results={}
        fp=episode_fingerprint(episodes)
        for mode in MODES:
            m=fit(seed,mode,batches);r=eval_model(m,episodes)
            mode_results[mode]=r
        rows.append({"seed":seed,"evaluation_fingerprint":fp,"train_dev_overlap":len(overlap),"modes":mode_results})
    out={"experiment_id":EXPERIMENT_ID,"status":"development",
      "provenance":{"git_commit":os.environ.get("GITHUB_SHA","unknown"),"workflow_run_id":os.environ.get("GITHUB_RUN_ID","unknown"),
        "python_version":platform.python_version(),"torch_version":torch.__version__,"contract_sha256":hashlib.sha256(CONTRACT_PATH.read_bytes()).hexdigest()},
      "protocol":{"seeds":list(SEEDS),"steps":STEPS,"batch_size":BATCH_SIZE,"eval_episodes":EVAL_EPISODES,"hidden_dim":64,
        "modes":list(MODES),"dev_compositions":[list(x) for x in DEV_COMBOS],"sealed_e2e005_excluded":True,"e2e006_heldout_excluded":True},
      "results":rows,
      "summary":{mode:{
        "mean_q2_accuracy":statistics.fmean(r["modes"][mode]["q2_accuracy"] for r in rows),
        "mean_xor_q2_accuracy":statistics.fmean(r["modes"][mode]["xor_q2_accuracy"] for r in rows),
        "mean_xor_bit_joint_accuracy":statistics.fmean(r["modes"][mode]["xor_queried_bit_joint_accuracy"] for r in rows),
        "mean_xor_state_error_rate":statistics.fmean(r["modes"][mode]["xor_state_error_rate"] for r in rows),
        "all_decision_mismatches_zero":all(r["modes"][mode]["classifier_decision_mismatches"]==0 for r in rows)
      } for mode in MODES},
      "selection_is_dev_only":True}
    p=ROOT/"artifacts"/f"{EXPERIMENT_ID}.json";p.parent.mkdir(exist_ok=True);p.write_text(json.dumps(out,indent=2,sort_keys=True)+"\n");print(json.dumps(out,indent=2,sort_keys=True))
if __name__=="__main__":run()
