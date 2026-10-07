#!/usr/bin/env python3
from __future__ import annotations
import hashlib, json, os, platform, random, statistics
from pathlib import Path
import torch
from torch import nn

ROOT=Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0,str(ROOT)); sys.path.insert(0,str(ROOT/"src"))

from tac_osm.integrated_e2e_005 import FunctionalConfig, FunctionalMultimodalPLM
from tac_osm.integrated_e2e_005_benchmark import ALL_COMBOS, HELDOUT, episode_fingerprint, episode_key, sample_episode

EXPERIMENT_ID="TACOSM-PLM-XOR-DIAG-001"
CONTRACT_PATH=ROOT/"contracts"/f"{EXPERIMENT_ID}.json"
SEEDS=(0,1,2,3,4)
STEPS=300
BATCH_SIZE=96
EVAL_EPISODES=200
DEV_COMBOS=(
 ("xor",0,5),("xor",0,6),("xor",0,7),("xor",0,8),("xor",0,9),("xor",0,10),("xor",0,11),("xor",1,4),
 ("and",0,4),("and",0,5),("and",0,6),("and",0,7),("and",0,8),("and",0,9),("and",0,10),("and",0,11),
 ("or",0,4),("or",0,5),("or",0,6),("or",0,7),("or",0,8),("or",0,9),("or",0,10),("or",0,11),
 ("xnor",0,4),("xnor",0,5),("xnor",0,6),("xnor",0,7),("xnor",0,8),("xnor",0,9),("xnor",0,10),("xnor",0,11),
)
EXCLUDED=set(HELDOUT)|set(DEV_COMBOS)
TRAIN_COMBOS=tuple(c for c in ALL_COMBOS if c not in EXCLUDED)

def build_batch(eps):
    from scripts.run_integrated_e2e_005 import build_batch
    return build_batch(eps)

def train_batch(model,batch):
    from scripts.run_integrated_e2e_005 import train_batch
    return train_batch(model,batch)

def train_seed(seed):
    torch.manual_seed(seed)
    rng=random.Random(seed+100700)
    model=FunctionalMultimodalPLM(config=FunctionalConfig(hidden_dim=64)).cpu()
    opt=torch.optim.AdamW(model.parameters(),lr=0.002,weight_decay=0.0001)
    keys=set()
    for _ in range(STEPS):
        eps=[sample_episode(rng,combo1=rng.choice(TRAIN_COMBOS),combo2=rng.choice(TRAIN_COMBOS)) for _ in range(BATCH_SIZE)]
        keys.update(episode_key(e) for e in eps)
        model.train(); opt.zero_grad(set_to_none=True)
        loss=train_batch(model,build_batch(eps)); loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
    return model,keys

def sample_dev(rng,n):
    out=[]
    for i in range(n):
        a=DEV_COMBOS[i%len(DEV_COMBOS)]
        b=DEV_COMBOS[(i*7+3)%len(DEV_COMBOS)]
        if a==b: b=DEV_COMBOS[(i*7+4)%len(DEV_COMBOS)]
        out.append(sample_episode(rng,combo1=a,combo2=b))
    return out

def evaluate(model,eps):
    model.eval()
    total=xor=state_wrong=correct_bits=exec_error_given_correct=0
    xor_cases={"both_bits_correct":0,"one_or_more_bits_wrong":0,"output_wrong_and_bits_correct":0}
    for ep in eps:
        mem=model.state.initial(1,torch.device("cpu"))
        for entity,text,image,audio in ep[0]:
            z=model.encode(text.unsqueeze(0),image.unsqueeze(0),audio.unsqueeze(0).unsqueeze(0))
            mem,_=model.state.write(mem,z,torch.tensor([entity],dtype=torch.long))
        entity,i,j,op,target=ep[2]
        if op!="xor": continue
        out=model.query(mem,torch.tensor([entity]),torch.tensor([i]),torch.tensor([j]),torch.tensor([0]))
        pbits=(out["bits"]>=0.5).long().squeeze(0)
        true_i=ep[3][entity][i]; true_j=ep[3][entity][j]
        bits_ok_i=int(pbits[i].item()==true_i); bits_ok_j=int(pbits[j].item()==true_j)
        bits_ok=bits_ok_i and bits_ok_j
        pred=int(out["logits"].argmax(-1).item())
        xor += int(pred==target); total += 1
        correct_bits += int(bits_ok)
        state_wrong += int(not bits_ok)
        if bits_ok:
            xor_cases["both_bits_correct"] += 1
            if pred != target:
                xor_cases["output_wrong_and_bits_correct"] += 1
                exec_error_given_correct += 1
        else:
            xor_cases["one_or_more_bits_wrong"] += 1
    return {
      "xor_q2_accuracy":xor/max(total,1),
      "queried_bit_joint_accuracy":correct_bits/max(total,1),
      "xor_state_error_rate":state_wrong/max(total,1),
      "xor_execution_error_given_correct_bits":exec_error_given_correct/max(xor_cases["both_bits_correct"],1),
      "xor_cases":xor_cases,
      "xor_episode_count":total
    }

def run():
    c=json.loads(CONTRACT_PATH.read_text())
    assert c["status"]=="pre-registered" and c["seeds"]==list(SEEDS) and c["steps"]==STEPS and c["eval_steps"]==EVAL_EPISODES
    rows=[]
    for seed in SEEDS:
        model,train_keys=train_seed(seed)
        eps=sample_dev(random.Random(seed+100800),EVAL_EPISODES)
        overlap=train_keys & {episode_key(e) for e in eps}
        if overlap: raise AssertionError(f"train/dev overlap seed {seed}")
        m=evaluate(model,eps)
        rows.append({"seed":seed,"evaluation_fingerprint":episode_fingerprint(eps),"train_dev_overlap":len(overlap),**m})
    mean=lambda k:statistics.fmean(r[k] for r in rows)
    out={
      "experiment_id":EXPERIMENT_ID,"status":"development",
      "provenance":{"git_commit":os.environ.get("GITHUB_SHA","unknown"),"workflow_run_id":os.environ.get("GITHUB_RUN_ID","unknown"),
        "python_version":platform.python_version(),"torch_version":torch.__version__,
        "contract_sha256":hashlib.sha256(CONTRACT_PATH.read_bytes()).hexdigest()},
      "protocol":{"seeds":list(SEEDS),"steps":STEPS,"batch_size":BATCH_SIZE,"eval_episodes":EVAL_EPISODES,"hidden_dim":64,
        "dev_compositions":[list(x) for x in DEV_COMBOS],"sealed_e2e005_excluded":True,"e2e006_heldout_excluded":True},
      "results":rows,
      "summary":{
        "mean_xor_q2_accuracy":mean("xor_q2_accuracy"),
        "mean_queried_bit_joint_accuracy":mean("queried_bit_joint_accuracy"),
        "mean_xor_state_error_rate":mean("xor_state_error_rate"),
        "mean_xor_execution_error_given_correct_bits":mean("xor_execution_error_given_correct_bits"),
        "all_train_dev_overlaps_zero":all(r["train_dev_overlap"]==0 for r in rows)
      }
    }
    p=ROOT/"artifacts"/f"{EXPERIMENT_ID}.json"; p.parent.mkdir(exist_ok=True); p.write_text(json.dumps(out,indent=2,sort_keys=True)+"\n")
    print(json.dumps(out,indent=2,sort_keys=True))
if __name__=="__main__": run()
