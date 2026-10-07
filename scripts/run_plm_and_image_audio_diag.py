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
from tac_osm.integrated_e2e_005_benchmark import ALL_COMBOS,episode_fingerprint,episode_key,sample_episode
from tac_osm.integrated_e2e_006_benchmark import HELDOUT as E2E006_HELDOUT
from tac_osm.integrated_e2e_007_benchmark import DEV_COMBOS,HELDOUT as E2E007_HELDOUT,SEALED_E2E005

ID="TACOSM-PLM-AND-IMAGE-AUDIO-DIAG-001"
CONTRACT=ROOT/"contracts"/f"{ID}.json"
SEEDS=(0,1,2,3,4);STEPS=300;BATCH=96;EVAL=200;OPS=("xor","and","or","xnor")
IA_HELDOUT=(("and",4,8),("and",4,9),("and",4,10),("and",5,8),("and",5,10),("and",5,11),
            ("and",6,8),("and",6,9),("and",6,10),("and",6,11),("and",7,8),("and",7,9),("and",7,10),("and",7,11))
EXCLUDED=set(SEALED_E2E005)|set(E2E006_HELDOUT)|set(E2E007_HELDOUT)|set(DEV_COMBOS)|set(IA_HELDOUT)
TRAIN_COMBOS=tuple(c for c in ALL_COMBOS if c not in EXCLUDED)
assert set(IA_HELDOUT).isdisjoint(EXCLUDED-set(IA_HELDOUT))
assert set(IA_HELDOUT).isdisjoint(set(TRAIN_COMBOS))

def build_batch(eps):
    from scripts.run_integrated_e2e_005 import build_batch
    return build_batch(eps)
def train_batch(model,batch):
    from scripts.run_integrated_e2e_005 import train_batch
    return train_batch(model,batch)

def train_seed(seed):
    torch.manual_seed(seed);rng=random.Random(seed+140700)
    m=FunctionalMultimodalPLM(config=FunctionalConfig(hidden_dim=64,state_write_mode="residual_linear")).cpu()
    opt=torch.optim.AdamW(m.parameters(),lr=.002,weight_decay=.0001);keys=set()
    for _ in range(STEPS):
        eps=[sample_episode(rng,combo1=rng.choice(TRAIN_COMBOS),combo2=rng.choice(TRAIN_COMBOS)) for _ in range(BATCH)]
        keys.update(episode_key(e) for e in eps);m.train();opt.zero_grad(set_to_none=True)
        loss=train_batch(m,build_batch(eps));loss.backward();nn.utils.clip_grad_norm_(m.parameters(),1.0);opt.step()
    return m,keys

def sample_eval(seed):
    rng=random.Random(seed+140800);out=[]
    for i in range(EVAL):
        q1=(DEV_COMBOS[(i*5)%len(DEV_COMBOS)])
        q2=IA_HELDOUT[i%len(IA_HELDOUT)]
        out.append(sample_episode(rng,combo1=q1,combo2=q2))
    return out

@torch.no_grad()
def evaluate(m,episodes):
    m.eval();total=correct=0;img_ok=aud_ok=joint_ok=state_err=exec_err=decision_mismatch=dispatch_mismatch=0
    img_n=aud_n=joint_n=correct_joint_cases=0;positive=0
    pair_stats={pair:[0,0,0,0] for pair in IA_HELDOUT}
    for ep in episodes:
        mem=m.state.initial(1,torch.device("cpu"))
        for entity,text,image,audio in ep[0]:
            z=m.encode(text.unsqueeze(0),image.unsqueeze(0),audio.unsqueeze(0).unsqueeze(0))
            mem,_=m.state.write(mem,z,torch.tensor([entity]))
        q=ep[2];entity,i,j,op,target=q;assert op=="and"
        out=m.query(mem,torch.tensor([entity]),torch.tensor([i]),torch.tensor([j]),torch.tensor([1]))
        bit=(out["bits"].squeeze(0)>=.5).long()
        ti,tj=ep[3][entity][i],ep[3][entity][j]
        io=int(bit[i].item()==ti);ao=int(bit[j].item()==tj);jo=io and ao
        pred=int(out["logits"].argmax(-1).item());thr=int((out["action"]>=.5).item())
        total+=1;correct+=int(pred==target);positive+=pred
        img_ok+=io;aud_ok+=ao;joint_ok+=int(jo);state_err+=int(not jo);decision_mismatch+=int(pred!=thr)
        img_n+=1;aud_n+=1;joint_n+=1
        if jo:
            correct_joint_cases+=1;exec_err+=int(pred!=target)
        expected_action=out["bits"].squeeze(0)[i]*out["bits"].squeeze(0)[j]
        dispatch_mismatch+=int(not torch.allclose(out["action"].squeeze(0),expected_action,atol=1e-7,rtol=1e-6))
        key=(op,i,j); pair_stats[key][0]+=int(pred==target);pair_stats[key][1]+=io;pair_stats[key][2]+=ao;pair_stats[key][3]+=int(jo)
    return {
      "image_audio_and_q2_accuracy":correct/total,
      "image_queried_bit_accuracy":img_ok/img_n,
      "audio_queried_bit_accuracy":aud_ok/aud_n,
      "joint_queried_bit_accuracy":joint_ok/joint_n,
      "state_error_rate":state_err/total,
      "execution_error_given_correct_bits":exec_err/max(correct_joint_cases,1),
      "positive_fraction":positive/total,
      "classifier_decision_mismatches":decision_mismatch,
      "and_dispatch_mismatches":dispatch_mismatch,
      "per_composition":{str(k):{
          "q2_accuracy":v[0]/200 if v[1]==200 else v[0]/max(1,v[1]),
          "image_bit_accuracy":v[1]/max(1,v[1]),
          "audio_bit_accuracy":v[2]/max(1,v[2]),
          "joint_bit_accuracy":v[3]/max(1,v[3])} for k,v in pair_stats.items()}
    }

def run():
    c=json.loads(CONTRACT.read_text())
    assert c["status"]=="pre-registered" and c["seeds"]==list(SEEDS) and c["steps"]==STEPS and c["eval_steps"]==EVAL
    rows=[]
    for seed in SEEDS:
        m,train_keys=train_seed(seed);eps=sample_eval(seed);eval_keys={episode_key(e) for e in eps}
        overlap=train_keys&eval_keys
        if overlap:raise AssertionError(f"training/evaluation overlap seed {seed}")
        rows.append({"seed":seed,"evaluation_fingerprint":episode_fingerprint(eps),"training_evaluation_overlap":len(overlap),**evaluate(m,eps)})
    mean=lambda k:statistics.fmean(r[k] for r in rows)
    out={"experiment_id":ID,"status":"development",
      "provenance":{"git_commit":os.environ.get("GITHUB_SHA","unknown"),"workflow_run_id":os.environ.get("GITHUB_RUN_ID","unknown"),
        "python_version":platform.python_version(),"torch_version":torch.__version__,"contract_sha256":hashlib.sha256(CONTRACT.read_bytes()).hexdigest()},
      "protocol":{"seeds":list(SEEDS),"steps":STEPS,"batch_size":BATCH,"evaluation_episodes_per_seed":EVAL,"hidden_dim":64,"state_write_mode":"residual_linear",
        "heldout_image_audio_and":[list(x) for x in IA_HELDOUT],"sealed_e2e005_excluded":True,"e2e006_excluded":True,"e2e007_excluded":True,"dev_excluded":True},
      "results":rows,"summary":{
        "mean_and_q2_accuracy":mean("image_audio_and_q2_accuracy"),
        "mean_image_queried_bit_accuracy":mean("image_queried_bit_accuracy"),
        "mean_audio_queried_bit_accuracy":mean("audio_queried_bit_accuracy"),
        "mean_joint_queried_bit_accuracy":mean("joint_queried_bit_accuracy"),
        "mean_state_error_rate":mean("state_error_rate"),
        "mean_execution_error_given_correct_bits":mean("execution_error_given_correct_bits"),
        "mean_positive_fraction":mean("positive_fraction"),
        "all_training_evaluation_overlap_zero":all(r["training_evaluation_overlap"]==0 for r in rows),
        "all_classifier_decision_mismatches_zero":all(r["classifier_decision_mismatches"]==0 for r in rows),
        "all_and_dispatch_mismatches_zero":all(r["and_dispatch_mismatches"]==0 for r in rows)
      },
      "claim_boundary":["development modality-specific diagnosis only","no confirmatory claim","no prior heldout outcomes used for tuning or selection"]}
    p=ROOT/"artifacts"/f"{ID}.json";p.parent.mkdir(exist_ok=True);p.write_text(json.dumps(out,indent=2,sort_keys=True)+"\n");print(json.dumps(out,indent=2,sort_keys=True))
if __name__=="__main__":run()
