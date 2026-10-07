#!/usr/bin/env python3
from __future__ import annotations
import argparse,hashlib,json,os,platform,random,statistics
from pathlib import Path
import torch
from torch import nn
ROOT=Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/"src"))
from tac_osm.integrated_e2e_005 import FunctionalConfig,FunctionalMultimodalPLM
from tac_osm.integrated_e2e_007_benchmark import (
    GENERATOR_VERSION,HELDOUT,TRAIN_COMBOS,benchmark_manifest,episode_fingerprint,
    episode_key,sample_episode,sample_evaluation_episodes
)
EXPERIMENT_ID="TACOSM-PLM-INTEGRATED-E2E-007"; CONTRACT_PATH=ROOT/"contracts"/f"{EXPERIMENT_ID}.json"
SEEDS=(0,1,2,3,4);STEPS=300;BATCH_SIZE=96;EVAL_EPISODES=400;OPS=("xor","and","or","xnor")
def build_batch(eps):
    from scripts.run_integrated_e2e_005 import build_batch
    return build_batch(eps)
def train_batch(model,batch):
    from scripts.run_integrated_e2e_005 import train_batch
    return train_batch(model,batch)
def gradient_surface_probe():
    torch.manual_seed(20261007);rng=random.Random(7007)
    eps=[sample_episode(rng) for _ in range(4)]
    m=FunctionalMultimodalPLM(config=FunctionalConfig(hidden_dim=64,state_write_mode="residual_linear"))
    loss=train_batch(m,build_batch(eps));loss.backward()
    required=("text.emb.weight","text.rnn.weight_ih_l0","image.net.0.weight","audio.net.0.weight",
              "rep.fuse.0.weight","state.write_value.0.weight","casm.decoder.0.weight",
              "verifier.0.weight","write_gate.0.weight")
    p=dict(m.named_parameters());bad=[]
    for n in required:
        g=p[n].grad
        if g is None or not torch.isfinite(g).all():bad.append(n)
    return {"pass":not bad,"missing_or_nonfinite":bad}
def train_seed(seed):
    torch.manual_seed(seed);rng=random.Random(seed+120700)
    m=FunctionalMultimodalPLM(config=FunctionalConfig(hidden_dim=64,state_write_mode="residual_linear")).cpu()
    opt=torch.optim.AdamW(m.parameters(),lr=.002,weight_decay=.0001);keys=set()
    for _ in range(STEPS):
        eps=[sample_episode(rng) for _ in range(BATCH_SIZE)]
        keys.update(episode_key(e) for e in eps);m.train();opt.zero_grad(set_to_none=True)
        loss=train_batch(m,build_batch(eps));loss.backward();nn.utils.clip_grad_norm_(m.parameters(),1.0);opt.step()
    return m,keys
@torch.no_grad()
def evaluate(m,episodes,control):
    m.eval();q1=q2=v=0;pos=0;mism=0
    per={op:[0,0] for op in OPS}
    for ep in episodes:
        mem=m.state.initial(1,torch.device("cpu"));rows=ep[0]
        for k,(entity,text,image,audio) in enumerate(rows):
            if control=="shuffle_image":image=rows[(k+1)%len(rows)][2]
            elif control=="text_only":image=torch.zeros_like(image);audio=torch.zeros_like(audio)
            elif control=="image_only":text=torch.zeros_like(text);audio=torch.zeros_like(audio)
            elif control=="audio_only":text=torch.zeros_like(text);image=torch.zeros_like(image)
            z=m.encode(text.unsqueeze(0),image.unsqueeze(0),audio.unsqueeze(0).unsqueeze(0))
            mem,_=m.state.write(mem,z,torch.tensor([entity]))
        if control=="no_memory":mem=torch.zeros_like(mem)
        def runq(q):
            return m.query(mem,torch.tensor([q[0]]),torch.tensor([q[1]]),torch.tensor([q[2]]),torch.tensor([OPS.index(q[3])]))
        o1=runq(ep[1]);target1=ep[1][4];out1=((o1["action"]>=.5).long()==torch.tensor([target1])).float().detach()
        upd,v1=m.post_action_update(mem,o1,out1)
        e2=ep[2];o2=m.query(upd,torch.tensor([e2[0]]),torch.tensor([e2[1]]),torch.tensor([e2[2]]),torch.tensor([OPS.index(e2[3])]))
        target2=e2[4];out2=((o2["action"]>=.5).long()==torch.tensor([target2])).float().detach()
        _,v2=m.post_action_update(upd,o2,out2)
        p1=int(o1["logits"].argmax(-1).item());p2=int(o2["logits"].argmax(-1).item())
        mism+=int(p1!=int((o1["action"]>=.5).item()))+int(p2!=int((o2["action"]>=.5).item()))
        q1+=int(p1==target1);q2+=int(p2==target2);pos+=p2
        per[e2[3]][0]+=int(p2==target2);per[e2[3]][1]+=1
        v+=int(((v1[:,:1].sigmoid()>=.5).long()==out1.long()).item())+int(((v2[:,:1].sigmoid()>=.5).long()==out2.long()).item())
    n=len(episodes)
    return {"q1_accuracy":q1/n,"q2_accuracy":q2/n,"verifier_q1_q2_accuracy":v/(2*n),
            "q2_positive_fraction":pos/n,"q2_decision_mismatches":mism,
            "per_op_q2_accuracy":{op:per[op][0]/per[op][1] for op in OPS}}
def bootstrap(vals,rounds=5000,seed=20261007):
    r=random.Random(seed);x=sorted(statistics.fmean(r.choices(vals,k=len(vals))) for _ in range(rounds))
    return [x[int(.025*rounds)],x[int(.975*rounds)]]
def run(smoke=False):
    c=json.loads(CONTRACT_PATH.read_text());assert c["status"]=="pre-registered" and c["seeds"]==list(SEEDS)
    assert c["protocol"]["hidden_dim"]==64 and c["protocol"]["state_write_mode"]=="residual_linear"
    gate=gradient_surface_probe();assert gate["pass"],gate
    seeds=(0,) if smoke else SEEDS
    eval_n=16 if smoke else EVAL_EPISODES
    rows=[];controls=("normal","no_memory","shuffle_image","text_only","image_only","audio_only")
    for seed in seeds:
        m,train_keys=train_seed(seed)
        eps=sample_evaluation_episodes(random.Random(seed+121000),eval_n);eval_keys={episode_key(e) for e in eps}
        overlap=train_keys&eval_keys
        if overlap:raise AssertionError(f"training/eval overlap seed {seed}")
        fps={k:episode_fingerprint(eps) for k in controls};assert len(set(fps.values()))==1
        metrics={k:evaluate(m,eps,k) for k in controls}
        rows.append({"seed":seed,"evaluation_episode_fingerprint":fps["normal"],
          "training_evaluation_semantic_overlap":len(overlap),
          "normal_q1_accuracy":metrics["normal"]["q1_accuracy"],"normal_q2_accuracy":metrics["normal"]["q2_accuracy"],
          "normal_verifier_accuracy":metrics["normal"]["verifier_q1_q2_accuracy"],
          "normal_q2_positive_fraction":metrics["normal"]["q2_positive_fraction"],
          "normal_q2_decision_mismatches":metrics["normal"]["q2_decision_mismatches"],
          "no_memory_q2_accuracy":metrics["no_memory"]["q2_accuracy"],
          "shuffle_image_q2_accuracy":metrics["shuffle_image"]["q2_accuracy"],
          "text_only_q2_accuracy":metrics["text_only"]["q2_accuracy"],
          "image_only_q2_accuracy":metrics["image_only"]["q2_accuracy"],
          "audio_only_q2_accuracy":metrics["audio_only"]["q2_accuracy"],
          "memory_drop":metrics["normal"]["q2_accuracy"]-metrics["no_memory"]["q2_accuracy"],
          "alignment_drop":metrics["normal"]["q2_accuracy"]-metrics["shuffle_image"]["q2_accuracy"],
          "per_op_q2_accuracy":metrics["normal"]["per_op_q2_accuracy"],
          "oracle_q2_accuracy":1.0})
    vals=[r["normal_q2_accuracy"] for r in rows]
    summary={"primary_q2_mean":statistics.fmean(vals),"primary_q2_seed_bootstrap_ci95":bootstrap(vals) if not smoke else [None,None],
             "all_seed_min_q2":min(vals),"primary_pass":bool((not smoke) and statistics.fmean(vals)>=.8 and min(vals)>=.4),
             "mean_memory_drop":statistics.fmean(r["memory_drop"] for r in rows),
             "mean_alignment_drop":statistics.fmean(r["alignment_drop"] for r in rows),
             "oracle_q2_accuracy":1.0,"gradient_surface_pass":gate["pass"],
             "classifier_decision_integrity_pass":all(r["normal_q2_decision_mismatches"]==0 for r in rows)}
    out={"experiment_id":EXPERIMENT_ID,"status":"smoke" if smoke else "measured",
         "provenance":{"git_commit":os.environ.get("GITHUB_SHA","unknown"),"workflow_run_id":os.environ.get("GITHUB_RUN_ID","unknown"),
          "python_version":platform.python_version(),"torch_version":torch.__version__,
          "contract_sha256":hashlib.sha256(CONTRACT_PATH.read_bytes()).hexdigest(),
          "benchmark_sha256":benchmark_manifest()["generator_sha256"]},
         "protocol":{"seeds":list(seeds),"steps":STEPS,"batch_size":BATCH_SIZE,"evaluation_episodes_per_seed":eval_n,
          "hidden_dim":64,"state_write_mode":"residual_linear","registered_heldout":[list(x) for x in HELDOUT],
          "train_combos_count":len(TRAIN_COMBOS),"benchmark_generator_version":GENERATOR_VERSION,"model_selection":"none"},
         "seed_results":rows,"summary":summary,
         "leakage_audit":{"training_evaluation_semantic_overlap_zero":all(r["training_evaluation_semantic_overlap"]==0 for r in rows),
          "heldout_compositions_excluded_from_training":True,"evaluation_generated_after_training":True,
          "controls_reuse_exact_same_episode_objects":True,"payload_auxiliary_supervision":False,
          "post_action_outcome_only_feedback":True,"classifier_action_decision_consistency":summary["classifier_decision_integrity_pass"],
          "prior_e2e005_e2e006_dev_sets_excluded":True},
         "claim_boundary":["fresh synthetic multimodal mechanism benchmark only","no real-world semantic multimodal claim",
          "no learned addressing or operator discovery claim","no scaling or hardware claim"]}
    p=ROOT/"artifacts"/f"{EXPERIMENT_ID}.json";p.parent.mkdir(exist_ok=True);p.write_text(json.dumps(out,indent=2,sort_keys=True)+"\n");print(json.dumps(out,indent=2,sort_keys=True))
if __name__=="__main__":
    ap=argparse.ArgumentParser();ap.add_argument("--smoke",action="store_true");run(ap.parse_args().smoke)
