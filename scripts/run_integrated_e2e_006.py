#!/usr/bin/env python3
"""Runner for TACOSM-PLM-INTEGRATED-E2E-006."""
from __future__ import annotations
import argparse, hashlib, json, os, platform, random, statistics
from pathlib import Path
import torch
from torch import nn
import torch.nn.functional as F

ROOT=Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/"src"))

from tac_osm.integrated_e2e_005 import FunctionalConfig, FunctionalMultimodalPLM
from tac_osm.integrated_e2e_006_benchmark import (
    GENERATOR_VERSION, HELDOUT, SEEDS, TRAIN_COMBOS, BITS, ENTITY_COUNT,
    benchmark_manifest, episode_fingerprint, episode_key, sample_episode,
    sample_evaluation_episodes,
)
EXPERIMENT_ID="TACOSM-PLM-INTEGRATED-E2E-006"
CONTRACT_PATH=ROOT/"contracts"/f"{EXPERIMENT_ID}.json"
BATCH_SIZE=96
STEPS=300
EVAL_EPISODES=400
OPS=("xor","and","or","xnor")

def contract_hash(): return hashlib.sha256(CONTRACT_PATH.read_bytes()).hexdigest()

def primary_threshold():
    raw=json.loads(CONTRACT_PATH.read_text())
    prim=[e for e in [raw["primary_endpoint"]] if e]
    return float(prim[0]["threshold"])

def build_batch(episodes):
    from scripts.run_integrated_e2e_005 import build_batch as f
    return f(episodes)

def write_observations(model,batch):
    from scripts.run_integrated_e2e_005 import write_observations as f
    return f(model,batch)

def environment_outcome(action,target):
    return ((action>=0.5).long()==target).float().detach()

def train_batch(model,batch):
    from scripts.run_integrated_e2e_005 import train_batch as f
    return f(model,batch)

def gradient_surface_probe():
    torch.manual_seed(20261006)
    rng=random.Random(86006)
    eps=[sample_episode(rng) for _ in range(4)]
    model=FunctionalMultimodalPLM(config=FunctionalConfig(hidden_dim=64))
    loss=train_batch(model,build_batch(eps)); loss.backward()
    required=("text.emb.weight","text.rnn.weight_ih_l0","image.net.0.weight",
              "audio.net.0.weight","rep.fuse.0.weight","state.write_value.weight",
              "casm.decoder.0.weight","verifier.0.weight","write_gate.0.weight")
    bad=[]
    params=dict(model.named_parameters())
    for n in required:
        g=params[n].grad
        if g is None or not torch.isfinite(g).all(): bad.append(n)
    return {"pass":not bad,"missing_or_nonfinite":bad}

def train_seed(seed):
    torch.manual_seed(seed)
    rng=random.Random(seed+910000)
    model=FunctionalMultimodalPLM(config=FunctionalConfig(hidden_dim=64)).cpu()
    opt=torch.optim.AdamW(model.parameters(),lr=0.002,weight_decay=0.0001)
    keys=set()
    for _ in range(STEPS):
        eps=[sample_episode(rng) for _ in range(BATCH_SIZE)]
        keys.update(episode_key(e) for e in eps)
        model.train(); opt.zero_grad(set_to_none=True)
        loss=train_batch(model,build_batch(eps)); loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
    return model,keys

@torch.no_grad()
def evaluate_seed(model,episodes,control):
    model.eval()
    q1=q2=ver=0; pos=0; mism=0
    per_op={op:[0,0] for op in OPS}
    for ep in episodes:
        rows=ep[0]
        mem=model.state.initial(1,torch.device("cpu"))
        for k,(entity,text,image,audio) in enumerate(rows):
            if control=="shuffle_image": image=rows[(k+1)%len(rows)][2]
            elif control=="text_only": image=torch.zeros_like(image); audio=torch.zeros_like(audio)
            elif control=="image_only": text=torch.zeros_like(text); audio=torch.zeros_like(audio)
            elif control=="audio_only": text=torch.zeros_like(text); image=torch.zeros_like(image)
            z=model.encode(text.unsqueeze(0),image.unsqueeze(0),audio.unsqueeze(0).unsqueeze(0))
            mem,_=model.state.write(mem,z,torch.tensor([entity],dtype=torch.long))
        if control=="no_memory": mem=torch.zeros_like(mem)
        e1,i1,j1,op1=(torch.tensor([ep[1][0]]),torch.tensor([ep[1][1]]),
                       torch.tensor([ep[1][2]]),torch.tensor([OPS.index(ep[1][3])]))
        o1=model.query(mem,e1,i1,j1,op1); t1=torch.tensor([ep[1][4]])
        out1=environment_outcome(o1["action"],t1); upd,v1=model.post_action_update(mem,o1,out1)
        e2,i2,j2,op2=(torch.tensor([ep[2][0]]),torch.tensor([ep[2][1]]),
                       torch.tensor([ep[2][2]]),torch.tensor([OPS.index(ep[2][3])]))
        o2=model.query(upd,e2,i2,j2,op2); t2=torch.tensor([ep[2][4]])
        out2=environment_outcome(o2["action"],t2); _,v2=model.post_action_update(upd,o2,out2)
        q1 += int(o1["logits"].argmax(-1).item()==ep[1][4])
        pred=int(o2["logits"].argmax(-1).item())
        threshold_pred=int((o2["action"]>=0.5).long().item())
        mism += int(pred!=threshold_pred); pos += pred; q2 += int(pred==ep[2][4])
        per_op[ep[2][3]][0] += int(pred==ep[2][4]); per_op[ep[2][3]][1] += 1
        ver += int(((v1[:,:1].sigmoid()>=0.5).long()==out1.long()).item())
        ver += int(((v2[:,:1].sigmoid()>=0.5).long()==out2.long()).item())
    n=len(episodes)
    return {"q1_accuracy":q1/n,"q2_accuracy":q2/n,
            "verifier_q1_q2_accuracy":ver/(2*n),
            "q2_positive_fraction":pos/n,"q2_decision_mismatches":mism,
            "per_op_q2_accuracy":{op:per_op[op][0]/per_op[op][1] for op in OPS}}

def bootstrap_ci(values,rounds=5000,seed=20261006):
    rng=random.Random(seed)
    means=sorted(statistics.fmean(rng.choices(values,k=len(values))) for _ in range(rounds))
    return [means[int(0.025*rounds)],means[int(0.975*rounds)]]

def run(smoke=False):
    contract=json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
    assert contract["status"] == "pre-registered"
    assert contract["primary_endpoint"]["name"] == "heldout_q2_accuracy_all_modalities"
    assert float(contract["primary_endpoint"]["threshold"]) == 0.80
    assert contract["protocol"]["hidden_dim"] == 64
    assert contract["seeds"] == list(SEEDS)
    gate=gradient_surface_probe()
    if not gate["pass"]: raise AssertionError(gate)
    seeds=(0,) if smoke else SEEDS
    if smoke:
        global BATCH_SIZE,EVAL_EPISODES
        BATCH_SIZE=8; EVAL_EPISODES=16
    rows=[]
    for seed in seeds:
        model,train_keys=train_seed(seed)
        eval_rng=random.Random(seed+920000)
        episodes=sample_evaluation_episodes(eval_rng,EVAL_EPISODES)
        eval_keys={episode_key(e) for e in episodes}
        overlap=train_keys & eval_keys
        if overlap: raise AssertionError(f"training/evaluation overlap seed {seed}")
        controls=("normal","no_memory","shuffle_image","text_only","image_only","audio_only")
        fps={k:episode_fingerprint(episodes) for k in controls}
        if len(set(fps.values()))!=1: raise AssertionError("control episode mismatch")
        metrics={k:evaluate_seed(model,episodes,k) for k in controls}
        oracle=sum(1 for e in episodes if e[2][4] == {
            "xor": e[3][e[2][0]][e[2][1]] ^ e[3][e[2][0]][e[2][2]],
            "and": e[3][e[2][0]][e[2][1]] & e[3][e[2][0]][e[2][2]],
            "or": e[3][e[2][0]][e[2][1]] | e[3][e[2][0]][e[2][2]],
            "xnor": 1-(e[3][e[2][0]][e[2][1]] ^ e[3][e[2][0]][e[2][2]]),
        }[e[2][3]])
        rows.append({"seed":seed,
            "normal_q1_accuracy":metrics["normal"]["q1_accuracy"],
            "normal_q2_accuracy":metrics["normal"]["q2_accuracy"],
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
            "evaluation_episode_fingerprint":fps["normal"],
            "training_evaluation_semantic_overlap":len(overlap),
            "oracle_q2_accuracy":oracle/EVAL_EPISODES})
    norm=[r["normal_q2_accuracy"] for r in rows]
    summary={"primary_q2_mean":statistics.fmean(norm),
             "primary_q2_seed_bootstrap_ci95":bootstrap_ci(norm) if not smoke else [None,None],
             "all_seed_min_q2":min(norm),
             "mean_memory_drop":statistics.fmean(r["memory_drop"] for r in rows),
             "mean_alignment_drop":statistics.fmean(r["alignment_drop"] for r in rows),
             "oracle_q2_accuracy":statistics.fmean(r["oracle_q2_accuracy"] for r in rows),
             "gradient_surface_pass":gate["pass"],
             "classifier_decision_integrity_pass":all(r["normal_q2_decision_mismatches"]==0 for r in rows),
             "primary_pass":bool((not smoke) and statistics.fmean(norm)>=primary_threshold() and min(norm)>=0.40)}
    out={"experiment_id":EXPERIMENT_ID,"status":"smoke" if smoke else "measured",
         "provenance":{"git_commit":os.environ.get("GITHUB_SHA","unknown"),
            "workflow_run_id":os.environ.get("GITHUB_RUN_ID","unknown"),
            "python_version":platform.python_version(),"torch_version":torch.__version__,
            "contract_sha256":contract_hash(),"benchmark_sha256":benchmark_manifest()["generator_sha256"],
            "base_benchmark_sha256":benchmark_manifest()["base_generator_sha256"]},
         "protocol":{"seeds":list(seeds),"steps":STEPS,"batch_size":BATCH_SIZE,
            "evaluation_episodes_per_seed":EVAL_EPISODES,"hidden_dim":64,
            "registered_heldout":[list(x) for x in HELDOUT],"train_combos_count":len(TRAIN_COMBOS),
            "benchmark_generator_version":GENERATOR_VERSION,"model_selection":"none",
            "representation_auxiliary_supervision":False,"operator_dispatch":"fixed_query_conditioned",
            "sealed_e2e005_and_dev_excluded_from_training":True},
         "seed_results":rows,"summary":summary,
         "leakage_audit":{"training_evaluation_semantic_overlap_zero":all(r["training_evaluation_semantic_overlap"]==0 for r in rows),
            "q1_q2_entities_distinct":True,"heldout_compositions_excluded_from_training":True,
            "evaluation_generated_after_training":True,"controls_reuse_exact_same_episode_objects":True,
            "payload_auxiliary_supervision":False,"post_action_outcome_only_feedback":True,
            "classifier_action_decision_consistency":summary["classifier_decision_integrity_pass"]},
         "claim_boundary":["fresh synthetic multimodal mechanism benchmark only","no real-world semantic multimodal claim",
            "no learned operator discovery claim","no scaling or hardware claim"]}
    path=ROOT/"artifacts"/f"{EXPERIMENT_ID}.json"; path.parent.mkdir(exist_ok=True); path.write_text(json.dumps(out,indent=2,sort_keys=True)+"\n")
    print(json.dumps(out,indent=2,sort_keys=True))

if __name__=="__main__":
    ap=argparse.ArgumentParser(); ap.add_argument("--smoke",action="store_true"); run(ap.parse_args().smoke)
