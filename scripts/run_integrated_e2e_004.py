#!/usr/bin/env python3
"""Authoritative runner for corrected typed-content E2E-004."""
from __future__ import annotations
import argparse,json,os,platform,random,statistics,sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/"src"))
from tac_osm.contract import load_contract
from tac_osm.integrated_e2e_content import ContentDiagnosisConfig,TypedContentModel
from tac_osm.integrated_e2e_benchmark import BATCH_SIZE,EVAL_EPISODES,HELDOUT,SEEDS,STEPS,batchify,sample_episode,validate_episode
ARMS=("typed_learned","typed_oracle")
def validate_contract(arm):
    c=load_contract("TACOSM-PLM-INTEGRATED-E2E-004"); c.require_levels((3,)); c.require_seeds(SEEDS); c.require_steps(STEPS); c.require_eval_steps(EVAL_EPISODES)
    if arm not in {a.name for a in c.arms}: raise RuntimeError(f"unregistered arm: {arm}")
    if c.check_consistency(): raise RuntimeError("contract consistency failure")
def train_model(seed):
    torch.manual_seed(seed); rng=random.Random(seed+12001)
    model=TypedContentModel(ContentDiagnosisConfig(content_mode="learned")).cpu()
    opt=torch.optim.AdamW(model.parameters(),lr=0.002,weight_decay=1e-4)
    for _ in range(STEPS):
        eps=[sample_episode(rng) for _ in range(BATCH_SIZE)]; batch=batchify(eps); opt.zero_grad(set_to_none=True)
        out=model.forward_episode({"text":torch.stack(batch["text"],0),"image":torch.stack(batch["image"],0),"audio":torch.stack(batch["audio"],0)},torch.stack(batch["entities"],0),torch.stack(batch["payloads"],0),batch["q1"],batch["q2"])
        out["loss"].backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
    return model
@torch.no_grad()
def evaluate_learned(model,episodes,no_memory=False):
    q1=q2=bit_correct=bit_total=0
    for ep in episodes:
        validate_episode(ep); batch=batchify([ep]); memory=model.state.initial(1,torch.device("cpu"))
        for k in range(3):
            z,_,_,_=model.encode(batch["text"][k],batch["image"][k],batch["audio"][k]); bits=torch.sigmoid(model.rep.bit_head(z))
            bit_correct += int(((bits>=0.5).int()==batch["payloads"][k].int()).sum().item()); bit_total += bits.numel()
            memory,_=model.observation_write(memory,z,batch["entities"][k],batch["payloads"][k])
        if no_memory: memory.zero_()
        oq1=model.query(memory,*batch["q1"][:-1]); oq2=model.query(memory,*batch["q2"][:-1])
        q1 += int(oq1["logits"].argmax(-1).item()==batch["q1"][-1].item()); q2 += int(oq2["logits"].argmax(-1).item()==batch["q2"][-1].item())
    n=len(episodes); return {"q1_accuracy":q1/n,"q2_accuracy":q2/n,"bit_accuracy":bit_correct/bit_total}
@torch.no_grad()
def evaluate_oracle(episodes):
    q1=q2=0
    for ep in episodes:
        validate_episode(ep)
        for idx,q in enumerate((ep[1],ep[2])):
            entity,i,j,op,target=q; payload=ep[3][entity]; a,b=payload[i],payload[j]; y={"xor":a^b,"and":a&b,"or":a|b,"xnor":1-(a^b)}[op]
            if y!=target: raise AssertionError("oracle benchmark outcome mismatch")
            if idx==0:q1+=1
            else:q2+=1
    n=len(episodes); return {"q1_accuracy":q1/n,"q2_accuracy":q2/n}
def bootstrap(values,samples=5000,seed=20261002):
    rng=random.Random(seed); n=len(values); draws=sorted(statistics.fmean(values[rng.randrange(n)] for _ in range(n)) for _ in range(samples))
    return [draws[int(.025*samples)],draws[int(.975*samples)-1]]
def main():
    p=argparse.ArgumentParser(); p.add_argument("--arm",choices=ARMS,required=True); p.add_argument("--output",default=""); args=p.parse_args(); validate_contract(args.arm)
    streams={s:random.Random(200000+s) for s in SEEDS}; rows=[]
    if args.arm=="typed_oracle":
        for seed in SEEDS:
            rng=streams[seed]; eps=[]
            for _ in range(EVAL_EPISODES):
                c1,c2=rng.sample(HELDOUT,2); eps.append(sample_episode(rng,combo1=c1,combo2=c2))
            out=evaluate_oracle(eps); rows.append({"seed":seed,"typed_oracle_q1_accuracy":out["q1_accuracy"],"typed_oracle_q2_accuracy":out["q2_accuracy"]})
    else:
        models={s:train_model(s) for s in SEEDS}
        for seed in SEEDS:
            rng=streams[seed]; eps=[]
            for _ in range(EVAL_EPISODES):
                c1,c2=rng.sample(HELDOUT,2); eps.append(sample_episode(rng,combo1=c1,combo2=c2))
            normal=evaluate_learned(models[seed],eps); ctrl=evaluate_learned(models[seed],eps,no_memory=True)
            rows.append({"seed":seed,"typed_learned_q1_accuracy":normal["q1_accuracy"],"typed_learned_q2_accuracy":normal["q2_accuracy"],"typed_learned_bit_accuracy":normal["bit_accuracy"],"typed_learned_no_memory_q2_accuracy":ctrl["q2_accuracy"]})
    key=f"{args.arm}_q2_accuracy"; vals=[r[key] for r in rows]
    output={"experiment_id":"TACOSM-PLM-INTEGRATED-E2E-004","status":"measured","arm":args.arm,"provenance":{"git_commit":os.environ.get("GITHUB_SHA","unknown"),"workflow_run_id":os.environ.get("GITHUB_RUN_ID","unknown"),"python_version":platform.python_version(),"torch_version":torch.__version__,"platform":platform.platform()},"protocol":{"seeds":SEEDS,"steps":STEPS,"batch_size":BATCH_SIZE,"eval_episodes":EVAL_EPISODES,"heldout_compositions":HELDOUT},"seed_results":rows,"summary":{"q2_accuracy_mean":statistics.fmean(vals),"q2_accuracy_seed_bootstrap_ci95":bootstrap(vals),"min_seed_q2_accuracy":min(vals)}}
    out=Path(args.output or f"artifacts/TACOSM-PLM-INTEGRATED-E2E-004-{args.arm}.json"); out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n"); print(json.dumps(output, indent=2, sort_keys=True))
"); print(json.dumps(output,indent=2,sort_keys=True))
if __name__=="__main__": main()
