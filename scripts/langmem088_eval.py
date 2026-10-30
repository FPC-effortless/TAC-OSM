#!/usr/bin/env python3
"""LANGMEM088 first-new-doc publisher evaluation of original +0 and +1 slow biases.

Only real source bytes [0..t] enter model. The trained +1 arm is frozen
before any new DEV file download; existing original +0 exact 087-640
checkpoint is immutable. Read attribution holds the current write and
original full next-state, not long-term causal deletion.
"""
from __future__ import annotations
import argparse,hashlib,json,sys,time
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/"src"),str(ROOT/"research/mathematical-engineering/langmem086"),str(ROOT/"research/mathematical-engineering/langmem087")]
from scripts import langmem086_run as original
from diagnostic_tac087 import DiagnosticTAC087
STUDY="LANGMEM088"
REG=ROOT/"research/mathematical-engineering/langmem088"
SEEDS=(1701,1702,1703)
START=256
STOP=2048

def frozen_models(seed,baseline,biased):
    evidence={}
    for name,path in (("baseline",baseline),("slow_bias_plus1",biased)):
        fpath=path/"FIT_RECEIPT.json"
        ckpath=path/"MODEL.pt"
        if not fpath.is_file() or not ckpath.is_file():raise ValueError("missing original model source checkpoint")
        r=json.loads(fpath.read_text())
        raw=ckpath.read_bytes()
        sha=hashlib.sha256(raw).hexdigest()
        if sha!=r["checkpoint_sha256"]:raise ValueError("original actual source checkpoint SHA")
        cp=torch.load(ckpath,map_location="cpu",weights_only=True)
        if cp["seed"]!=seed or cp["steps"]!=640 or r["trained_steps"]!=640 or cp["model_source_sha256"]!=original.model_source_hash():
            raise ValueError("not same 086 original graph or 640 step seed")
        if r["fit_source_sha256"]!="911981cc63fd5436e0495cb067784e53475971131c811b7026e07d97a886d2d4":
            raise ValueError("original FIT owners differ")
        if name=="baseline":
            if cp["study"]!="LANGMEM087" or r["study"]!="LANGMEM087":
                raise ValueError("not original 087 640 baseline")
        else:
            if cp["study"]!=STUDY or r["study"]!=STUDY or r["bias_delta_initial"]!=1.:
                raise ValueError("not original new bias-only fit")
            if not r["checkpoint_and_optimizer_cold_replay_bitwise"] or r["unseen_DEV_accessed_before_fit_seal"]:
                raise ValueError("bias trained checkpoint not sealed before unseen")
        m=DiagnosticTAC087().eval()
        m.load_state_dict(cp["model"],strict=True)
        if m.counts()["trainable"]!=100521 or m.start(("fixture",)).tensor_bytes()!=1920:
            raise ValueError("registered source model count/memory changed")
        if original.weight_hash(m)!=r["trained_model_hash" if name=="baseline" else "final_trained_model_sha256"]:
            raise ValueError("model weights provenance mismatch")
        fit_plan=json.loads((ROOT/"research/mathematical-engineering/langmem087/FIT_PLAN.json").read_text())
        expected=next(z for z in fit_plan["by_seed"] if z["seed"]==seed)["checkpoints"][1]
        if cp["sample_receipt"]!=expected:raise ValueError("no exactly equal original FIT sample byte/order schedule")
        if name=="slow_bias_plus1":
            for arm,step in cp["optimizer"]["state"].items():
                if int(step["step"])!=640:raise ValueError("biased original optimizer step mismatch")
        evidence[name]={"model":m,"checkpoint_sha256":sha,
            "model_sha256":original.weight_hash(m),
            "final_slow_bias":float(m.read_mix.bias[1]),
            "frozen_original_FIT_source_sha256":r["fit_source_sha256"]}
    return evidence

def load_new_docs(folder):
    reg=json.loads((REG/"NEW_DEV_SOURCE.json").read_text())
    if reg["publisher_commit"]!="2771df165f092a4fbb9f46b5b2e7450f19010adb":
        raise ValueError("unregistered Rust publisher Git commit")
    names={x["path"] for x in reg["documents"]}
    if names!={"src/ch05-01-defining-structs.md","src/ch08-01-vectors.md"}:
        raise ValueError("unregistered Rust path or changed holdout")
    data={}
    for item in reg["documents"]:
        path=folder/Path(item["path"]).name
        if not path.is_file():raise ValueError("registered source missing")
        raw=path.read_bytes()
        if len(raw)<item["must_have_at_least_bytes"]:raise ValueError("new source too short")
        git_blob=hashlib.sha1(b"blob "+str(len(raw)).encode()+b"\0"+raw).hexdigest()
        sha=hashlib.sha256(raw).hexdigest()
        data[item["path"]]={"raw":raw,"sha256":sha,"git_blob":git_blob,"size":len(raw)}
    if len(list(folder.glob("*.md")))!=2:raise ValueError("extra Rust source outside registry")
    return data

@torch.no_grad()
def eval_once(seed,baseline,biased,documents,out):
    if seed not in SEEDS or out.exists():raise ValueError("unregistered source seed or output overwrite")
    torch.set_num_threads(2)
    models=frozen_models(seed,baseline,biased)
    # Pinned original publisher source is opened only AFTER both source-model
    # checkpoint FIT seals (all three bias seeds required by GitHub needs).
    docs=load_new_docs(documents)
    rows=[]
    target_sha=hashlib.sha256()
    start=time.process_time()
    prefix_max=0.
    for owner,record in sorted(docs.items()):
        raw=record["raw"]
        if len(raw)<STOP+1:raise ValueError("actual original new owner insufficient clock")
        states={name:x["model"].start((owner,)) for name,x in models.items()}
        for t in range(STOP):
            x=torch.tensor([[raw[t]]],dtype=torch.long)
            outcomes={}
            for name,obj in models.items():
                model=obj["model"]
                prior_state=states[name]
                logits,nxt,trace=model(x,state=prior_state,owners=(owner,),return_trace=True)
                if t==127:
                    bulk,_=model(torch.tensor([list(raw[:128])]),owners=(owner,))
                    err=float((bulk[:,-1]-logits[:,0]).abs().max())
                    prefix_max=max(prefix_max,err)
                    if err>3e-5:raise ValueError("real original source 128 prefix graph mismatch")
                if nxt.clock!=t+1 or nxt.tensor_bytes()!=1920:raise ValueError("original model state fixed-size broken")
                if t>=START:
                    alt,_=model(x,state=prior_state,owners=(owner,),lesion="old_slow_zero")
                    outcomes[name]={"full_logits":logits[0,0],"old_zero_logits":alt[0,0],
                        "slow_share":float(trace[0]["mix"][0,1]),
                        "old_read_L2":float(torch.linalg.vector_norm(trace[0]["old_slow_component"][0]@trace[0]["query"][0])),
                        "current_write_read_L2":float(torch.linalg.vector_norm(trace[0]["current_slow_component"][0]@trace[0]["query"][0]))}
                states[name]=nxt
            if t<START:continue
            # True next token is used ONLY after baseline/biased original
            # learned predictions and old-content interventions are chosen.
            y=int(raw[t+1])
            r={"owner":owner,"clock":t,"actual_next_byte":y}
            for name,outcome in outcomes.items():
                lp=outcome["full_logits"].log_softmax(0)
                op=outcome["old_zero_logits"].log_softmax(0)
                r[name+"_nll"]=-float(lp[y])
                r[name+"_accuracy"]=int(int(lp.argmax())==y)
                r[name+"_old_slow_zero_minus_full_nll"]=-float(op[y])+float(lp[y])
                r[name+"_slow_mix_share"]=outcome["slow_share"]
                r[name+"_old_slow_read_L2"]=outcome["old_read_L2"]
                r[name+"_current_slow_write_read_L2"]=outcome["current_write_read_L2"]
            rows.append(r)
            target_sha.update(owner.encode()+b":"+str(t).encode()+b":"+bytes([y])+b";")
    if len(rows)!=2*(STOP-START):raise ValueError("real source owner/clock/byte grid changed")
    if any(original.weight_hash(o["model"])!=o["model_sha256"] for o in models.values()):
        raise ValueError("model weights altered by unseen source evaluation")
    metrics={}
    for name in models:
        metrics[name]={}
        for key,cells in (("overall",rows),
                          ("early",[r for r in rows if r["clock"]<512]),
                          ("middle",[r for r in rows if 512<=r["clock"]<1024]),
                          ("late",[r for r in rows if r["clock"]>=1024])):
            n=len(cells)
            def avg(f):return sum(float(r[name+"_"+f]) for r in cells)/n
            metrics[name][key]={"n":n,
                "nll":avg("nll"),"accuracy":avg("accuracy"),
                "old_slow_zero_minus_full_nll":avg("old_slow_zero_minus_full_nll"),
                "slow_mix_share":avg("slow_mix_share"),
                "old_slow_read_L2":avg("old_slow_read_L2"),
                "current_slow_write_read_L2":avg("current_slow_write_read_L2")}
        metrics[name]["per_owner"]={owner:{"n":len(cells),"nll":sum(r[name+"_nll"] for r in cells)/len(cells),
                "old_slow_zero_minus_full_nll":sum(r[name+"_old_slow_zero_minus_full_nll"] for r in cells)/len(cells)}
               for owner in sorted(docs)
               for cells in [[r for r in rows if r["owner"]==owner]]}
    result={"study":STUDY,"seed":seed,"status":"FIRST_UNSEEN_RUST_PAIRED_MIX_BIAS_MATCHED640",
        "intervention":"read_mix.bias[1] +1 ONLY AT_INITIALIZATION",
        "registered_seed_count":3,"trained_640_steps_per_arm":True,
        "bias_arm_extra_parameters":0,"all_arms_trainable_parameters":100521,
        "model_state_tensor_bytes":1920,
        "source_new_docs":{owner:{k:record[k] for k in ("sha256","git_blob","size")} for owner,record in docs.items()},
        "original_Rust_git_commit":"2771df165f092a4fbb9f46b5b2e7450f19010adb",
        "source_nextbyte_target_sha256":target_sha.hexdigest(),"original_nextbyte_targets_per_seed":len(rows),
        "original_prefix_max_abs_error":prefix_max,
        "both_models_frozen_and_sha_verified_before_new_DEV":True,
        "original_baseline_model_checkpoint_sha256":models["baseline"]["checkpoint_sha256"],
        "bias_plus1_model_checkpoint_sha256":models["slow_bias_plus1"]["checkpoint_sha256"],
        "baseline_final_read_bias":models["baseline"]["final_slow_bias"],
        "biased_final_read_bias":models["slow_bias_plus1"]["final_slow_bias"],
        "metrics":metrics,"rows":rows,
        "cpu_eval_seconds":time.process_time()-start,
        "protected_TEST":"NOT_ACCESSED","gpu_minutes":0,"independently_qualified":False,
        "limitations":["Same Rust publisher used in prior studies but two distinct previously-unused chapters",
          "old_slow_zero changes only instantaneous read with original current write, not long-term factual recall",
          "All source nextbyte targets repeated across three model seed predictions",
          "Reference unmodified 087 checkpoints originally saw OTHER Rust documents, not these holdout documents",
          "Bias+1 is trained against exact original FIT schedule; no extra param, no model family superiority",
          "Unseen Rust documents fixed by Git commit and path; raw SHA measured only after fitting"],
    }
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(result,indent=2,sort_keys=True,allow_nan=False)+"\n")
    print("LANGMEM088_PAIRED_NEW_DEV "+json.dumps({"seed":seed,"target_sha":result["source_nextbyte_target_sha256"],
       "baseline":metrics["baseline"]["overall"],"plus1":metrics["slow_bias_plus1"]["overall"]},sort_keys=True),flush=True)

def main():
    p=argparse.ArgumentParser();p.add_argument("--seed",required=True,type=int,choices=SEEDS)
    p.add_argument("--baseline-dir",required=True,type=Path)
    p.add_argument("--biased-dir",required=True,type=Path)
    p.add_argument("--documents",required=True,type=Path)
    p.add_argument("--output",required=True,type=Path)
    a=p.parse_args();eval_once(a.seed,a.baseline_dir,a.biased_dir,a.documents,a.output)
if __name__=="__main__":main()
