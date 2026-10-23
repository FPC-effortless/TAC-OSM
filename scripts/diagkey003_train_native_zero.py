"""DIAGKEY003 owner-requested original 100k native zero-distractor recall fit.

Synthetic mechanism TRAINING, never evidence of real world or architecture win.
Original five guarded model/optimizer/RNG checkpoint bytes are Git-owned.
Train only 0 distractors; 1..16 evaluations are explicitly OOD probes.
"""
from __future__ import annotations
import argparse,hashlib,json,random,subprocess,time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from scripts.diagkey001_associative_recall import model_for,model_counts
from scripts.diagkey002_no_fit_mechanism import original_key_query,wilson
from tac_osm.parameter_budget import budgeted_optimizer,checked_optimizer_step,save_training_checkpoint

STUDY="DIAGKEY003_NATIVE_100K_ZERO_DISTRACTOR_MECHANISM_ONLY"
SEEDS=(75001,75002,75003,75004,75005)
STEPS=640
BATCH=16
NVAL=32
CHANCE=1./NVAL
OUT=Path("research/replay-bundles/DIAGKEY003")
REGISTRATION=Path("research/mathematical-engineering/diagkey003/FAIR_COMPARISON.json")
CURVE=(1,16,32,64,128,256,512,640)
PROBE_DELAYS=(0,1,2,4,8,16)

@torch.no_grad()
def evaluate(model,seed,delay,*,cases,tag):
    model.eval()
    correct=0;total=0;nll=0.
    for i in range(cases//32):
        t,k,v,y=original_key_query(seed+30000,i+700000,delay,batch_size=32,split=tag)
        scores=model(t,k,v)
        correct+=int((scores.argmax(-1)==y).sum())
        nll+=float(F.cross_entropy(scores,y,reduction="sum"))
        total+=len(y)
    assert total==cases
    return {"cases":total,"correct":correct,"accuracy":correct/total,
            "wilson95":wilson(correct,total),"NLL":nll/total}

def counts_guard():
    counts=model_counts()
    if counts["tac"]!=101554:
        raise ValueError("original native 101554 active/deployed parameter model changed")
    return counts["tac"]

def fit(seed):
    if seed not in SEEDS:raise ValueError("undeclared seed")
    torch.set_num_threads(1)
    count=counts_guard()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    model=model_for("tac")
    folder=OUT/f"seed-{seed}"
    folder.mkdir(parents=True,exist_ok=True)
    target=folder/"tac.pt"
    scores_path=folder/"tac.json"
    receipt_path=folder/"tac-live-receipt.json"
    if any(p.exists() for p in (target,scores_path,receipt_path)):
        raise FileExistsError("original guarded zero-distractor model/receipt already exists")
    optimizer=budgeted_optimizer(model,torch.optim.AdamW,model.parameters(),
                                lr=.001,weight_decay=.0001,
                                budget_manifest=REGISTRATION,
                                budget_receipt=receipt_path,
                                budget_seed=seed,budget_architecture="tac")
    learning_curve=[]
    trained_examples=0
    start=time.perf_counter()
    for step in range(1,STEPS+1):
        kind,key,value,y=original_key_query(seed,step-1,0,batch_size=BATCH,split="FIT_ZERO")
        model.train()
        logits=model(kind,key,value)
        loss=F.cross_entropy(logits,y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradnorm=float(torch.nn.utils.clip_grad_norm_(model.parameters(),1.))
        checked_optimizer_step(optimizer)
        trained_examples+=len(y)
        if step in CURVE:
            learning_curve.append({
                "completed_guarded_updates":step,
                "original_fit_batch_NLL":float(loss.detach()),
                "original_fit_batch_accuracy":float((logits.argmax(-1)==y).float().mean()),
                "gradient_norm_before_clip":gradnorm,
                "fixed_original_DEV_128":evaluate(model,seed,0,cases=128,tag="DEV_ZERO")
            })
        if step in (1,128,256,512,640):
            print("DIAGKEY003_NATIVE_ORIGINAL_TRAINING="+json.dumps({
                "seed":seed,"step":step,"fit_NLL":float(loss.detach()),
                "guarded":True}),flush=True)
    result={
        "study":STUDY,"mechanism_only":True,"real_world_data":False,
        "arm":"tac","seed":seed,"trainable_parameters":count,
        "deployed_parameters":count,"optimizer_updates":STEPS,
        "examples_seen":trained_examples,
        "trained_delay":0,"trained_on_multiple_lengths":False,
        "training_budget":{"batch":BATCH,"AdamW_lr":.001,"weight_decay":.0001,
                           "gradient_clip_norm":1.},
        "original_learning_curve":learning_curve,
        "final_original_sealed_TEST_512":evaluate(model,seed,0,cases=512,tag="TEST_ZERO"),
        "after_fit_out_of_distribution_ladder":{
            str(d):evaluate(model,seed,d,cases=512,tag="TEST_LADDER_OUT_OF_FIT" ) for d in PROBE_DELAYS[1:]},
        "chance":CHANCE,"training_wall_seconds":time.perf_counter()-start,
        "interpretation":["No fair head-to-head model comparison trained here",
                          "Actual guarded PyTorch parameter and checkpoint receipts are required",
                          "No real-world model capability established",
                          "Original ME074 real forecast cannot be qualified by synthetic scores"]
    }
    save_training_checkpoint({
        "original_synthetic_mechanism_only":True,
        "weights":model.state_dict(),
        "optimizer":optimizer.state_dict(),
        "torch_rng_state":torch.get_rng_state(),
        "numpy_rng_state":np.random.get_state(),
        "spec":result
    },target)
    receipt=json.loads(receipt_path.read_text())
    if receipt["status"]!="CHECKPOINT_VERIFIED" or receipt["successful_optimizer_steps"]!=STEPS:
        raise ValueError("mandatory original live optimizer prefit-to-checkpoint proof FAILED")
    result["original_model_SHA256"]=hashlib.sha256(target.read_bytes()).hexdigest()
    result["original_runtime_receipt_SHA256"]=hashlib.sha256(receipt_path.read_bytes()).hexdigest()
    scores_path.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
    print("DIAGKEY003_ACTUAL_SYNTHETIC_FIT_ONLY="+json.dumps({
        "seed":seed,"original_TEST_ZERO":result["final_original_sealed_TEST_512"],
        "original_model_SHA256":result["original_model_SHA256"],
        "training_wall_seconds":result["training_wall_seconds"]}),flush=True)

@torch.no_grad()
def replay():
    torch.set_num_threads(1)
    counts_guard()
    evidence={}
    for seed in SEEDS:
        folder=OUT/f"seed-{seed}"
        checkpoint=folder/"tac.pt"
        scorepath=folder/"tac.json"
        receiptpath=folder/"tac-live-receipt.json"
        if not all(p.is_file() for p in (checkpoint,scorepath,receiptpath)):
            raise ValueError("all five original guarded model/receipt/score sets required")
        git_bytes=subprocess.check_output(["git","show","HEAD:"+checkpoint.as_posix()])
        if git_bytes!=checkpoint.read_bytes():
            raise ValueError("checkpoint bytes NOT original Git-owned")
        row=json.loads(scorepath.read_text())
        if row["original_model_SHA256"]!=hashlib.sha256(checkpoint.read_bytes()).hexdigest():
            raise ValueError("model hash changed")
        if row["original_runtime_receipt_SHA256"]!=hashlib.sha256(receiptpath.read_bytes()).hexdigest():
            raise ValueError("runtime receipt hash changed")
        receipt=json.loads(receiptpath.read_text())
        if receipt["status"]!="CHECKPOINT_VERIFIED" or receipt["successful_optimizer_steps"]!=STEPS:
            raise ValueError("runtime budget first-update proof missing")
        saved=torch.load(checkpoint,map_location="cpu",weights_only=False)
        model=model_for("tac")
        model.load_state_dict(saved["weights"],strict=True)
        final=evaluate(model,seed,0,cases=512,tag="TEST_ZERO")
        original=row["final_original_sealed_TEST_512"]
        if final["correct"]!=original["correct"] or abs(final["NLL"]-original["NLL"])>1e-5:
            raise ValueError("original fresh TEST mismatch")
        evidence[str(seed)]={"accuracy":final["accuracy"],"correct":final["correct"],
                             "cases":final["cases"],"NLL":final["NLL"],
                             "original_checkpoint_SHA256":row["original_model_SHA256"]}
    vals=[v["accuracy"] for v in evidence.values()]
    report={"study":STUDY,"status":"QUALIFIED_SYNTHETIC_SOFTWARE_DIAGNOSTIC_ONLY",
            "original_fits_replayed":len(SEEDS),
            "new_model_fits_in_replay":0,"retrained_real_world_model":False,
            "trainable_parameters_per_model":counts_guard(),
            "original_live_guarded_training_and_checkpoint_bytes_verified":True,
            "chance":CHANCE,
            "original_zero_distractor_exact_recall_by_seed":evidence,
            "mean_original_five_seed_accuracy":float(np.mean(vals)),
            "all_five_original_seed_90_percent_gate_pass":all(v>=.9 for v in vals),
            "real_architectural_superiority_claim":False,
            "long_memory_256_recall_claim":False,
            "next_gate":"If 0-delay learned ≥90% for each of five seeds, preregister new 1/2/4/8/16 curriculum. Otherwise stop and diagnose optimization/key binding before another long-delay fit."}
    dest=OUT/"INDEPENDENT_SYNTHETIC_ORIGINAL_ZERO_REPLAY.json"
    if dest.exists():raise FileExistsError("do not rewrite original replay evidence")
    dest.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    print("DIAGKEY003_ACTUAL_ORIGINAL_FIVE_GUARDED_ZERO_RESULTS="+json.dumps(report),flush=True)

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--seed",type=int,default=75001)
    p.add_argument("--mode",choices=("smoke","fit","replay"),default="smoke")
    args=p.parse_args()
    if args.mode=="replay":return replay()
    if args.mode=="smoke":
        print("NATIVE_ZERO_COUNT="+str(counts_guard()))
        t,k,v,y=original_key_query(75001,0,0,batch_size=4,split="SMOKE")
        m=model_for("tac")
        loss=F.cross_entropy(m(t,k,v),y)
        loss.backward()
        if any(z.grad is None or not bool(torch.isfinite(z.grad).all())
               for z in m.parameters() if z.requires_grad):
            raise ValueError("bad synthetic zero-memory loss/gradient path")
        print("DIAGKEY003_ZERO_SMOKE_PASS_NO_FIT")
        return
    fit(args.seed)
if __name__=="__main__":main()
