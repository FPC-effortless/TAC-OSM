#!/usr/bin/env python3
"""CODEVALUE002 read-only seal/replay + first-open, real-source developer-code prediction.

Never trains. Evaluate real held-out publisher code only AFTER all original model,
optimizer, RNG and FIT-schedule checkpoints are source controlled and validated.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import statistics
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import torch
from torch.nn import functional as F
CODE = Path(__file__).resolve().parent
ROOT = CODE.parents[2]
sys.path.insert(0,str(CODE))
from train import REG, PARAMS, ARMS, SEEDS, STEPS, WINDOW, make, fit_sources, fetch_files, gitblob, sha256, schedule_hash, tensor_hash, preflight

ARCHIVE = ROOT / "research/replay-bundles/CODEVALUE002"
SEAL = ARCHIVE / "SEAL.json"
SCORES = CODE / "RESULTS.json"

def verify_all(archive):
    preflight()
    expected = {(a,s) for a in ARMS for s in SEEDS}
    found = set()
    receipts = {}
    for arm,seed in sorted(expected):
        folder = archive / f"{arm}-{seed}"
        cpfile = folder / "MODEL.pt"
        recfile = folder / "RECEIPT.json"
        if not cpfile.is_file() or not recfile.is_file():
            raise FileNotFoundError("missing original model/receipt " + folder.name)
        rec = json.loads(recfile.read_text())
        if (rec["arm"],rec["seed"],rec["study"],rec["steps"],rec["params"]) != (arm,seed,"CODEVALUE002",STEPS,PARAMS[arm]):
            raise ValueError("wrong original train identity")
        cpdata = cpfile.read_bytes()
        if rec["checkpoint_sha256"] != sha256(cpdata):raise ValueError("original checkpoint digest mismatch")
        cp = torch.load(cpfile,map_location="cpu",weights_only=True)
        if (cp["arm"],cp["seed"],cp["study"],cp["steps"],cp["params"])!=(arm,seed,"CODEVALUE002",STEPS,PARAMS[arm]):
            raise ValueError("original metadata mismatch")
        if "optimizer" not in cp or not cp["optimizer"]["state"] or "torch_rng" not in cp or "python_rng" not in cp or "numpy_rng" not in cp:
            raise ValueError("missing real training optimizer/RNG")
        model = make(arm);model.load_state_dict(cp["model"],strict=True)
        if tensor_hash(model.state_dict()) != rec["model_state_sha256"]:raise ValueError("original weights digest mismatch")
        if cp["sample"] != rec["sample"]:raise ValueError("original sample source mismatch")
        found.add((arm,seed));receipts[(arm,seed)] = rec
    if found!=expected:raise ValueError("missing model-seed")
    for seed in SEEDS:
        paired = [receipts[(arm,seed)]["sample"] for arm in ARMS]
        if any(x != paired[0] for x in paired[1:]): raise ValueError("training sample mismatch within seed")
    return receipts

def seal(archive, output):
    if output.exists():raise FileExistsError("do not overwrite archive seal")
    receipts=verify_all(archive)
    info={
        "study":"CODEVALUE002","sealed_before_DEV_source_read":True,
        "original_models":{f"{a}-{s}":{"checkpoint_sha256":r["checkpoint_sha256"],
                  "model_state_sha256":r["model_state_sha256"],"params":r["params"],
                  "fit_source_offsets_sha256":r["sample"]["owner_offsets_sha256"]}
                  for (a,s),r in sorted(receipts.items())},
        "model_count":len(receipts),"matched_schedules":True,
        "training_FLOPs":"UNMATCHED","protected_TEST":"NOT_OPENED","gpu_minutes":0
    }
    output.write_text(json.dumps(info,sort_keys=True,indent=2)+"\n")
    print(json.dumps({"seal":"PASS","models":len(receipts)}),flush=True)

class P3:
    """Fixed FIT-only trigram count law, no tuning on DEV."""
    def __init__(self, docs, tau=4.):
        self.tau=tau;self.ct=defaultdict(Counter);b=Counter()
        for raw in docs.values():
            b.update(raw)
            for t in range(3,len(raw)):
                self.ct[raw[t-3:t]][raw[t]]+=1
        denom=sum(b.values())+256
        self.prior=[(b[i]+1)/denom for i in range(256)]
        self.logical_estimate_bytes = sum(3+6*len(x) for x in self.ct.values()) + 2048
    def probs(self,context):
        rows=self.ct.get(context[-3:],{})
        count=sum(rows.values())
        return [(rows.get(i,0)+self.tau*self.prior[i])/(count+self.tau) for i in range(256)]
    def next(self,context):
        probs=self.probs(context)
        return max(range(256),key=lambda i:(probs[i],-i))

def score_neural_window(m,raw,offset):
    x=torch.tensor([list(raw[offset:offset+WINDOW])],dtype=torch.long)
    y=torch.tensor([list(raw[offset+1:offset+WINDOW+1])],dtype=torch.long)
    start=time.perf_counter()
    logits=m(x)
    process_ms=(time.perf_counter()-start)*1000
    if logits.shape != (1,WINDOW,256):raise ValueError("model/eval shape mismatch")
    lp=F.log_softmax(logits,-1)
    nll = float((-lp.gather(2,y.unsqueeze(-1))).sum())
    acc = int((lp.argmax(-1)==y).sum())
    results={}
    for prefix_len in REG["evaluation"]["offer_prefix_lengths"]:
        prefix=list(raw[offset:offset+prefix_len])
        generated=[]
        for _ in range(6):
            probe = torch.tensor([prefix+generated],dtype=torch.long)
            one=m(probe)
            generated.append(int(one[0,-1].argmax()))
        # ONLY NOW compare against unknown future source labels.
        target=raw[offset+prefix_len:offset+prefix_len+6]
        for n in REG["evaluation"]["greedy_completion_lengths"]:
            results[n]=results.get(n,0)+int(bytes(generated[:n])==target[:n])
    return {"nll_sum":nll,"top1_count":acc,"forward_ms":process_ms,"exact":results}

def score_p3_window(p3,raw,offset):
    losses=0.;accuracy=0;exact={}
    for i in range(WINDOW):
        context=raw[offset:offset+i+1]
        p=p3.probs(context)
        truth=raw[offset+i+1]
        losses-=math.log(p[truth])
        accuracy+=int(max(range(256),key=lambda k:(p[k],-k))==truth)
    for prefix_len in REG["evaluation"]["offer_prefix_lengths"]:
        ctx=raw[offset:offset+prefix_len]
        generated=[]
        for _ in range(6):
            c=p3.next(ctx+bytes(generated))
            generated.append(c)
        target=raw[offset+prefix_len:offset+prefix_len+6]
        for n in REG["evaluation"]["greedy_completion_lengths"]:
            exact[n]=exact.get(n,0)+int(bytes(generated[:n])==target[:n])
    return {"nll_sum":losses,"top1_count":accuracy,"exact":exact}

def aggregate(rows):
    if len(rows)!=len(REG["dev"]["files"])*len(REG["evaluation"]["window_offsets"]):
        raise ValueError("missing fixed source/window denominator")
    total=len(rows)*WINDOW
    result={
        "targets":total,"offers":len(rows)*len(REG["evaluation"]["offer_prefix_lengths"]),
        "nll":sum(z["nll_sum"] for z in rows)/total,
        "top1":sum(z["top1_count"] for z in rows)/total,
        "exact":{str(k):sum(z["exact"].get(k,0) for z in rows) for k in (2,4,6)},
        "by_publisher":{}
    }
    result["oracle_net_keys"]= {str(k):result["exact"][str(k)]*(k-1) for k in (2,4,6)}
    for group in REG["evaluation"]["groups"]:
        own=[r for r in rows if r["group"]==group]
        if len(own)!=2*len(REG["evaluation"]["window_offsets"]):raise ValueError("publisher cohort mismatch")
        result["by_publisher"][group]= {
            "nll":sum(z["nll_sum"] for z in own)/(len(own)*WINDOW),
            "top1":sum(z["top1_count"] for z in own)/(len(own)*WINDOW),
            "exact4":sum(z["exact"][4] for z in own),
            "offers":len(own)*2
        }
    times=[z.get("forward_ms") for z in rows if "forward_ms" in z]
    if times:
        result["full_window_forward_p50_ms"]=statistics.median(times)
        result["full_window_forward_p95_ms"]=sorted(times)[int(.95*(len(times)-1))]
    return result

@torch.inference_mode()
def evaluate(archive, out):
    if out.exists():raise FileExistsError("no DEV reruns or changed results")
    if not (archive/"SEAL.json").is_file():raise ValueError("all original checkpoints must be sealed BEFORE DEV")
    seal_info=json.loads((archive/"SEAL.json").read_text())
    receipts=verify_all(archive)
    if (seal_info["study"],seal_info["model_count"],seal_info["matched_schedules"]) != ("CODEVALUE002",9,True):
        raise ValueError("invalid preregistered original seal")
    for (a,s),r in receipts.items():
        if seal_info["original_models"][f"{a}-{s}"]["checkpoint_sha256"] != r["checkpoint_sha256"]:
            raise ValueError("original seal altered")
    torch.set_num_threads(2)
    fit=fit_sources()
    for seed in SEEDS:
        expected=schedule_hash(fit,seed)
        for arm in ARMS:
            if receipts[(arm,seed)]["sample"] != expected:raise ValueError("original training source replay mismatch")
    p3=P3(fit)  # all learning from CPython only
    # FIRST-OPEN real held-out source bytes only AFTER all original checkpoints validated.
    dev=fetch_files(REG["dev"]["files"])
    def all_rows(evaluator):
        rows=[]
        for source in REG["dev"]["files"]:
            owner=source["repo"]+":"+source["path"]
            for offset in REG["evaluation"]["window_offsets"]:
                r=evaluator(dev[owner],offset)
                r.update({"group":source["group"],"owner":owner,"offset":offset})
                rows.append(r)
        return rows
    baseline=aggregate(all_rows(lambda raw,offset:score_p3_window(p3,raw,offset)))
    result_models={}
    for arm in ARMS:
        each=[]
        for seed in SEEDS:
            cp=torch.load(archive/f"{arm}-{seed}"/"MODEL.pt",map_location="cpu",weights_only=True)
            m=make(arm).eval();m.load_state_dict(cp["model"],strict=True)
            scored=aggregate(all_rows(lambda raw,offset:score_neural_window(m,raw,offset)))
            scored.update({"seed":seed,"checkpoint_sha256":receipts[(arm,seed)]["checkpoint_sha256"],
                           "weights_bytes":sum(t.numel()*t.element_size() for t in m.state_dict().values())})
            each.append(scored)
        result_models[arm]={
            "seeds":each,
            "nll_mean":statistics.mean(x["nll"] for x in each),
            "top1_mean":statistics.mean(x["top1"] for x in each),
            "exact4_mean":statistics.mean(x["exact"]["4"] for x in each),
            "oracle4_net_keys_mean":statistics.mean(x["oracle_net_keys"]["4"] for x in each)
        }
    tac=result_models["tac"]
    p3keys=baseline["oracle_net_keys"]["4"]
    groupwins=sum(
        statistics.mean(s["by_publisher"][k]["exact4"] for s in tac["seeds"])>
        baseline["by_publisher"][k]["exact4"] for k in REG["evaluation"]["groups"])
    seedwins=sum(s["oracle_net_keys"]["4"]>p3keys for s in tac["seeds"])
    advance=tac["oracle4_net_keys_mean"]>p3keys and groupwins>=2 and seedwins>=2
    outcome={
        "study":"CODEVALUE002", "decision":"PRELIMINARY_UTILITY_SCREEN_PASS" if advance else "UTILITY_GATE_FAIL",
        "commercial_value_proved":False,"revenue_proved":False,"physical_mobile_test":"NOT_RUN",
        "source_type":"real licensed publisher source code, new first-open DEV, not protected blind TEST",
        "dev_source_blobs":[{"repo":x["repo"],"path":x["path"],"git_blob":x["git_blob_sha1"],
                    "sha256":sha256(dev[x["repo"]+":"+x["path"]])} for x in REG["dev"]["files"]],
        "training_FLOPs":"NOT_MATCHED", "all_checkpoints_original_replayed":True,
        "gpu_minutes":0,"training_target_bytes_per_model":STEPS*8*128,
        "scoring_targets_per_model":len(REG["dev"]["files"])*len(REG["evaluation"]["window_offsets"])*WINDOW,
        "baseline_fit_only_p3":baseline, "p3_logical_estimate_bytes":p3.logical_estimate_bytes,
        "results":result_models,"tac_primary4_group_wins":groupwins,"tac_primary4_seed_wins":seedwins,
        "limitations":["No user-accepted completion data, no actual time saved or sales","Exact-match oracle ignores distraction",
         "Reset 128-byte windows are not long-history streaming or Android",
         "3 publishers 2 files each, repeated seeds are not independent source data",
         "Only fixed source/code training, no matching full FLOPs, SSM not Mamba3",
         "No customer pricing/energy/export/SDK/product qualification"]
    }
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(outcome,sort_keys=True,indent=2,allow_nan=False)+"\n")
    print(json.dumps({"status":outcome["decision"],"p3":{"nll":baseline["nll"],"oracle4":p3keys},
       "tac":{"nll":tac["nll_mean"],"oracle4_mean":tac["oracle4_net_keys_mean"]},
       "models":{a:{"nll":v["nll_mean"],"keys":v["oracle4_net_keys_mean"]} for a,v in result_models.items()}}),flush=True)

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--seal-dir",type=Path)
    p.add_argument("--seal-out",type=Path)
    p.add_argument("--evaluate-dir",type=Path)
    p.add_argument("--output",type=Path)
    a=p.parse_args()
    if a.seal_dir and a.seal_out and not a.evaluate_dir:seal(a.seal_dir,a.seal_out)
    elif a.evaluate_dir and a.output and not a.seal_dir:evaluate(a.evaluate_dir,a.output)
    else:raise ValueError("choose seal-dir + seal-out OR evaluate-dir + output")
if __name__=="__main__":
    main()
