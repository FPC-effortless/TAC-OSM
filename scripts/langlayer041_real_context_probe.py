#!/usr/bin/env python3
"""LANGLAYER041 trained original-checkpoint history intervention, NO NEW FIT.

At each authentic exposed DEV next-byte target, compare original contiguous
left prefix to a same-length, source-owned *distant earlier* donor prefix
with identical last64 genuine observed bytes. Evaluate same target offsets
at each length 128/256/512. All models reset window state; no carried stream.
"""
from __future__ import annotations
import argparse,hashlib,io,json,math,sys
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
ROOT=Path(__file__).resolve().parents[1]
for p in (ROOT,ROOT/"research/mathematical-engineering/langbase024",
          ROOT/"research/mathematical-engineering/langlayer034",
          ROOT/"research/mathematical-engineering/langlayer038",
          ROOT/"research/mathematical-engineering/langlayer040"):
    sys.path.insert(0,str(p))
from byte_next_reference import original_source
from layered_transition import LayeredContextTAC034
from recurrence_only import RecurrenceOnlyDepth038
from factorial_depth import FactorialDepthControl040
from scripts.langlayer035_sampling import fit_sources,schedule_receipt

CONFIG={
 "h48_graph":("LANGLAYER035","tac_layer2",499725),
 "h48_plain":("LANGLAYER040","state48_no_graph",499623),
 "h59_graph":("LANGLAYER040","state59_full_graph",499848),
 "h59_plain":("LANGLAYER039","tac_recuronly038",499901)
}
SEEDS=(1701,1702,1703,1704,1705)
LENGTHS=(128,256,512)
RECENT=64
ANCHORS_PER_DOC=8

def model_for(arm):
    if arm=="h48_graph":return LayeredContextTAC034(depth=2)
    if arm=="h48_plain":return FactorialDepthControl040(state_width=48,upper_graph=False)
    if arm=="h59_graph":return FactorialDepthControl040(state_width=59,upper_graph=True)
    if arm=="h59_plain":return RecurrenceOnlyDepth038()
    raise ValueError("unrecognized trained model family")

def aligned_source_targets():
    _,dev,_=original_source(ROOT)
    schedule={}
    for name,raw in sorted(dev.items()):
        if len(raw)<1280+ANCHORS_PER_DOC:raise ValueError("not enough earlier real history for frozen source")
        anchors=np.linspace(1280,len(raw)-2,ANCHORS_PER_DOC,dtype=np.int64).tolist()
        if len(set(anchors))!=ANCHORS_PER_DOC:raise ValueError("duplicate selected target")
        schedule[name]={"size":len(raw),"bytes_sha256":hashlib.sha256(raw).hexdigest(),
            "target_offsets":anchors}
    if len(schedule)!=4:raise ValueError("original four DEV owners changed")
    return dev,schedule

def evaluate(arm,seed,checkpoint,receipt):
    if arm not in CONFIG or seed not in SEEDS:raise ValueError("not frozen configuration")
    study,model_arm,params=CONFIG[arm]
    old=json.loads(Path(receipt).read_text())
    if (old["study"],old["arm"],old["seed"],old["updates"])!=(study,model_arm,seed,320):
        raise ValueError("wrong historical original checkpoint/seed/study")
    raw=Path(checkpoint).read_bytes()
    digest=hashlib.sha256(raw).hexdigest()
    if digest!=old["checkpoint_sha256"]:raise ValueError("sha256 invalid original checkpoint")
    saved=torch.load(io.BytesIO(raw),map_location="cpu",weights_only=True)
    if (saved["study"],saved["arm"],saved["seed"],saved["updates"])!=(study,model_arm,seed,320):
        raise ValueError("checkpoint metadata mismatch")
    if saved["source_commit"]!="60403a5409ff2c3f3b07dd2ca91a7a3e096839c7":
        raise ValueError("publisher training source commit mismatch")
    source=schedule_receipt(fit_sources(),seed,steps=320)
    if (saved["sample_order_sha256"]!=source["sample_order_sha256"] or
        old["source_sample_order_sha256"]!=source["sample_order_sha256"] or
        old["source_bytes_sha256"]!=source["sample_bytes_sha256"]):
        raise ValueError("original training sample source mismatch")
    torch.set_num_threads(2)
    m=model_for(arm).eval()
    if sum(p.numel() for p in m.parameters())!=params:raise ValueError("source model count mismatch")
    m.load_state_dict(saved["model"],strict=True)
    dev,source_docs=aligned_source_targets()
    result={"study":"LANGLAYER041","arm":arm,"seed":seed,
            "original_checkpoint_sha256":digest,
            "original_train_run_study":study,
            "source_docs":source_docs,
            "target_count":4*ANCHORS_PER_DOC,
            "training_updates":0,"GPU_minutes":0,"protected_TEST":"NOT_ACCESSED",
            "whole_model_carried_state_test":False,
            "fresh_independent_generalization":False,
            "lengths":{}}
    with torch.no_grad():
        for L in LENGTHS:
            sums={"genuine_nll_sum":0.,"donor_nll_sum":0.,
                  "genuine_top1_correct":0,"donor_top1_correct":0}
            count=0;clock=0.
            import time
            for name,doc in sorted(dev.items()):
                anchors=source_docs[name]["target_offsets"]
                authentic=[];changed=[];gold=[]
                for pos in anchors:
                    observed=doc[pos-L:pos]
                    distant=doc[:L-RECENT]
                    tail=doc[pos-RECENT:pos]
                    if len(observed)!=L or len(distant)!=L-RECENT or len(tail)!=RECENT:
                        raise ValueError("original prefix source bounds changed")
                    if pos-L<=L-RECENT or distant+tail==observed:
                        raise ValueError("ineligible or ineffective donor intervention")
                    authentic.append(list(observed))
                    changed.append(list(distant+tail))
                    gold.append(doc[pos])
                x=torch.tensor(authentic,dtype=torch.long)
                z=torch.tensor(changed,dtype=torch.long)
                y=torch.tensor(gold,dtype=torch.long)
                start=time.process_time()
                a=m(x)[:,-1].double()
                b=m(z)[:,-1].double()
                clock+=time.process_time()-start
                la=F.log_softmax(a,dim=-1).gather(-1,y[:,None]).neg().sum()
                lb=F.log_softmax(b,dim=-1).gather(-1,y[:,None]).neg().sum()
                sums["genuine_nll_sum"]+=float(la);sums["donor_nll_sum"]+=float(lb)
                sums["genuine_top1_correct"]+=int((a.argmax(-1)==y).sum())
                sums["donor_top1_correct"]+=int((b.argmax(-1)==y).sum())
                count+=len(anchors)
            if count!=32 or not all(math.isfinite(q) for q in sums.values()):
                raise ValueError("real DEV intervention incomplete or invalid")
            result["lengths"][str(L)]={
                "n":count,
                "genuine_nll":sums["genuine_nll_sum"]/count,
                "distant_donor_nll":sums["donor_nll_sum"]/count,
                "history_utility_delta_donor_minus_genuine":
                    (sums["donor_nll_sum"]-sums["genuine_nll_sum"])/count,
                "genuine_top1":sums["genuine_top1_correct"]/count,
                "distant_donor_top1":sums["donor_top1_correct"]/count,
                "cpu_process_seconds_two_conditions":clock,
                "control":"same byte target offsets, same last64 observed bytes and local positions; distant earlier prefix replaced, state reset per window"}
    return result

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--arm",choices=tuple(CONFIG),required=True)
    p.add_argument("--seed",choices=SEEDS,type=int,required=True)
    p.add_argument("--checkpoint",type=Path,required=True)
    p.add_argument("--receipt",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    result=evaluate(a.arm,a.seed,a.checkpoint,a.receipt)
    if a.output.exists():raise FileExistsError("refuse overwrite")
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,sort_keys=True,indent=2,allow_nan=False)+"\n")
    print(json.dumps(result,sort_keys=True,allow_nan=False))
if __name__=="__main__":main()
