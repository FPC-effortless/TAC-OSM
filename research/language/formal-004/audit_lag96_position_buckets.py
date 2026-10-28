"""MLAG004 additional fixed-position diagnostic on real 320-char DEV windows.

Ten original MLAG001 dual-horizon 128-trained checkpoints are frozen.
Divide the *same* 96-event historical query into training-position and
post-training-position blocks. No fitting, protected TEST or model promotion.
"""
import argparse,json,sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from train_multilag_real_fit import SHA,sha,chars

BLOCKS=((96,127),(128,191),(192,255),(256,319))

def audit(original,mlag,source,output):
    torch.set_num_threads(1)
    proto=json.loads((original/"PROTOCOL.json").read_text())
    old=json.loads((original/"RESULTS.json").read_text())
    new=json.loads((mlag/"RESULTS.json").read_text())
    if old["protocol"]!=proto or sha(original/"parallel_tac_language_gate.py")!=SHA:
        raise ValueError("original source provenance failure")
    if new["protocol"]["original_protocol_sha256"]!=sha(original/"PROTOCOL.json"):
        raise ValueError("paired source provenance failure")
    for name,h in proto["source"]["byte_sha256"].items():
        if Path(name).name!=name or sha(source/name)!=h:
            raise ValueError("authentic source chapter hash failure")
    fit=[chars((source/n).read_bytes()) for n in proto["source"]["doc_partitions"]["fit"]]
    counts=np.full(128,.5,dtype=np.float64)
    for d in fit:counts+=np.bincount(d,minlength=128)
    prior=torch.tensor(counts/counts.sum(),dtype=torch.float32)
    windows=[]
    for name in proto["source"]["doc_partitions"]["dev"]:
        d=chars((source/name).read_bytes())
        if len(d)<=322:raise ValueError("DEV source does not cover context320")
        for start in np.linspace(0,len(d)-322,4,dtype=np.int64):
            windows.append(torch.tensor(d[start:start+320],dtype=torch.long))
    dev=torch.stack(windows)
    sys.path.insert(0,str(original))
    from parallel_tac_language_gate import ParallelTACLanguage
    result={"study":"MLAG004-positions","scope":"real, previously exposed DEV; 128-trained frozen checkpoints on UNTRAINED context320",
            "lag":96,"blocks":BLOCKS,"runs":[]}
    with torch.inference_mode():
      for row in new["runs"]:
        name=row["checkpoint"];f=mlag/name
        if f.name!=name or sha(f)!=row["checkpoint_sha256"]:
            raise ValueError("checkpoint SHA256 failure")
        net=ParallelTACLanguage(decoder=75,position_mode="multiscale",
                                gate_mode=row["mode"]).eval()
        if sum(p.numel() for p in net.parameters() if p.requires_grad)!=100680:
            raise ValueError("parameter budget changed")
        net.load_state_dict(torch.load(f,weights_only=True,map_location="cpu")["model"],strict=True)
        sums={f"{lo}-{hi}":{k:[0.,0] for k in ("correct","zero","prior")}
              for lo,hi in BLOCKS}
        for x in dev.split(2):
            _,trace=net(x,return_trace=True)
            logits,labels=net.historical_logits(x,trace,96)
            blind=dict(trace);blind["aligned"]=torch.zeros_like(trace["aligned"])
            no_memory,_=net.historical_logits(x,blind,96)
            for lo,hi in BLOCKS:
                i,j=lo-96,hi-96+1
                target=labels[:,i:j];p=logits[:,i:j];z=no_memory[:,i:j]
                bucket=sums[f"{lo}-{hi}"]
                for mode,value in (("correct",p),("zero",z)):
                    bucket[mode][0]+=F.cross_entropy(value.reshape(-1,128),
                        target.reshape(-1),reduction="sum").item()
                    bucket[mode][1]+=target.numel()
                bucket["prior"][0]+=float((-prior.log()[target]).sum())
                bucket["prior"][1]+=target.numel()
        result["runs"].append({"seed":row["seed"],"mode":row["mode"],
              "checkpoint_sha256":row["checkpoint_sha256"],
              "blocks":{block:{key:{"nll":loss/n,"n":n}
                        for key,(loss,n) in vals.items()}
                        for block,vals in sums.items()}})
    result["mean"]={f"{lo}-{hi}":{mode:float(np.mean([
        item["blocks"][f"{lo}-{hi}"][mode]["nll"]
        for item in result["runs"]])) for mode in ("correct","zero","prior")}
        for lo,hi in BLOCKS}
    output.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
    return result

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--original",type=Path,required=True)
    p.add_argument("--mlag",type=Path,required=True)
    p.add_argument("--source",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    print(json.dumps(audit(a.original,a.mlag,a.source,a.output)["mean"],indent=2))
