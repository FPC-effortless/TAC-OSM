"""MLAG004 frozen 320-character longer-context preflight on authentic Vim DEV.

ALL 20 original+dual-64/96 models were trained on context128. This evaluates
out-of-training context T=320 and lags 64/96/128/256, NOT 256-trained recall,
independent TEST, or any new model. Source and every checkpoint SHA256 verified.
"""
import argparse,json,sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from train_multilag_real_fit import SHA,chars,sha

def run(original,dual,source,output):
    torch.set_num_threads(1)
    proto=json.loads((original/"PROTOCOL.json").read_text())
    old=json.loads((original/"RESULTS.json").read_text())
    new=json.loads((dual/"RESULTS.json").read_text())
    if old["protocol"]!=proto or sha(original/"parallel_tac_language_gate.py")!=SHA:
        raise ValueError("source model/protocol mismatch")
    if new["protocol"]["original_protocol_sha256"]!=sha(original/"PROTOCOL.json"):
        raise ValueError("dual source protocol mismatch")
    if len(proto["source"]["byte_sha256"])!=36:
        raise ValueError("missing original source hashes")
    for name,expected in proto["source"]["byte_sha256"].items():
        if Path(name).name!=name or sha(source/name)!=expected:
            raise ValueError("source chapter mismatch "+name)
    docs=proto["source"]["doc_partitions"]
    fit=[chars((source/n).read_bytes()) for n in docs["fit"]]
    counts=np.full(128,.5,dtype=np.float64)
    for d in fit:counts+=np.bincount(d,minlength=128)
    prior=torch.tensor(counts/counts.sum(),dtype=torch.float32)
    windows=[]
    for name in docs["dev"]:
        data=chars((source/name).read_bytes())
        if len(data)<=322:raise ValueError("DEV text too short for context320")
        for start in np.linspace(0,len(data)-322,4,dtype=np.int64):
            windows.append(torch.tensor(data[start:start+320],dtype=torch.long))
    dev=torch.stack(windows)
    sys.path.insert(0,str(original))
    from parallel_tac_language_gate import ParallelTACLanguage
    result={"study":"MLAG004","scope":"previously exposed real DEV; UNTRAINED context320 extrapolation only",
            "source_sha256":SHA,"source_chapter_count":36,
            "untrained_context":320,"original_training_context":128,
            "lags":[64,96,128,256],"runs":[]}
    with torch.inference_mode():
      for arm,rows in (("original",old["runs"]),("dual_horizon",new["runs"])):
        if len(rows)!=10:raise ValueError("ten checkpoints missing in "+arm)
        for item in rows:
            name=item["checkpoint_file"] if arm=="original" else item["checkpoint"]
            path=(original if arm=="original" else dual)/name
            if path.name!=name or sha(path)!=item["checkpoint_sha256"]:
                raise ValueError("checkpoint hash drift")
            ck=torch.load(path,weights_only=True,map_location="cpu")
            model=ParallelTACLanguage(decoder=75,position_mode="multiscale",
                gate_mode=item["mode"]).eval()
            if sum(p.numel() for p in model.parameters() if p.requires_grad)!=100680:
                raise ValueError("model geometry mismatch")
            model.load_state_dict(ck["state"] if arm=="original" else ck["model"],strict=True)
            agg={k:[0.,0] for k in ["next"]+
                [f"lag{lag}_{mode}" for lag in (64,96,128,256)
                    for mode in ("correct","zero","fit_prior")]}
            def accumulate(key,loss,count):
                agg[key][0]+=float(loss);agg[key][1]+=int(count)
            for x in dev.split(2):
                out,trace=model(x,return_trace=True)
                y=x[:,1:]
                accumulate("next",F.cross_entropy(out[:,:-1].reshape(-1,128),
                    y.reshape(-1),reduction="sum").item(),y.numel())
                blind=dict(trace);blind["aligned"]=torch.zeros_like(trace["aligned"])
                for lag in (64,96,128,256):
                    h,t=model.historical_logits(x,trace,lag)
                    z,_=model.historical_logits(x,blind,lag)
                    accumulate(f"lag{lag}_correct",F.cross_entropy(h.reshape(-1,128),
                        t.reshape(-1),reduction="sum").item(),t.numel())
                    accumulate(f"lag{lag}_zero",F.cross_entropy(z.reshape(-1,128),
                        t.reshape(-1),reduction="sum").item(),t.numel())
                    accumulate(f"lag{lag}_fit_prior",(-prior.log()[t]).sum().item(),t.numel())
            result["runs"].append({"arm":arm,"seed":item["seed"],"mode":item["mode"],
                "checkpoint_sha256":item["checkpoint_sha256"],
                "nll":{k:v/n for k,(v,n) in agg.items()},
                "target_count":{k:n for k,(_,n) in agg.items()}})
    means={}
    for arm in ("original","dual_horizon"):
        rr=[r for r in result["runs"] if r["arm"]==arm]
        means[arm]={k:float(np.mean([r["nll"][k] for r in rr]))
                    for k in rr[0]["nll"]}
    result["means"]=means
    result["warning"]="Models were NEVER trained at 320 positions: out-of-distribution context length invalidates claims of successful long-context retrieval."
    output.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
    return result

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--original",type=Path,required=True)
    p.add_argument("--mlag",type=Path,required=True)
    p.add_argument("--source",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    args=p.parse_args()
    print(json.dumps(run(args.original,args.mlag,args.source,args.output)["means"],indent=2))
