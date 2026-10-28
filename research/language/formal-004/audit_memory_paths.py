"""MLAG-002 frozen real Vim FIT/DEV native vs source-aligned path audit.

Authentic source only. Requires original and MLAG001 checkpoint archives,
and byte-identical Vim9.1 manual. Never synthesizes missing rows, trains,
touches independent TEST or replaces the original PyTorch model.
"""
import argparse, hashlib, json, sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from train_multilag_real_fit import SHA,chars,batches,sha

def one_model(model, dev):
    sums={k:[0.,0] for k in ("full","no_native","no_aligned","no_both",
                              "history64","history96","history64_no_aligned",
                              "history96_no_aligned","history64_wrong1",
                              "history96_wrong1","history64_wrong8",
                              "history96_wrong8")}
    def record(k,logit,target):
        sums[k][0]+=F.cross_entropy(logit.reshape(-1,128),
                target.reshape(-1),reduction="sum").item()
        sums[k][1]+=target.numel()
    with torch.inference_mode():
        for x,y in dev:
            out,trace=model(x,return_trace=True)
            record("full",out,y)
            record("no_native",model(x,disable_native=True),y)
            record("no_aligned",model(x,disable_aligned=True),y)
            record("no_both",model(x,disable_native=True,disable_aligned=True),y)
            blind=dict(trace);blind["aligned"]=torch.zeros_like(trace["aligned"])
            for lag in (64,96):
                hist,target=model.historical_logits(x,trace,lag)
                record(f"history{lag}",hist,target)
                zeros,_=model.historical_logits(x,blind,lag)
                record(f"history{lag}_no_aligned",zeros,target)
                for offset in (1,8):
                    wrong,_=model.historical_logits(x,trace,lag+offset)
                    record(f"history{lag}_wrong{offset}",wrong,target[:,:-offset])
    return {key:{"nll":value/n,"n":n} for key,(value,n) in sums.items()}

def audit(original_dir,mlag_dir,vim_root,output):
    torch.set_num_threads(1)
    if sha(original_dir/"parallel_tac_language_gate.py")!=SHA:
        raise ValueError("original source hash drift")
    original=json.loads((original_dir/"RESULTS.json").read_text())
    protocol=json.loads((original_dir/"PROTOCOL.json").read_text())
    mlag=json.loads((mlag_dir/"RESULTS.json").read_text())
    if original["protocol"]!=protocol:
        raise ValueError("original protocol drift")
    if mlag["protocol"]["original_protocol_sha256"]!=sha(original_dir/"PROTOCOL.json"):
        raise ValueError("dual-horizon protocol drift")
    if len(protocol["source"]["byte_sha256"])!=36:
        raise ValueError("missing real source chapter hashes")
    for name,h in protocol["source"]["byte_sha256"].items():
        if Path(name).name!=name or sha(vim_root/name)!=h:
            raise ValueError("source document hash drift "+name)
    dev=batches([chars((vim_root/name).read_bytes())
             for name in protocol["source"]["doc_partitions"]["dev"]])
    sys.path.insert(0,str(original_dir))
    from parallel_tac_language_gate import ParallelTACLanguage
    result={"study":"MLAG002","scope":"already-examined genuine Vim DEV; frozen lesion, no training/TEST",
            "source_sha256":SHA,"checkpoint_count":20,"results":[]}
    for arm,rows in (("original",original["runs"]),("dual_horizon",mlag["runs"])):
        if len(rows)!=10:raise ValueError("missing original or dual checkpoints")
        for row in rows:
            name=row["checkpoint_file"] if arm=="original" else row["checkpoint"]
            f=(original_dir if arm=="original" else mlag_dir)/name
            if f.name!=name or sha(f)!=row["checkpoint_sha256"]:
                raise ValueError("checkpoint SHA256 drift")
            ck=torch.load(f,map_location="cpu",weights_only=True)
            net=ParallelTACLanguage(decoder=75,position_mode="multiscale",
                gate_mode=row["mode"]).eval()
            if sum(p.numel() for p in net.parameters() if p.requires_grad)!=100680:
                raise ValueError("learned parameter count drift")
            net.load_state_dict(ck["state"] if arm=="original" else ck["model"],strict=True)
            measures=one_model(net,dev)
            target=row["heldout"]["nll_nats"] if arm=="original" else row["dev"]["char_nll"]
            if abs(measures["full"]["nll"]-target)>=1e-5:
                raise ValueError("original DEV replay mismatch")
            for lag in (64,96):
                expected=(row["historical"][str(lag)]["nll"] if arm=="original"
                          else row["dev"][f"hist{lag}_nll"])
                if abs(measures[f"history{lag}"]["nll"]-expected)>=1e-5:
                    raise ValueError("archived history replay mismatch")
            result["results"].append({"arm":arm,"seed":row["seed"],
                "mode":row["mode"],"sha256":row["checkpoint_sha256"],
                "metrics":measures})
    result["means"]={}
    for arm in ("original","dual_horizon"):
        rows=[r for r in result["results"] if r["arm"]==arm]
        result["means"][arm]={key:float(np.mean([r["metrics"][key]["nll"] for r in rows]))
                for key in rows[0]["metrics"]}
    result["warning"]="Frozen no-memory lesions are NOT separately trained pruned models. Historical classifier uses ONLY source-aligned memory."
    output.write_text(json.dumps(result,sort_keys=True,indent=2)+"\n")
    return result

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--original",type=Path,required=True)
    p.add_argument("--mlag",type=Path,required=True)
    p.add_argument("--source",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    result=audit(a.original,a.mlag,a.source,a.output)
    print(json.dumps(result["means"],indent=2,sort_keys=True))
