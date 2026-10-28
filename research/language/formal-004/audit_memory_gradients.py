"""MLAG002: FIT-only source-native and aligned memory gradient-path audit.

Frozen 10 original + 10 paired MLAG001 checkpoints. Only actual Vim FIT
source; NO model fit, no independent TEST, no optimizer update.
Check all 36 authentic document byte hashes and 20 checkpoint hashes.
"""
import argparse,hashlib,json,sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from train_multilag_real_fit import SHA,sha,chars

GROUPS={
 "native_params":("key_native.","value_native.","native_gates."),
 "source_shared":("rotation","source_content.","source_router.","source_retention.",
                  "source_strength.","source_silu_mod."),
 "history_read_and_router":("hist_project.","read_router."),
 "next_head":("read_in.","read_out.","res_in.","res_out.","norm."),
 "common_embedding":("embedding.","time_proj.")
}

def group_stats(main,history,names,prefixes):
    ids=[i for i,name in enumerate(names) if name.startswith(prefixes)]
    mag_main=sum(float(main[i].detach().square().sum()) for i in ids if main[i] is not None)**.5
    mag_hist=sum(float(history[i].detach().square().sum()) for i in ids if history[i] is not None)**.5
    overlap=sum(float((main[i].detach()*history[i].detach()).sum()) for i in ids
                if main[i] is not None and history[i] is not None)
    return {"main_grad_l2":mag_main,"history96_grad_l2":mag_hist,
            "cosine":(overlap/(mag_main*mag_hist) if mag_main*mag_hist else None)}

def audit(original,dual,source,output):
    torch.set_num_threads(1)
    if sha(original/"parallel_tac_language_gate.py")!=SHA:
        raise ValueError("original model source hash mismatch")
    op=json.loads((original/"PROTOCOL.json").read_text())
    old=json.loads((original/"RESULTS.json").read_text())
    new=json.loads((dual/"RESULTS.json").read_text())
    if op!=old["protocol"] or new["protocol"]["original_protocol_sha256"]!=sha(original/"PROTOCOL.json"):
        raise ValueError("experiment provenance mismatch")
    for name,expected in op["source"]["byte_sha256"].items():
        if Path(name).name!=name or sha(source/name)!=expected:
            raise ValueError("original Vim chapter hash mismatch "+name)
    xs=[];ys=[]
    docs=["usr_01.txt","usr_02.txt","usr_03.txt","usr_04.txt"]
    if not set(docs).issubset(op["source"]["doc_partitions"]["fit"]):
        raise ValueError("FIT-only probe leaked out of training partition")
    for name in docs:
        data=chars((source/name).read_bytes())
        pos=min(512,len(data)-130)
        xs.append(torch.tensor(data[pos:pos+128],dtype=torch.long))
        ys.append(torch.tensor(data[pos+1:pos+129],dtype=torch.long))
    x=torch.stack(xs);y=torch.stack(ys)
    sys.path.insert(0,str(original))
    from parallel_tac_language_gate import ParallelTACLanguage
    runs=[]
    for arm,rows in (("original",old["runs"]),("dual_horizon",new["runs"])):
        if len(rows)!=10:raise ValueError("10 checkpoints missing per arm")
        for item in rows:
            filename=item["checkpoint_file"] if arm=="original" else item["checkpoint"]
            f=(original if arm=="original" else dual)/filename
            if f.name!=filename or sha(f)!=item["checkpoint_sha256"]:
                raise ValueError("checkpoint SHA256 mismatch")
            ck=torch.load(f,weights_only=True,map_location="cpu")
            net=ParallelTACLanguage(decoder=75,position_mode="multiscale",
                                    gate_mode=item["mode"]).eval()
            if sum(p.numel() for p in net.parameters())!=100680:
                raise ValueError("model parameter budget drift")
            net.load_state_dict(ck["state"] if arm=="original" else ck["model"],strict=True)
            out,trace=net(x,return_trace=True)
            logits,history_target=net.historical_logits(x,trace,96)
            main=F.cross_entropy(out.reshape(-1,128),y.reshape(-1))
            hist=F.cross_entropy(logits.reshape(-1,128),history_target.reshape(-1))
            named=list(net.named_parameters())
            ps=[p for _,p in named];names=[name for name,_ in named]
            gm=torch.autograd.grad(main,ps,retain_graph=True,allow_unused=True)
            gh=torch.autograd.grad(hist,ps,allow_unused=True)
            groups={name:group_stats(gm,gh,names,prefixes)
                    for name,prefixes in GROUPS.items()}
            if groups["native_params"]["history96_grad_l2"]!=0:
                raise ValueError("historical head unexpectedly depends on native memory")
            runs.append({"arm":arm,"mode":item["mode"],"seed":item["seed"],
                         "checkpoint_sha256":item["checkpoint_sha256"],
                         "groups":groups,"main_nll":float(main.detach()),
                         "history96_nll":float(hist.detach())})
    means={}
    for arm in ("original","dual_horizon"):
        rows=[r for r in runs if r["arm"]==arm]
        means[arm]={g:{k:float(np.mean([r["groups"][g][k] for r in rows
                    if r["groups"][g][k] is not None])) if any(
                    r["groups"][g][k] is not None for r in rows) else None
                    for k in ("main_grad_l2","history96_grad_l2","cosine")}
                    for g in GROUPS}
    result={"scope":"authentic source-hash-verified FIT 4-doc gradient diagnostic, not independent TEST",
            "no_model_training":True,"source_sha256":SHA,"n_checkpoints":len(runs),
            "source_documents":docs,"runs":runs,"means":means,
            "qualification":"Gradient norms and cosines are one FIT slice; no proof of general training improvement."}
    output.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
    return result

if __name__=="__main__":
    a=argparse.ArgumentParser()
    a.add_argument("--original",type=Path,required=True)
    a.add_argument("--mlag",type=Path,required=True)
    a.add_argument("--source",type=Path,required=True)
    a.add_argument("--output",type=Path,required=True)
    v=a.parse_args()
    print(json.dumps(audit(v.original,v.mlag,v.source,v.output)["means"],indent=2))
