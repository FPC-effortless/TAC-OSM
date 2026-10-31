"""LANGCAP105 genuine FIT-only owner-authorized *exploratory* trainer.

User authorization permits exploratory optimization without falsely forging
the separate independent assessor required for scientific qualification.
The historical signed runner and LANGCAP103 drafts are NOT amended.
"""
from __future__ import annotations
import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import random
import shutil
import sys
import tarfile
import time

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT/"research/langcap103"))

def require(value,message):
    if not value:raise ValueError(message)

def digest(path):
    h=sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda:f.read(1048576),b""):h.update(chunk)
    return h.hexdigest()

def load_protocol():
    path=HERE/"OWNER_PROTOCOL.json"
    p=json.loads(path.read_text())
    require(p.get("schema")=="langcap105.owner-authorized-exploratory-fit.v1","wrong protocol")
    require(p.get("owner_authorized") is True,"owner authorization missing")
    require(p.get("independent_attestation") is False and
            p.get("architecture_superiority_eligible") is False and
            p.get("dev_scoring_in_training") is False and
            p.get("protected_test_access") is False,
            "cannot represent exploratory training as qualified evidence")
    require(p["fit"]["steps"]==2560 and p["fit"]["batch_size"]==4 and
            p["fit"]["sequence_length"]==128,"unexpected training regime")
    require(set(p["seeds"])=={10301,10302,10303},"seed matrix changed")
    return p,digest(path)

def verify_source(receipt_path,archive_path,protocol):
    from source import canonical,digest as sha_raw,git_blob
    receipt=json.loads(Path(receipt_path).read_text())
    require(receipt["source_sha256"]==protocol["source"]["source_sha256"] and
            sha_raw(canonical(receipt["identity"]))==receipt["source_sha256"],
            "source identity changed")
    fit={r["path"]:r for r in receipt["identity"]["files"] if r["split"]=="FIT"}
    require(len(fit)==560,"not the original exact 560 FIT file cohort")
    data={}
    with tarfile.open(archive_path,"r") as t:
        for member in t:
            require(member.isfile() and member.name in fit and
                    member.name not in data and not member.issym() and
                    not member.islnk(),"unsafe or DEV input in model archive")
            value=t.extractfile(member).read()
            row=fit[member.name]
            require(len(value)==row["bytes"] and sha_raw(value)==row["sha256"] and
                    git_blob(value)==row["git_blob"],"FIT file original bytes differ")
            data[member.name]=value
    require(set(data)==set(fit),"FIT-only archive missing files")
    return data

def blocks_for(data,length):
    # Sequence + next-token label stays WITHIN an original publisher file.
    return [(name,offset) for name,raw in sorted(data.items())
            for offset in range(0,len(raw)-length,length)]

def training_stream(blocks,seed):
    rng=random.Random(seed)
    while True:
        current=list(blocks)
        rng.shuffle(current)
        yield from current

def fit(args):
    p,protocol_sha=load_protocol()
    require(args.seed in p["seeds"] and args.scale in p["models"],
            "unregistered seed or size")
    model_rows=p["models"][args.scale]["configurations"]
    rows=[r for r in model_rows if r["arm"]==args.arm]
    require(len(rows)==1,"unregistered model")
    spec=rows[0]
    require(not args.output.exists(),"refuse overwrite of existing model evidence")
    require(args.max_steps is None or (1<=args.max_steps<=p["fit"]["steps"]),
            "smoke steps exceed predeclared training bounds")
    smoke=args.max_steps is not None and args.max_steps<p["fit"]["steps"]
    require(args.smoke or not smoke,"partial fit must be labeled smoke")
    if args.smoke:
        require(smoke and args.max_steps<=8,"smoke must be <=8 steps")
    else:
        require(args.max_steps is None,"full fit cannot silently truncate")
    source=verify_source(args.source_receipt,args.fit_archive,p)
    import torch
    import torch.nn.functional as F
    import preflight
    torch.set_num_threads(p["fit"]["cpu_threads"])
    torch.manual_seed(args.seed)
    torch.use_deterministic_algorithms(True)
    random.seed(args.seed)
    config={k:v for k,v in spec.items() if k!="trainable_parameters"}
    target=p["models"][args.scale]["target_parameters"]
    model=preflight.build(config,target,preflight.load_native(),
                          preflight.comparator_classes(torch)).float().train()
    exact=sum(x.numel() for x in model.parameters() if x.requires_grad)
    require(exact==spec["trainable_parameters"],"frozen live model parameter count changed")
    optimizer=torch.optim.AdamW(model.parameters(),
        lr=p["fit"]["lr"],betas=tuple(p["fit"]["betas"]),eps=p["fit"]["eps"],
        weight_decay=p["fit"]["weight_decay"],foreach=False)
    length=p["fit"]["sequence_length"];batch=p["fit"]["batch_size"]
    all_blocks=blocks_for(source,length)
    require(len(all_blocks)>=100,"insufficient authentic FIT chunks")
    stream=training_stream(all_blocks,args.seed)
    steps_goal=args.max_steps if smoke else p["fit"]["steps"]
    maximum=p["resource_bounds"]["per_job_soft_deadline_minutes"]*60
    started=time.monotonic();losses=[];steps=0;observed=0;state="STARTED"
    args.output.mkdir(parents=True)
    def checkpoint(name):
        tmp=args.output/"CHECKPOINT.tmp"
        payload={"model":model.state_dict(),"optimizer":optimizer.state_dict(),
                 "rng_cpu":torch.get_rng_state(),"seed":args.seed,
                 "arm":args.arm,"scale":args.scale,"steps":steps,
                 "source_sha256":p["source"]["source_sha256"],
                 "protocol_sha256":protocol_sha,
                 "scientific_status":"OWNER_AUTHORIZED_UNQUALIFIED_EXPLORATORY_FIT"}
        torch.save(payload,tmp)
        tmp.replace(args.output/name)
        return digest(args.output/name)
    def report(checksum):
        receipt={"status":state,"arm":args.arm,"scale":args.scale,
                 "seed":args.seed,"actual_steps":steps,
                 "planned_steps":p["fit"]["steps"],"smoke":smoke,
                 "fit_loss_first":losses[0] if losses else None,
                 "fit_loss_last":losses[-1] if losses else None,
                 "fit_loss_series":losses,"observed_source_bytes":observed,
                 "elapsed_seconds":time.monotonic()-started,
                 "checkpoint_sha256":checksum,"source_sha256":p["source"]["source_sha256"],
                 "protocol_sha256":protocol_sha,
                 "full_training_flops":"NOT_MEASURED",
                 "dev_scored":False,"protected_test_access":False,
                 "independently_qualified":False,
                 "architecture_superiority":False}
        (args.output/"FIT_RECEIPT.json").write_text(json.dumps(receipt,indent=2)+"\n")
        return receipt
    try:
        for i in range(1,steps_goal+1):
            require(time.monotonic()-started<maximum,"predeclared wall budget exceeded")
            chosen=[next(stream) for _ in range(batch)]
            x=[];y=[]
            for name,offset in chosen:
                raw=source[name][offset:offset+length+1]
                require(len(raw)==length+1,"sample crosses original file edge")
                x.append(list(raw[:-1]));y.append(list(raw[1:]))
            x=torch.tensor(x,dtype=torch.long);y=torch.tensor(y,dtype=torch.long)
            optimizer.zero_grad(set_to_none=True)
            if args.arm=="tac_associative_stack":
                logits,_=model(x,owners=tuple(f"fit-{j}" for j in range(batch)))
            elif args.arm=="stacked_gru":
                logits,_=model(x)
            else:
                logits=model(x)
            require(logits.shape==(batch,length,256),"unexpected model output geometry")
            loss=F.cross_entropy(logits.reshape(-1,256),y.reshape(-1))
            require(bool(torch.isfinite(loss)),"nonfinite FIT loss")
            loss.backward()
            require(all(v.grad is not None and bool(torch.isfinite(v.grad).all())
                        for v in model.parameters() if v.requires_grad),
                    "missing or nonfinite trained gradient")
            optimizer.step()
            steps=i;observed+=batch*length;losses.append(float(loss.detach()))
            if i in (1,8,64,128,256,512,1024,2048,2560):
                print(json.dumps({"status":"FIT_SMOKE" if smoke else "FIT_PROGRESS",
                                  "scale":args.scale,"arm":args.arm,"seed":args.seed,
                                  "step":i,"loss":losses[-1]}),flush=True)
            if not smoke and i in (256,512,1024,2048):
                checkpoint("LATEST.pt")
        state="SMOKE_COMPLETED_NOT_TRAINED" if smoke else "FIT_TRAINING_COMPLETED_UNQUALIFIED"
        checksum=checkpoint("SMOKE.pt" if smoke else "TRAINED.pt")
        if not smoke and (args.output/"LATEST.pt").exists():
            (args.output/"LATEST.pt").unlink()
        receipt=report(checksum)
    except Exception as exc:
        state="FIT_FAILED_RETAINED_UNQUALIFIED"
        checksum=checkpoint("FAILED.pt") if steps else None
        receipt=report(checksum)
        (args.output/"FAILURE.json").write_text(json.dumps(
             {"type":type(exc).__name__,"message":str(exc),"steps":steps},indent=2)+"\n")
        raise
    print(json.dumps({k:receipt[k] for k in ("status","arm","scale","seed","actual_steps","checkpoint_sha256")}),flush=True)

def main():
    a=argparse.ArgumentParser()
    a.add_argument("--source-receipt",type=Path,required=True)
    a.add_argument("--fit-archive",type=Path,required=True)
    a.add_argument("--scale",choices=("small","medium"),required=True)
    a.add_argument("--arm",choices=("tac_associative_stack","causal_transformer","stacked_gru"),required=True)
    a.add_argument("--seed",type=int,required=True)
    a.add_argument("--output",type=Path,required=True)
    a.add_argument("--smoke",action="store_true")
    a.add_argument("--max-steps",type=int)
    fit(a.parse_args())

if __name__=="__main__":
    main()
