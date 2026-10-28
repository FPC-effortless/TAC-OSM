"""LANGBASE024: original CPython publisher byte-complete next-byte references.

Only authentic Git-owned 13 FIT, four already-exposed DEV and registered Unicode
FIT supplement. No neural training, no protected TEST, no tuning on DEV labels.
Frozen algorithm: 256-way Dirichlet(0.5) next-byte P0, hierarchical k=1..3
source-owned conditional counts with tau=8. Source hashes/Git blobs checked.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import hashlib
import heapq
import json
import math
from pathlib import Path, PurePosixPath

V=256
ALPHA=.5
TAU=8.
SOURCE_RECEIPT="research/mathematical-engineering/language_source023/SOURCE_RECEIPT.json"
SUPPLEMENT_RECEIPT="research/mathematical-engineering/language_source023/UTF8_SUPPLEMENT_RECEIPT.json"
SOURCE_DIR="research/source-bundles/LANGUAGE-SOURCE023"
EXPOSED_DEV=True

def sha256(data:bytes)->str:
    return hashlib.sha256(data).hexdigest()

def git_blob_sha(data:bytes)->str:
    return hashlib.sha1(b"blob "+str(len(data)).encode()+b"\0"+data).hexdigest()

def read_sha_source(root:Path,relative:str,entry:dict)->bytes:
    p=PurePosixPath(relative)
    if (not relative or p.is_absolute() or ".." in p.parts
        or "\\" in relative or p.as_posix()!=relative):
        raise ValueError("unsafe source path")
    path=root/SOURCE_DIR/relative
    if not path.is_file() or path.is_symlink():
        raise ValueError("missing canonical Git-owned publisher original: "+relative)
    raw=path.read_bytes()
    if (len(raw)!=entry["bytes"] or sha256(raw)!=entry["sha256"]
        or git_blob_sha(raw)!=entry["git_blob"]):
        raise ValueError("original publisher SHA, Git-blob, or byte size mismatch: "+relative)
    if len(raw)<2:raise ValueError("missing next-byte source outcome")
    return raw

def original_source(root:Path)->tuple[dict[str,bytes],dict[str,bytes],tuple[str,...]]:
    receipt=json.loads((root/SOURCE_RECEIPT).read_text())
    if (receipt.get("release")!="v3.13.0" or
        receipt.get("commit")!="60403a5409ff2c3f3b07dd2ca91a7a3e096839c7" or
        len(receipt.get("files",[]))!=17):
        raise ValueError("frozen original publisher source receipt changed")
    fit,dev={},{}
    all_names=[]
    for entry in receipt["files"]:
        name=entry["path"]
        if name in all_names:raise ValueError("duplicate frozen publisher source")
        all_names.append(name)
        if entry["role"] not in ("FIT","DEV"):
            raise ValueError("original source role changed")
        raw=read_sha_source(root,name,entry)
        (fit if entry["role"]=="FIT" else dev)[name]=raw
    if len(fit)!=13 or len(dev)!=4 or set(fit)&set(dev):
        raise ValueError("original 13 FIT/4 EXPOSED DEV split changed")
    # Original frozen split: sorted tutorial path ordinal modulo4==3 is DEV.
    sorted_names=sorted(all_names)
    if tuple(sorted_names)!=tuple(all_names):
        raise ValueError("frozen source ordering changed")
    for i,name in enumerate(sorted_names):
        if (name in dev)!=(i%4==3):
            raise ValueError("frozen publisher source partition changed")
    supplement_receipt=json.loads((root/SUPPLEMENT_RECEIPT).read_text())
    if (supplement_receipt.get("path")!="Doc/howto/unicode.rst" or
        supplement_receipt.get("role")!="FIT_SUPPLEMENT" or
        supplement_receipt.get("publisher_commit")!=receipt["commit"]):
        raise ValueError("Unicode supplement role or publisher changed")
    supplement=read_sha_source(root,supplement_receipt["path"],supplement_receipt)
    if supplement_receipt["path"] in fit or supplement_receipt["path"] in dev:
        raise ValueError("Unicode supplement overlaps original file identities")
    return fit,dev,(supplement_receipt["path"],supplement)

def train_count_model(docs:dict[str,bytes]):
    if not docs:raise ValueError("nonempty FIT source required")
    global_next=Counter()
    context_counts={k:defaultdict(Counter) for k in (1,2,3)}
    targets=0
    for owner in sorted(docs):
        data=docs[owner]
        if not isinstance(data,bytes) or len(data)<2:
            raise ValueError("original raw bytes required")
        for t in range(len(data)-1):
            # Same-source, next-byte target from FIT only; never concatenate documents.
            y=data[t+1]
            global_next[y]+=1
            targets+=1
            for k in (1,2,3):
                if t+1>=k:
                    context_counts[k][data[t-k+1:t+1]][y]+=1
    den=targets+ALPHA*V
    prior=tuple((global_next[y]+ALPHA)/den for y in range(V))
    if abs(math.fsum(prior)-1.)>1e-12 or min(prior)<=0:
        raise ValueError("invalid smoothed FIT byte prior")
    return {"prior":prior,"counts":context_counts,"targets":targets,
            "nonascii_fit_targets":sum(v for y,v in global_next.items() if y>=128),
            "fit_docs":len(docs)}

def law(model,prefix:bytes,order:int):
    if order not in (0,3) or not isinstance(prefix,bytes):
        raise ValueError("registered byte context/order only")
    p=model["prior"]
    if order==3:
        for k in (1,2,3):
            if len(prefix)<k:break
            ctx=prefix[-k:]
            c=model["counts"][k].get(ctx)
            if c:
                n=sum(c.values())
                p=tuple((c.get(y,0)+TAU*p[y])/(n+TAU) for y in range(V))
    if min(p)<=0 or not all(math.isfinite(a) for a in p):
        raise ValueError("invalid positive law")
    if not math.isclose(math.fsum(p),1.,abs_tol=5e-12):
        raise ValueError("prediction law fails normalization")
    return p

def score_document(model,data:bytes,order:int):
    if not isinstance(data,bytes) or len(data)<2:
        raise ValueError("observed original source bytes required")
    total={"n":0,"nll_sum":0.,"brier_sum":0.,"top1":0,"top5":0,
           "nonascii_n":0,"nonascii_nll_sum":0.}
    for t in range(len(data)-1):
        # No source is visible after position t. The label at t+1 is read
        # only after the probability law has been produced for the prefix.
        p=law(model,data[:t+1],order)
        target=data[t+1]
        total["n"]+=1
        nll=-math.log(p[target])
        total["nll_sum"]+=nll
        total["brier_sum"]+=1-2*p[target]+math.fsum(x*x for x in p)
        leaders=heapq.nlargest(5,range(V),key=lambda j:(p[j],-j))
        total["top1"]+=target==leaders[0]
        total["top5"]+=target in leaders
        if target>=128:
            total["nonascii_n"]+=1
            total["nonascii_nll_sum"]+=nll
    n=total["n"]
    return {"targets":n,"nll_nats_per_byte":total["nll_sum"]/n,
            "bits_per_original_byte":total["nll_sum"]/n/math.log(2),
            "brier":total["brier_sum"]/n,
            "top1":total["top1"]/n,"top5":total["top5"]/n,
            "nonascii_targets":total["nonascii_n"],
            "nonascii_nll_nats_per_byte":total["nonascii_nll_sum"]/total["nonascii_n"] if total["nonascii_n"] else None,
            "_sums":total}

def benchmark(root:Path):
    fit,dev,supplement=original_source(root)
    suppl_path,suppl_bytes=supplement
    arms={"original_fit":fit,"fit_plus_registered_unicode":{**fit,suppl_path:suppl_bytes}}
    all_results={}
    for arm,train in arms.items():
        model=train_count_model(train)
        rows={}
        for order in (0,3):
            per_file={name:score_document(model,dev[name],order)
                      for name in sorted(dev)}
            sums=[r["_sums"] for r in per_file.values()]
            n=sum(x["n"] for x in sums)
            nll=sum(x["nll_sum"] for x in sums)/n
            brier=sum(x["brier_sum"] for x in sums)/n
            rare=sum(x["nonascii_n"] for x in sums)
            rare_loss=sum(x["nonascii_nll_sum"] for x in sums)
            for r in per_file.values():del r["_sums"]
            rows[f"P{order}"]={"targets":n,"nll_nats_per_byte":nll,
                 "bits_per_original_byte":nll/math.log(2),"brier":brier,
                 "top1":sum(x["top1"] for x in sums)/n,
                 "top5":sum(x["top5"] for x in sums)/n,
                 "nonascii_targets":rare,
                 "nonascii_nll_nats_per_byte":rare_loss/rare if rare else None,
                 "unweighted_document_macro_nll":sum(x["nll_nats_per_byte"] for x in per_file.values())/len(per_file),
                 "per_document":per_file}
        all_results[arm]={"fit_docs":model["fit_docs"],
          "fit_next_byte_targets":model["targets"],
          "fit_nonascii_next_byte_targets":model["nonascii_fit_targets"],
          "models":rows}
    return {"study":"LANGBASE024_SOURCE_VERIFIED_BYTE_NGRAM_REFERENCE",
      "status":"AUTHENTIC_EXPOSED_INTERNAL_DEV_ONLY_NOT_NATIVE_TAC_GAIN",
      "publisher":"PSF CPython3.13.0 at pinned commit 60403a5409ff2c3f3b07dd2ca91a7a3e096839c7",
      "source_fit_original_documents":13,"source_dev_exposed_documents":4,
      "supplement_role":"FIT_ONLY_REGISTERED_UTF8",
      "prediction":"next original byte from within-document input prefix through t only",
      "classes":256,"alpha":ALPHA,"tau":TAU,"max_history":3,
      "fits_neural_models":False,"protected_test_access":False,
      "original_TAC_Git_replay":"BLOCKED_NOT_PART_OF_THIS_STUDY",
      "byte_TAC_499978_train_status":"UNTRAINED",
      "results":all_results}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo",type=Path,default=Path(__file__).resolve().parents[2])
    p.add_argument("--output",required=True,type=Path)
    a=p.parse_args()
    result=benchmark(a.repo.resolve())
    a.output.parent.mkdir(parents=True,exist_ok=True)
    if a.output.exists():raise ValueError("refuse to overwrite earlier scientific receipt")
    a.output.write_text(json.dumps(result,indent=2,sort_keys=True,allow_nan=False)+"\n")
    for name,v in result["results"].items():
        for model,r in v["models"].items():
            print(f"{name} {model}: {r['nll_nats_per_byte']:.8f} NLL "
                  f"{r['bits_per_original_byte']:.8f} bits/byte "
                  f"targets={r['targets']} nonascii={r['nonascii_targets']}",flush=True)
if __name__=="__main__":main()
