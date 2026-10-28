#!/usr/bin/env python3
"""LANGLAYER035 reused frozen 5-seed FIT sampler — never trains or evaluates.

Sole permitted training input schedule: genuine already-registered
publisher FIT bytes, no document-boundary crossing, same indices per seed
and architecture. All targets are the actual NEXT byte of the same source.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import sys
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
SOURCE_MODULE=ROOT/"research/mathematical-engineering/langbase024"
SEEDS=(1701,1702,1703,1704,1705)
UPDATES=320
BATCH=8
WINDOW=129
NEXT_BYTE_TARGETS=128

def fit_sources():
    sys.path.insert(0,str(SOURCE_MODULE))
    from byte_next_reference import original_source
    fit0,dev,supp=original_source(ROOT)
    fit={**fit0,supp[0]:supp[1]}
    if len(fit)!=14 or len(dev)!=4 or set(fit)&set(dev):
        raise ValueError("source owner/split mismatch")
    if not all(len(v)>=WINDOW for v in fit.values()):
        raise ValueError("FIT original shorter than source window")
    return fit

def iter_batches(fit,seed,steps=UPDATES):
    if seed not in SEEDS or not (0<=steps<=UPDATES):
        raise ValueError("nonregistered seed/steps")
    names=tuple(sorted(fit))
    rng=np.random.default_rng(seed)
    for step in range(steps):
        indices=[]
        for _ in range(BATCH):
            name=names[int(rng.integers(len(names)))]
            original=fit[name]
            start=int(rng.integers(len(original)-WINDOW+1))
            raw=original[start:start+WINDOW]
            if len(raw)!=WINDOW:
                raise ValueError("source owner/window mismatch")
            indices.append((name,start,raw))
        yield step,indices

def schedule_receipt(fit,seed,steps=UPDATES):
    h=hashlib.sha256()
    byte_h=hashlib.sha256()
    seen=set()
    count=0
    for _,batch in iter_batches(fit,seed,steps):
        if len(batch)!=BATCH:raise ValueError("batch geometry")
        for name,start,raw in batch:
            h.update(f"{name}:{start};".encode())  # preserves LANGTRAIN025 sampling digest convention
            byte_h.update(raw)
            seen.add(name)
            count+=1
    return {"seed":seed,"steps":steps,"examples_seen":count,
            "input_bytes_seen":count*NEXT_BYTE_TARGETS,
            "next_byte_targets_seen":count*NEXT_BYTE_TARGETS,
            "source_window_bytes_seen":count*WINDOW,
            "sample_order_sha256":h.hexdigest(),
            "sample_bytes_sha256":byte_h.hexdigest(),
            "distinct_fit_owners_seen":len(seen),
            "nontrainable_external_knowledge":"NONE"}

def frozen_plan():
    fit=fit_sources()
    results=[schedule_receipt(fit,s) for s in SEEDS]
    # Independent arms MUST reuse each exact seed schedule.
    return {"study_id":"LANGLAYER035","execution_level":"internal_research",
            "source_partition":"publisher-authenticated PSF FIT+UTF8 supplement only",
            "num_fit_owners":len(fit),"seeds":list(SEEDS),
            "sampler":"numpy.default_rng(seed): uniform sorted FIT filename then uniform offset",
            "window_bytes":WINDOW,"model_input_bytes":NEXT_BYTE_TARGETS,
            "label_rule":"same original filename raw bytes at offsets +1..+128",
            "training_updates":UPDATES,"batch_size":BATCH,
            "optimizer_steps_performed":0,"DEV_predictions":"NOT_RUN",
            "gpu_minutes":0,"protected_TEST":"NOT_ACCESSED",
            "study_status":"FROZEN_NO_FIT_SCHEDULE",
            "by_seed":results}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output",type=Path)
    a=p.parse_args()
    r=frozen_plan()
    raw=json.dumps(r,sort_keys=True,indent=2,allow_nan=False)+"\n"
    if a.output:
        if a.output.exists():raise ValueError("refuse to overwrite existing receipt")
        a.output.parent.mkdir(parents=True,exist_ok=True)
        a.output.write_text(raw)
    print(raw,end="")
if __name__=="__main__":main()
