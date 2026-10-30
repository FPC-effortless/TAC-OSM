"""Metered raw-ring Witten--Bell control; observed bytes only, no target API."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import resource
import time

import numpy as np

RING_BYTES = 3904
CAPACITY = RING_BYTES - 16
ORDER = 6


def distribution(observed):
    """Order-six suffix counts with Witten--Bell escape to a uniform base."""
    history = bytes(observed)
    probability = np.full(256, 1/256, dtype=np.float64)
    for order in range(min(ORDER, len(history))+1):
        suffix = history[-order:] if order else b""
        counts = np.zeros(256, dtype=np.int64)
        for end in range(order, len(history)):
            if order == 0 or history[end-order:end] == suffix:
                counts[history[end]] += 1
        total = int(counts.sum())
        types = int(np.count_nonzero(counts))
        if total:
            probability = counts/(total+types) + types/(total+types)*probability
    if not np.isfinite(probability).all() or not np.isclose(probability.sum(), 1):
        raise ValueError("invalid raw-history distribution")
    return probability


def predict(acquisition, queries):
    """Fixed-allocation ring, one condition; no gold or file metadata input."""
    if acquisition.dtype != np.uint8 or queries.dtype != np.uint8:
        raise ValueError("raw history requires byte arrays")
    if acquisition.ndim != 2 or acquisition.shape[1] != 128 or queries.shape != (len(acquisition),64):
        raise ValueError("raw history requires registered128/64 geometry")
    n = len(acquisition)
    start = time.perf_counter()
    ring = np.zeros((n,CAPACITY),dtype=np.uint8)
    cursor = np.zeros(n,dtype=np.int64)
    size = np.zeros(n,dtype=np.int64)
    acquisition_seconds = 0.
    query_seconds = 0.
    generation_seconds = 0.
    log_probs = np.empty((1,n,256),dtype=np.float64)
    greedy = np.empty((1,n,4),dtype=np.uint8)

    def admit(i, values):
        for value in values:
            ring[i,cursor[i]] = value
            cursor[i] = (cursor[i]+1) % CAPACITY
            size[i] = min(CAPACITY,size[i]+1)

    def read(i):
        if size[i] < CAPACITY:
            return ring[i,:size[i]].tobytes()
        return np.concatenate((ring[i,cursor[i]:],ring[i,:cursor[i]])).tobytes()

    for i in range(n):
        tic=time.perf_counter(); admit(i,acquisition[i]); acquisition_seconds += time.perf_counter()-tic
        tic=time.perf_counter(); admit(i,queries[i]); p=distribution(read(i)); query_seconds += time.perf_counter()-tic
        log_probs[0,i] = np.log(p)
        tic=time.perf_counter()
        for step in range(4):
            value=int(p.argmax()); greedy[0,i,step]=value
            if step < 3:
                admit(i,[value]); p=distribution(read(i))
        generation_seconds += time.perf_counter()-tic
    payload=ring.nbytes+cursor.nbytes+size.nbytes
    if payload != n*RING_BYTES:
        raise AssertionError("ring-budget accounting failed")
    costs={"setup_seconds":max(0.,time.perf_counter()-start-acquisition_seconds-query_seconds-generation_seconds),
           "acquisition_seconds":acquisition_seconds,"query_seconds":query_seconds,
           "generation_seconds":generation_seconds,"model_bytes":0,
           "state_payload_bytes":payload,"serialized_metadata_bytes":0,
           "peak_rss_bytes":int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)}
    metadata={"conditions":["carry"],"costs":costs,"ring_bytes_per_stream":RING_BYTES,
              "order":ORDER,"evidence":"observed-prefix raw storage, no learned neural parameters",
              "optimizer_steps":0}
    for _ in range(10):
        encoded=json.dumps(metadata,sort_keys=True,allow_nan=False)
        size=len(encoded.encode())
        if costs['serialized_metadata_bytes']==size: break
        costs['serialized_metadata_bytes']=size
    else:
        raise AssertionError('metadata byte count did not converge')
    return {"log_probs":log_probs,"greedy":greedy,"metadata":metadata}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--acquisition',type=Path,required=True)
    parser.add_argument('--queries',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists(): raise FileExistsError(args.output)
    result=predict(np.load(args.acquisition,allow_pickle=False),np.load(args.queries,allow_pickle=False))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('xb') as handle:
        np.savez_compressed(handle,log_probs=result['log_probs'],greedy=result['greedy'],
                            metadata=np.array(json.dumps(result['metadata'],sort_keys=True,allow_nan=False)))
    print(json.dumps({'status':'RAW_HISTORY_PREDICTIONS_SAVED','output':str(args.output)}))


if __name__=='__main__':
    main()
