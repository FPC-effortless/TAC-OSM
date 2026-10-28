"""GRAPH011 reproducible authentic-source (checked-in Python code) memory/latency benchmark.

No model fitting or independent next-token accuracy claims. Comparison
measures exact same frozen backbone + source cache under dense vs sparse
replay policies. The full, publicly checked-in source Python text remains
available to every arm and its byte cost is ALWAYS reported.
"""
import hashlib
import json
import statistics
import time
from pathlib import Path

import torch

from source_checkpoint_replay_v7 import VerifiedSparseReplay
from source_calibrated_tac_v5 import SourceCalibratedTAC
from tac_wave_ced_500k_v4 import SlotTAC500K


def bench():
    torch.set_num_threads(1)
    path=Path(__file__).with_name("bayesian_cache_scan_v5.py")
    original=path.read_bytes()
    if any(byte>=128 for byte in original):
        raise ValueError("recorded fixture not in source 128-character vocabulary")
    # 128 real consecutive chars from the original checked-in Python program.
    # No made-up labels or future source is used.
    source=original[:128]
    digest=hashlib.sha256(source).hexdigest()
    torch.manual_seed(711)
    model=SourceCalibratedTAC(
        SlotTAC500K().eval(),torch.ones(128)/128,capacity=32).eval()
    model.requires_grad_(False)
    queries=(1,17,32,49,64,81,97,113,128)
    results=[]
    for block in (1,8,32,128):
        t0=time.perf_counter()
        sparse=VerifiedSparseReplay(
            model,source,owner="GRAPH009-source-implementation",
            block=block,source_sha256=digest).prepare()
        index_s=time.perf_counter()-t0
        query_s=[];replayed=0
        checksum=hashlib.sha256()
        for target in queries:
            pred,state,receipt=sparse.predict_at(
                target,owner="GRAPH009-source-implementation",
                expected_sha256=digest)
            query_s.append(receipt.seconds)
            replayed+=receipt.replayed_tokens
            checksum.update(bytes(pred.detach().cpu().contiguous().view(torch.uint8).reshape(-1).tolist()))
        results.append({
            "block":block,"checkpoint_count":len(sparse.checkpoints),
            "checkpoint_serialized_bytes":sparse.resident_checkpoint_bytes(),
            "source_bytes":len(source),
            "checkpoint_plus_source_bytes":sparse.resident_checkpoint_bytes()+len(source),
            "model_parameter_bytes":sparse.model_weight_bytes,
            "index_wall_s":index_s,
            "mean_query_replay_wall_s":statistics.mean(query_s),
            "total_query_replayed_tokens":replayed,
            "max_replayed_per_query":max((t-1)%block+1 for t in queries),
            "source_sha256":digest,
            # Different TF/BLAS chunk schedules may differ at FP roundoff.
            "query_result_checksum_noncanonical":checksum.hexdigest()})
    return {
        "experiment":"GRAPH011_READONLY_SOURCE_REPLAY",
        "source":"authentic checked-in GRAPH009 Python implementation, first 128 ASCII bytes",
        "scope":"engineering memory/time tradeoff only; NOT language quality or independent TEST",
        "theorem_claim":"NONE: not Ryan Williams universal square-root-space simulation",
        "platform":{"torch":torch.__version__,"device":"CPU","threads":1},
        "query_ordinals":queries,"results":results}


if __name__=="__main__":
    print(json.dumps(bench(),indent=2,sort_keys=True))
