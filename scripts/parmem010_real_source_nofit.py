"""PAR-MEM-010 authentic publisher-source, no-fit integration and parity preflight.

Uses only the already-authorized March-April 2014 original PRSA Beijing
FIT observations and 12 externally supplied station IDs. The event index
is the genuine original publisher row hour. No target/future measurement
is ever used for prediction or scoring in this diagnostic; no optimizer.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import torch

from scripts.partime005_continuous_real_training import (
    source_and_times, observed_source_step,
)
from tac_osm.variants.source_aligned_parallel_world import SourceAlignedParallelWorld

def run(source:Path):
    torch.manual_seed(10010)
    torch.set_num_threads(1)
    data,fit,dev=source_and_times(source)
    batch=len(fit)
    if batch!=12 or min(len(x) for x in fit)<16:
        raise RuntimeError("authentic original 12 owner FIT history unavailable")
    model=SourceAlignedParallelWorld()
    contract=model.contract()
    model.eval()
    owners=torch.arange(batch,dtype=torch.int64)
    state=model.init_stream(owners)
    bobs,bcov,bdt,bmask,bindex=[],[],[],[],[]
    stream_preds=[]
    prev=None
    with torch.no_grad():
        for step in range(16):
            clocks=[fit[s][step] for s in range(batch)]
            obs,cov,dt,admit=observed_source_step(data,clocks,prev)
            true_event_ordinal=torch.tensor(clocks,dtype=torch.int64)
            pred,state=model.stream_step(
                state,obs,cov,elapsed_dt=dt,write_mask=admit,
                source_ids=owners,event_index=true_event_ordinal)
            stream_preds.append(pred)
            bobs.append(obs);bcov.append(cov);bdt.append(dt)
            bmask.append(admit);bindex.append(true_event_ordinal)
            prev=clocks
        obs=torch.stack(bobs,1)
        cov=torch.stack(bcov,1)
        dt=torch.stack(bdt,1)
        mask=torch.stack(bmask,1)
        idx=torch.stack(bindex,1)
        batch_pred,trace=model(obs,cov,elapsed_dt=dt,write_mask=mask,
            source_ids=owners,event_index=idx,return_trace=True)
        errors=(batch_pred-torch.stack(stream_preds,1)).abs().max().item()
        matrix_errors=(state.aligned_memory-trace["aligned_memory"][:,-1]).abs().max().item()
        if errors>1e-4 or matrix_errors>1e-4:
            raise RuntimeError("genuine source scan/stream mismatch beyond contract")
        if not torch.equal(idx[:,1:]>idx[:,:-1],
                           torch.ones_like(idx[:,1:],dtype=torch.bool)):
            raise RuntimeError("genuine source event indices not chronological")
        if not mask.all().item():
            raise RuntimeError("unrecorded source event write in genuine FIT cohort")
        if not (torch.isfinite(batch_pred).all().item()
                and torch.isfinite(state.aligned_memory).all().item()):
            raise RuntimeError("source model emitted nonfinite original-sensor state")
        past=model.historic_query(state,source_ids=owners,
            requested_event_index=idx[:,0])
        if past.shape!=(12,14) or not torch.isfinite(past).all():
            raise RuntimeError("true-source historic request failed shape")
        original_source_state=state.aligned_memory.clone()
        if not torch.equal(original_source_state,state.aligned_memory):
            raise RuntimeError("historical read unexpectedly mutated state")
    out=dict(
      status="GENUINE_SOURCE_NO_FIT_OPERATOR_PREFLIGHT",
      source_station_entities=batch,
      genuine_observed_source_events_per_entity=16,
      real_source_writable_events=int(mask.sum()),
      source_first_original_hours=idx[:,0].tolist(),
      source_last_original_hours=idx[:,-1].tolist(),
      trainable_parameter_count=contract["learned_trainable_parameters"],
      original_plus_aligned_fp32_stream_scalars=contract["total_fp32_per_stream"],
      scan_stream_max_prediction_difference=errors,
      scan_stream_max_memory_difference=matrix_errors,
      genuine_publisher_next_outcomes_scored=False,
      historical_factual_content_retrieval_qualified=False,
      source_numerical_2015_2017_read=False,
      optimizer_updates=0)
    print("PAR_MEM_010_REAL_SOURCE_ORIGINAL_NO_FIT="+json.dumps(out,sort_keys=True))
    return out

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--source",type=Path,required=True)
    args=parser.parse_args()
    run(args.source)
