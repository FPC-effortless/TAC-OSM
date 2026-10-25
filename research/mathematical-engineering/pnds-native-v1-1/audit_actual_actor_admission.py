#!/usr/bin/env python3
"""Diagnostic ONLY: source-authorized fixed-slot admissions on frozen real
INTERACTION traffic DEV case *inputs* from the existing ESWAM012 evidence.

Source is the immutable original single-site previously-exposed DEV,
not a new independent test and not the real 256-observation training stream.
No future source labels or predictions are fed to this mechanism.

This tests whether source actor identity and observed current position/speed
can be routed through the admission protocol, and counts rejected/replaced
actors. It cannot establish better physical forecasts, long-memory recall,
decision costs, or parallel batch time legality.
"""
from __future__ import annotations
import hashlib,json,sys,urllib.request
from pathlib import Path
import torch

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parents[2]/"src"))
from tac_osm.variants.bounded_entity_admission import init_entity_state,entity_admit_step

REPO="FPC-effortless/TAC-OSM"
SOURCE_COMMIT="d9db3a0ac68d2bbabe3febe6cfe3d40bf322ea1d"
SOURCE_PATH="research/replay-bundles/ESWAM012/native-86001-ORIGINAL_DEV_CASES.json"


def run():
    url=f"https://raw.githubusercontent.com/{REPO}/{SOURCE_COMMIT}/{SOURCE_PATH}"
    request=urllib.request.Request(url,headers={"User-Agent":"TAC-OSM-source-pinned-audit"})
    with urllib.request.urlopen(request,timeout=40) as response:
        payload=response.read()
    rows=json.loads(payload)
    if not isinstance(rows,list) or len(rows)<266:
        raise ValueError("insufficient real source cases; no synthetic fallback")
    # Source-factual unique actor/key/time positions; do NOT inspect
    # true_original_future_xy_m or model predictions.
    bykey={}
    for r in rows:
        if int(r["horizon_frames"])!=10:continue
        cluster=str(r["source_actor_cluster"])
        position=r["observed_current_xy_m"]
        speed=float(r["current_measured_source_velocity_norm_m_per_s"])
        if not(0<=speed<50 and len(position)==2 and
               all(abs(float(x))<1e5 for x in position)):
            raise ValueError("invalid observed measured source evidence")
        key=(int(r["source_original_recording"]),
             int(r["source_original_frame"]),cluster)
        bykey[key]=(position,speed)
    ids=sorted({k[2] for k in bykey})
    if len(ids)<30:raise ValueError("real source actor IDs absent")
    encoded={actor:i+1 for i,actor in enumerate(ids)}
    state=init_entity_state(1,slots=8,width=8)
    received=replaced=denied=0
    for (recording,frame,cluster),(xy,speed) in sorted(bykey.items()):
        entity_id=torch.tensor([encoded[cluster]],dtype=torch.long)
        # Fixed physically bounded feature coding from CURRENT observation,
        # not learned from future DEV labels.
        position=[float(x)/10000 for x in xy]
        value=torch.tensor([position+[min(speed/20,1.)]+[0.]*5])
        score=torch.tensor([min(speed/2,1.)])
        writable=torch.tensor([True])
        prev_ids=state.entity_ids.clone()
        prev_values=state.values.clone()
        state,_,admit,evict=entity_admit_step(
            state,entity_id,value,score,writable,
            beta=.5,replacement_margin=.15)
        received+=int(admit[0])
        replaced+=int(evict[0])
        denied+=int(not admit[0])
        # Source-grounded invariance: whenever an old identity survives,
        # its slot cannot be silently overwritten by the current new ID.
        for slot in range(8):
            if (prev_ids[0,slot]!=-1 and state.entity_ids[0,slot]==prev_ids[0,slot]
                and state.entity_ids[0,slot]!=entity_id[0]):
                if not torch.equal(prev_values[0,slot],state.values[0,slot]):
                    raise AssertionError("unrelated real identity memory eroded")
    print(json.dumps({
      "status":"AUTHENTIC_EXPOSED_DEV_ENTITY_ADMISSION_MECHANISM_ONLY",
      "source_ref":SOURCE_COMMIT,"source_path":SOURCE_PATH,
      "source_blob_sha256":hashlib.sha256(payload).hexdigest(),
      "observed_distinct_actor_clusters":len(ids),
      "unique_observed_source_events":len(bykey),
      "real_source_accepted_writes":received,
      "real_source_rejected_writes":denied,
      "real_source_slot_evictions":replaced,
      "memory_slots":8,"memory_width":8,
      "real_source_authoritative_actor_ids":True,
      "future_truth_or_model_prediction_used_as_input":False,
      "source_is_independent_new_test":False,
      "model_parameters_fitted":False,
      "forecast_or_decision_success_established":False},sort_keys=True))

if __name__=="__main__":run()
