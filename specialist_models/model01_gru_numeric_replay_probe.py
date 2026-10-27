#!/usr/bin/env python3
"""No-training diagnosis of varying CPU GRU replay tolerances on original UCI.

Does not relax original strict replay. Reports quantitative differences for
native Git-owned original GRU trained checksums, with default vs disabled
oneDNN execution, to determine cross-runner numerical behavior. Read-only.
"""
from __future__ import annotations
import argparse,base64,gzip,hashlib,io,json,math
from pathlib import Path
import numpy as np
import torch
import model01_matched_gru_control as control

ROOT=Path(__file__).resolve().parents[1]
ORIGINAL=ROOT/"research/replay-bundles/MODEL01-GRU-103K-CONTROL-001/ORIGINAL_FIT.json"

def run_probe():
    original,parity=control.preflight()
    saved=json.loads(ORIGINAL.read_bytes())
    assert saved["source_zip_sha256"]==control.tac.SOURCE
    raw=control.tac.uci.fetch_official_zip()
    if control.tac.uci.digest(raw)!=control.tac.SOURCE:
        raise control.ControlError("publisher source hash mismatch")
    y=np.asarray(control.tac.uci.original_rows(raw),dtype=np.float64)
    mean=float(y[:control.tac.uci.FIT_END].mean())
    std=float(y[:control.tac.uci.FIT_END].std())
    cal=control.tac.windows(y,control.tac.uci.eligible_indices(
        control.tac.uci.FIT_END,control.tac.uci.CAL_END),mean,std)
    dev=control.tac.windows(y,control.tac.uci.eligible_indices(
        control.tac.uci.CAL_END,control.tac.uci.DEV_END),mean,std)
    torch.set_num_threads(2)
    recordings=[]
    for flag in [True,False]:
        torch.backends.mkldnn.enabled=flag
        for bundle in saved["runs"]:
            b=gzip.decompress(base64.b64decode(bundle["checkpoint_b64_gzip"],validate=True))
            if hashlib.sha256(b).hexdigest()!=bundle["checkpoint_sha256"]:
                raise control.ControlError("saved GRU model bytes mismatch")
            state=torch.load(io.BytesIO(b),map_location="cpu",weights_only=True)
            net=control.ControlGRU()
            net.load_state_dict(state["model"],strict=True)
            ca,ae=control.tac.score(net,cal,mean,std)
            q=float(np.sort(ae)[math.ceil((len(ae)+1)*.9)-1])
            de,_=control.tac.score(net,dev,mean,std,q)
            metric_delta={k:de[k]-v for k,v in bundle["dev"].items()}
            ratio={k:(de[k]-v)/max(abs(v),1e-9) for k,v in bundle["dev"].items()}
            recordings.append({
                "seed":bundle["seed"],
                "mkldnn_enabled":flag,
                "original_checkpoint_sha256":bundle["checkpoint_sha256"],
                "original_pinned_q90_wh":bundle["q90_wh"],
                "current_q90_wh":q,
                "delta_q90_wh":q-bundle["q90_wh"],
                "original_cal":bundle["cal"],"current_cal":ca,
                "original_dev":bundle["dev"],"current_dev":de,
                "signed_absolute_dev_deltas":metric_delta,
                "relative_dev_deltas":ratio,
                "passes_existing_1e_minus_6_relative_tolerance":all(
                    math.isclose(de[k],v,rel_tol=1e-6,abs_tol=1e-6)
                    for k,v in bundle["dev"].items()),
            })
    return {
        "schema":"model01-gru-original-cross-runner-numeric-investigation-v1",
        "source_zip_sha256":control.tac.SOURCE,
        "original_parameter_gap":parity,
        "trained_during_probe":False,
        "independent_forecasting_qualification":False,
        "checked_unchanged_original_three_GRU_checkpoints":True,
        "original_strict_replay_requirement_weakened":False,
        "results":recordings,
        "warning":"Disabling oneDNN for diagnostic execution does not recreate training code path and is not automatic replay qualification"
    }

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args()
    if a.out.exists():raise SystemExit("refusing overwrite")
    report=run_probe()
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(report,indent=2,sort_keys=True,allow_nan=False)+"\n")
    print(json.dumps({"study":report["schema"],"results":[{
      "seed":x["seed"],"mkldnn_enabled":x["mkldnn_enabled"],
      "delta_q90_wh":x["delta_q90_wh"],"relative_dev_deltas":x["relative_dev_deltas"],
      "passes_strict":x["passes_existing_1e_minus_6_relative_tolerance"]
      } for x in report["results"]]},sort_keys=True))
