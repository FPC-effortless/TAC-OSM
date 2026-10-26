"""LATFUSE001 no-fit, source-cutoff diagnostic on authenticated real UCI501 FIT rows.

This script NEVER trains or scores targets. Structural state/gate observations of
random initialized parameters do not measure prediction skill. Data are read
from first 512 chronological rows per station in original 2013 FIT; first 256
rows normalize, next 256 are a read-only diagnostic. No 2015+ numerical access.
The original source archive is not modified. Exits fail-closed on SHA mismatch.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import statistics
import time
import zipfile

import torch
from tac_osm.latent_fusion_001 import FusedLatentCandidate, affine_prefix, gated_delta

SOURCE_SHA = "d1b9261c54132f04c374f762f1e5e512af19f95c95fd6bfa1e8ac7e927e3b0b8"
SITES = ("Aotizhongxin","Changping","Dingling","Dongsi","Guanyuan",
         "Gucheng","Huairou","Nongzhanguan","Shunyi","Tiantan","Wanliu","Wanshouxigong")
CHANNELS = ("PM2.5","PM10","SO2","NO2","CO","O3","TEMP","PRES","DEWP","RAIN","WSPM")
WIND = ("N","NNE","NE","ENE","E","ESE","SE","SSE","S","SSW","SW","WSW","W","WNW","NW","NNW")


def load_first_real_fit(source: Path):
    actual = hashlib.sha256(source.read_bytes()).hexdigest()
    if actual != SOURCE_SHA:
        raise RuntimeError("SOURCE_SHA_MISMATCH: refusing source diagnostics")
    records = []
    with zipfile.ZipFile(source) as z:
        for site in SITES:
            filenames = [name for name in z.namelist() if name.endswith(
                f"PRSA_Data_{site}_20130301-20170228.csv")]
            if len(filenames)!=1:
                raise RuntimeError(f"source file missing or duplicated: {site}")
            rows = []
            with z.open(filenames[0]) as f:
                reader=csv.DictReader(io.TextIOWrapper(f,encoding="utf-8-sig"))
                for row in reader:
                    # Deliberately leave 2015-17 source rows untouched; read only 512 hours of 2013.
                    if len(rows) == 512:
                        break
                    if row["station"] != site or int(row["year"]) != 2013 or int(row["month"]) != 3:
                        raise RuntimeError("unexpected chronology/station in original FIT source")
                    # Input only, no +6 or other future labels.
                    x = [float(row[c]) if row[c] != "NA" else float("nan") for c in CHANNELS]
                    direction = row["wd"]
                    if direction == "NA":
                        x.extend([float("nan"),float("nan")])
                    else:
                        if direction not in WIND:
                            raise RuntimeError("unregistered wind direction")
                        angle = 2*math.pi*WIND.index(direction)/16
                        x.extend([math.sin(angle),math.cos(angle)])
                    if any(math.isinf(v) for v in x):
                        raise RuntimeError("nonfinite physical source entry")
                    rows.append(x)
            if len(rows)!=512:
                raise RuntimeError("expected exactly 512 chronological FIT rows per station")
            records.append(rows)
    return torch.tensor(records,dtype=torch.float64),actual


def source_diagnostic(source:Path):
    torch.set_num_threads(1)
    raw,source_sha = load_first_real_fit(source)
    # Only *past* 256 original FIT input hours normalize the next 256 FIT hours.
    normalized=raw.clone()
    for k in range(raw.shape[-1]):
        if k<6:
            x=raw[:,:,k]
            normalized[:,:,k]=torch.where(torch.isfinite(x), torch.log1p(x.clamp_min(0)), x)
        # Physical units of other dimensions may vary widely; FIT-only normalization.
    fit=normalized[:,:256]
    mask=torch.isfinite(fit)
    safe=torch.where(mask,fit,torch.zeros_like(fit))
    counts=mask.sum(dim=(0,1)).clamp_min(1)
    mu=safe.sum(dim=(0,1))/counts
    variance=(torch.where(mask,fit-mu,torch.zeros_like(fit)).square().sum(dim=(0,1))/counts).clamp_min(1e-6)
    sigma=variance.sqrt()
    std=(normalized-mu)/sigma
    # Never write NaN observed values to a state.
    m=FusedLatentCandidate(obs_dim=13,action_dim=1,hidden=32,key_dim=16,val_dim=16,entity_count=16).to(dtype=torch.float64)
    with torch.no_grad():
        state=m.initial(12,dtype=torch.float64)
        prev=torch.zeros(12,13,dtype=torch.float64)
        age=torch.full((12,13),256,dtype=torch.float64)
        alphas=[]; betas=[]; fast=[];slow=[];mem=[]; elapsed=[]
        for t in range(256,512):
            x=std[:,t]
            observed=torch.isfinite(x)
            age=torch.where(observed,torch.zeros_like(age),torch.clamp(age+1,max=256))
            masked_x=torch.where(observed,x,torch.zeros_like(x))
            u=m.encode(masked_x,observed,age,prev)
            alphas.append(torch.sigmoid(m.retention_map(u)).view(-1))
            betas.append(torch.sigmoid(m.write_map(u)).view(-1))
            state=m.step(state,masked_x,observed,age,prev,torch.arange(12))
            prev=torch.where(observed,masked_x,prev)
            age=torch.where(observed,torch.zeros_like(age),age)
            fast.append(state.fast.norm(dim=-1))
            slow.append(state.slow.norm(dim=-1))
            mem.append(state.associative.norm(dim=(-2,-1)))
            elapsed.append(int((~observed).sum()))
        alphas=torch.stack(alphas)
        betas=torch.stack(betas)
        missing=int(sum(elapsed))
        occupancy=state.active.sum(dim=1).tolist()
    def summ(t):
        x=t.flatten().double()
        return {"mean":round(float(x.mean()),7),"min":round(float(x.min()),7),
                "max":round(float(x.max()),7)}
    state_float_values=2*m.hidden+m.key_dim*m.val_dim+m.hidden*m.entity_count
    state_bytes=state_float_values*4+m.entity_count  # nominal FP32 reference; run uses FP64
    return dict(source="UCI501 original Beijing stations",source_sha256=source_sha,
                source_scope="2013-03 first512 real chronological hours/site only, 256 normalization + 256 diagnostic",
                stations=len(SITES),feature_dim=13,observed_channel_values=int(12*256*13-missing),
                missing_channel_values=missing,parameters=sum(p.numel() for p in m.parameters()),
                recurrent_float_values=state_float_values,
                nominal_fp32_state_bytes=state_bytes,
                numerical_run_dtype="float64",
                alpha=summ(alphas),beta=summ(betas),
                fast_norm=summ(torch.stack(fast)),slow_norm=summ(torch.stack(slow)),
                associative_norm=summ(torch.stack(mem)),
                theoretical_retention_only_h64_meanalpha=round(float(alphas.mean()**64),12),
                slots_occupied_per_site=occupancy,
                checked_forecast_accuracy=False,checked_training=False,
                checked_calibration=False,has_logged_candidate_actions=False)


def constructed_stress():
    torch.set_num_threads(1)
    torch.manual_seed(20261026)
    ret = []
    for n in (8,16,32,64):
        dim=16
        keys=torch.nn.functional.normalize(torch.randn(n,dim,dtype=torch.float64),dim=-1)
        vals=torch.randn(n,8,dtype=torch.float64)
        m=torch.zeros(1,8,dim,dtype=torch.float64)
        for k,v in zip(keys,vals):
            m=gated_delta(m,k[None],v[None],
                          torch.ones(1,1,dtype=torch.float64),torch.ones(1,1,dtype=torch.float64))
        recalled=(m[0]@keys.T).T
        rel=(recalled-vals).square().sum().sqrt()/vals.square().sum().sqrt()
        ret.append({"n_items":n,"key_dim":dim,"relative_retrieval_l2":round(float(rel),6)})
    # Benchmark the *reference* prefix only, never assert production latency.
    a=(torch.rand(64,8,32)*.4-.2).double()
    b=torch.randn(64,8,32).double()
    z=torch.randn(8,32).double()
    with torch.no_grad():
        s=time.perf_counter()
        h=z
        for t in range(64):
            h=a[t]*h+b[t]
        serial=time.perf_counter()-s
        s=time.perf_counter()
        full=affine_prefix(a,b,z)
        parallel=time.perf_counter()-s
    return {"purpose":"constructed mechanism diagnostic NOT real-data prediction accuracy",
            "retrieval":ret,"affine_ref_serial_ms":round(serial*1e3,4),
            "affine_ref_prefix_ms":round(parallel*1e3,4),
            "affine_final_max_absolute_error":round(float((full[-1]-h).abs().max()),12),
            "benchmark_scope":"single CPU session, unoptimized Python reference, timing illustrative only"}


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--source",type=Path,required=True)
    args=ap.parse_args()
    torch.manual_seed(20261026)
    result={"study":"LATFUSE001-SOURCE-ONLY-DIAG-001","training_updates":0,
            "evidence_tier":"genuine FIT inputs + constructed mechanism falsifiers; NO capability/evaluation",
            "real_fit":source_diagnostic(args.source),"constructed_mechanisms":constructed_stress(),
            "promotion_eligible":False}
    print("LATFUSE001_DIAG_JSON="+json.dumps(result,sort_keys=True))


if __name__=="__main__":
    main()
