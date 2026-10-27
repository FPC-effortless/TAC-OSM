"""PAR-TIME-002 authentic UCI Beijing air sensor source adapter.

Source is the publisher PRSA2017 original archive pinned by SHA256; only
original 2013-2014 records are parsed. No invented actions: the model's
historical 'observed_action' input is explicitly an observation-availability
mask plus a legally known hour-of-day feature. This adapter demonstrates
physical next-observation prediction, NOT action-conditioned causality.

No future numerical target is an input, no future-based row admission, no
negative-event selection, no protected 2015-2017 numeric target access.
Numeric source years 2015-2017 are never parsed. Historical 2014 development
is already exposed elsewhere and is NEVER independent generalization proof.
"""
from __future__ import annotations
import csv
import datetime as dt
import hashlib
import io
import math
import zipfile
from dataclasses import dataclass
from pathlib import Path

import torch

SHA256 = "d1b9261c54132f04c374f762f1e5e512af19f95c95fd6bfa1e8ac7e927e3b0b8"
SITES = ("Aotizhongxin","Changping","Dingling","Dongsi","Guanyuan",
         "Gucheng","Huairou","Nongzhanguan","Shunyi","Tiantan",
         "Wanliu","Wanshouxigong")
CHANNELS = ("PM2.5","PM10","SO2","NO2","CO","O3",
            "TEMP","PRES","DEWP","RAIN","WSPM")
WINDS = ("N","NNE","NE","ENE","E","ESE","SE","SSE","S","SSW","SW","WSW",
         "W","WNW","NW","NNW")
START = dt.datetime(2013,3,1)
NORMALIZE_STOP = dt.datetime(2014,3,1)
FIT_START = NORMALIZE_STOP
FIT_STOP = dt.datetime(2014,4,18)
EXPOSED_START = dt.datetime(2014,5,1)
EXPOSED_STOP = dt.datetime(2014,6,10)
PARSE_STOP = dt.datetime(2015,1,1)
HISTORY = 12

def hour(when):
    return int((when - START).total_seconds() // 3600)

def read_source(path: Path):
    if hashlib.sha256(path.read_bytes()).hexdigest() != SHA256:
        raise RuntimeError("publisher source ZIP has wrong immutable SHA")
    expected = hour(PARSE_STOP)
    records=[]
    with zipfile.ZipFile(path) as z:
        for site in SITES:
            matches=[n for n in z.namelist()
                     if n.endswith(f"PRSA_Data_{site}_20130301-20170228.csv")]
            if len(matches)!=1:
                raise RuntimeError("station original source CSV missing/ambiguous")
            site_rows=[]
            with z.open(matches[0]) as fp:
                for row in csv.DictReader(io.TextIOWrapper(fp,encoding="utf-8-sig")):
                    # MUST break before reading any protected year numeric columns.
                    year=int(row["year"])
                    if year>=2015:
                        break
                    if row["station"]!=site:
                        raise RuntimeError("station identity mismatch")
                    stamp=dt.datetime(year,int(row["month"]),int(row["day"]),int(row["hour"]))
                    if hour(stamp)!=len(site_rows):
                        raise RuntimeError("untrusted source chronological index/gap")
                    vals=[float(row[k]) if row[k]!="NA" else float("nan")
                          for k in CHANNELS]
                    w=row["wd"]
                    if w=="NA":
                        vals.extend((float("nan"),float("nan")))
                    else:
                        if w not in WINDS:
                            raise RuntimeError("unknown source wind direction")
                        theta=2*math.pi*WINDS.index(w)/16
                        vals.extend((math.sin(theta),math.cos(theta)))
                    if not all(math.isfinite(v) or math.isnan(v) for v in vals):
                        raise RuntimeError("nonfinite infinite sensor reading")
                    if any(v<0 for v in vals[:6] if math.isfinite(v)):
                        raise RuntimeError("negative source pollutant")
                    site_rows.append(vals)
            if len(site_rows)!=expected:
                raise RuntimeError("original 2013/2014 source length mismatch")
            records.append(site_rows)
    raw=torch.tensor(records,dtype=torch.float32)
    x=raw.clone()
    x[:,:,:6]=torch.where(torch.isfinite(x[:,:,:6]),
                          torch.log1p(x[:,:,:6].clamp_min(0)),x[:,:,:6])
    early=x[:,:hour(NORMALIZE_STOP)]
    early_ok=torch.isfinite(early)
    count=early_ok.sum((0,1)).clamp_min(1)
    center=torch.where(early_ok,early,0).sum((0,1))/count
    spread=(torch.where(early_ok,early-center,0).square().sum((0,1))/count).clamp_min(1e-6).sqrt()
    normalized=(x-center)/spread
    valid=torch.isfinite(normalized)
    values=torch.where(valid,normalized,torch.zeros_like(normalized))
    # Authentic event admission criterion uses ONLY observations already
    # recorded at that instant, never future target presence or level.
    event_eligible=valid[:,:,0] & (valid.sum(-1)>=12)
    events=[]
    for s in range(12):
        times=torch.nonzero(event_eligible[s],as_tuple=False).flatten().tolist()
        events.append(times)
    return {"raw":raw,"values":values,"valid":valid,"events":events,
            "center":center,"spread":spread}

def chronology(src, *, history=HISTORY):
    if history != HISTORY:
        raise ValueError("registered source history changed")
    all_events=[]
    train=[]
    exposed=[]
    for s,times in enumerate(src["events"]):
        for idx in range(history-1,len(times)):
            t=times[idx]
            if t+1>=hour(PARSE_STOP): continue
            # Skip overlaps into normalization or exposed development targets.
            if hour(FIT_START)<=t<hour(FIT_STOP)-1:
                train.append((s,idx,t))
            elif hour(EXPOSED_START)<=t<hour(EXPOSED_STOP)-1:
                # Deterministic source-clock every 48h, NEVER target-based.
                if (t-hour(EXPOSED_START))%48==0:
                    exposed.append((s,idx,t))
        all_events.append(times)
    if len(train)<100 or len(exposed)<10:
        raise RuntimeError("insufficient genuine selected source cases")
    if max(t for _,_,t in train)+1>=min(t for _,_,t in exposed):
        raise RuntimeError("source outcome temporal overlap")
    return train,exposed

def batch_from_cases(src,cases,*,history=HISTORY):
    b=len(cases)
    if b<1 or history!=HISTORY:
        raise ValueError("nonempty registered source cases required")
    t_idx=torch.tensor([src["events"][s][ix-history+1:ix+1]
                       for s,ix,_ in cases],dtype=torch.long)
    if t_idx.shape!=(b,history):
        raise RuntimeError("malformed genuine chronological history")
    for i,(s,ix,t) in enumerate(cases):
        if t_idx[i,-1].item()!=t or t_idx[i].diff().le(0).any().item():
            raise RuntimeError("ordered per-station evidence breached")
    sites=torch.tensor([s for s,_,_ in cases],dtype=torch.long)
    sid=sites[:,None].expand(b,history)
    v=src["values"][sid,t_idx]
    available=src["valid"][sid,t_idx]
    observed_frac=available.float().mean(-1,keepdim=True)
    obs=torch.cat((v,observed_frac),-1)
    # The model argument is named observed_action for historic API reasons.
    # It carries ONLY real observation-presence flags and known clock phase.
    hour_phase=torch.sin((t_idx%24).float()*2*math.pi/24).unsqueeze(-1)
    availability_covariates=torch.cat((available.float(),hour_phase),-1)
    dt_hours=torch.zeros(b,history,dtype=torch.float32)
    dt_hours[:,1:]=t_idx.diff(dim=1).float()
    if (dt_hours[:,1:]<=0).any().item():
        raise RuntimeError("source event chronology invalid")
    anchor=torch.tensor([t for _,_,t in cases],dtype=torch.long)
    yt=src["values"][sites,anchor+1]
    future_valid=src["valid"][sites,anchor+1]
    target=torch.cat((yt,future_valid.float().mean(-1,keepdim=True)),-1)
    mask=torch.cat((future_valid,torch.ones(b,1,dtype=torch.bool)),-1)
    writes=torch.ones(b,history,dtype=torch.bool)
    return obs,availability_covariates,dt_hours,writes,target,mask

def score(pred,target,mask):
    # Outcome accessed only AFTER called prediction. Missing physical channels
    # cannot contribute an invented zero label to the loss.
    loss=torch.nn.functional.smooth_l1_loss(
        pred[:,:13],target[:,:13],reduction="none"
    )
    physical=(loss*mask[:,:13]).sum()/mask[:,:13].sum().clamp_min(1)
    available=(pred[:,13]-target[:,13]).square().mean()
    return physical+0.1*available
