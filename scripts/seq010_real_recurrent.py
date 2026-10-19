#!/usr/bin/env python3
"""SEQ010 exploratory: train GRU and TAC-inspired gated readout on REAL SCANIA.

Only official publisher TRAIN operational readouts are opened; an internal
vehicle-heldout cohort is used for exploratory scoring. No official validation
or test bodies are opened. Synthetic data is permitted only in software tests.
"""
from __future__ import annotations
import argparse
from collections import defaultdict, deque
import csv
import hashlib
import json
from pathlib import Path
import random
import sys
import time
import urllib.request
import zipfile
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

SOURCE_URL = "https://api.researchdata.se/dataset/2024-34/3/file/zip"
TRAIN_CSV = "train_operational_readouts.csv"
SOURCE_DOI = "10.5878/bnh5-ka77"
N_FEATURES = 24
HISTORY = 8
TRAIN_STEPS = 200
BATCH = 64
SEEDS = (810101, 810102)
MODELS = ("gru", "tac_gated_readout")
HIDDEN = 32
SPLIT_SALT = "seq010-v1-internal-split-20261019"
MAX_ZIP_BYTES = 2_600_000_000
MAX_SOURCE_BYTES = 2_000_000_000

def digest_partition(vehicle_id: str) -> bool:
    # True => train-internal validation. Decided solely by vehicle identity.
    digest = hashlib.sha256((SPLIT_SALT + ":" + vehicle_id).encode()).digest()
    return int.from_bytes(digest[:8], "big") % 5 == 0

def sample_window(vehicle_id: str, index: int) -> bool:
    digest = hashlib.sha256(f"seq010-sample:{vehicle_id}:{index}".encode()).digest()
    return int.from_bytes(digest[:8], "big") % 31 == 0

def signed_log(raw):
    return np.sign(raw) * np.log1p(np.abs(raw))

def acquire_training_csv(root: Path):
    path = root / "official-v3.zip"
    hash_zip = hashlib.sha256()
    count = 0
    request = urllib.request.Request(
        SOURCE_URL, headers={"User-Agent":"TAC-OSM-SEQ010-real/1.0"})
    try:
        with urllib.request.urlopen(request, timeout=120) as response, path.open("wb") as out:
            if response.status != 200:
                raise ValueError("Publisher ZIP HTTP non-200")
            while (chunk := response.read(2*1024*1024)):
                count += len(chunk)
                if count > MAX_ZIP_BYTES: raise ValueError("Excess ZIP size")
                hash_zip.update(chunk)
                out.write(chunk)
        file = root / TRAIN_CSV
        with zipfile.ZipFile(path) as archive:
            matches = [i for i in archive.infolist()
                       if Path(i.filename).name == TRAIN_CSV]
            if len(matches) != 1: raise ValueError("Missing/ambiguous TRAIN telemetry")
            info = matches[0]
            if not 0 < info.file_size <= MAX_SOURCE_BYTES:
                raise ValueError("Invalid TRAIN telemetry size")
            h = hashlib.sha256()
            size = 0
            with archive.open(info) as inp, file.open("wb") as dst:
                while (chunk := inp.read(2*1024*1024)):
                    size += len(chunk)
                    if size > MAX_SOURCE_BYTES: raise ValueError("Exceeded file bound")
                    h.update(chunk)
                    dst.write(chunk)
            if size != info.file_size: raise ValueError("Truncated publisher TRAIN member")
        return file, {"publisher_zip_url": SOURCE_URL, "dataset_doi": SOURCE_DOI,
                      "dataset_version": 3, "zip_sha256": hash_zip.hexdigest(),
                      "zip_bytes": count, "train_member_sha256": h.hexdigest(),
                      "train_member_bytes":size,"test_entries_read":False,
                      "publisher_validation_entries_read":False}
    finally:
        path.unlink(missing_ok=True)

def parse_sensor_vector(row, indices):
    z = np.full((len(indices),), np.nan, dtype=np.float32)
    for j, i in enumerate(indices):
        t = row[i].strip()
        if t and t not in ("NA","NaN","nan","null","None"):
            val = float(t)
            if np.isfinite(val): z[j] = np.sign(val)*np.log1p(abs(val))
    return z

def window(history):
    # history contains prior readouts only; never the target at its last index.
    obs = [o for _,o in list(history)[-HISTORY:]]
    x = np.full((HISTORY, len(obs[0])), np.nan, dtype=np.float32)
    x[-len(obs):] = np.stack(obs)
    return x

def load_real_windows(csv_path):
    with csv_path.open(newline="", encoding="utf-8-sig") as f:
        columns = next(csv.reader(f))
    if len(set(columns)) != len(columns) or "vehicle_id" not in columns or "time_step" not in columns:
        raise ValueError("Wrong training schema")
    ix_id, ix_time = columns.index("vehicle_id"), columns.index("time_step")
    sensor_ix = [i for i,c in enumerate(columns) if c not in ("vehicle_id","time_step")][:N_FEATURES]
    if len(sensor_ix) != N_FEATURES: raise ValueError("Insufficient real sensor features")
    sensor_names = [columns[i] for i in sensor_ix]
    states = defaultdict(lambda:deque(maxlen=HISTORY+1))
    counts = defaultdict(int)
    training = []
    all_ids=set()
    row_count=0
    with csv_path.open(newline="", encoding="utf-8-sig") as f:
        reader=csv.reader(f)
        if next(reader) != columns: raise ValueError("Header changed")
        for row in reader:
            if len(row)!=len(columns):raise ValueError("Invalid row length")
            vehicle=row[ix_id]
            if not vehicle:raise ValueError("Missing vehicle identity")
            t=float(row[ix_time])
            if not np.isfinite(t):raise ValueError("Invalid timestamp")
            value=parse_sensor_vector(row,sensor_ix)
            history=states[vehicle]
            if history and not t > history[-1][0]:
                raise ValueError("Non-increasing within-vehicle time: "+vehicle)
            counts[vehicle]+=1
            all_ids.add(vehicle)
            if not digest_partition(vehicle) and len(history) >= 1 and sample_window(vehicle, counts[vehicle]):
                training.append((window(history), value))
            history.append((t,value))
            row_count+=1
    # Pre-registered inclusion of the final transition for each TRAIN vehicle
    # only; truly held-out vehicle IDs contribute no training examples.
    validation=[]
    for vehicle,history in states.items():
        if len(history)<2: continue
        pair=(window(list(history)[:-1]),history[-1][1])
        (validation if digest_partition(vehicle) else training).append(pair)
    training = [(a,b) for a,b in training if np.isfinite(a[-1]).any() and np.isfinite(b).any()]
    if len(training) < 10000 or len(validation) < 2000:
        raise ValueError("Too few real vehicle windows")
    train_ids={v for v in all_ids if not digest_partition(v)}
    val_ids=all_ids-train_ids
    if train_ids & val_ids:raise ValueError("Train/heldout split overlap")
    def stack(records):
        return (np.stack([x for x,_ in records]),np.stack([y for _,y in records]))
    trainx,trainy=stack(training)
    valx,valy=stack(validation)
    if not np.isfinite(valx[:,-1,:]).any() or not np.isfinite(valy).any():
        raise ValueError("No observed validation readouts")
    return (trainx,trainy), (valx,valy), {
        "train_vehicles":len(train_ids), "internal_heldout_vehicles":len(val_ids),
        "original_train_vehicle_count":len(all_ids),"readout_rows":row_count,
        "train_windows":len(trainx),"internal_heldout_windows":len(valx),
        "feature_names":sensor_names,"feature_count":len(sensor_names),
        "vehicle_split":"sha256(vehicle_id + fixed salt) heldout 20%",
        "official_validation_used":False,"official_test_used":False,
    }

def prepare(inputs,targets):
    last=inputs[:,-1,:]
    last_ok=np.isfinite(last)
    target_ok=np.isfinite(targets)
    mask=(last_ok & target_ok)
    if not mask.any():raise ValueError("No paired sensor targets")
    inp=np.concatenate((np.clip(np.nan_to_num(inputs,nan=0),-12,12)/12,
                        np.isfinite(inputs).astype(np.float32)),axis=-1)
    delta=np.nan_to_num(targets-np.where(last_ok,last,0),nan=0)
    return (torch.from_numpy(inp.astype(np.float32)),
            torch.from_numpy(delta.astype(np.float32)),
            torch.from_numpy(mask.astype(np.bool_)),
            last, targets, mask)

class ResidualRecurrent(nn.Module):
    def __init__(self,kind):
        super().__init__()
        if kind not in MODELS:raise ValueError(kind)
        self.kind=kind
        self.gru=nn.GRU(2*N_FEATURES,HIDDEN,batch_first=True)
        if kind=="tac_gated_readout":
            self.delta=nn.Linear(HIDDEN+2*N_FEATURES,HIDDEN)
            self.gate=nn.Linear(HIDDEN+2*N_FEATURES,HIDDEN)
        self.head=nn.Linear(HIDDEN,N_FEATURES)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)
    def forward(self,x):
        state=self.gru(x)[0][:,-1,:]
        if self.kind=="tac_gated_readout":
            j=torch.cat((state,x[:,-1,:]),-1)
            state=state+torch.sigmoid(self.gate(j))*torch.tanh(self.delta(j))
        return self.head(state)

def score(model,prepared):
    x,delta,mask=prepared[:3]
    model.eval()
    errors=[]
    counts=[]
    with torch.inference_mode():
        for i in range(0,len(x),256):
            xx,yy,mm=x[i:i+256],delta[i:i+256],mask[i:i+256]
            pred=model(xx)
            err=torch.abs(pred-yy).double()
            errors.append((err*mm).sum(0))
            counts.append(mm.sum(0))
    total=torch.stack(errors).sum(0).numpy()
    n=torch.stack(counts).sum(0).numpy()
    if (n<100).any():raise ValueError("Some features lack enough heldout targets")
    return {"macro_mae":float(np.mean(total/n)),"micro_mae":float(total.sum()/n.sum()),
            "min_sensor_targets":int(n.min())}

def persistence_score(prepared):
    _,delta,mask=prepared[:3]
    err=delta.abs().double()
    m=mask.double()
    total=(err*m).sum(0).numpy()
    n=m.sum(0).numpy()
    return {"macro_mae":float(np.mean(total/n)),"micro_mae":float(total.sum()/n.sum()),
            "min_sensor_targets":int(n.min())}

def train_model(kind, seed, train, validation, root, steps=TRAIN_STEPS):
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    model=ResidualRecurrent(kind)
    opt=torch.optim.AdamW(model.parameters(),lr=0.001,weight_decay=0.0001)
    gen=torch.Generator().manual_seed(seed+1000)
    x,delta,mask=train[:3]
    loss_curve=[]
    tic=time.monotonic()
    for step in range(steps):
        index=torch.randint(0,len(x),(BATCH,),generator=gen)
        xb,db,mb=x[index],delta[index],mask[index]
        model.train()
        prediction=model(xb)
        # Huber loss on signed-log residuals, normalized by observed entries.
        element=F.smooth_l1_loss(prediction,db,beta=0.1,reduction="none")
        loss=(element*mb).sum()/mb.sum().clamp_min(1)
        if not torch.isfinite(loss).item():raise ArithmeticError("Nonfinite training loss")
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
        opt.step()
        if step+1 in (1,steps//2,steps):
            loss_curve.append({"step":step+1,"loss":float(loss.detach())})
    results=score(model,validation)
    weights=root/"checkpoints"/f"{kind}-{seed}.pt"
    weights.parent.mkdir(parents=True,exist_ok=True)
    torch.save({"kind":kind,"seed":seed,"weights":model.state_dict(),"steps":steps},weights)
    return {"kind":kind,"seed":seed,"steps":steps,"parameters":sum(p.numel() for p in model.parameters()),
            "heldout":results,"loss_curve":loss_curve,
            "training_wall_seconds":round(time.monotonic()-tic,3),
            "weights":str(weights.relative_to(root)),
            "checkpoint_sha256":hashlib.sha256(weights.read_bytes()).hexdigest()}

def run(root, steps=TRAIN_STEPS, seeds=SEEDS):
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    archive=root/"official-v3.zip"
    try:
        file,source=acquire_training_csv(root)
    finally:
        archive.unlink(missing_ok=True)
    (root/"source.json").write_text(json.dumps(source,sort_keys=True,indent=2)+"\n")
    tr,va,cohort=load_real_windows(file)
    prepared_train=prepare(*tr)
    prepared_val=prepare(*va)
    baseline=persistence_score(prepared_val)
    report={"study":"SEQ010","stage":"TRAIN_INTERNAL_EXPLORATORY_REAL",
            "source":source, "cohort":cohort,"baseline":{"persistence":baseline},
            "recurrent":[],"official_publisher_validation_used":False,
            "official_publisher_test_used":False,"commercial_qualified":False}
    for seed in seeds:
        for kind in MODELS:
            r=train_model(kind,seed,prepared_train,prepared_val,root,steps)
            report["recurrent"].append(r)
            print(json.dumps(r,sort_keys=True),flush=True)
    path=root/"results.json"
    path.write_text(json.dumps(report,sort_keys=True,indent=2)+"\n")
    print(json.dumps({"status":"REAL_INTERNAL_VALIDATION_DONE","baseline":baseline,
                      "fits":len(report["recurrent"]),"cohort":cohort},sort_keys=True),flush=True)
    return report

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--output",default="results/seq010")
    args=p.parse_args()
    run(args.output)

if __name__=="__main__":
    main()
