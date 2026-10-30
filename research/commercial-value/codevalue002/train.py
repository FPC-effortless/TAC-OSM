#!/usr/bin/env python3
"""CODEVALUE002: real-source code-domain 500K language model training, no TEST.

One source-pinned training job (arm, seed) at a time; evaluator runs only after
all 9 full model/optimizer/RNG checkpoints are archived and SHA-sealed in Git.
This is research, not source-quality or commercial certification.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import random
import resource
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[3]
CODE = ROOT / "research/commercial-value/codevalue002"
for path in (ROOT, ROOT / "src",
             ROOT / "research/mathematical-engineering/langscale070",
             ROOT / "research/mathematical-engineering/langhead094",
             ROOT / "research/mathematical-engineering/langhead093",
             ROOT / "research/mathematical-engineering/langmem086"):
    if str(path) not in sys.path: sys.path.insert(0, str(path))
from scaled_recurrence import ScaledRecurrenceTAC070
from model094 import SelectiveSSMLanguage094

REG = json.loads((CODE / "REGISTER.json").read_text())
PARAMS = {k: v["params"] for k, v in REG["arms"].items()}
SEEDS = tuple(REG["seeds"])
STEPS = REG["budget"]["updates"]
BATCH = REG["budget"]["batch"]
WINDOW = REG["budget"]["context"]
ARMS = tuple(sorted(PARAMS))

def sha256(b):
    return hashlib.sha256(b).hexdigest()

def gitblob(b):
    return hashlib.sha1(b"blob " + str(len(b)).encode() + b"\0" + b).hexdigest()

class GRU500(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(256, 48)
        self.gru = nn.GRU(48, 211, num_layers=2, batch_first=True)
        self.head = nn.Linear(211, 256)

    def forward(self, x):
        out, _ = self.gru(self.embedding(x))
        return self.head(out)

class SimpleSSM500(SelectiveSSMLanguage094):
    def __init__(self):
        super().__init__()
        self.ffn_up = nn.Linear(128, 931)
        self.ffn_gate = nn.Linear(128, 931)
        self.ffn_out = nn.Linear(931, 256)

    def forward(self, x):
        logits, _ = super().forward(x)
        return logits

def make(arm):
    if arm == "tac": m = ScaledRecurrenceTAC070(scale="p500k")
    elif arm == "gru": m = GRU500()
    elif arm == "ssm": m = SimpleSSM500()
    else: raise ValueError("unregistered architecture")
    n = sum(p.numel() for p in m.parameters() if p.requires_grad)
    d = sum(p.numel() for p in m.parameters())
    if (n, d) != (PARAMS[arm], PARAMS[arm]): raise ValueError(f"trainable/deployed params: {arm} {n} {d}")
    if abs(n / PARAMS["tac"] - 1) > .01: raise ValueError("parameter parity gap >1%")
    return m

def fetch_files(rows):
    files = {}
    for item in rows:
        url = "https://raw.githubusercontent.com/" + item["repo"] + "/" + item["commit"] + "/" + item["path"]
        with urllib.request.urlopen(url, timeout=45) as f: raw = f.read()
        if len(raw) != item["size"] or gitblob(raw) != item["git_blob_sha1"]:
            raise ValueError("original publisher immutable source mismatch: " + item["path"])
        if len(raw) < 129 + 768 + 6:
            raise ValueError("original source too short")
        key = item["repo"] + ":" + item["path"]
        if key in files: raise ValueError("repeated source owner")
        files[key] = raw
    return files

def fit_sources():
    rows = [dict(x, repo=REG["train"]["repo"], commit=REG["train"]["commit"])
            for x in REG["train"]["files"]]
    return fetch_files(rows)

def batches(docs, seed, steps=STEPS):
    rng = np.random.default_rng(seed)
    names = sorted(docs)
    if seed not in SEEDS or steps != STEPS: raise ValueError("unregistered seed/steps")
    for s in range(steps):
        sample = []
        for _ in range(BATCH):
            owner = names[int(rng.integers(len(names)))]
            raw = docs[owner]
            offset = int(rng.integers(len(raw) - (WINDOW+1) + 1))
            sample.append((owner, offset, raw[offset:offset+WINDOW+1]))
        yield s, sample

def schedule_hash(docs, seed):
    h, hb = hashlib.sha256(), hashlib.sha256()
    for _, batch in batches(docs, seed):
        for owner, offset, raw in batch:
            h.update(f"{owner}:{offset};".encode()); hb.update(raw)
    return {"owner_offsets_sha256": h.hexdigest(), "target_windows_sha256": hb.hexdigest()}

def tensor_hash(state):
    h = hashlib.sha256()
    for k,v in sorted(state.items()):
        h.update(k.encode() + b"\0" + v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()

def preflight():
    if REG["study"] != "CODEVALUE002" or REG["status"] != "PREREGISTERED_NOT_TRAINED":
        raise ValueError("changed original preregistration")
    if len(REG["train"]["files"]) != 14 or len(REG["dev"]["files"]) != 6:
        raise ValueError("changed fixed source cohort")
    if REG["budget"]["updates"] != 320 or REG["budget"]["batch"] != 8 or REG["budget"]["context"] != 128:
        raise ValueError("changed training exposure")
    if REG["evaluation"]["offer_prefix_lengths"] != [64, 120] or REG["evaluation"]["greedy_completion_lengths"] != [2, 4, 6]:
        raise ValueError("changed offer policy")
    if sorted(set(row["group"] for row in REG["dev"]["files"])) != ["click", "httpx", "requests"]:
        raise ValueError("missing DEV publisher")
    for a in ARMS: make(a)
    return True

def train(arm, seed, output):
    if output.exists(): raise FileExistsError("do not overwrite original artifacts")
    preflight()
    docs = fit_sources()  # never fetch DEV in training
    planned = schedule_hash(docs, seed)
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.set_num_threads(2)
    model = make(arm).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.01)
    sha_order, sha_bytes = hashlib.sha256(), hashlib.sha256()
    loss_first = loss_last = None
    start = time.process_time()
    for step, rows in batches(docs, seed):
        windows = []
        for owner, offset, raw in rows:
            sha_order.update(f"{owner}:{offset};".encode())
            sha_bytes.update(raw)
            windows.append(list(raw))
        z = torch.tensor(windows, dtype=torch.long)
        logits = model(z[:, :-1])
        if logits.shape != (BATCH, WINDOW, 256): raise ValueError("wrong model logits geometry")
        loss = F.cross_entropy(logits.reshape(-1, 256), z[:, 1:].reshape(-1))
        if not torch.isfinite(loss): raise ValueError("nonfinite real-source loss")
        optimizer.zero_grad(set_to_none=True); loss.backward()
        if any(p.grad is None or not torch.isfinite(p.grad).all() for p in model.parameters()):
            raise ValueError("missing or nonfinite trained gradients")
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        if not math.isfinite(float(norm)): raise ValueError("nonfinite gradients")
        optimizer.step()
        if step == 0: loss_first = float(loss.detach())
        loss_last = float(loss.detach())
        if (step+1)%80 == 0:
            print(json.dumps({"arm":arm, "seed":seed, "update":step+1, "loss":loss_last}),flush=True)
    if planned != {"owner_offsets_sha256": sha_order.hexdigest(), "target_windows_sha256": sha_bytes.hexdigest()}:
        raise ValueError("paired training schedule mismatch")
    output.mkdir(parents=True)
    cp = {
        "study":REG["study"],"arm":arm,"seed":seed,"steps":STEPS,
        "source":"python/cpython pinned original", "sample":planned, "params":PARAMS[arm],
        "model":model.state_dict(),"optimizer":optimizer.state_dict(),
        "torch_rng":torch.get_rng_state(),"numpy_rng":np.random.get_state()[1].tolist(),
        "python_rng":random.getstate(),
    }
    checkpoint = output / "MODEL.pt"
    torch.save(cp, checkpoint)
    raw = checkpoint.read_bytes()
    cold_cp = torch.load(io.BytesIO(raw),map_location="cpu",weights_only=True)
    cold = make(arm).eval();cold.load_state_dict(cold_cp["model"],strict=True)
    model.eval()
    with torch.no_grad():
        ex = torch.tensor([list(next(iter(docs.values()))[:32])],dtype=torch.long)
        if (model(ex)-cold(ex)).abs().max().item() > 1e-6:raise ValueError("cold score replay failed")
    receipt = {
        "study":REG["study"],"arm":arm,"seed":seed,"params":PARAMS[arm],
        "steps":STEPS,"targets":STEPS*BATCH*WINDOW,
        "sample":planned,"checkpoint_sha256":sha256(raw),
        "model_state_sha256":tensor_hash(cold.state_dict()),"fit_loss_first":loss_first,"fit_loss_last":loss_last,
        "train_cpu_seconds":time.process_time()-start,
        "peak_rss_kb":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "gpu_minutes":0,"protected_TEST":"NOT_OPENED","evaluation":"NOT_OPENED","training_flops":"NOT_MATCHED",
        "original_checkpoint_optimizer_rng_cold_replay":"PASS"
    }
    (output / "RECEIPT.json").write_text(json.dumps(receipt,indent=2,sort_keys=True,allow_nan=False)+"\n")
    print(json.dumps({"arm":arm,"seed":seed,"fit_first":loss_first,"fit_last":loss_last,"checkpoint":receipt["checkpoint_sha256"]}),flush=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=ARMS, required=True)
    ap.add_argument("--seed", type=int, choices=SEEDS, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a=ap.parse_args()
    train(a.arm,a.seed,a.out)

if __name__ == "__main__":
    main()
