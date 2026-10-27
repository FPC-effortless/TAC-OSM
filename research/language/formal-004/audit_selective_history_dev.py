#!/usr/bin/env python3
"""TAC-LANG-REPAIR-018 replay on EXACT archived Vim9.1 DEV windows.

Purpose: evaluate a predeclared fallback that (a) trusts 64-token historical
queries, the original training target; (b) abstains and emits a FIT-only
smoothed unigram distribution on untrained 96-token historical queries.

This is previously exposed DEVELOPMENT evaluation, NOT independent TEST,
NOT proof of improved useful long-history memory. No checkpoint updates.

Example:
  python audit_selective_history_dev.py \
    --archive-dir /path/to/extracted/tac_lang_gate003 \
    --vim-root /usr/share/vim/vim91/doc \
    --output /tmp/selective_history_dev.json
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from parallel_tac_language_gate_original import ParallelTACLanguage
from selective_history import (TRAINED_HISTORICAL_LAG, SMOOTHING,
                               fit_only_frequency_prior, query_selective_history)

ORIGINAL_SHA = "835212d74cc0d2d4d22ff7e75beb12771586cce58f2e63943690e5d859417457"
PARAMETERS = 100680
SEEDS = (1701,1702,1703,1704,1705)
MODES = ("tanh","silu")
V = 128
CONTEXT = 128
DEV_WINDOWS_PER_DOC = 8

def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

def ascii_doc(raw: bytes) -> np.ndarray:
    return np.frombuffer(bytes(b if b < 128 else 63 for b in raw),
                         dtype=np.uint8).astype(np.int64)

def make_batches(docs: list[np.ndarray]) -> list[torch.Tensor]:
    samples=[]
    for doc in docs:
        if len(doc) <= CONTEXT + 2:
            raise ValueError("DEV document shorter than frozen context")
        # Exact pre-exposed source DEV sampler from original experiment.
        starts=np.linspace(0,len(doc)-CONTEXT-2,DEV_WINDOWS_PER_DOC,dtype=np.int64)
        for i in starts:
            samples.append(torch.from_numpy(doc[i:i+CONTEXT]))
    if len(samples) != 48:
        raise ValueError("incorrect frozen DEV windows")
    return [torch.stack(samples[i:i+8]) for i in range(0,48,8)]

def replay(archive_dir:Path,vim_root:Path) -> dict:
    torch.set_num_threads(1)
    original=Path(__file__).with_name("parallel_tac_language_gate_original.py")
    if digest(original) != ORIGINAL_SHA or digest(archive_dir/"parallel_tac_language_gate.py") != ORIGINAL_SHA:
        raise ValueError("immutable original Python source SHA mismatch")
    protocol=json.loads((archive_dir/"PROTOCOL.json").read_text())
    archived=json.loads((archive_dir/"RESULTS.json").read_text())
    if protocol != archived["protocol"]:
        raise ValueError("archive protocol/results mismatch")
    if protocol["aux_historical_lag"] != TRAINED_HISTORICAL_LAG or protocol["context"] != CONTEXT:
        raise ValueError("unmatched original training horizon/context")
    if tuple(protocol["seeds"]) != SEEDS or set(protocol["arms"]) != set(MODES):
        raise ValueError("unexpected archived model family")
    if protocol["active_learned_params_per_arm"] != PARAMETERS:
        raise ValueError("parameter budget not source-matched")
    partitions=protocol["source"]["doc_partitions"]
    fit_names=partitions["fit"];dev_names=partitions["dev"]
    if len(fit_names)!=30 or len(dev_names)!=6 or set(fit_names)&set(dev_names):
        raise ValueError("unexpected/reused FIT/DEV partitions")
    for name,sha in protocol["source"]["byte_sha256"].items():
        if "/" in name or digest(vim_root/name) != sha:
            raise ValueError(f"source chapter not authorized or hash mismatch: {name}")
    fit=[ascii_doc((vim_root/n).read_bytes()) for n in fit_names]
    dev=[ascii_doc((vim_root/n).read_bytes()) for n in dev_names]
    prior=fit_only_frequency_prior(fit,SMOOTHING)
    batches=make_batches(dev)
    measurements=[]
    with torch.inference_mode():
      for row in archived["runs"]:
        if row["mode"] not in MODES or row["seed"] not in SEEDS:
            raise ValueError("not a frozen eligible checkpoint")
        file=archive_dir/row["checkpoint_file"]
        if file.name != row["checkpoint_file"] or digest(file) != row["checkpoint_sha256"]:
            raise ValueError("checkpoint SHA256 mismatch")
        checkpoint=torch.load(file,map_location="cpu",weights_only=True)
        if checkpoint["seed"]!=row["seed"] or checkpoint["mode"]!=row["mode"] or (
           checkpoint["sample_sha256"] != row["sample_sha256"]):
            raise ValueError("checkpoint training provenance mismatch")
        torch.manual_seed(row["seed"])
        model=ParallelTACLanguage(decoder=75,position_mode="multiscale",
                                    gate_mode=row["mode"]).eval()
        if sum(p.numel() for p in model.parameters() if p.requires_grad) != PARAMETERS:
            raise ValueError("100680 trainable geometry changed")
        model.load_state_dict(checkpoint["state"],strict=True)
        entry={"mode":row["mode"],"seed":row["seed"],
               "checkpoint_sha256":row["checkpoint_sha256"],"lags":{}}
        for lag in (64,96):
            n=0; old_nll=0.; routed_nll=0.; uniform_nll=0.
            abstained=None
            for tokens in batches:
                _,trace=model(tokens,return_trace=True)
                logits,expected=model.historical_logits(tokens,trace,lag)
                if not torch.equal(expected,tokens[:,:CONTEXT-lag]):
                    raise ValueError("historical source target mismatch")
                old_nll+=float(F.cross_entropy(logits.reshape(-1,V),expected.reshape(-1),
                                                reduction="sum"))
                selected=query_selective_history(model,tokens,trace,lag,prior)
                if abstained is not None and selected.abstained != abstained:
                    raise ValueError("routing changed despite same lag")
                abstained=selected.abstained
                targets=expected.unsqueeze(-1)
                routed_nll+=float(-selected.log_probabilities.gather(-1,targets).sum())
                n+=expected.numel()
                uniform_nll+=expected.numel()*math.log(V)
            archived_loss=row["historical"][str(lag)]["nll"]
            if n != (3072 if lag==64 else 1536):
                raise ValueError("lost source DEV history samples")
            if abs(old_nll/n-archived_loss) >= 1e-5:
                raise ValueError(f"frozen historical loss replay failed: {row['mode']}/{row['seed']}/{lag}")
            if abstained != (lag>TRAINED_HISTORICAL_LAG):
                raise ValueError("untrained lag used memory or trained lag abstained")
            entry["lags"][str(lag)]={
                "n":n,"frozen_source_nll":old_nll/n,"selective_nll":routed_nll/n,
                "uniform_nll":uniform_nll/n,"abstained":abstained,
                "delta_source_minus_selective":(old_nll-routed_nll)/n,
                "original_archive_nll":archived_loss}
        measurements.append(entry)
    if len(measurements)!=10 or set((r["mode"],r["seed"]) for r in measurements) != set(
        (m,s) for m in MODES for s in SEEDS):
        raise ValueError("missing or duplicate original checkpoints")
    summary={}
    for lag in ("64","96"):
        rows=[r["lags"][lag] for r in measurements]
        summary[lag]={
          "n_checkpoints":len(rows), "n_targets_per_checkpoint":rows[0]["n"],
          "source_mean_nll":sum(r["frozen_source_nll"] for r in rows)/len(rows),
          "selective_mean_nll":sum(r["selective_nll"] for r in rows)/len(rows),
          "source_minus_selective_mean":sum(r["delta_source_minus_selective"] for r in rows)/len(rows),
          "selective_wins_in_checkpoint_pairs":sum(r["selective_nll"]<r["frozen_source_nll"] for r in rows),
          "selective_abstentions_in_checkpoint_pairs":sum(r["abstained"] for r in rows)}
    return {
       "study":"TAC-LANG-REPAIR-018", "evaluation_scope":"previously exposed original Vim9.1 DEV, NOT independent TEST",
       "training":"none; immutable frozen 10 original checkpoints",
       "trained_historical_lag":TRAINED_HISTORICAL_LAG,
       "fallback":"FIT-only 128-class token unigram +0.5 per category; nonlearned static prior",
       "metadata_overhead_fp32_bytes":512,"learned_params":PARAMETERS,
       "original_source_sha256":ORIGINAL_SHA,"authorized_chapter_hash_count":len(protocol["source"]["byte_sha256"]),
       "fit_names":fit_names,"dev_names":dev_names,
       "source_nll_better_at_64_and_prior_better_at_96_are_exploratory_dev_observations":True,
       "summary":summary,"runs":measurements}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--archive-dir",type=Path,required=True)
    ap.add_argument("--vim-root",type=Path,required=True)
    ap.add_argument("--output",type=Path,required=True)
    args=ap.parse_args()
    result=replay(args.archive_dir,args.vim_root)
    args.output.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
    print(json.dumps(result["summary"],sort_keys=True,indent=2))

if __name__=="__main__":
    main()
