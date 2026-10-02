#!/usr/bin/env python3
"""Aggregate the three independently executed E2E-003 arm artifacts."""
from __future__ import annotations

import json
import statistics
from pathlib import Path
import random
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))

from tac_osm.contract import load_contract

ARMS=("explicit_write","explicit_read","explicit_both")
SEEDS=(0,1,2,3,4)


def bootstrap(values:list[float],samples:int=5000,seed:int=20261002)->list[float]:
    rng=random.Random(seed)
    n=len(values)
    means=sorted(
        statistics.fmean(values[rng.randrange(n)] for _ in range(n))
        for _ in range(samples)
    )
    return [means[int(.025*samples)],means[int(.975*samples)-1]]


def main() -> None:
    contract=load_contract("TACOSM-PLM-INTEGRATED-E2E-003")
    contract.require_levels((3,))
    contract.require_seeds(SEEDS)
    contract.require_steps(300)
    contract.require_eval_steps(400)
    if contract.check_consistency():
        raise RuntimeError("contract consistency failure")

    records={}
    for arm in ARMS:
        p=ROOT/"artifacts"/f"TACOSM-PLM-INTEGRATED-E2E-003-{arm}.json"
        data=json.loads(p.read_text())
        if data["arm"]!=arm:
            raise RuntimeError(f"artifact arm mismatch for {arm}")
        if data["experiment_id"]!="TACOSM-PLM-INTEGRATED-E2E-003":
            raise RuntimeError(f"experiment mismatch for {arm}")
        if tuple(data["protocol"]["seeds"]) != SEEDS:
            raise RuntimeError(f"seed mismatch for {arm}")
        if data["protocol"]["steps"] != 300 or data["protocol"]["evaluation_episodes_per_seed"] != 400:
            raise RuntimeError(f"protocol mismatch for {arm}")
        rows=data["seed_results"]
        got=sorted(int(r["seed"]) for r in rows)
        if got!=list(SEEDS):
            raise RuntimeError(f"seed coverage mismatch for {arm}: {got}")
        if len(rows)!=5 or len({r["seed"] for r in rows})!=5:
            raise RuntimeError(f"duplicate/missing seed rows for {arm}")
        records[arm]=data

    provs=[records[a]["provenance"] for a in ARMS]
    pr_heads={p.get("pr_head_sha") for p in provs}
    commits={p.get("git_commit") for p in provs}
    if len(pr_heads)!=1 or "unknown" in pr_heads:
        raise RuntimeError(f"arm PR-head provenance mismatch: {sorted(pr_heads)}")
    if len(commits)!=1 or "unknown" in commits:
        raise RuntimeError(f"arm checkout provenance mismatch: {sorted(commits)}")

    both=records["explicit_both"]
    seed_rows=both["seed_results"]
    q2=[float(r["explicit_both"]) for r in seed_rows]
    no_mem=[float(r["no_memory_explicit_both_q2"]) for r in seed_rows]
    shuffle=[float(r["shuffle_image_explicit_both_q2"]) for r in seed_rows]
    text_only=[float(r["text_only_explicit_both_q2"]) for r in seed_rows]
    image_only=[float(r["image_only_explicit_both_q2"]) for r in seed_rows]
    audio_only=[float(r["audio_only_explicit_both_q2"]) for r in seed_rows]

    output={
        "experiment_id":"TACOSM-PLM-INTEGRATED-E2E-003",
        "status":"measured",
        "provenance":{
            "arm_artifacts":{
                arm:records[arm]["provenance"] for arm in ARMS
            }
        },
        "protocol":{
            "seeds":SEEDS,
            "steps":300,
            "eval_episodes":400,
            "heldout_compositions":both["protocol"]["heldout_compositions"]
        },
        "arm_seed_results":{arm:records[arm]["seed_results"] for arm in ARMS},
        "summary":{
            "explicit_both_q2_mean":statistics.fmean(q2),
            "explicit_both_q2_seed_bootstrap_ci95":bootstrap(q2),
            "explicit_both_min_seed":min(q2),
            "explicit_write_q2_mean":statistics.fmean(float(r["explicit_write"]) for r in records["explicit_write"]["seed_results"]),
            "explicit_read_q2_mean":statistics.fmean(float(r["explicit_read"]) for r in records["explicit_read"]["seed_results"]),
            "no_memory_q2_mean":statistics.fmean(no_mem),
            "shuffle_image_q2_mean":statistics.fmean(shuffle),
            "text_only_q2_mean":statistics.fmean(text_only),
            "image_only_q2_mean":statistics.fmean(image_only),
            "audio_only_q2_mean":statistics.fmean(audio_only),
            "memory_drop":statistics.fmean(q2)-statistics.fmean(no_mem),
            "alignment_drop":statistics.fmean(q2)-statistics.fmean(shuffle),
            "text_only_gap":statistics.fmean(q2)-statistics.fmean(text_only),
            "image_only_gap":statistics.fmean(q2)-statistics.fmean(image_only),
            "audio_only_gap":statistics.fmean(q2)-statistics.fmean(audio_only),
            "primary_pass":statistics.fmean(q2)>=0.80,
            "min_seed_pass":min(q2)>=0.40
        },
        "decision_rule":{
            "primary":"explicit_both mean q2 >= 0.80 and every seed >= 0.40",
            "integrity":"passed before arm execution"
        }
    }
    out=ROOT/"artifacts"/"TACOSM-PLM-INTEGRATED-E2E-003-COMBINED.json"
    out.write_text(json.dumps(output,indent=2,sort_keys=True)+"\n")
    print(json.dumps(output,indent=2,sort_keys=True))


if __name__=="__main__":
    main()
