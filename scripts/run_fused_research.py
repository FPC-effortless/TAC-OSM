#!/usr/bin/env python3
"""Run the fused TAC-OSM ablation research chain."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from tac_osm.fusion.benchmark import memory_decay_profile, run_history_contrast, run_multi_seed_suite, summarize_metrics, write_results
from tac_osm.fusion.gates import all_gates

def _seeds(value: str) -> tuple[int, ...]:
    try:
        seeds=tuple(int(x.strip()) for x in value.split(",") if x.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("seeds must be comma-separated integers") from exc
    if not seeds:
        raise argparse.ArgumentTypeError("at least one seed is required")
    return seeds

def main() -> int:
    parser=argparse.ArgumentParser()
    parser.add_argument("--suite",default="smoke")
    parser.add_argument("--steps",type=int,default=24)
    parser.add_argument("--seeds",type=_seeds,default=(0,1,2,3,4))
    parser.add_argument("--output",default="")
    args=parser.parse_args()
    if args.steps < 1:
        parser.error("--steps must be >= 1")

    gates=all_gates()
    if not all(g.passed for g in gates):
        print(json.dumps({"status":"gate_failed","gates":[{"name":g.name,"passed":g.passed,"detail":g.detail} for g in gates]},indent=2,sort_keys=True))
        return 2

    metrics=run_multi_seed_suite(args.suite,seeds=args.seeds,n_steps=args.steps)
    payload={
        "status":"ok",
        "suite":args.suite,
        "seeds":list(args.seeds),
        "steps":args.steps,
        "gates":[{"name":g.name,"passed":g.passed} for g in gates],
        "summary":summarize_metrics(metrics),
        "history_contrast": [run_history_contrast(seed=seed) .__dict__ for seed in args.seeds],
        "memory_decay": memory_decay_profile(),
        "results":[m.to_dict() for m in metrics],
    }
    if args.output:
        write_results(metrics,args.output)
        payload["output"]=str(Path(args.output))
    print(json.dumps(payload,indent=2,sort_keys=True))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
