#!/usr/bin/env python3
"""Run the fused TAC-OSM ablation research chain."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from tac_osm.fusion.benchmark import run_suite, write_results
from tac_osm.fusion.gates import all_gates

def main() -> int:
    parser=argparse.ArgumentParser()
    parser.add_argument("--suite",default="smoke")
    parser.add_argument("--steps",type=int,default=24)
    parser.add_argument("--seed",type=int,default=0)
    parser.add_argument("--output",default="")
    args=parser.parse_args()
    if args.steps < 1: parser.error("--steps must be >= 1")
    gates=all_gates()
    if not all(g.passed for g in gates):
        print(json.dumps({"status":"gate_failed","gates":[{"name":g.name,"passed":g.passed,"detail":g.detail} for g in gates]},indent=2,sort_keys=True))
        return 2
    metrics=run_suite(args.suite,seed=args.seed,n_steps=args.steps)
    payload={"status":"ok","suite":args.suite,"seed":args.seed,"steps":args.steps,"gates":[{"name":g.name,"passed":g.passed} for g in gates],"results":[m.to_dict() for m in metrics]}
    if args.output:
        write_results(metrics,args.output)
        payload["output"]=str(Path(args.output))
    print(json.dumps(payload,indent=2,sort_keys=True))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
