#!/usr/bin/env python3
"""Inspect existing engineering traces without new training or scientific data."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from tac_osm.development.pnds_conformance import inspect_engineering_report


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args()
    output=Path(a.output)
    if output.exists():raise ValueError('preserve earlier evidence')
    if subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=ROOT,text=True).strip():raise ValueError('freeze source first')
    path=ROOT/'docs/DEV-032-TEST6-ENGINEERING.json'
    result=inspect_engineering_report(json.loads(path.read_text()))
    result.update(experiment='DEV-033 PNDS trace conformance',
        source=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        protocol_sha256=hashlib.sha256((ROOT/'research/dev033/protocol.json').read_bytes()).hexdigest(),
        engineering_report_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        training_performed=False,scientific_test_accessed=False,learned_capability_established=False)
    output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');print(json.dumps(result))
