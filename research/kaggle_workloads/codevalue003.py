"""Dedicated entrypoint for the reviewed private offline CODEVALUE003 bundle.

Unlike the generic seed worker launcher, this coordinator itself owns both
physical GPUs and enforces the global quota deadline. Packaging calls this
entrypoint directly, not through a second two-worker wrapper.
"""
import argparse
from pathlib import Path
import runpy
import sys

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--bundle-root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    coordinator=a.bundle_root/'repo/research/commercial-value/codevalue003/coordinator.py'
    sys.path.insert(0,str(coordinator.parent))
    sys.argv=[str(coordinator),'--data',str(a.bundle_root/'data'),'--out',str(a.output)]
    runpy.run_path(str(coordinator),run_name='__main__')
