"""Registered CODEVALUE004 workload identity; exact bootstrap owns both T4s.

The reviewed bootstrap invokes run004.run(data,out,max_seconds=7020) directly.
Do not wrap it in the generic per-seed launcher, which would duplicate GPU owners.
"""
from pathlib import Path
import runpy
import sys

if __name__=='__main__':
    root=Path(__file__).resolve().parents[2]
    script=root/'research/commercial-value/codevalue004/run004.py'
    sys.path.insert(0,str(script.parent))
    runpy.run_path(str(script),run_name='__main__')
