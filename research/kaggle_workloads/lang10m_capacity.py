"""Registered LANG10M001 no-update capacity entry; bootstrap owns both GPUs."""
from pathlib import Path
import runpy
import sys

if __name__=='__main__':
    root=Path(__file__).resolve().parents[2]
    target=root/'research/language-scaling/lang10m/capacity_profile.py'
    sys.path.insert(0,str(target.parent));runpy.run_path(str(target),run_name='__main__')
