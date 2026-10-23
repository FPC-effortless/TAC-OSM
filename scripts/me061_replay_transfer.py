"""Shared artifact replay, calibration and frozen target-task evaluation."""
import argparse
from pathlib import Path
from scripts.me061_train_reliable import ROOT,build
from scripts.me060_summarize_replay import summarize
from scripts.me060_transfer import reserve,run

if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-cache",type=Path,required=True)
    p.add_argument("--output",type=Path,default=Path("results/me061"))
    p.add_argument("--register-transfer",action="store_true")
    p.add_argument("--transfer",action="store_true")
    a=p.parse_args()
    if a.register_transfer:
        reserve(a.source_cache,ROOT,"me061",("hang_clothes_on_drying_rack",))
    elif a.transfer:run(a.source_cache,a.output,ROOT,build)
    else:summarize(a.output,a.source_cache,build)
