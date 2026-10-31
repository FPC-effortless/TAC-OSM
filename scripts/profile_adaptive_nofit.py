#!/usr/bin/env python3
"""No-fit measured CPU loop time. Not science, FLOP parity or energy savings."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import statistics
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from diagnose_adaptive_byte_token_nofit import SOURCE_BLOBS, git_blob


def profile(source_root: Path, *, max_bytes=256, repeats=5):
    import torch
    from tac_osm.adaptive_byte_token import AdaptiveByteTokenGRU
    from tac_osm.patch_sparse_byte_probe import PatchSparseByteGRU
    from tac_osm.causal_byte_probe import CausalByteGRU, predecision_examples
    if not (32 <= max_bytes <= 1024 and 2 <= repeats <= 15):
        raise ValueError('bounded no-fit profiler only')
    torch.set_num_threads(1)
    torch.manual_seed(20261031)
    adaptive=AdaptiveByteTokenGRU().eval()
    sparse=PatchSparseByteGRU().eval()
    dense=CausalByteGRU().eval()
    files=[]
    with torch.no_grad():
        for name, expected in SOURCE_BLOBS.items():
            path=source_root/name
            if path.is_symlink() or not path.is_file():
                raise ValueError('missing publisher source: '+name)
            source=path.read_bytes()
            if git_blob(source)!=expected:
                raise ValueError('source blob mismatch, fail closed: '+name)
            raw=source[:max_bytes]
            x, features, _=predecision_examples(raw, 'sensory_v2')
            arms={
                'dense_sensory_v2': lambda: dense(x[None],features[None]),
                'sparse_sensory_v2': lambda: sparse(x[None],variant='sensory_v2'),
                'adaptive_byte': lambda: adaptive(x[None],force='byte'),
                'adaptive_token': lambda: adaptive(x[None],force='token'),
                'adaptive_untrained': lambda: adaptive(x[None]),
            }
            times={}
            # Every arm sees exact same initial prefix, same CPU. Warmups don't fit.
            for arm,run in arms.items():
                run()
                series=[]
                for _ in range(repeats):
                    start=time.perf_counter_ns()
                    run()
                    series.append((time.perf_counter_ns()-start)/1e6)
                times[arm]={'median_ms':statistics.median(series),'min_ms':min(series),
                            'repeat_count':repeats}
            files.append({'file':name,'bytes':len(raw),'times':times})
    aggregates={name:statistics.median(f['times'][name]['median_ms'] for f in files)
               for name in files[0]['times']}
    return {'id':'SENSE-TOK-004-NOFIT-CPU','authorization':'NO_FIT_ZERO_OPTIMIZER_STEPS',
            'publisher':'CPython v3.13.5 6 Git blob matches','threads':1,'hardware':'current_local_cpu',
            'sample_bytes':max_bytes,'repeats':repeats,'input_files':len(files),
            'median_ms_across_sources':aggregates,'files':files,
            'warning':'Uncontrolled Python/CPU profiling; no GPU, trained utility, amortized prefill, decoder optimization or FLOP parity; all source prefixes exposed DEV diagnostics.'}


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--source-root',type=Path,default=Path('/usr/lib/python3.13'))
    ap.add_argument('--output',type=Path)
    args=ap.parse_args()
    r=profile(args.source_root)
    encoded=json.dumps(r,indent=2,sort_keys=True)+'\n'
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(encoded,encoding='utf8')
    else:print(encoded,end='')

if __name__=='__main__':main()
