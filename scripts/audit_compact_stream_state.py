#!/usr/bin/env python3
"""Original untrained R2 carried-state audit, not model quality evaluation."""
import argparse
from dataclasses import asdict, fields, is_dataclass
import hashlib
import json
from pathlib import Path
import platform
import resource
import statistics
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import torch
from tac_osm.csl125_r2 import CSL125R2
from tac_osm.compact_stream_state import compact_stream_state, stream_state_storage


def exact(a,b):
    if type(a) is not type(b):
        return False
    if isinstance(a,torch.Tensor):
        return (a.shape==b.shape and a.dtype==b.dtype and a.device==b.device and
                torch.equal(a.contiguous().view(torch.uint8),b.contiguous().view(torch.uint8)))
    if is_dataclass(a):
        return all(exact(getattr(a,f.name),getattr(b,f.name)) for f in fields(a))
    if isinstance(a,(tuple,list)):
        return len(a)==len(b) and all(exact(x,y) for x,y in zip(a,b))
    if isinstance(a,dict):
        return a.keys()==b.keys() and all(exact(a[k],b[k]) for k in a)
    return a==b


def run(output):
    if output.exists():
        raise FileExistsError('preserve prior audit')
    protocol=json.loads((ROOT/'research/compact_state_20261108/PROTOCOL.json').read_text())
    for path,pin in protocol['source_pins'].items():
        if hashlib.sha256((ROOT/path).read_bytes()).hexdigest()!=pin:
            raise ValueError('original source pin mismatch: '+path)
    torch.set_num_threads(1);torch.manual_seed(71)
    model=CSL125R2().eval()
    count=sum(p.numel() for p in model.parameters())
    assert count==123822574
    def weights_digest():
        h=hashlib.sha256()
        for name,t in model.state_dict().items():
            h.update(name.encode());h.update(t.contiguous().numpy().tobytes())
        return h.hexdigest()
    weights_before=weights_digest()
    rows=[]
    with torch.no_grad():
        for depth in model.config.rungs:
            for length in (64,128):
                tokens=torch.arange(length+1).remainder(model.config.vocab_size).reshape(1,-1)
                _,state=model(tokens[:,:length],depth=depth,stream_ids=('fixture',),all_rungs=True)
                before=stream_state_storage(state)
                detached=stream_state_storage(state.detached())
                rng=torch.get_rng_state().clone()
                compact=compact_stream_state(state)
                after=stream_state_storage(compact)
                unchanged=exact(state,compact)
                for _ in range(2):
                    compact_stream_state(state)
                samples=[]
                for _ in range(7):
                    begin=time.perf_counter_ns(); temporary=compact_stream_state(state)
                    samples.append((time.perf_counter_ns()-begin)/1e6)
                    del temporary
                a,sa=model(tokens[:,length:],depth=depth,state=state,stream_ids=('fixture',),all_rungs=True)
                b,sb=model(tokens[:,length:],depth=depth,state=compact,stream_ids=('fixture',),all_rungs=True)
                checks={'state_bitwise_identical':unchanged,'all_emitted_outputs_bitwise_identical':exact(a,b),
                        'next_state_bitwise_identical':exact(sa,sb),
                        'compact_retained_equals_logical':after['retained_storage_bytes']==after['logical_bytes'],
                        'detach_did_not_compact':detached==before,
                        'copy_uses_no_rng':torch.equal(rng,torch.get_rng_state()),
                        'retained_storage_reduced':after['retained_storage_bytes']<before['retained_storage_bytes']}
                rows.append({'depth':depth,'prefix_tokens':length,'before':before,'compact':after,
                    'source_plus_one_copy_storage_bytes':before['retained_storage_bytes']+after['retained_storage_bytes'],
                    'copy_samples_ms':samples,'copy_median_ms':statistics.median(samples),'checks':checks,
                    'status':'PASS' if all(checks.values()) else 'FAIL'})
                del state,compact,a,b,sa,sb
    weights_after=weights_digest()
    if weights_before!=weights_after:
        raise RuntimeError('model weights changed')
    result={'study_id':'COMPACT_STATE_20261108','execution_level':'internal_research',
        'status':'PASS' if all(r['status']=='PASS' for r in rows) else 'FAIL',
        'qualification':'engineering fixtures only; not independently qualified; not learned capability',
        'source_pins':protocol['source_pins'],'config':asdict(model.config),'untrained_parameters':count,
        'model_weights_changed':False,'weights_before_sha256':weights_before,'weights_after_sha256':weights_after,
        'optimizer_steps':0,'GPU_minutes':0,'corpus_or_TEST_access':False,
        'rows':rows,'torch':torch.__version__,'python':platform.python_version(),
        'process_lifetime_peak_RSS_KiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        'RSS_scope':'Linux process lifetime high-water including weights/tests/runtime; NOT isolated copy peak or device RAM',
        'allocator_reserved_and_isolated_peak_copy_RAM':'NOT_RUN',
        'math':'Identity value copy with identity Jacobian tested; inherited recurrence/IEEE/AD/PNDS qualification OPEN',
        'device':{k:'NOT_RUN' for k in ('target','operators','numerical_agreement','physical_p50_p95','RAM','energy','airplane_mode')},
        'limits':'State-only backing bytes exclude weights, allocations and autograd. Old aliases/attached graphs may retain allocations; release old state. Copy adds time and transient storage. No latency superiority or model intelligence claim.'}
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    print(json.dumps({'status':result['status'],'rows':len(rows),'untrained_parameters':count}))
    if result['status']!='PASS':
        raise SystemExit(1)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    run(p.parse_args().output)
