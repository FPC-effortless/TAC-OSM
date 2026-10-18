"""Same-weight dense/sparse CPU measurements; allocator bytes are not DRAM."""
import argparse
import json
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import numpy as np
import torch
from tac_osm.development.learned_agent import AgentConfig
from tac_osm.development.temporal_agent import SparseTemporalAgent
from tac_osm.development.learned_agent_world import World
from run_learned_stack_dev055 import write,digest,git


def measure(output,checkpoint=None):
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    checkpoint=Path(checkpoint or ROOT/'results/dev058-main/r0-7581/agent.pt')
    dense=SparseTemporalAgent(AgentConfig(),seed=0,sparse=False,temporal=False).eval()
    loaded=dense.load_state_dict(torch.load(checkpoint,weights_only=True,map_location='cpu'),strict=False)
    assert not loaded.missing_keys and not loaded.unexpected_keys
    sparse=SparseTemporalAgent(AgentConfig(),seed=0,sparse=True,temporal=False).eval();sparse.load_state_dict(dense.state_dict())
    world=World(6063003,0,6,128);events=[torch.from_numpy(e[None]) for e in world.events]
    with torch.inference_mode():
        a=dense.initial(1);b=sparse.initial(1);rows={name:{'admitted':0,'encoder_rows':0,'address_rows':0,'bank_clone_bytes':0} for name in ('dense','sparse')}
        for event in events:
            a,ta=dense.observe(a,event);b,tb=sparse.observe(b,event)
            assert all(torch.equal(x,y) for x,y in zip(a,b)),'inference-state mismatch'
            assert torch.equal(ta['address'],tb['address']) and torch.equal(ta['gate'],tb['gate'])
            for name,trace in (('dense',ta),('sparse',tb)):
                rows[name]['admitted']+=trace['writes']
                for key in ('encoder_rows','address_rows','bank_clone_bytes'):rows[name][key]+=trace[key]
        d,v,q,qv,_=world.candidates();args=[torch.from_numpy(x[None]) for x in (d,v,q,qv)]
        assert torch.equal(dense(a,*args)['logits'],sparse(b,*args)['logits'])
        results={}
        for name,model in (('dense',dense),('sparse',sparse)):
            times=[]
            for trial in range(35):
                state=model.initial(1);start=time.perf_counter_ns()
                for e in events:state,_=model.observe(state,e)
                if trial>=5:times.append((time.perf_counter_ns()-start)/1e6)
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU],profile_memory=True) as profiler:
                state=model.initial(1)
                for e in events:state,_=model.observe(state,e)
            allocations=sum(max(0,e.self_cpu_memory_usage) for e in profiler.key_averages())
            results[name]={'latency_ms_per128_records':times,'p50_ms':statistics.median(times),
                'p95_ms':float(np.quantile(times,.95)),'records_per_second_at_median':128000/statistics.median(times),
                'positive_net_operator_allocator_bytes':allocations,**rows[name]}
    perf=shutil.which('perf');counter={'available':False,'reason':'perf not installed; physical DRAM traffic not measured'}
    if perf:
        observed=subprocess.run([perf,'stat','-e','cache-references,cache-misses','true'],capture_output=True,text=True)
        counter={'available':observed.returncode==0,'probe':observed.stderr[-1500:],
                 'reason':'Cache counters alone do not measure DRAM bytes; no physical traffic claim'}
    result={'source':git('rev-parse','HEAD'),'checkpoint_sha256':digest(checkpoint),'events':128,'batch':1,'threads':1,
            'temporal_bias_enabled':False,'semantics':'Original temporal-disabled weights. Identical weights. Exact state/address/gate and final action logits; copy-on-write. Admission runs on every event. Timestamp feature is public.',
            'results':results,'physical_counter_probe':counter,'physical_dram_bytes':None,
            'limitations':'Single CPU microbenchmark of observe only, no end-to-end speed or total memory-traffic claim. Sparse admitted writes still clone the entire bank. Allocator statistic is sum of positive net grouped-operator self allocations, not peak RSS or DRAM traffic. Timing is descriptive and not a CI threshold.'}
    write(output,result);print(json.dumps({k:{m:v[m] for m in ('p50_ms','p95_ms','encoder_rows','address_rows','positive_net_operator_allocator_bytes')} for k,v in results.items()}))
    return result


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--output',required=True);ap.add_argument('--checkpoint');a=ap.parse_args();measure(a.output,a.checkpoint)
