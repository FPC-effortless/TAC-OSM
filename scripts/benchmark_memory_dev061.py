"""Idle-host timing, declared state and limited profiler FLOPs; no DRAM claim."""
import argparse
import json
from pathlib import Path
import statistics
import time
import numpy as np
import torch
from run_structured_memory_dev061 import ROOT,ResearchWorld,create
from tac_osm.development.temporal_agent import SparseTemporalAgent
from tac_osm.development.inference_memory import InferenceMemoryAgent
from tac_osm.development.learned_agent import AgentConfig
from run_learned_stack_dev055 import write


def timed(function,repeats=7):
    for _ in range(2):function()
    values=[]
    for _ in range(repeats):
        start=time.perf_counter_ns();function();values.append((time.perf_counter_ns()-start)/1e6)
    return {'median_ms':statistics.median(values),'min_ms':min(values),'max_ms':max(values),'samples_ms':values}


@torch.no_grad()
def run(directory,output):
    torch.set_num_threads(1);root=Path(directory);result={'models':{},'admission_sweep':[],
      'scope':'CPU wall time on idle host. FLOPs are profiler-supported operators only, not complete algorithm/hardware counts. State tensor bytes include public metadata; exclude weights/intermediates/allocator overhead. No physical DRAM counter available; DRAM=UNKNOWN. No speed or allocation thresholds.',
      'hardware':{'threads':torch.get_num_threads(),'torch':torch.__version__,'numpy':np.__version__}}
    for folder in sorted(root.glob('*-9101')):
        report=json.loads((folder/'report.json').read_text());arm=report['arm'];m=create(arm,report['config'],9101).eval();m.load_state_dict(torch.load(folder/'agent.pt',weights_only=True))
        if arm=='generic':
            original=m;m=InferenceMemoryAgent(report['config'],'generic').eval();m.load_state_dict(original.state_dict())
        measured=[]
        for batch in (1,8,32):
            worlds=[ResearchWorld(5611120,i,4,128) for i in range(batch)];events=torch.from_numpy(np.stack([w.events for w in worlds])).transpose(0,1)
            args=[torch.from_numpy(np.stack([w.candidates()[i] for w in worlds])) for i in range(4)]
            def build():
                state=m.initial(batch)
                for e in events:state,_=m.observe(state,e)
                return state
            state=build()
            def forward():return m(state,*args)
            def end_to_end():return m(build(),*args)
            measured.append({'batch':batch,'history':128,'observe':timed(build),'decision_with_auxiliary_probe':timed(forward),
                             'end_to_end':timed(end_to_end),'tensor_state_bytes_per_world':sum(v.numel()*v.element_size() for v in state)//batch})
        state=m.initial(1);w=ResearchWorld(5611120,0,4,128)
        for e in w.events:state,_=m.observe(state,torch.from_numpy(e[None]))
        args=[torch.from_numpy(v[None]) for v in w.candidates()[:4]]
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU],with_flops=True,profile_memory=True) as prof:m(state,*args)
        supported_flops=sum(e.flops for e in prof.key_averages())
        result['models'][arm]={'seed':9101,'runtime':'skip_closed_metadata_copy' if arm=='generic' else 'registered','parameters':report['parameters'],'timing':measured,'decision_supported_operator_flops':supported_flops,'training_cpu_seconds':report['cpu_seconds']}
    # Keep frozen DEV058 weights. Same event set and exact resulting outputs per dense/sparse pair.
    old=ROOT/'results/dev058-main/r0-7581';config=AgentConfig(**json.loads((old/'report.json').read_text())['config'])
    dense=SparseTemporalAgent(config,temporal=False,sparse=False).eval();sparse=SparseTemporalAgent(config,temporal=False,sparse=True).eval()
    for m in (dense,sparse):m.load_state_dict(torch.load(old/'agent.pt',weights_only=True))
    for batch in (1,8,32):
        for rate in (0.,.1,.5,1.):
            rng=np.random.default_rng(611213);events=np.zeros((128,batch,22),np.float32);kinds=np.full((128,batch),3)
            selected=rng.permutation(128*batch)[:round(rate*128*batch)];kinds.reshape(-1)[selected]=rng.integers(3,size=len(selected))
            for t in range(128):
                for i in range(batch):
                    e=events[t,i];e[kinds[t,i]]=1;e[4+int(rng.integers(8))]=1;e[12+int(rng.integers(8))]=1;e[20]=rng.integers(2);e[21]=t/256
            stream=torch.from_numpy(events)
            def process(model):
                state=model.initial(batch)
                for e in stream:state,_=model.observe(state,e)
                return state
            a,b=process(dense),process(sparse)
            assert all(torch.equal(x,y) for x,y in zip(a,b))
            admitted=int((sparse.admission(stream).squeeze(-1)>=0).sum());type_signals=int((stream[:,:,:4].argmax(-1)!=3).sum())
            agree=bool(torch.equal(sparse.admission(stream).squeeze(-1)>=0,stream[:,:,:4].argmax(-1)!=3))
            def admission_only():return sparse.admission(stream)
            def admission_streamed():
                for e in stream:sparse.admission(e)
            def type_filter():return stream[:,:,:4].argmax(-1)!=3
            def last_filter():
                # Explicit public task parser CONTROL, not deployed in a learned arm.
                kinds=stream[:,:,:4].argmax(-1);keys=stream[:,:,4:12].argmax(-1);kept=[]
                for i in range(batch):
                    seen=set()
                    for t in range(127,-1,-1):
                        kind=int(kinds[t,i]);key=int(keys[t,i]);route=(int(kind==1),key)
                        if kind!=3 and route not in seen:seen.add(route);kept.append((t,i))
                return kept
            row={'batch':batch,'history':128,'requested_rate':rate,'actual_admitted':admitted,'observations':128*batch,
                 'public_type_signals':type_signals,'type_gate_exact_agreement':agree,'public_latest_records':len(last_filter()),
                 'dense':timed(lambda:process(dense)),'sparse':timed(lambda:process(sparse)),'gate_batched':timed(admission_only),'gate_streamed':timed(admission_streamed),
                 'public_type_filter_batched':timed(type_filter),'public_latest_filter_python':timed(last_filter),
                 'public_latest_filter_caveat':'Changes retained history/address allocation; no learned-policy output equivalence claimed. Selection timing alone is not end-to-end speed.'}
            row['dense_over_sparse']=row['dense']['median_ms']/row['sparse']['median_ms'];result['admission_sweep'].append(row)
    write(output,result)


if __name__=='__main__':
    a=argparse.ArgumentParser(description=__doc__);a.add_argument('--directory',required=True);a.add_argument('--output',required=True);args=a.parse_args();run(args.directory,args.output)
