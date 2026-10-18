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
from tac_osm.development.inference_public_features import InferencePublicFeatureMLP
from tac_osm.development.learned_agent import AgentConfig
from run_learned_stack_dev055 import write


def timed(function,repeats=7):
    for _ in range(2):function()
    values=[]
    for _ in range(repeats):
        start=time.perf_counter_ns();function();values.append((time.perf_counter_ns()-start)/1e6)
    return {'median_ms':statistics.median(values),'min_ms':min(values),'max_ms':max(values),'samples_ms':values}


class PublicIncrementalReference:
    """Benchmark-only public rule; never imported by a learned artifact."""
    def __init__(self):self.bits={};self.parents={}
    def observe(self,e):
        kind=int(e[:4].argmax())
        if kind==3:return
        key=int(e[4:12].argmax())
        if kind==1:self.parents[key]=int(e[12:20].argmax())
        else:self.bits[key]=int(e[20])
    def choose(self,descriptors,valid):
        active={int(d[3:].argmax()) for d,v in zip(descriptors,valid) if v and not d[0]}
        healthy=not any(self.bits[k] for k in active)
        for i,(d,v) in enumerate(zip(descriptors,valid)):
            if not v:continue
            kind=int(d[:3].argmax());key=int(d[3:].argmax())
            if healthy and kind==0:return i
            if kind==2 and self.bits[key] and (key not in self.parents or not self.bits[self.parents[key]]):return i
        raise AssertionError('no public safe-progress action')


@torch.no_grad()
def run(directory,output):
    torch.set_num_threads(1);root=Path(directory);result={'models':{},'admission_sweep':[],
      'scope':'CPU wall time on idle host. FLOPs are profiler-supported operators only, not complete algorithm/hardware counts. State tensor bytes include public metadata; exclude weights/intermediates/allocator overhead. No physical DRAM counter available; DRAM=UNKNOWN. No speed or allocation thresholds.',
      'hardware':{'threads':torch.get_num_threads(),'torch':torch.__version__,'numpy':np.__version__}}
    for folder in sorted(root.glob('*-9101')):
        report=json.loads((folder/'report.json').read_text());arm=report['arm'];m=create(arm,report['config'],9101).eval();m.load_state_dict(torch.load(folder/'agent.pt',weights_only=True))
        if arm=='generic':
            original=m;m=InferenceMemoryAgent(report['config'],'generic').eval();m.load_state_dict(original.state_dict())
        if arm=='public_mlp':
            original=m;m=InferencePublicFeatureMLP(report['config']['width']).eval();m.load_state_dict(original.state_dict())
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
        result['models'][arm]={'seed':9101,'runtime':'skip_closed_copy' if arm in ('generic','public_mlp') else 'registered','parameters':report['parameters'],'timing':measured,'decision_supported_operator_flops':supported_flops,'training_cpu_seconds':report['cpu_seconds']}
    from audit_evidence_gaps_dev059 import public_reference
    result['optimized_public_reference']=[]
    for batch in (1,8,32):
        worlds=[ResearchWorld(5611120,i,4,128) for i in range(batch)]
        def public_build():
            states=[PublicIncrementalReference() for _ in worlds]
            for state,world in zip(states,worlds):
                for event in world.events[:128]:state.observe(event)
            return states
        states=public_build();candidates=[w.candidates() for w in worlds]
        for state,world,public in zip(states,worlds,candidates):
            assert state.choose(public[0],public[1])==public_reference(world.events,public[0],public[1])
        def public_decide():return [state.choose(public[0],public[1]) for state,public in zip(states,candidates)]
        def public_complete():
            return [state.choose(public[0],public[1]) for state,public in zip(public_build(),candidates)]
        result['optimized_public_reference'].append({'batch':batch,'history':128,'observe':timed(public_build),'decision':timed(public_decide),'end_to_end':timed(public_complete),
            'scope':'Explicit incremental Python/NumPy public-rule control; different representation prior. Input conversion excluded as for Torch timing. No neural architecture contribution.'})
    # Keep frozen DEV058 weights. Record bitwise and bounded float equivalence
    # separately: different admitted batch shapes can change CPU GEMM rounding.
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
            bitwise=all(torch.equal(x,y) for x,y in zip(a,b))
            state_drift=max(float((x-y).abs().max()) for x,y in zip(a,b))
            occupancy_equal=bool(torch.equal(a[1],b[1]))
            state_bounded=all(torch.allclose(x,y,atol=1e-4,rtol=1e-5) for x,y in zip(a,b))
            probes=[ResearchWorld(5611120,i,4,128) for i in range(batch)]
            arguments=[torch.from_numpy(np.stack([w.candidates()[i] for w in probes])) for i in range(4)]
            left,right=dense(a,*arguments),sparse(b,*arguments)
            actions_equal=bool(torch.equal(left['logits'].argmax(-1),right['logits'].argmax(-1)))
            output_drift={key:float((left[key]-right[key]).abs().max()) for key in ('logits','completion','outcomes','cost')}
            outputs_bounded=all(torch.allclose(left[key],right[key],atol=1e-4,rtol=1e-5) for key in output_drift)
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
                 'bitwise_state_equal':bitwise,'maximum_state_absolute_difference':state_drift,'exact_occupancy':occupancy_equal,'exact_action_choices':actions_equal,'output_maximum_absolute_difference':output_drift,
                 'state_bounded_equivalence':state_bounded,'outputs_bounded_equivalence':outputs_bounded,
                 'bounded_equivalence_pass':occupancy_equal and actions_equal and state_bounded and outputs_bounded,
                 'equivalence_scope':'Unchanged atol1e-4/rtol1e-5; failed cases remain false. Raw latency ratios do not establish output-preserving speedup when equivalence fails. Bitwise equality is separate.',
                 'public_type_signals':type_signals,'type_gate_exact_agreement':agree,'public_latest_records':len(last_filter()),
                 'dense':timed(lambda:process(dense)),'sparse':timed(lambda:process(sparse)),'gate_batched':timed(admission_only),'gate_streamed':timed(admission_streamed),
                 'public_type_filter_batched':timed(type_filter),'public_latest_filter_python':timed(last_filter),
                 'public_latest_filter_caveat':'Changes retained history/address allocation; no learned-policy output equivalence claimed. Selection timing alone is not end-to-end speed.'}
            row['dense_over_sparse']=row['dense']['median_ms']/row['sparse']['median_ms'];result['admission_sweep'].append(row)
    result['all_sweep_bounded_equivalence_pass']=all(row['bounded_equivalence_pass'] for row in result['admission_sweep'])
    result['all_sweep_bitwise_states_equal']=all(row['bitwise_state_equal'] for row in result['admission_sweep'])
    write(output,result)


if __name__=='__main__':
    a=argparse.ArgumentParser(description=__doc__);a.add_argument('--directory',required=True);a.add_argument('--output',required=True);args=a.parse_args();run(args.directory,args.output)
