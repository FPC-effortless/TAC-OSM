"""Actual-FIT complete normalized/physical/context/state execution and restoration."""
import argparse,json,time,statistics,warnings
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
import torch
import scripts.me095_train_world as R
from scripts.me094_physical_boundary import physical_inputs as original_physical_inputs
from tac_osm.variants.context_filter_execution import NormalizedContextGraph,PhysicalContextGraph,ContextInitializer,PointContextGraph,capture_clock_features
from tac_osm.variants.context_filter_world import ContextFilterState
from tac_osm.variants.median_anchored_distribution import mixture_quantiles as original_quantiles


def physical_inputs(cache,rows,entries,history=32):
    original=original_physical_inputs(cache,rows,entries,history)
    tasks=torch.tensor([rows[(task,ep)]['task_id'] for task,ep,t in entries],dtype=torch.int64);clocks=[]
    for task,ep,t in entries:
        path=cache/task/f'data/chunk-{ep//1000:03d}'/f'episode_{ep:06d}.parquet'
        clock=np.asarray(pq.read_table(path,columns=['capture_time'])['capture_time'].to_numpy(),dtype=np.float64)
        clocks.append(torch.from_numpy(clock[t-history+1:t+1].copy())[:,None])
    return (*original,tasks,torch.stack(clocks))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--verify',action='store_true');parser.add_argument('--cache',type=Path,default=Path('/tmp/me077-original-source'));args=parser.parse_args()
    R.configure('me095');R.integrity();torch.set_num_threads(1)
    if args.verify:R.frozen()
    _,fit,_,identified,_=R.load_original(args.cache);entries=R.schedule(fit,identified,R.SEEDS[0])[0][0]
    checkpoint=R.OUT/f'seed-{R.SEEDS[0]}'/'final.pt';m=R.ContextFilterWorld(identified);m.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=True)['weights']);m.double().eval()
    wrappers={'normalized':NormalizedContextGraph(m).eval(),'physical':PhysicalContextGraph(m).eval()};wrappers['point_query']=PointContextGraph(wrappers['physical']).eval();initializer=ContextInitializer(m).eval()
    audit_path=R.ROOT/'COMPLETE_EXECUTION_AUDIT.json';original=json.loads(audit_path.read_text()) if args.verify else None;records={}
    if original and original['checkpoint_sha256']!=R.sha(checkpoint):raise RuntimeError('Original execution checkpoint differs')
    with torch.no_grad():
        for offset in ((0,2) if args.verify else (0,)):
            chosen=entries[offset:offset+2];x,a,y,valid,dt,images,task_ids,ctx=R.batch(fit,chosen);physical=physical_inputs(args.cache,fit,chosen)
            torch.testing.assert_close(capture_clock_features(physical[-1]),ctx.double(),rtol=0,atol=2*torch.finfo(torch.float32).eps)
            init=tuple(m.initial_state(x[:,0].double(),task_ids));init_path=R.ROOT/'CONTEXT_INITIALIZER.pt';init_inputs=(physical[0][:,0],task_ids)
            if args.verify:
                if R.sha(init_path)!=original['initializer_sha256']:raise RuntimeError('Initializer program changed')
                init_program=torch.jit.load(str(init_path))
            else:
                with warnings.catch_warnings(record=True):warnings.simplefilter('always');init_program=torch.jit.freeze(torch.jit.trace(initializer,init_inputs,strict=True,check_trace=True).eval())
                init_program.save(str(init_path))
            for u,v in zip(init,init_program(*init_inputs)):torch.testing.assert_close(u,v,rtol=1e-5,atol=1e-6)
            _,carried=m(x[:,:-1].double(),a[:,:-1].double(),elapsed=dt[:,:-1].double(),images=images[:,:-1].double(),task_ids=task_ids,clock_context=ctx[:,:-1].double(),return_state=True)
            for profile,norm,phys in [
                ('prefill32',(x.double(),a.double(),dt.double(),images.double(),task_ids,ctx.double(),init),(*physical,init)),
                ('incremental_step',(x[:,-1:].double(),a[:,-1:].double(),dt[:,-1:].double(),images[:,-1:].double(),task_ids,ctx[:,-1:].double(),tuple(carried)),(*(v[:,-1:] for v in physical[:4]),task_ids,physical[-1][:,-1:],tuple(carried)))]:
                p,state=m.visual_tensor_graph(*norm[:4],ContextFilterState(*norm[6]),last_only=True,task_ids=task_ids,clock_context=norm[5]);qq=original_quantiles(p)
                reference=(*qq,p['motion_logit'].sigmoid(),*tuple(state));physical_reference=tuple(v*m.state_scale+m.state_mean for v in qq)+reference[3:]
                for name,inputs,expected in [('normalized',norm,reference),('physical',phys,physical_reference),('point_query',phys,(physical_reference[1],physical_reference[3],*physical_reference[4:]))]:
                    label=name+'_'+profile;wrapper=wrappers[name];path=R.ROOT/(label.upper()+'.pt')
                    for u,v in zip(expected,wrapper(*inputs)):torch.testing.assert_close(u,v,rtol=1e-5,atol=1e-6)
                    if args.verify:
                        if R.sha(path)!=original['profiles'][label]['executable_sha256']:raise RuntimeError('Original execution program changed')
                        program=torch.jit.load(str(path))
                    else:
                        with warnings.catch_warnings(record=True) as caught:
                            warnings.simplefilter('always');program=torch.jit.freeze(torch.jit.trace(wrapper,inputs,strict=True,check_trace=True).eval())
                        program.save(str(path))
                    for u,v in zip(expected,program(*inputs)):torch.testing.assert_close(u,v,rtol=1e-5,atol=1e-6)
                    if not args.verify:
                        timings={'eager':[],'compiled':[]}
                        for _ in range(5):wrapper(*inputs);program(*inputs)
                        for repeat in range(31):
                            order=[('eager',wrapper),('compiled',program)]
                            if repeat%2:order.reverse()
                            for key,fn in order:
                                start=time.perf_counter_ns();fn(*inputs);timings[key].append((time.perf_counter_ns()-start)/1e6)
                        medians={k:statistics.median(v) for k,v in timings.items()}
                        records[label]={'original_all_output_fields':'PASS','output_fields':len(expected),'executable_sha256':R.sha(path),'median_ms':medians,'speedup_vs_own_eager':medians['eager']/medians['compiled'],'raw_timings_ms':timings,'nodes_before_save':len(list(program.graph.nodes())),'trace_warnings':list(dict.fromkeys(str(w.message) for w in caught))}
    result={'study':'ME095','status':'PASS','no_fit':True,'checkpoint_sha256':R.sha(checkpoint),'initializer_sha256':R.sha(init_path),'rtol':1e-5,'atol':1e-6,'scope':'First-original actual-FIT fixedB2T32/B2T1 context-aware original normalized, raw physical/capture-clock/pixel, point-query and15state initializer functions. No full-REPORT accelerated qualification or all-shape latency claim. Model104689parameters; derived views add no independent learned values.','context_preprocessing':'originalNumPyfloat64 sin/cos thenfloat32 vsTorchfloat64 sin/cos thenfloat32, metadata check2float32epsilon, final outputs unchanged1e-5/1e-6','measurement_limitations':'Shared/unpinned CPU, concurrent fits;31alternating rounds; excludes video/I/O/entity-handler/authority costs. Pointquery has fewer outputs, not equal-workload architecture comparison.'}
    if args.verify:print('CONTEXT_FILTER_COMPLETE_EXECUTION_REPLAY='+json.dumps(result,sort_keys=True))
    else:
        result['profiles']=records;R.write(audit_path,result);print(json.dumps({k:{'ms':v['median_ms'],'speedup':v['speedup_vs_own_eager']} for k,v in records.items()}),flush=True)

if __name__=='__main__':main()
