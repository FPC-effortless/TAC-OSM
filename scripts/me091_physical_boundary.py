"""No-fit raw physical/pixel-to-physical-output compiled graph and restore audit."""
import argparse,json,statistics,time,warnings
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
import torch
import scripts.me091_train_world as R
from scripts.me091_compile import CompleteGraph
from tac_osm.variants.visual_execution import PhysicalVisualGraph


def physical_inputs(cache,rows,entries,history=32):
    values=[];commands=[]
    for task,ep,t in entries:
        p=cache/task/f'data/chunk-{ep//1000:03d}'/f'episode_{ep:06d}.parquet'
        table=pq.read_table(p,columns=['observations.state.qpos','action.qpos'])
        x=np.asarray(table['observations.state.qpos'].to_pylist(),dtype=np.float32)
        a=np.asarray(table['action.qpos'].to_pylist(),dtype=np.float32)
        values.append(torch.from_numpy(x[t-history+1:t+1]));commands.append(torch.from_numpy(a[t-history:t]))
    dt=torch.stack([rows[(task,ep)]['elapsed'][t-history+1:t+1] for task,ep,t in entries])
    pixels=torch.stack([rows[(task,ep)]['images'][t-history+1:t+1] for task,ep,t in entries])
    return torch.stack(values),torch.stack(commands),dt,pixels


def main():
    p=argparse.ArgumentParser();p.add_argument('--cache',type=Path,default=Path('/tmp/me077-original-source'));p.add_argument('--verify',action='store_true');args=p.parse_args()
    torch.set_num_threads(1);R.configure('me091');R.integrity()
    _,fit,_,identified,_=R.load_original(args.cache)
    path=R.OUT/f'seed-{R.SEEDS[0]}'/'final.pt';m=R.VisualWorld(identified);m.load_state_dict(torch.load(path,map_location='cpu',weights_only=True)['weights'],strict=True);m.double().eval()
    normalized=CompleteGraph(m).eval();whole=PhysicalVisualGraph(m).eval();records={};entries=R.schedule(fit,identified,R.SEEDS[0])[0][0]
    audit_path=R.ROOT/'PHYSICAL_BOUNDARY_GRAPH_AUDIT.json'
    original=json.loads(audit_path.read_text()) if args.verify else None
    if original and original['checkpoint_sha256']!=R.sha(path):raise RuntimeError('Original physical executable checkpoint differs')
    with torch.no_grad():
        for offset in ((0,2) if args.verify else (0,)):
            chosen=entries[offset:offset+2];x,a,_,_,dt,images=R.batch(fit,chosen)
            physical=physical_inputs(args.cache,fit,chosen);xp,ap,dtp,pixels=physical
            torch.testing.assert_close((xp-m.state_mean.float())/m.state_scale.float(),x,rtol=0,atol=0)
            torch.testing.assert_close((ap-m.action_mean.float())/m.action_scale.float(),a,rtol=0,atol=0)
            _,carried=m(x[:,:-1].double(),a[:,:-1].double(),elapsed=dt[:,:-1].double(),images=images[:,:-1].double(),return_state=True)
            for label,raw_inputs,norm_inputs in [
                ('physical_prefill32',(*physical,tuple(m.initial_state(x[:,0].double()))),(x.double(),a.double(),dt.double(),images.double(),tuple(m.initial_state(x[:,0].double())))),
                ('physical_incremental_step',(xp[:,-1:],ap[:,-1:],dtp[:,-1:],pixels[:,-1:],tuple(carried)),(x[:,-1:].double(),a[:,-1:].double(),dt[:,-1:].double(),images[:,-1:].double(),tuple(carried)))]:
                expected=normalized(*norm_inputs);expected=tuple(v*m.state_scale+m.state_mean for v in expected[:3])+expected[3:]
                actual=whole(*raw_inputs)
                for u,v in zip(expected,actual):torch.testing.assert_close(u,v,rtol=1e-5,atol=1e-6)
                executable=R.ROOT/(label.upper()+'_EXECUTABLE.pt')
                if args.verify:
                    if R.sha(executable)!=original['profiles'][label]['executable_sha256']:raise RuntimeError('Physical executable bytes differ')
                    program=torch.jit.load(str(executable),map_location='cpu')
                else:
                    with warnings.catch_warnings(record=True) as caught:
                        warnings.simplefilter('always');program=torch.jit.freeze(torch.jit.trace(whole,raw_inputs,strict=True,check_trace=True).eval())
                    program.save(str(executable))
                for u,v in zip(expected,program(*raw_inputs)):torch.testing.assert_close(u,v,rtol=1e-5,atol=1e-6)
                if not args.verify:
                    times={'eager':[],'compiled':[]}
                    for _ in range(5):whole(*raw_inputs);program(*raw_inputs)
                    for i in range(31):
                        order=[('eager',whole),('compiled',program)]
                        if i%2:order.reverse()
                        for name,fn in order:
                            start=time.perf_counter_ns();fn(*raw_inputs);times[name].append((time.perf_counter_ns()-start)/1e6)
                    med={k:statistics.median(v) for k,v in times.items()}
                    records[label]={'input_state_shape':list(xp.shape if label=='physical_prefill32' else xp[:,-1:].shape),'pixels_dtype':'uint8','preprocessing':'float32 original subtraction/division, uint8/255, then float64 native reference','point_interval_units':'original physical qpos units','state_units':'internal normalized reference state','complete_output_fields':len(expected),'normative_equivalence':'PASS','graph_nodes':len(list(program.graph.nodes())),'median_ms':med,'speedup':med['eager']/med['compiled'],'raw_timings_ms':times,'executable_sha256':R.sha(executable),'trace_warnings':list(dict.fromkeys(str(w.message) for w in caught))}
    result={'study':'ME091','status':'PASS','no_fit':True,'checkpoint_sha256':R.sha(path),'rtol':1e-5,'atol':1e-6,'scope':'First original actual FIT observed physical qpos, lagged references, elapsed and uint8 pixels through exact original normalization, complete reference model, physical-unit quantiles and13state outputs. FixedB2T32/B2T1; not all-shape or full REPORT compiled qualification.','measurement_limitations':'Shared CPU, concurrent original training, unpinned affinity; math boundary only, excludes video decoding, storage/network and runtime-handler hashing.'}
    if args.verify:print('VISUAL_PHYSICAL_BOUNDARY_REPLAY='+json.dumps(result,sort_keys=True))
    else:
        result['profiles']=records;R.write(audit_path,result);print(json.dumps({k:{'speedup':v['speedup'],'status':v['normative_equivalence'],'nodes':v['graph_nodes']} for k,v in records.items()}),flush=True)

if __name__=='__main__':main()
