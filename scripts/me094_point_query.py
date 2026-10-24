"""Derived query-selective execution: same original median/risk/state, no tail solve.

No new learned model or training. Complete uncertainty remains available through
the original complete executable; this view prunes calculations not requested.
"""
import argparse,json,time,statistics,warnings
from pathlib import Path
import torch
import scripts.me094_train_world as R
from scripts.me094_physical_boundary import physical_inputs
from tac_osm.variants.median_anchored_execution import PhysicalMedianAnchoredGraph


class PointQuery(torch.nn.Module):
    def __init__(self,complete):super().__init__();self.complete=complete
    def forward(self,x,a,dt,pixels,state):
        result=self.complete(x,a,dt,pixels,state)
        return (result[1],result[3],*result[4:])


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--verify',action='store_true');parser.add_argument('--cache',type=Path,default=Path('/tmp/me077-original-source'));args=parser.parse_args()
    R.configure('me094');R.integrity();torch.set_num_threads(1)
    if args.verify:R.frozen()
    _,fit,_,identified,_=R.load_original(args.cache);entries=R.schedule(fit,identified,R.SEEDS[0])[0][0]
    checkpoint=R.OUT/f'seed-{R.SEEDS[0]}'/'final.pt';m=R.MedianAnchoredWorld(identified);m.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=True)['weights']);m.double().eval()
    complete=PhysicalMedianAnchoredGraph(m).eval();view=PointQuery(complete).eval();audit=R.ROOT/'POINT_QUERY_GRAPH_AUDIT.json'
    original=json.loads(audit.read_text()) if args.verify else None
    if original and original['checkpoint_sha256']!=R.sha(checkpoint):raise RuntimeError('Point-query original checkpoint differs')
    reports={}
    with torch.no_grad():
        for offset in ((0,2) if args.verify else (0,)):
            chosen=entries[offset:offset+2];x,a,_,_,dt,images=R.batch(fit,chosen);physical=physical_inputs(args.cache,fit,chosen)
            _,carried=m(x[:,:-1].double(),a[:,:-1].double(),elapsed=dt[:,:-1].double(),images=images[:,:-1].double(),return_state=True)
            for label,inputs in [('prefill32',(*physical,tuple(m.initial_state(x[:,0].double())))),('incremental_step',(*(v[:,-1:] for v in physical),tuple(carried)))]:
                full_path=R.ROOT/('PHYSICAL_PREFILL32_EXECUTABLE.pt' if label=='prefill32' else 'PHYSICAL_INCREMENTAL_STEP_EXECUTABLE.pt');full=torch.jit.load(str(full_path));outputs=full(*inputs);expected=(outputs[1],outputs[3],*outputs[4:]);path=R.ROOT/('POINT_QUERY_'+label.upper()+'.pt')
                if args.verify:
                    if R.sha(path)!=original['profiles'][label]['executable_sha256']:raise RuntimeError('Point program changed')
                    program=torch.jit.load(str(path))
                else:
                    with warnings.catch_warnings(record=True) as caught:
                        warnings.simplefilter('always');program=torch.jit.freeze(torch.jit.trace(view,inputs,strict=True,check_trace=True).eval())
                    program.save(str(path))
                for u,v in zip(expected,program(*inputs)):torch.testing.assert_close(u,v,rtol=1e-5,atol=1e-6)
                if not args.verify:
                    timings={'complete':[],'point_only':[]}
                    for _ in range(5):full(*inputs);program(*inputs)
                    for repeat in range(31):
                        order=[('complete',full),('point_only',program)]
                        if repeat%2:order.reverse()
                        for name,fn in order:
                            start=time.perf_counter_ns();fn(*inputs);timings[name].append((time.perf_counter_ns()-start)/1e6)
                    medians={k:statistics.median(v) for k,v in timings.items()}
                    reports[label]={'original_point_risk_and_all13state_outputs':'PASS','executable_sha256':R.sha(path),'complete_executable_sha256':R.sha(full_path),'graph_nodes_before_save':len(list(program.graph.nodes())),'complete_restored_graph_nodes':len(list(full.graph.nodes())),'median_ms':medians,'speedup_when_only_point_requested':medians['complete']/medians['point_only'],'raw_timings_ms':timings,'trace_warnings':list(dict.fromkeys(str(w.message) for w in caught))}
    result={'study':'ME094','status':'PASS','no_fit':True,'checkpoint_sha256':R.sha(checkpoint),'rtol':1e-5,'atol':1e-6,'output_fields':15,'scope':'First-original fixedB2T32/B2T1 real-FIT point/risk/carried-state query view of the SAME101173parameter full model. Complete probabilistic capability remains available in complete program. No new learned parameters, ensemble, model budget exception, predictive skill or autonomous routing. Explicit caller output request; no learned adaptive-compute claim.','measurement_limitations':'Shared/unpinned CPU with concurrent fits, alternating31rounds, excludes source/network/handler costs.'}
    if args.verify:print('MEDIAN_ANCHORED_POINT_QUERY_REPLAY='+json.dumps(result,sort_keys=True))
    else:result['profiles']=reports;R.write(audit,result);print(json.dumps({k:{'median_ms':v['median_ms'],'speedup':v['speedup_when_only_point_requested']} for k,v in reports.items()}),flush=True)

if __name__=='__main__':main()
