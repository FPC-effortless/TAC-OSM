"""No-fit fixed-profile compiled complete input-to-output graph audit on real FIT."""
import argparse,json,time,statistics,warnings
from pathlib import Path
import torch
from torch import nn
import scripts.me094_train_world as R
from tac_osm.variants.median_anchored_world import MedianAnchoredWorld
from tac_osm.variants.clock_value_world import ClockValueState
from tac_osm.variants.median_anchored_execution import execution_quantiles as mixture_quantiles
from tac_osm.variants.median_anchored_distribution import mixture_quantiles as original_quantiles


class CompleteGraph(nn.Module):
    def __init__(self,model):super().__init__();self.model=model
    def forward(self,x,a,dt,images,state_values):
        p,s=self.model.visual_tensor_graph(x,a,dt,images,ClockValueState(*state_values),last_only=True)
        q=mixture_quantiles(p)
        return (q[0],q[1],q[2],p['motion_logit'].sigmoid(),*tuple(s))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--study',choices=tuple(R.CONFIG),default='me094');args=parser.parse_args()
    R.configure(args.study);R.integrity();torch.set_num_threads(1)
    _,fit,_,identified,_=R.load_original(Path('/tmp/me077-original-source'))
    entries=R.schedule(fit,identified,R.SEEDS[0])[0][0][:2];x,a,_,_,dt,images=R.batch(fit,entries)
    path=R.OUT/f'seed-{R.SEEDS[0]}'/'final.pt';model=MedianAnchoredWorld(identified)
    model.load_state_dict(torch.load(path,map_location='cpu',weights_only=True)['weights'],strict=True);model.double().eval();wrapper=CompleteGraph(model).eval()
    x,a,dt,images=x.double(),a.double(),dt.double(),images.double();reports={}
    with torch.no_grad():
        _,carried=model(x[:,:-1],a[:,:-1],elapsed=dt[:,:-1],images=images[:,:-1],return_state=True)
        for label,inputs in [('prefill32',(x,a,dt,images,tuple(model.initial_state(x[:,0])))),('incremental_step',(x[:,-1:],a[:,-1:],dt[:,-1:],images[:,-1:],tuple(carried)))]:
            # Tracing is intentionally fixed-profile. Shape warnings are retained,
            # never advertised as arbitrary-length/variable-batch compilation.
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always')
                traced=torch.jit.trace(wrapper,inputs,strict=True,check_trace=True)
            compiled=torch.jit.freeze(traced.eval())
            expected=wrapper(*inputs);actual=compiled(*inputs)
            raw,state=model.visual_tensor_graph(*inputs[:4],ClockValueState(*inputs[4]),last_only=True)
            original=(*original_quantiles(raw),raw['motion_logit'].sigmoid(),*tuple(state))
            for u,v in zip(original,expected):torch.testing.assert_close(u,v,rtol=1e-5,atol=1e-6)
            for u,v in zip(expected,actual):torch.testing.assert_close(u,v,rtol=1e-5,atol=1e-6)
            # Different actual FIT observations at the same profile, not just the
            # trace values, must execute through the compiled tensor program.
            changed=tuple(v.flip(0) if isinstance(v,torch.Tensor) else tuple(t.flip(0) for t in v) for v in inputs)
            for u,v in zip(wrapper(*changed),compiled(*changed)):torch.testing.assert_close(u,v,rtol=1e-5,atol=1e-6)
            timing={'eager':[],'compiled':[]}
            for _ in range(5):wrapper(*inputs);compiled(*inputs)
            for repeat in range(31):
                order=[('eager',wrapper),('compiled',compiled)]
                if repeat%2:order.reverse()
                for name,fn in order:
                    start=time.perf_counter_ns();fn(*inputs);timing[name].append((time.perf_counter_ns()-start)/1e6)
            executable=R.ROOT/f'{label.upper()}_REFERENCE_EXECUTABLE.pt'
            compiled.save(str(executable));restored=torch.jit.load(str(executable))
            for u,v in zip(expected,restored(*inputs)):torch.testing.assert_close(u,v,rtol=1e-5,atol=1e-6)
            med={k:statistics.median(v) for k,v in timing.items()}
            reports[label]={'fixed_input_shape':list(inputs[0].shape),'normative_original_tolerance':'PASS','roundtrip_executable':'PASS',
               'median_ms':med,'raw_timings_ms':timing,'speedup':med['eager']/med['compiled'],'torchscript_nodes':len(list(compiled.graph.nodes())),
               'trace_warnings':list(dict.fromkeys(str(w.message) for w in caught)),'executable_sha256':R.sha(executable)}
    result={'study':R.STUDY,'no_fit':True,'checkpoint_sha256':R.sha(path),'source':'real FIT only; no REPORT or protected-source scoring',
       'dtype':'float64 complete graph','torch':torch.__version__,'threads':1,'profiles':reports,
       'scope':'derived fixed-shape first-original executables. No new training, separate learned parameters or changed forecasts. NOT full original REPORT replay qualification of this optional compiled path.',
       'measurement_limitations':'Concurrent training in shared workspace, CPU affinity unpinned. Alternating within-process timing; no rival/hardware-general speed claim.'}
    R.write(R.ROOT/'COMPILED_GRAPH_AUDIT.json',result)
    print(json.dumps({k:{'median_ms':v['median_ms'],'speedup':v['speedup'],'status':v['normative_original_tolerance']} for k,v in reports.items()}))

if __name__=='__main__':main()
