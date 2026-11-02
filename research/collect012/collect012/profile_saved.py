"""One instrumented CPU pass per saved graph; no optimizer or timing ranking."""
import json,time
import torch
from .common import ROOT,build,SPECS
from .train import fingerprint

def main():
    torch.set_num_threads(1);rows=[]
    for arm in SPECS:
        root=ROOT.parent/'collect012_downloads'/f'{arm}_12003'
        saved=torch.load(root/'checkpoint_1024.pt',weights_only=True,map_location='cpu')
        replay=torch.load(root/'REPLAY.pt',weights_only=True,map_location='cpu')
        m=build(arm,12003);m.load_state_dict(saved['model']);before=fingerprint(m)
        def run():
            m.zero_grad(set_to_none=True);y,_=m(replay['tokens'])
            torch.nn.functional.cross_entropy(y.flatten(0,1),replay['targets'].flatten()).backward()
        run();tic=time.perf_counter()
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU],record_shapes=True) as p:run()
        elapsed=time.perf_counter()-tic;events=p.key_averages();total=sum(e.self_cpu_time_total for e in events)
        top=sorted(events,key=lambda e:e.self_cpu_time_total,reverse=True)[:20]
        assert before==fingerprint(m)
        rows.append(dict(arm=arm,seed=12003,shape=list(replay['tokens'].shape),instrumented_wall_seconds=elapsed,
            total_operator_self_cpu_us=total,top_operators=[dict(name=e.key,count=e.count,self_cpu_us=e.self_cpu_time_total,
                self_cpu_fraction=e.self_cpu_time_total/total) for e in top],weights_unchanged=True))
    result=dict(status='MEASURED_SINGLE_CPU_PROFILE_NOT_GPU_OR_THROUGHPUT_BENCHMARK',torch=torch.__version__,
        optimizer_updates=0,method='one warmup plus one instrumented saved-minibatch forward/backward per graph; FP32, one CPU thread',rows=rows)
    (ROOT/'results/COLLECT013_CPU_PROFILE.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))

if __name__=='__main__':main()
