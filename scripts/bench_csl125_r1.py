"""Single-thread CPU no-fit reference timings, B1/T64 prefill + T8 decode.

Runs the preserved CSL125 control and corrected CSL125R1 with the same
shapes, random input and exit depths; not a trained-model accuracy comparison.
"""
import hashlib,json,platform,statistics,sys,time,resource
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from tac_osm.csl125 import CSL125,CSL125Config
from tac_osm.csl125_r1 import CSL125R1,CSL125R1Config


def state_bytes(s):
    total=0
    for layer in s.layers:
        for k in ('fast','slow','conv_tail','fast_mass','slow_mass'):
            t=getattr(layer,k,None)
            if t is not None:total+=t.numel()*t.element_size()
    for a in s.attention:
        if a is not None:total+=a.k.numel()*a.k.element_size()+a.v.numel()*a.v.element_size()
    return total


def run():
    torch.set_num_threads(1)
    torch.manual_seed(20261105)
    xx=torch.randint(0,32768,(1,64)); yy=torch.randint(0,32768,(1,8))
    r={'stage':'CPU_FP32_B1_NO_FIT_REFERENCE','torch':str(torch.__version__),'platform':platform.platform(),'threads':1,'prompt_tokens':64,'decode_tokens':8,'repetitions':3,'optimizer_updates':0,'arms':{},'limitations':'No fitting, fused kernel, GPU, INT8 or physical device. Time includes PyTorch CPU overhead. Do not compare language capability.'}
    with torch.inference_mode():
        for label,model in (('CSL125_CONTROL',CSL125(CSL125Config()).eval()),('CSL125_R1',CSL125R1(CSL125R1Config()).eval())):
            row={'parameter_count':model.parameter_count(),'depths':{}}
            for depth in model.config.rungs:
                p_times=[];d_times=[]
                for trial in range(3):
                    t=time.perf_counter()
                    output,state=model(xx,depth=depth,stream_ids=('bench',))
                    p_times.append(time.perf_counter()-t)
                    t=time.perf_counter()
                    for j in range(8):
                        output,state=model(yy[:,j:j+1],depth=depth,state=state,stream_ids=('bench',))
                    d_times.append(time.perf_counter()-t)
                row['depths'][str(depth)]={'prefill_samples_ms':[round(x*1000,3) for x in p_times],
                    'decode_samples_ms':[round(x*1000,3) for x in d_times],
                    'prefill_tokens_per_s':round(64/statistics.median(p_times),1),
                    'decode_tokens_per_s':round(8/statistics.median(d_times),1),
                    'state_bytes':state_bytes(state)}
            r['arms'][label]=row
            print(label,json.dumps(row['depths']),flush=True)
            del model
    r['peak_process_rss_kib']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    r['source_sha256']={v:hashlib.sha256((ROOT/'src/tac_osm'/f).read_bytes()).hexdigest() for v,f in [('control','csl125.py'),('r1','csl125_r1.py')]}
    out=ROOT/'research/mathematical-engineering/csl125_r1/CPU_ENGINEERING_BENCHMARK.json'
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(r,indent=2)+'\n')
if __name__=='__main__':run()
