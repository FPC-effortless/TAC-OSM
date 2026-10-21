"""Second registered construction: immutable physical serving graph."""
import json,math,time
import numpy as np
import torch
import scripts.me039_research as b
from tac_osm.me039_native_execution import freeze_serving


def run():
    torch.set_num_threads(1);rows=b.source();old=json.loads((b.ROOT/'results/me038/runtime.json').read_text())
    original={r['key']:r for r in old['models']};exports=[];cases=[];values=0
    for r in rows:
        o=original[r['key']];p=b.ROOT/o['physical_path'];assert b.sha(p)==o['physical_sha256']
        folder=b.OUT/'frozen-serving'/r['key'];folder.mkdir(parents=True,exist_ok=True)
        old_folder=b.ROOT/'results/me038'/r['key'];x=torch.from_numpy(np.load(old_folder/'serving-input.npy'));expected=np.load(old_folder/'serving-output.npy')
        t=time.process_time();eager=torch.jit.load(str(p)).eval();loaded=time.process_time()-t
        t=time.process_time();frozen=freeze_serving(eager);freeze_cpu=time.process_time()-t
        with torch.no_grad():
            t=time.perf_counter();y=frozen(x);cold_ms=(time.perf_counter()-t)*1000
            assert np.array_equal(y.numpy(),expected),r['key']
        path=folder/'physical.jit.pt';frozen.save(str(path))
        restored=torch.jit.load(str(path)).eval()
        with torch.no_grad():assert np.array_equal(restored(x).numpy(),expected)
        np.save(folder/'input.npy',x.numpy());np.save(folder/'output.npy',expected);values+=expected.size
        exports.append({'key':r['key'],'dataset':r['dataset'],'horizon':r['horizon'],'seed':r['seed'],'lane':r['lane'],
            'original_path':o['physical_path'],'original_sha256':o['physical_sha256'],'original_bytes':o['physical_bytes'],
            'path':str(path.relative_to(b.ROOT)),'sha256':b.sha(path),'bytes':path.stat().st_size,'values_bit_exact':expected.size,
            'original_learned_parameters':r['parameters'],'new_learned_parameters':r['parameters'],
            'serialized_load_CPU_seconds':loaded,'freeze_CPU_seconds':freeze_cpu,'first_B2_call_ms':cold_ms,
            'state_coordinates_per_channel':128,'original_training_weights_unchanged':True})
        if r['lane']=='exposure' and r['dataset'] in ('ETTh1','electricity') and r['horizon'] in (96,720):
            repeats=[]
            with torch.no_grad():
                for repeat in range(3):
                    for _ in range(5):eager(x[:1]);restored(x[:1])
                    samples={'original':[],'frozen':[]}
                    for trial in range(30):
                        for name in (('original','frozen') if (trial+repeat)%2==0 else ('frozen','original')):
                            model=eager if name=='original' else restored
                            t=time.perf_counter();model(x[:1]);samples[name].append((time.perf_counter()-t)*1000)
                    med={k:float(np.median(v)) for k,v in samples.items()}
                    repeats.append({'repeat':repeat,'median_ms':med,'speedup':med['original']/med['frozen'],'samples_ms':samples})
            speed=float(np.median([v['speedup'] for v in repeats]));saved_ms=float(np.median([v['median_ms']['original']-v['median_ms']['frozen'] for v in repeats]))
            cases.append({'case':r['key'],'speedup':speed,'repeats':repeats,'freeze_CPU_seconds':freeze_cpu,
                'estimated_freeze_break_even_calls':freeze_cpu*1000/saved_ms if saved_ms>0 else None})
        print('FREEZE',r['key'],'PASS',flush=True)
    geomean=math.exp(np.mean([math.log(c['speedup']) for c in cases]))
    output={'status':'PASS','protocol_sha256':b.sha(b.REPORT/'FROZEN-SERVING-PROTOCOL.md'),'exports':exports,
        'cases':cases,'serving_geometric_speedup':geomean,'target_pass':bool(geomean>=1.1),
        'physical_values_bit_exact':values,'old_export_bytes':sum(r['original_bytes'] for r in exports),
        'new_export_bytes':sum(r['bytes'] for r in exports),'weights_as_graph_constants':True,
        'new_accuracy_or_training_speed_claim':False,'model_source_sha256':b.sha(b.ROOT/'src/tac_osm/me039_native_execution.py')}
    b.dump(b.OUT/'frozen-serving.json',output);print('FROZEN SERVING SUMMARY',geomean,output['target_pass'],flush=True)

if __name__=='__main__':run()
