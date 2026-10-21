"""Export selected TAC models and verify package-independent checkpoint serving."""
import json, os, subprocess, sys, time
from pathlib import Path
import numpy as np
import torch
import scripts.me036_research as base
from tac_osm.me036_patch_forecast import build_model


def run():
    torch.set_num_threads(1)
    out=base.ROOT/'results/me036b';frozen=json.loads((out/'frozen.json').read_text())
    assert frozen['protocol_sha256']==base.sha(base.REPORT/'PROTOCOL.md')
    assert frozen['continuation_protocol_sha256']==base.sha(base.REPORT/'CONTINUATION.md')
    for r in frozen['fits']:
        for e in r['endpoints']:assert base.sha(base.ROOT/e['checkpoint'])==e['sha256']
    exports=[]
    for name in base.DATASETS:
        z,meta=base.load_data(name);train=meta['split_edges'][0]
        for r in frozen['fits']:
            if r['dataset']!=name or r['arm']!='tac_patch' or r['status']!='PASS':continue
            model=build_model(r['arm'],r['horizon'],r['seed']).eval()
            model.load_state_dict(torch.load(base.ROOT/r['selected']['checkpoint'],weights_only=False)['state_dict'])
            x,_=base.batch(z,[train,train+24],r['horizon'])
            with torch.no_grad():expected=model(x).numpy()
            folder=out/r['key'];path=folder/'selected.jit.pt'
            export=torch.jit.trace(model,x,check_trace=False)
            export.save(str(path))
            with torch.no_grad():actual=export(x).numpy()
            assert np.array_equal(expected,actual),r['key']
            np.save(folder/'serving-input.npy',x.numpy());np.save(folder/'serving-output.npy',expected)
            code='''import sys,numpy as np,torch
torch.set_num_threads(1)
p=sys.argv[1]; m=torch.jit.load(p+'/selected.jit.pt').eval()
with torch.no_grad(): a=m(torch.from_numpy(np.load(p+'/serving-input.npy'))).numpy()
assert np.array_equal(a,np.load(p+'/serving-output.npy'))
'''
            env=dict(os.environ);env.pop('PYTHONPATH',None)
            subprocess.run([sys.executable,'-c',code,str(folder)],cwd='/tmp',env=env,check=True)
            sample=x[:1]
            with torch.no_grad():
                for _ in range(5):export(sample)
                times=[]
                for _ in range(20):
                    start=time.perf_counter();export(sample);times.append(time.perf_counter()-start)
            exports.append({'key':r['key'],'dataset':name,'horizon':r['horizon'],'seed':r['seed'],
                'path':str(path.relative_to(base.ROOT)),'sha256':base.sha(path),'bytes':path.stat().st_size,
                'forecast_values_replayed':expected.size,'package_independent_exact':True,
                'B1_full_window_median_ms':float(np.median(times)*1000),
                'weight_bytes':sum(p.numel()*p.element_size() for p in model.parameters()),
                'input_window_bytes':96*z.size(1)*4,'core_state_bytes_all_channels':64*z.size(1)*4,
                'output_bytes':r['horizon']*z.size(1)*4,'streaming_state_reused':False})
            print('EXPORTED',r['key'],round(exports[-1]['B1_full_window_median_ms'],3),flush=True)
    base.dump(out/'runtime.json',{'exports':exports,'CPU_model':next((l.split(':',1)[1].strip() for l in Path('/proc/cpuinfo').read_text().splitlines() if l.startswith('model name')),'unknown'),
        'threads':1,'total_exact_export_forecasts':sum(e['forecast_values_replayed'] for e in exports),
        'end_to_end_contract':'Inputs and predictions globally train-standardized. Raw-unit conversion requires saved per-channel train mean/scale in data.json; runtime includes local normalization. Full96-step window replay.'})


if __name__=='__main__':run()
