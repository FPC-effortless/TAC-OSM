"""Original parent/adapter forecast replay and standalone native export."""
import argparse,copy,json,os,subprocess,sys,time
from pathlib import Path
import numpy as np
import torch
import scripts.me037_research as b
from tac_osm.me036_patch_forecast import build_model as old_model
from tac_osm.me037_horizon_forecast import build_model


def dependencies():
    for item in json.loads((b.REPORT/'dependencies.json').read_text())['unchanged_dependencies']:
        assert b.sha(b.ROOT/item['path'])==item['sha256'],item['path']


def audit():
    torch.set_num_threads(1);dependencies();frozen=b.verify()
    parents=json.loads((b.OUT/'parents.json').read_text())['parents']
    checkpoints=values=initial_values=0
    for parent in parents:
        z,meta=b.data(parent['dataset']);tr=meta['split_edges'][0];h=parent['horizon']
        x,_=b.batch(z,[tr,tr+24],h)
        for e in parent['endpoints']:
            path=b.ROOT/e['checkpoint'];assert b.sha(path)==e['sha256']
            state=torch.load(path,weights_only=False);assert state['step']==e['step']
            model=old_model('ar',h,parent['seed']).eval();model.load_state_dict(state['state_dict'])
            with torch.no_grad():p=model(x).numpy()
            assert np.array_equal(p,np.load(path.parent/f"u{e['step']}-probe.npy")),str(path)
            assert all(float(v['step'])==e['step'] for v in state['optimizer']['state'].values())
            checkpoints+=1;values+=p.size
    for r in frozen['fits']:
        if r['status']!='PASS':continue
        z,meta=b.data(r['dataset']);tr=meta['split_edges'][0];h=r['horizon'];x,_=b.batch(z,[tr,tr+24],h)
        parent_state=torch.load(b.ROOT/r['parent']['selected']['checkpoint'],weights_only=False)['state_dict']
        initial=build_model(r['arm'],h,r['seed']).eval();initial.initialize_ar(parent_state)
        reference=old_model('ar',h,r['parent']['seed']).eval();reference.load_state_dict(parent_state)
        with torch.no_grad():p=initial(x).numpy();ref=reference(x).numpy()
        assert np.array_equal(p,ref) and np.array_equal(p,np.load(b.OUT/r['key']/'initial-probe.npy'))
        initial_values+=p.size
        trace=json.loads((b.OUT/r['key']/'trace.json').read_text())
        assert len(trace)==500 and [t['step'] for t in trace]==list(range(1,501))
        assert all(np.isfinite(t['loss']) and np.isfinite(t['gradient_norm']) for t in trace)
        for e in r['endpoints']:
            path=b.ROOT/e['checkpoint'];state=torch.load(path,weights_only=False)
            assert state['step']==e['step'] and state['seed']==r['seed']
            assert all(float(v['step'])==e['step'] for v in state['optimizer']['state'].values())
            model=build_model(r['arm'],h,r['seed']).eval();model.load_state_dict(state['state_dict'])
            with torch.no_grad():p=model(x).numpy()
            assert np.array_equal(p,np.load(path.parent/f"u{e['step']}-probe.npy")),r['key']
            checkpoints+=1;values+=p.size
    receipt={'status':'PASS','parent_checkpoints':len(parents)*6,'adaptation_checkpoints':checkpoints-len(parents)*6,
        'original_checkpoints_replayed':checkpoints,'bit_exact_validation_forecast_values':values,
        'bit_exact_initial_AR_forecast_values':initial_values,'freeze_sha256':b.sha(b.OUT/'frozen.json'),
        'test_metrics_recomputed':False}
    b.dump(b.OUT/'audit.json',receipt);print(json.dumps(receipt,indent=2),flush=True)


class PhysicalForecast(torch.nn.Module):
    def __init__(self,model,mean,scale):
        super().__init__();self.model=model
        self.register_buffer('mean',torch.tensor(mean,dtype=torch.float32))
        self.register_buffer('scale',torch.tensor(scale,dtype=torch.float32))
    def forward(self,x):return self.model((x-self.mean)/self.scale)*self.scale+self.mean


def export():
    torch.set_num_threads(1);dependencies();frozen=b.verify();rows=[]
    for r in frozen['fits']:
        if r['arm']!='tac_horizon' or r['status']!='PASS':continue
        z,meta=b.data(r['dataset']);h=r['horizon'];tr=meta['split_edges'][0];folder=b.OUT/r['key']
        model=build_model(r['arm'],h,r['seed']).eval()
        model.load_state_dict(torch.load(b.ROOT/r['selected']['checkpoint'],weights_only=False)['state_dict'])
        x,_=b.batch(z,[tr,tr+24],h)
        with torch.no_grad():expected=model(x).numpy()
        # Training-only anchors are not part of serving; learned weights unchanged.
        del model.anchor_weight;del model.anchor_bias
        std=torch.jit.trace(model,x,check_trace=False);std_path=folder/'selected.jit.pt';std.save(str(std_path))
        with torch.no_grad():assert np.array_equal(std(x).numpy(),expected)
        physical=PhysicalForecast(model,meta['mean'],meta['scale']).eval();raw=x*physical.scale+physical.mean
        with torch.no_grad():gold=physical(raw).numpy()
        export=torch.jit.trace(physical,raw,check_trace=False);path=folder/'physical-selected.jit.pt';export.save(str(path))
        with torch.no_grad():assert np.array_equal(export(raw).numpy(),gold)
        np.save(folder/'serving-input.npy',raw.numpy());np.save(folder/'serving-output.npy',gold)
        code='''import sys,numpy as np,torch
torch.set_num_threads(1)
p=sys.argv[1];m=torch.jit.load(p+'/physical-selected.jit.pt').eval()
with torch.no_grad(): y=m(torch.from_numpy(np.load(p+'/serving-input.npy'))).numpy()
assert np.array_equal(y,np.load(p+'/serving-output.npy'))
'''
        env=dict(os.environ);env.pop('PYTHONPATH',None)
        subprocess.run([sys.executable,'-c',code,str(folder)],cwd='/tmp',env=env,check=True)
        times=[]
        with torch.no_grad():
            for _ in range(5):export(raw[:1])
            for _ in range(20):
                t=time.perf_counter();export(raw[:1]);times.append(time.perf_counter()-t)
        rows.append({'key':r['key'],'dataset':r['dataset'],'horizon':h,'seed':r['seed'],'selected_step':r['selected']['step'],
            'physical_path':str(path.relative_to(b.ROOT)),'physical_sha256':b.sha(path),'physical_bytes':path.stat().st_size,
            'standardized_path':str(std_path.relative_to(b.ROOT)),'standardized_sha256':b.sha(std_path),
            'standardized_values_exact':expected.size,'physical_values_exact':gold.size,'standalone_physical_replay':True,
            'B1_full_window_median_ms':float(np.median(times)*1000),'trainable_parameters':r['parameters'],
            'learned_weight_bytes':sum(p.numel()*p.element_size() for p in model.parameters()),
            'serving_buffer_bytes':sum(p.numel()*p.element_size() for p in physical.buffers()),
            'training_anchor_buffers_removed':True,'core_state_bytes_per_window':64*z.size(1)*4,
            'input_window_bytes':96*z.size(1)*4,'output_bytes':h*z.size(1)*4,'streaming_state_reused':False})
        print('EXPORT',r['key'],round(rows[-1]['B1_full_window_median_ms'],3),flush=True)
    b.dump(b.OUT/'runtime.json',{'models':rows,'total_standardized_values_exact':sum(r['standardized_values_exact'] for r in rows),
        'total_physical_values_exact':sum(r['physical_values_exact'] for r in rows),'same_CPU_environment':True,
        'physical_contract':'Raw publisher units; bit exact against eager physical wrapper on same raw input. No equality claim to separately standardized input after float32 roundtrip.'})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('audit','export'));args=p.parse_args();globals()[args.mode]()
