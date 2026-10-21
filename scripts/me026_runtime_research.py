"""Preregistered zero-fit native execution study on original ME025 checkpoints."""
from __future__ import annotations
import argparse
import json
import platform
import resource
import subprocess
import time
from pathlib import Path
import numpy as np
import torch
from scripts.me025_real_transformer_parity import model_for,fit_only_popularity,prob_scores
from scripts.me022_native_readout import save_json
from scripts.me023_predictive_research import sha,load_source
from scripts.me025_recorded_audit import C,audit
from tac_osm.native_forecast_runtime import NativeForecastRuntime,kernel_segment,compose_kernel_segments


@torch.no_grad()
def measured(call,repeats):
    rows=[]
    for _ in range(repeats):
        w,c=time.perf_counter(),time.process_time(); call()
        rows.append({'wall_seconds':time.perf_counter()-w,'cpu_seconds':time.process_time()-c})
    return rows


def statistics(rows):
    return {name:float(np.median([r[name] for r in rows])) for name in ['wall_seconds','cpu_seconds']}


def run(root,parent,out,archive):
    root,out=Path(root),Path(out); out.mkdir(parents=True,exist_ok=True)
    if (out/'runtime-results.json').exists(): raise FileExistsError('Retain original results; use a fresh output directory')
    torch.set_num_threads(C['torch_threads'])
    audit(root,parent,out,archive)
    source=load_source(parent); train,dev=source['train'],source['dev']
    original=json.loads((root/'manifest.json').read_text())
    rate=fit_only_popularity(train); mean=original['duration_fit_mean']; scale=original['duration_fit_scale']
    cpu=next((s.split(':',1)[1].strip() for s in Path('/proc/cpuinfo').read_text().splitlines() if s.startswith('model name')),'unknown')
    report={'study':C['study'],'contract':C,'contract_sha256':sha(Path(__file__).resolve().parents[1]/'research/mathematical-engineering/me026/contract.json'),
        'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        'environment':{'python':platform.python_version(),'torch':str(torch.__version__),'numpy':np.__version__,
            'platform':platform.platform(),'cpu':cpu,'torch_threads':torch.get_num_threads()},
        'models':[],'new_fits':0,'new_test_outcomes':0,'architecture_promoted':False}
    models=[]
    # Every original and derived numerical gate must pass before any timing samples.
    for row in original['model_results']:
        if row['arm'] not in C['native_arms']: continue
        ck=torch.load(root/row['checkpoint'],weights_only=True)
        net=model_for(row['arm'],mean,scale,rate).eval(); net.load_state_dict(ck['state_dict'],strict=True)
        w,c=time.perf_counter(),time.process_time()
        compiled=torch.jit.script(NativeForecastRuntime(net,row['arm']=='tac_kernel')).eval()
        compile_wall,compile_cpu=time.perf_counter()-w,time.process_time()-c
        name=f"{row['arm']}-seed{row['seed']}"; path=out/f'{name}.jit.pt'
        torch.jit.save(compiled,str(path)); compiled=torch.jit.load(str(path)).eval()
        cp=prob_scores(compiled,dev); old=np.load(root/row['prediction_file'])
        error=float(np.max(np.abs(cp-old))); assert error<=C['runtime_probability_atol']
        np.save(out/f'{name}-compiled-predictions.npy',cp)
        n=C['stream_probe_windows']; state=torch.zeros(n,compiled.state_width)
        with torch.no_grad():
            for t in range(128): state=compiled.stream_step(dev['items'][:n,t],dev['metadata'][:n,t],state)
            prior=(dev['metadata'][:n,-32:,1].sum(-1,keepdim=True)+1)/34
            streaming=compiled.readout(state,dev['action'][:n],dev['duration'][:n],prior)
            direct=net(dev['items'][:n],dev['metadata'][:n],dev['action'][:n],dev['duration'][:n])
            le=float((streaming-direct).abs().max()); pe=float((streaming.sigmoid()-direct.sigmoid()).abs().max())
            assert le<=C['runtime_logit_atol'] and pe<=C['runtime_probability_atol']
            ne=float((state[:,:64]-net.encode(dev['items'][:n],dev['metadata'][:n])).abs().max())
            assert ne<=C['runtime_logit_atol']
            segment=kernel_segment(dev['metadata'][:n,:0,1])
            for a,b in [(0,11),(11,32),(32,97),(97,128)]:
                segment=compose_kernel_segments(segment,kernel_segment(dev['metadata'][:n,a:b,1]))
            raw=kernel_segment(dev['metadata'][:n,:,1])
            segment_error=float((segment[1]/segment[2]-raw[1]/raw[2]).abs().max())
            assert segment_error<=C['runtime_probability_atol']
        item={'arm':row['arm'],'seed':row['seed'],'parameters':sum(p.numel() for p in compiled.parameters()),
            'nontrainable_buffer_bytes':sum(b.numel()*b.element_size() for b in compiled.buffers()),
            'state_coordinates':compiled.state_width,'float32_state_bytes_per_entity':compiled.state_width*4,
            'separate_prior_fifo_float32_bytes_per_entity':128,'original_checkpoint_sha256':row['checkpoint_sha256'],
            'standalone':path.name,'standalone_sha256':sha(path),'compile_wall_seconds':compile_wall,
            'compile_cpu_seconds':compile_cpu,'maximum_original_probability_error':error,
            'maximum_stream_logit_error':le,'maximum_stream_probability_error':pe,
            'maximum_native_state_error':ne,'maximum_segment_normalized_error':segment_error,'timings':[]}
        assert item['parameters']==row['parameters']
        report['models'].append(item); models.append((net,compiled)); save_json(out/'numerical-gates.json',report)
    assert len(models)==6
    with torch.no_grad():
        for i,((net,compiled),item) in enumerate(zip(models,report['models'])):
            for b in C['benchmark_batches']:
                args=tuple(dev[k][:b] for k in ['items','metadata','action','duration'])
                eager=lambda:net(*args); optimized=lambda:compiled(*args)
                for _ in range(C['benchmark_warmup']): eager(); optimized()
                samples={'eager':[],'compiled':[]}
                for j in range(C['benchmark_repeats']):
                    order=['eager','compiled'] if (i+j)%2==0 else ['compiled','eager']
                    for name in order: samples[name].extend(measured(eager if name=='eager' else optimized,1))
                state=torch.zeros(b,compiled.state_width)
                for t in range(127): state=compiled.stream_step(args[0][:,t],args[1][:,t],state)
                prior=(args[1][:,-32:,1].sum(-1,keepdim=True)+1)/34
                def incremental():
                    advanced=compiled.stream_step(args[0][:,127],args[1][:,127],state)
                    return compiled.readout(advanced,args[2],args[3],prior)
                for _ in range(C['benchmark_warmup']): incremental()
                samples['incremental_step_and_forecast']=measured(incremental,C['benchmark_repeats'])
                stats={k:statistics(v) for k,v in samples.items()}
                item['timings'].append({'batch':b,'samples':samples,'medians':stats,
                    'equal_work_full_window_speedup':stats['eager']['wall_seconds']/stats['compiled']['wall_seconds'],
                    'incremental_is_different_work':True})
            save_json(out/'partial-runtime-results.json',report)
            print(json.dumps({'arm':item['arm'],'seed':item['seed'],'batch128_speedup':item['timings'][-1]['equal_work_full_window_speedup']}),flush=True)
    ratios=[m['timings'][-1]['equal_work_full_window_speedup'] for m in report['models']]
    report['primary_engineering_gate_passed']=all(v>=1/.9 for v in ratios)
    report['batch128_speedup_range']=[min(ratios),max(ratios)]
    report['batch128_geometric_mean_speedup']=float(np.exp(np.mean(np.log(ratios))))
    report['process_lifetime_peak_RSS_KiB']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    report['isolated_inference_memory_measured']=False
    save_json(out/'runtime-results.json',report)
    print(json.dumps({k:v for k,v in report.items() if k not in ['models','environment','contract']}))
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',default='/workspace/me025-workflow-original/results/me025')
    p.add_argument('--parent',default='results/me022');p.add_argument('--out',default='results/me026')
    p.add_argument('--archive',default='/workspace/me025-original-workflow.zip');a=p.parse_args()
    try:run(a.root,a.parent,a.out,a.archive)
    except Exception as e:
        save_json(Path(a.out)/'failure.json',{'error':repr(e),'new_fits':0,'instruction':'preserve this attempt; no silent retuning'})
        raise
