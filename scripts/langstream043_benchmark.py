#!/usr/bin/env python3
"""Frozen original-checkpoint incremental parity and same-history speed audit."""
import argparse,hashlib,importlib.util,io,json,os,platform,resource,statistics,sys,tempfile,time,urllib.request
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
STUDY=ROOT/'research/mathematical-engineering/langstream043'
sys.path[:0]=[str(ROOT),str(ROOT/'src'),str(STUDY)]
from incremental import IncrementalTAC043,SHAPES
from stream_probe import adapter
from tac_osm.research_probe import probe_memory
from scripts.autoresearch_memory042 import download,unique,sha
from scripts.langlayer035_sampling import fit_sources,schedule_receipt
from scripts.langlayer041_real_context_probe import model_for
from candidate import controlled_position


def load_reference():
    link=json.loads((STUDY/'PNDS_MATH_LINK.json').read_text());contents={}
    for path,key in [(link['python_path'],'python_sha256'),(link['lean_path'],'lean_sha256'),(link['record_path'],'record_sha256')]:
        with urllib.request.urlopen(f"https://raw.githubusercontent.com/FPC-effortless/pnds-lean/{link['pnds_commit']}/{path}",timeout=30) as response: raw=response.read()
        if sha(raw)!=link[key]: raise ValueError('pinned PNDS source differs')
        contents[path]=raw
    record=json.loads(contents[link['record_path']])
    if record['tac_commit']!=link['tac_commit'] or record['tac_source_sha256']!=sha((STUDY/'incremental.py').read_bytes()): raise ValueError('reciprocal source differs')
    with tempfile.TemporaryDirectory() as tmp:
        p=Path(tmp)/'reference.py';p.write_bytes(contents[link['python_path']])
        spec=importlib.util.spec_from_file_location('independent_stream043',p);ref=importlib.util.module_from_spec(spec);spec.loader.exec_module(ref)
    return ref


def compare(a,b,tol):
    if a.shape!=b.shape or not torch.isfinite(a).all() or not torch.isfinite(b).all(): raise ValueError('invalid correspondence')
    d=(a-b).abs();scale=tol['atol']+tol['rtol']*torch.maximum(a.abs(),b.abs())
    absolute=float(d.max());scaled=float((d/scale).max())
    if scaled>1: raise ValueError(f'parity failed: abs={absolute}, scaled={scaled}')
    return absolute,scaled


def parity(model,raw,owner,mode,ref,tol):
    runner=IncrementalTAC043(model,mode);x=torch.tensor([list(raw)],dtype=torch.long)
    with torch.no_grad(),controlled_position(model,mode): expected,tr=model(x,return_trace=True)
    state=runner.start((owner,));rs=ref.initial(dict(model.named_parameters()),1)
    errors={'logits':0.,'independent_logits':0.,'native_trace':0.,'all_state':0.,'scaled':0.}
    with torch.no_grad():
        for t in range(x.shape[1]):
            mask=torch.ones(1,dtype=torch.bool)
            out,state=runner.step(x[:,t],state,owners=(owner,))
            rout,rs=ref.step(dict(model.named_parameters()),rs,x[:,t],t,mask,mode)
            for name,a,b in [('logits',out,expected[:,t]),('independent_logits',out,rout),('native_trace',state.tensors['native'],tr['native_memory'][:,t])]:
                absolute,scaled=compare(a,b,tol);errors[name]=max(errors[name],absolute);errors['scaled']=max(errors['scaled'],scaled)
            for name in SHAPES:
                absolute,scaled=compare(state.tensors[name],rs[name],tol);errors['all_state']=max(errors['all_state'],absolute);errors['scaled']=max(errors['scaled'],scaled)
        last,bulk=runner.prefill(x,owners=(owner,))
        compare(last,expected[:,-1],tol)
        for name in SHAPES: compare(bulk.tensors[name],state.tensors[name],tol)
    if state.tensor_bytes()!=976*model.embedding.weight.element_size(): raise ValueError('retained payload grew')
    return {'owner':owner,'source_prefix_sha256':sha(raw),'bytes':len(raw),'position_mode':mode,'dtype':str(model.embedding.weight.dtype),'errors':errors,'state_tensor_bytes':state.tensor_bytes()}


def distribution(samples):
    return {'median_ms':statistics.median(samples),'p95_ms':float(np.percentile(samples,95)),'raw_ms':samples}


def timed_decode(model,prompt,owner,mode,stream):
    x=torch.tensor([list(prompt)],dtype=torch.long);setup=time.perf_counter_ns()
    runner=IncrementalTAC043(model,mode) if stream else None
    setup_ms=(time.perf_counter_ns()-setup)/1e6
    start=time.perf_counter_ns()
    with torch.no_grad():
        if stream: logits,state=runner.prefill(x,owners=(owner,))
        else:
            with controlled_position(model,mode): logits=model(x)[:,-1]
        prefill=(time.perf_counter_ns()-start)/1e6
        outputs=[];times=[];full=x
        for _ in range(16):
            token=logits.argmax(-1);outputs.append(int(token[0]));tick=time.perf_counter_ns()
            if stream: logits,state=runner.step(token,state,owners=(owner,))
            else:
                full=torch.cat((full,token[:,None]),1)
                with controlled_position(model,mode): logits=model(full)[:,-1]
            times.append((time.perf_counter_ns()-tick)/1e6)
    return {'engine_setup_ms':setup_ms,'prefill_ms':prefill,'decode_ms':times,'total_prefill_and_updates_ms':prefill+sum(times),'generated_bytes':outputs,'last_logits':logits,'state_tensor_bytes':state.tensor_bytes() if stream else None}


def benchmark(model,fit):
    owner=sorted(fit)[0];raw=fit[owner];rows=[]
    for mode in ('native','cap_all'):
        for length in (32,128,512):
            full=[];incremental=[]
            for trial in range(9):
                # Alternate execution order to reduce systematic warm/cache bias.
                paths=(False,True) if trial%2==0 else (True,False)
                runs={stream:timed_decode(model,raw[:length],owner,mode,stream) for stream in paths}
                a,b=runs[False],runs[True]
                if a['generated_bytes']!=b['generated_bytes']: raise ValueError('greedy sequence mismatch; speed claim blocked')
                compare(a['last_logits'],b['last_logits'],{'atol':3e-5,'rtol':1e-4})
                if trial>=2:
                    for target,record in ((full,a),(incremental,b)):
                        target.append({k:v for k,v in record.items() if k!='last_logits'})
            original=distribution([v for row in full for v in row['decode_ms']])
            updated=distribution([v for row in incremental for v in row['decode_ms']])
            rows.append({'position_mode':mode,'owner':owner,'prompt_bytes':length,'generated_bytes_per_repeat':16,'measured_repeats':7,'full_prefix_decode':original,'incremental_decode':updated,'decode_speedup':original['median_ms']/updated['median_ms'],
                         'incremental_engine_setup':distribution([r['engine_setup_ms'] for r in incremental]),'full_prefix_prefill':distribution([r['prefill_ms'] for r in full]),'incremental_bulk_prefill':distribution([r['prefill_ms'] for r in incremental]),
                         'full_prefix_total':distribution([r['total_prefill_and_updates_ms'] for r in full]),'incremental_total':distribution([r['total_prefill_and_updates_ms'] for r in incremental]),
                         'greedy_identity':True,'retained_tensor_bytes':incremental[0]['state_tensor_bytes'],'original_generated_bytes':full[0]['generated_bytes']})
    return rows


def main():
    p=argparse.ArgumentParser();p.add_argument('--output-dir',type=Path,required=True);a=p.parse_args();out=a.output_dir
    if out.exists(): raise FileExistsError('preserve evidence')
    out.mkdir(parents=True);torch.set_num_threads(2)
    pins=json.loads((STUDY/'SOURCE_PINS.json').read_text())
    for path,digest in pins.items():
        if sha((ROOT/path).read_bytes())!=digest: raise ValueError('frozen source differs '+path)
    ref=load_reference();protocol=json.loads((STUDY/'PROTOCOL.json').read_text())
    fit=fit_sources();artifacts=json.loads((ROOT/'research/mathematical-engineering/langmemory042/ORIGINAL_ARTIFACTS.json').read_text())
    rows=[];originals=[];timings=None;probe=None
    def record_case(seed,m,data,owner,mode,tol):
        try: row={'seed':seed,**parity(m,data,owner,mode,ref,tol),'status':'PASS_BOUNDED_NUMERICAL'}
        except Exception as error:
            with (out/'PARITY.jsonl').open('a') as handle: handle.write(json.dumps({'seed':seed,'owner':owner,'bytes':len(data),'mode':mode,'dtype':str(m.embedding.weight.dtype),'status':'FAIL_PRESERVED','reason':str(error)})+'\n')
            raise
        with (out/'PARITY.jsonl').open('a') as handle: handle.write(json.dumps(row,sort_keys=True,allow_nan=False)+'\n')
        rows.append(row)
    for seed in range(1701,1706):
        item=next(row for row in artifacts if row['name'].endswith('seed'+str(seed)))
        root=download(item,out/f'original-{seed}',40797105091);cp=unique(root,'*.pt');raw=cp.read_bytes()
        receipt=json.loads(unique(root,'FIT_AND_EXPOSED_DEV_RECEIPT.json').read_text());saved=torch.load(io.BytesIO(raw),map_location='cpu',weights_only=True)
        if sha(raw)!=receipt['checkpoint_sha256']: raise ValueError('original checkpoint hash')
        for entry in (receipt,saved):
            if (entry['study'],entry['arm'],entry['seed'],entry['updates'])!=('LANGLAYER040','state48_no_graph',seed,320): raise ValueError('wrong original model')
        source=schedule_receipt(fit,seed,steps=320)
        if saved['source_commit']!='60403a5409ff2c3f3b07dd2ca91a7a3e096839c7' or saved['sample_order_sha256']!=source['sample_order_sha256'] or receipt['source_sample_order_sha256']!=source['sample_order_sha256'] or receipt['source_bytes_sha256']!=source['sample_bytes_sha256']: raise ValueError('original source schedule')
        if any(k not in saved for k in ('optimizer','rng_python','rng_numpy','rng_torch')) or not saved['optimizer']['state']: raise ValueError('missing optimizer/RNG')
        originals.append({'seed':seed,'checkpoint_sha256':sha(raw),'sample_order_sha256':source['sample_order_sha256']})
        for precision,tol in protocol['precisions'].items():
            m=model_for('h48_plain').to(getattr(torch,precision)).eval();m.load_state_dict(saved['model'],strict=True)
            if sum(p.numel() for p in m.parameters())!=499623: raise ValueError('model count changed')
            for mode in protocol['position_modes']:
                for owner,source_raw in sorted(fit.items()):
                    record_case(seed,m,source_raw[:33],owner,mode,tol)
                owner=sorted(fit)[0];record_case(seed,m,fit[owner][:513],owner,mode,tol)
        if seed==1701:
            m=model_for('h48_plain').eval();m.load_state_dict(saved['model'],strict=True)
            owner=sorted(fit)[0];x=torch.tensor([list(fit[owner][:17])])
            sample={'tokens':x,'owners':(owner,),'write_mask':torch.ones_like(x,dtype=torch.bool)}
            probe=probe_memory(m,sample,adapter('cap_all'))
            r=IncrementalTAC043(m,'cap_all');_,state=r.prefill(x[:,:7],owners=(owner,));path=out/'inference-state.pt';r.save_state(state,path)
            restored=IncrementalTAC043(m,'cap_all').load_state(path,owners=(owner,))
            for t in range(7,17):
                y,state=r.step(x[:,t],state,owners=(owner,));z,restored=r.step(x[:,t],restored,owners=(owner,))
                if not torch.equal(y,z) or any(not torch.equal(state.tensors[n],restored.tensors[n]) for n in SHAPES): raise ValueError('snapshot restart not exact')
            timings=benchmark(m,fit)
            if any(not torch.equal(v,saved['model'][n]) for n,v in m.state_dict().items()): raise ValueError('weights mutated')
        if sha(cp.read_bytes())!=originals[-1]['checkpoint_sha256']: raise ValueError('original binary mutated')
    result={'study':'LANGSTREAM043','status':'PASS_ORIGINAL_INCREMENTAL_PARITY_AND_SAME_HISTORY_SPEED','source_checkpoint_count':5,'parity_cases':rows,'benchmark':timings,'originals':originals,'state_payload_float_coordinates':976,'recurrent_and_context_coordinates':720,'current_logit_coordinates':256,'state_payload_bytes_FP32':3904,'learned_parameters':499623,'original_weight_tensor_bytes':sum(v.numel()*v.element_size() for v in saved['model'].values()),'training_updates':0,'source_partition':'AUTHENTIC_PSF_FIT_NO_DEV_TEST_SCORING','upper_and_base_probe':probe,'restart':'BITWISE_EXACT_ON_TESTED_ORIGINAL1701_SEQUENCE','process_peak_RSS_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,'hardware':{'machine':platform.machine(),'processor':next((line.split(':',1)[1].strip() for line in Path('/proc/cpuinfo').read_text().splitlines() if line.startswith('model name')),platform.processor()),'logical_cpus':os.cpu_count(),'torch':str(torch.__version__),'threads':torch.get_num_threads()},'semantics':'growing source prefix; no rolling-window eviction','limits':['one CPU timing seed, seven repeats,16 update steps','no KV-Transformer/BDH/SSM ranking','no fresh accuracy or learned long-memory claim','no mobile/RAM/energy qualification','FP reassociation tolerance only; formal IEEE/AD/source bridge OPEN'],'protected_TEST':False,'GPU_minutes':0,'checkpoint_promotion':False,'independently_qualified':False}
    (out/'RESULTS.json').write_text(json.dumps(result,indent=2,sort_keys=True,allow_nan=False)+'\n')
    print(json.dumps({'status':result['status'],'cases':len(rows),'benchmark':[{'mode':r['position_mode'],'prompt':r['prompt_bytes'],'decode_speedup':r['decode_speedup'],'baseline_ms':r['full_prefix_decode']['median_ms'],'incremental_ms':r['incremental_decode']['median_ms'],'baseline_total_ms':r['full_prefix_total']['median_ms'],'incremental_total_ms':r['incremental_total']['median_ms']} for r in timings]}))

if __name__=='__main__': main()
