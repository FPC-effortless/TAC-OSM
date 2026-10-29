#!/usr/bin/env python3
"""Frozen bounded harness: FIT selects one of two maps, then DEV confirms.
Only candidate.py is the trial edit surface. No optimizer/TEST/promotion.
"""
import argparse,hashlib,io,json,math,os,resource,statistics,subprocess,sys,time,zipfile
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
ROOT=Path(__file__).resolve().parents[1]
STUDY=ROOT/'research/mathematical-engineering/langmemory042'
sys.path[:0]=[str(ROOT),str(STUDY)]
from candidate import controlled_position,MODES
from scripts.langlayer041_real_context_probe import model_for,aligned_source_targets
from scripts.langlayer035_sampling import fit_sources,schedule_receipt

SEEDS=(1701,1702,1703,1704,1705)

def sha(raw): return hashlib.sha256(raw).hexdigest()

def download(row,dest,run):
    meta=json.loads(subprocess.check_output(['gh','api',f'repos/FPC-effortless/TAC-OSM/actions/artifacts/{row["id"]}']))
    if meta['id']!=row['id'] or meta['name']!=row['name'] or meta['expired'] or meta['digest']!=row['digest'] or meta['workflow_run']['id']!=run:
        raise ValueError('original artifact identity changed')
    zipped=dest.with_suffix('.zip')
    with zipped.open('xb') as f:
        subprocess.run(['gh','api',f'repos/FPC-effortless/TAC-OSM/actions/artifacts/{row["id"]}/zip'],stdout=f,check=True)
    if 'sha256:'+sha(zipped.read_bytes())!=row['digest']: raise ValueError('original ZIP differs')
    dest.mkdir()
    with zipfile.ZipFile(zipped) as z:
        if sum(i.file_size for i in z.infolist())>64*1024**2: raise ValueError('oversize archive')
        for i in z.infolist():
            p=Path(i.filename)
            if p.is_absolute() or '..' in p.parts or (i.external_attr>>16)&0o170000==0o120000: raise ValueError('unsafe original archive')
        z.extractall(dest)
    return dest


def unique(root,pattern):
    files=list(root.rglob(pattern))
    if len(files)!=1: raise ValueError('ambiguous original payload '+pattern)
    return files[0]


def load_original(seed,directory,fit):
    manifest=json.loads((STUDY/'ORIGINAL_ARTIFACTS.json').read_text())
    row=next(r for r in manifest if r['name'].endswith('seed'+str(seed)))
    original=download(row,directory/f'original-{seed}',40797105091)
    cp=unique(original,'*.pt');receipt=unique(original,'FIT_AND_EXPOSED_DEV_RECEIPT.json')
    old=json.loads(receipt.read_text());raw=cp.read_bytes();saved=torch.load(io.BytesIO(raw),map_location='cpu',weights_only=True)
    if sha(raw)!=old['checkpoint_sha256']: raise ValueError('original checkpoint differs')
    for item in (old,saved):
        if (item['study'],item['arm'],item['seed'],item['updates'])!=('LANGLAYER040','state48_no_graph',seed,320): raise ValueError('original study differs')
    source=schedule_receipt(fit,seed,steps=320)
    if saved['source_commit']!='60403a5409ff2c3f3b07dd2ca91a7a3e096839c7' or saved['sample_order_sha256']!=source['sample_order_sha256'] or old['source_sample_order_sha256']!=source['sample_order_sha256'] or old['source_bytes_sha256']!=source['sample_bytes_sha256']:
        raise ValueError('original source/schedule differs')
    if any(k not in saved for k in ('optimizer','rng_torch','rng_numpy','rng_python')) or not saved['optimizer']['state']: raise ValueError('missing original optimizer/RNG')
    model=model_for('h48_plain').eval()
    if sum(p.numel() for p in model.parameters())!=499623: raise ValueError('model size differs')
    model.load_state_dict(saved['model'],strict=True)
    hrows=json.loads((STUDY/'HISTORY_ARTIFACTS.json').read_text());hrow=next(r for r in hrows if r['name'].endswith('seed'+str(seed)))
    history=download(hrow,directory/f'history-{seed}',40797642418)
    historical=json.loads(unique(history,'*.json').read_text())
    if (historical['study'],historical['arm'],historical['seed'],historical['original_checkpoint_sha256'])!=('LANGLAYER041','h48_plain',seed,sha(raw)): raise ValueError('historical replay receipt differs')
    return model,cp,saved,{'seed':seed,'checkpoint_sha256':sha(raw),'receipt_sha256':sha(receipt.read_bytes()),'sample_order_sha256':source['sample_order_sha256'],'historical':historical}


def score(model,docs,schedule,length,mode,donor=False):
    loss=0.;correct=0;count=0;clock=time.process_time()
    with torch.no_grad(),controlled_position(model,mode):
        for owner in sorted(schedule):
            raw=docs[owner];xs=[];ys=[]
            for pos in schedule[owner]['target_offsets']:
                observed=raw[pos-length:pos]
                if donor: observed=raw[:length-64]+raw[pos-64:pos]
                if len(observed)!=length: raise ValueError('source boundary differs')
                xs.append(list(observed));ys.append(raw[pos])
            logits=model(torch.tensor(xs,dtype=torch.long))[:,-1].double()
            gold=torch.tensor(ys,dtype=torch.long)
            loss+=float(-F.log_softmax(logits,-1).gather(1,gold[:,None]).sum())
            correct+=int((logits.argmax(-1)==gold).sum());count+=len(ys)
    if not count or not math.isfinite(loss): raise ValueError('invalid real score')
    return {'n':count,'NLL':loss/count,'BPB':loss/count/math.log(2),'top1':correct/count,'cpu_process_seconds':time.process_time()-clock}


def main():
    p=argparse.ArgumentParser();p.add_argument('--output-dir',type=Path,required=True);a=p.parse_args()
    out=a.output_dir
    if out.exists(): raise FileExistsError('preserve existing experiment')
    out.mkdir(parents=True);torch.set_num_threads(2)
    pins=json.loads((STUDY/'SOURCE_PINS.json').read_text())
    link=json.loads((STUDY/'PNDS_MATH_LINK.json').read_text())
    import urllib.request,importlib.util,tempfile
    maps={}
    for path,key in [(link['python_path'],'python_sha256'),(link['lean_path'],'lean_sha256'),(link['record_path'],'record_sha256')]:
        with urllib.request.urlopen(f"https://raw.githubusercontent.com/FPC-effortless/pnds-lean/{link['pnds_commit']}/{path}",timeout=30) as response: content=response.read()
        if sha(content)!=link[key]: raise ValueError('pinned PNDS formulation differs')
        maps[path]=content
    rec=json.loads(maps[link['record_path']])
    if rec['tac_commit']!=link['tac_commit'] or rec['tac_source_sha256']!=pins['research/mathematical-engineering/langmemory042/candidate.py']: raise ValueError('reciprocal candidate pin differs')
    with tempfile.TemporaryDirectory() as temp:
        path=Path(temp)/'independent.py';path.write_bytes(maps[link['python_path']])
        spec=importlib.util.spec_from_file_location('independent042',path);ref=importlib.util.module_from_spec(spec);spec.loader.exec_module(ref)
        from candidate import transform
        from context_transition import phi_position
        features=phi_position(torch.arange(513),torch.float64)
        for mode in MODES:
            expected=torch.tensor([ref.transform(row.tolist(),mode) for row in features],dtype=torch.float64)
            if not torch.allclose(transform(features,mode),expected,atol=1e-12,rtol=1e-12): raise ValueError('independent feature map differs')
    for path,digest in pins.items():
        if sha((ROOT/path).read_bytes())!=digest: raise ValueError('frozen source differs: '+path)
    fit=fit_sources();cal={}
    for name,raw in sorted(fit.items()):
        if len(raw)>=1282:
            cal[name]={'bytes_sha256':sha(raw),'target_offsets':np.linspace(1024,len(raw)-2,4,dtype=np.int64).tolist()}
    if len(cal)<8: raise ValueError('too few preregistered FIT owners')
    loaded=[];fitrows=[]
    def event(row):
        with (out/'events.jsonl').open('a') as handle: handle.write(json.dumps(row,sort_keys=True,allow_nan=False)+'\n')
    for seed in SEEDS:
        m,cp,saved,receipt=load_original(seed,out,fit);loaded.append((m,cp,saved,receipt))
        for mode in MODES:
            event({'seed':seed,'mode':mode,'partition':'FIT_SELECTION','status':'RUNNING'})
            try: row={'seed':seed,'mode':mode,'partition':'FIT_SELECTION',**score(m,fit,cal,512,mode)}
            except Exception as error:
                event({'seed':seed,'mode':mode,'status':'FAILED_PRESERVED','reason':str(error)});raise
            fitrows.append(row);event({**row,'status':'MEASURED_INTERNAL'})
    means={mode:statistics.mean(r['NLL'] for r in fitrows if r['mode']==mode) for mode in MODES}
    selected='native';eligible=[]
    for mode in MODES[1:]:
        wins=sum(next(r['NLL'] for r in fitrows if r['seed']==s and r['mode']==mode)<next(r['NLL'] for r in fitrows if r['seed']==s and r['mode']=='native') for s in SEEDS)
        if means[mode]<=means['native']-.005 and wins>=4: eligible.append(mode)
    if eligible: selected=min(eligible,key=lambda name:(means[name],MODES.index(name)))
    selection={'study':'LANGMEMORY042','selected':selected,'FIT_means':means,'rows':fitrows,'source_docs':cal,'candidate_trial_count':2,'DEV_inference_started':False,'promotion':False}
    selection_path=out/'SELECTION_BEFORE_DEV.json';selection_path.write_text(json.dumps(selection,indent=2,sort_keys=True,allow_nan=False)+'\n')
    selection_hash=sha(selection_path.read_bytes())
    # Only now may the confirmation phase call DEV models. Never select on DEV.
    dev,schedule=aligned_source_targets();rows=[]
    for m,cp,saved,receipt in loaded:
        seed=receipt['seed'];historical=receipt.pop('historical')
        if historical['source_docs']!=schedule: raise ValueError('old041 targets differ')
        x=torch.tensor([list(dev[owner][pos-128:pos]) for owner,item in schedule.items() for pos in item['target_offsets']])
        with torch.no_grad():
            native=m(x)
            for mode in MODES[1:]:
                with controlled_position(m,mode):
                    if not torch.equal(native,m(x)): raise ValueError('short-window identity FAIL')
        for mode in dict.fromkeys(('native',selected)):
            for length in (128,256,512):
                genuine=score(m,dev,schedule,length,mode)
                donor=score(m,dev,schedule,length,mode,True)
                if mode=='native':
                    old=historical['lengths'][str(length)]
                    if abs(genuine['NLL']-old['genuine_nll'])>1e-6 or abs(donor['NLL']-old['distant_donor_nll'])>1e-6: raise ValueError('original041 numerical replay FAIL')
                rows.append({'seed':seed,'mode':mode,'length':length,'genuine':genuine,'donor':donor,'donor_minus_genuine':donor['NLL']-genuine['NLL']})
        if sha(cp.read_bytes())!=receipt['checkpoint_sha256'] or any(not torch.equal(value,saved['model'][name]) for name,value in m.state_dict().items()): raise ValueError('original weights mutated')
    if sha(selection_path.read_bytes())!=selection_hash: raise ValueError('FIT selection mutated after DEV')
    ledger=[]
    for mode in MODES:
        ledger.append({'mode':mode,'partition':'FIT_SELECTION','mean_NLL':means[mode],'BPB':means[mode]/math.log(2),'status':'baseline' if mode=='native' else 'selected_for_internal_confirmation' if mode==selected else 'not_selected_preserved','promotion':False})
    summary={mode:{str(length):{'NLL':statistics.mean(r['genuine']['NLL'] for r in rows if r['mode']==mode and r['length']==length),'BPB':statistics.mean(r['genuine']['BPB'] for r in rows if r['mode']==mode and r['length']==length),'donor_minus_genuine':statistics.mean(r['donor_minus_genuine'] for r in rows if r['mode']==mode and r['length']==length)} for length in (128,256,512)} for mode in dict.fromkeys(('native',selected))}
    result={'study':'LANGMEMORY042','status':'COMPLETE_INTERNAL_FROZEN_CHECKPOINT_DIAGNOSTIC','selected_by_FIT':selected,'selection_sha256_before_DEV':selection_hash,'DEV_sources':schedule,'summary':summary,'rows':rows,'originals':[r for _,_,_,r in loaded],'ledger':ledger,'learned_parameters':499623,'weight_tensor_bytes':sum(p.numel()*p.element_size() for p in loaded[0][0].parameters()),'process_peak_RSS_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,'RSS_scope':'entire Python process including runtime, all five loaded models, artifacts and scan temporaries; not per-arm RAM or mobile energy','training_updates':0,'candidate_trials':2,'DEV_targets_per_seed':32,'protected_TEST':False,'checkpoint_promotion':False,'independently_qualified':False,'long_history_intelligence':'NOT_ESTABLISHED_BY_CLOCK_DIAGNOSTIC','scientific_limits':['exposed small DEV cohort','selection uses training-exposed FIT','five seeds share32 source targets','not matched quality/cost versus SSM/Transformer','full model IEEE/AD remains OPEN','no dual-memory/reconstruction implemented']}
    (out/'RESULTS.json').write_text(json.dumps(result,indent=2,sort_keys=True,allow_nan=False)+'\n')
    (out/'results.jsonl').write_text(''.join(json.dumps(row,sort_keys=True)+'\n' for row in ledger))
    print(json.dumps({'status':result['status'],'selected':selected,'summary':summary}))

if __name__=='__main__': main()
