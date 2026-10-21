"""ME037 complete-model training and prospective/new-site evaluation."""
import argparse,concurrent.futures,csv,json,math,multiprocessing,platform,resource,time,traceback
from datetime import datetime
from pathlib import Path
import numpy as np
import torch
import scripts.me036_research as old
from tac_osm.me036_patch_forecast import build_model as old_model
from tac_osm.me037_horizon_forecast import ARMS,build_model

ROOT=old.ROOT;OUT=ROOT/'results/me037';REPORT=ROOT/'research/mathematical-engineering/me037'
DATA=ROOT/'results/me037-source'
DATASETS=(*old.DATASETS,'appliances');HORIZONS=old.HORIZONS;SEEDS=(37001,37002)
CACHE={}
sha,dump,batch=old.sha,old.dump,old.batch


def data(name):
    if name in CACHE:return CACHE[name]
    if name!='appliances':value=old.load_data(name)
    else:
        manifest=json.loads((REPORT/'sources.json').read_text())
        path=DATA/'energydata_complete.csv'
        assert sha(path)==manifest[path.name]['sha256']
        columns=['Appliances','lights']+[v for i in range(1,10) for v in (f'T{i}',f'RH_{i}')]
        reader=csv.DictReader(path.open());rows=list(reader)
        a=np.array([[float(r[k]) for k in columns] for r in rows],dtype=np.float32)
        dates=[datetime.fromisoformat(r['date']) for r in rows]
        assert len(a)==19735 and np.isfinite(a).all()
        assert all((b-a).total_seconds()==600 for a,b in zip(dates,dates[1:])), 'unexpected source cadence'
        n=len(a);train=int(.7*n);val=n-int(.2*n)
        mean=a[:train].mean(0,dtype=np.float64).astype(np.float32)
        scale=np.maximum(a[:train].std(0,dtype=np.float64),1e-5).astype(np.float32)
        z=torch.from_numpy(((a-mean)/scale).copy())
        value=(z,{'name':name,'source':manifest[path.name],'rows':n,'channels':len(columns),'columns':columns,
            'split_edges':[train,val,n],'mean':mean.tolist(),'scale':scale.tolist(),
            'first_timestamp':str(dates[0]),'last_timestamp':str(dates[-1]),'cadence_seconds':600,
            'timezone':'publisher unspecified','excluded_columns':['rv1','rv2','T_out','Press_mm_hg','RH_out','Windspeed','Visibility','Tdewpoint'],
            'publisher':'UCI374 / Luis Candanedo, DOI10.24432/C5VC8G','license':'CC BY4.0, publisher page link verified',
            'measurements':'Energy meters and ZigBee temperature/humidity; no action/cost labels'})
    CACHE[name]=value
    return value


@torch.no_grad()
def score(model,z,first,end,h,stride=1):
    start=time.process_time();model.eval();squared=np.zeros(z.size(1));absolute=np.zeros(z.size(1));n=0
    ix=np.arange(first,end-h+1,stride)
    for i in range(0,len(ix),16):
        x,y=batch(z,ix[i:i+16],h);p=model(x)
        if not torch.isfinite(p).all():raise ArithmeticError('nonfinite forecasts')
        e=(p-y).double();squared+=e.square().sum((0,1)).numpy();absolute+=e.abs().sum((0,1)).numpy();n+=e.size(0)*h
    return {'MSE':float(squared.sum()/(n*z.size(1))),'MAE':float(absolute.sum()/(n*z.size(1))),
        'channel_MSE':(squared/n).tolist(),'channel_MAE':(absolute/n).tolist(),
        'scalar_targets':n*z.size(1),'windows':len(ix),'stride':stride,'CPU_seconds':time.process_time()-start}


def save_endpoint(folder,model,opt,gen,step,arm,h,seed,z,meta,charged):
    path=folder/f'u{step}.pt'
    torch.save({'state_dict':model.state_dict(),'optimizer':opt.state_dict(),'sampler_rng':gen.get_state(),
        'model_rng':torch.get_rng_state(),'step':step,'arm':arm,'horizon':h,'seed':seed},path)
    tr,va,_=meta['split_edges'];metric=score(model,z,tr,va,h,24)
    x,_=batch(z,[tr,tr+24],h)
    with torch.no_grad():np.save(folder/f'u{step}-probe.npy',model(x).numpy())
    return {'step':step,'validation':metric,'checkpoint':str(path.relative_to(ROOT)),'sha256':sha(path),
        'training_CPU_seconds':charged,'probe_sha256':sha(folder/f'u{step}-probe.npy')}


def parent_job(task):
    torch.set_num_threads(1);name,h,seed=task;z,meta=data(name)
    folder=OUT/f'parent-{name}-h{h}-s{seed}';folder.mkdir(parents=True,exist_ok=True)
    if (folder/'fit.json').exists():return json.loads((folder/'fit.json').read_text())
    start=time.process_time();model=old_model('ar',h,seed)
    opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
    gen=torch.Generator().manual_seed(seed+1000);charged=time.process_time()-start;trace=[];ends=[];step=0
    try:
        for step in range(1,1001):
            t=time.process_time();lr=.001 if step<=100 else .0001+.0009*.5*(1+math.cos(math.pi*(step-101)/899))
            for group in opt.param_groups:group['lr']=lr
            model.train();x,y=batch(z,torch.randint(96,meta['split_edges'][0]-h+1,(4,),generator=gen),h)
            opt.zero_grad(set_to_none=True);loss=(model(x)-y).square().mean()
            if not torch.isfinite(loss):raise ArithmeticError('nonfinite parent loss')
            loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            if not torch.isfinite(norm):raise ArithmeticError('nonfinite parent gradient')
            opt.step();dt=time.process_time()-t;charged+=dt
            trace.append({'step':step,'MSE':loss.item(),'CPU_seconds':dt,'learning_rate':lr})
            if step in (25,50,100,250,500,1000):ends.append(save_endpoint(folder,model,opt,gen,step,'ar',h,seed,z,meta,charged))
            if charged>120:raise RuntimeError('registered parent120CPU-second limit')
        rec={'status':'PASS','dataset':name,'horizon':h,'seed':seed,'updates':1000,'windows_seen':4000,
            'training_CPU_seconds':charged,'endpoints':ends,'selected':min(ends,key=lambda e:(e['validation']['MSE'],e['step'])),
            'origin':'new_household_pretraining'}
    except Exception:
        torch.save({'state_dict':model.state_dict(),'optimizer':opt.state_dict(),'step':step,'sampler_rng':gen.get_state()},folder/'partial.pt')
        rec={'status':'FAIL','dataset':name,'horizon':h,'seed':seed,'updates':step,'endpoints':ends,'error':traceback.format_exc()}
    dump(folder/'trace.json',trace);dump(folder/'fit.json',rec);print('PARENT',name,h,seed,rec['status'],flush=True)
    return rec


def prepare():
    OUT.mkdir(parents=True,exist_ok=True)
    dump(OUT/'data.json',[data(d)[1] for d in DATASETS])
    original=json.loads((ROOT/'results/me036b/frozen.json').read_text());parents=[]
    for r in original['fits']:
        if r['arm']!='ar':continue
        for e in r['endpoints']:assert sha(ROOT/e['checkpoint'])==e['sha256']
        parents.append({'status':r['status'],'dataset':r['dataset'],'horizon':r['horizon'],'seed':r['seed'],
            'updates':r['updates'],'windows_seen':r['examples_seen'],'training_CPU_seconds':r['charged_training_CPU_seconds'],
            'selected':r['selected'],'origin':'cached_ME036_AR_only','endpoints':r['endpoints']})
    with concurrent.futures.ProcessPoolExecutor(max_workers=2,mp_context=multiprocessing.get_context('spawn')) as pool:
        parents+=list(pool.map(parent_job,[('appliances',h,s) for h in HORIZONS for s in (36001,36002)]))
    assert len(parents)==48
    dump(OUT/'parents.json',{'protocol_sha256':sha(REPORT/'PROTOCOL.md'),'parents':parents})


def fit_job(task):
    torch.set_num_threads(1);name,h,arm,seed,parent=task;z,meta=data(name)
    key=f'{name}-h{h}-{arm}-s{seed}';folder=OUT/key;folder.mkdir(parents=True,exist_ok=True)
    if (folder/'fit.json').exists():return json.loads((folder/'fit.json').read_text())
    start=time.process_time();selected=parent['selected'];assert sha(ROOT/selected['checkpoint'])==selected['sha256']
    state=torch.load(ROOT/selected['checkpoint'],weights_only=False)['state_dict']
    model=build_model(arm,h,seed);model.initialize_ar(state)
    reference=old_model('ar',h,parent['seed']);reference.load_state_dict(state);reference.eval()
    tr,va,_=meta['split_edges'];probe,_=batch(z,[tr,tr+24],h)
    with torch.no_grad():
        initial=model(probe);assert torch.equal(initial,reference(probe)), 'initialAR bit equality failed'
        np.save(folder/'initial-probe.npy',initial.numpy())
    extra=[p for n,p in model.named_parameters() if not n.startswith('ar.')]
    groups=[{'params':list(model.ar.parameters()),'lr':.0001}]
    if extra:groups.append({'params':extra,'lr':.001})
    opt=torch.optim.AdamW(groups,weight_decay=.0001);gen=torch.Generator().manual_seed(seed+2000)
    charged=time.process_time()-start;trace=[];ends=[];step=0
    try:
        for step in range(1,501):
            t=time.process_time();model.train();x,y=batch(z,torch.randint(96,tr-h+1,(4,),generator=gen),h)
            opt.zero_grad(set_to_none=True);pred,delta,_=model.parts(x)
            mse=(pred-y).square().mean();anchor=model.anchor_loss();penalty=delta.square().mean()
            loss=mse+10.*anchor+.01*penalty
            if not torch.isfinite(loss):raise ArithmeticError('nonfinite complete-model loss')
            loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            if not torch.isfinite(norm):raise ArithmeticError('nonfinite complete-model gradient')
            memory=sum(p.grad.square().sum().item() for n,p in model.named_parameters() if n.startswith(('memory.','gru.')) and p.grad is not None)**.5
            opt.step();dt=time.process_time()-t;charged+=dt
            trace.append({'step':step,'MSE':mse.item(),'loss':loss.item(),'anchor_loss':anchor.item(),
                'correction_energy':penalty.item(),'CPU_seconds':dt,'gradient_norm':norm.item(),'memory_gradient_after_clip':memory})
            if step in (100,250,500):ends.append(save_endpoint(folder,model,opt,gen,step,arm,h,seed,z,meta,charged))
            if charged>120:raise RuntimeError('registered adaptation120CPU-second cap')
        if arm in ('tac_horizon','gru_horizon'):assert any(t['memory_gradient_after_clip']>0 for t in trace[1:])
        rec={'status':'PASS','key':key,'dataset':name,'horizon':h,'arm':arm,'seed':seed,'updates':step,
            'parameters':sum(p.numel() for p in model.parameters()),'joint_windows_seen':step*4,'joint_CPU_seconds':charged,
            'parent':parent,'pipeline_training_CPU_seconds':charged+parent['training_CPU_seconds'],
            'pipeline_windows_seen':step*4+parent['windows_seen'],'initial_AR_bit_exact':True,
            'endpoints':ends,'selected':min(ends,key=lambda e:(e['validation']['MSE'],e['step'])),
            'process_high_water_RSS_KiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
    except Exception:
        torch.save({'state_dict':model.state_dict(),'optimizer':opt.state_dict(),'step':step,
            'sampler_rng':gen.get_state(),'model_rng':torch.get_rng_state()},folder/'partial.pt')
        rec={'status':'FAIL','key':key,'dataset':name,'horizon':h,'arm':arm,'seed':seed,'updates':step,
            'joint_CPU_seconds':charged,'endpoints':ends,'error':traceback.format_exc()}
    dump(folder/'trace.json',trace);dump(folder/'fit.json',rec)
    print('FIT',key,rec['status'],round(charged,2),rec.get('selected',{}).get('step'),flush=True)
    return rec


def train():
    prepared=json.loads((OUT/'parents.json').read_text());assert prepared['protocol_sha256']==sha(REPORT/'PROTOCOL.md')
    parents={(r['dataset'],r['horizon'],r['seed']):r for r in prepared['parents']}
    assert all(r['status']=='PASS' for r in parents.values()),'registered parent failure; no replacement fit'
    tasks=[(d,h,a,s,parents[(d,h,s-1000)]) for d in DATASETS for h in HORIZONS for a in ARMS for s in SEEDS]
    # Independent fits scheduled once, largest cohort first; two single-thread workers.
    tasks.sort(key=lambda t:(t[0]=='electricity',t[1]),reverse=True)
    with concurrent.futures.ProcessPoolExecutor(max_workers=2,mp_context=multiprocessing.get_context('spawn')) as pool:
        records=list(pool.map(fit_job,tasks))
    dump(OUT/'frozen.json',{'protocol_sha256':sha(REPORT/'PROTOCOL.md'),'source_manifest_sha256':sha(REPORT/'sources.json'),
        'parents_sha256':sha(OUT/'parents.json'),'fits':records,'data_sha256':sha(OUT/'data.json'),
        'model_source_sha256':sha(ROOT/'src/tac_osm/me037_horizon_forecast.py'),'execution_source_sha256':sha(Path(__file__)),
        'threads_per_fit':1,'max_concurrent_fit_workers':2,'torch':torch.__version__,'platform':platform.platform(),
        'CPU_model':next((s.split(':',1)[1].strip() for s in Path('/proc/cpuinfo').read_text().splitlines() if s.startswith('model name')),'unknown'),
        'final_test_scored':False})


class Persistence(torch.nn.Module):
    def __init__(self,h):super().__init__();self.h=h
    def forward(self,x):return x[:,-1:].expand(-1,self.h,-1)


def test_job(task):
    torch.set_num_threads(1);kind,r,freeze=task;name=r['dataset'];h=r['horizon']
    key=r['key'] if kind=='models' else f'persistence-{name}-h{h}'
    path=OUT/'scores'/(key+'.json')
    if path.exists():
        cached=json.loads(path.read_text());assert cached['freeze_sha256']==freeze;return cached
    z,meta=data(name);_,va,end=meta['split_edges']
    if kind=='models':
        model=build_model(r['arm'],h,r['seed']);model.load_state_dict(torch.load(ROOT/r['selected']['checkpoint'],weights_only=False)['state_dict'])
        row={k:r[k] for k in ('key','dataset','horizon','arm','seed','parameters','joint_CPU_seconds','pipeline_training_CPU_seconds')}
        row['selected_step']=r['selected']['step']
    else:model=Persistence(h);row=dict(r)
    metric=score(model,z,va,end,h)
    if name=='appliances':
        metric['energy_MSE']=float(np.mean(metric['channel_MSE'][:2]));metric['energy_MAE']=float(np.mean(metric['channel_MAE'][:2]))
    row['test']=metric;receipt={'kind':kind,'row':row,'freeze_sha256':freeze};dump(path,receipt)
    print('TEST',key,round(metric['MSE'],6),round(metric['CPU_seconds'],2),flush=True)
    return receipt


def verify():
    frozen=json.loads((OUT/'frozen.json').read_text())
    assert frozen['protocol_sha256']==sha(REPORT/'PROTOCOL.md')
    assert frozen['source_manifest_sha256']==sha(REPORT/'sources.json')
    assert frozen['model_source_sha256']==sha(ROOT/'src/tac_osm/me037_horizon_forecast.py')
    assert frozen['execution_source_sha256']==sha(Path(__file__))
    assert frozen['parents_sha256']==sha(OUT/'parents.json') and frozen['data_sha256']==sha(OUT/'data.json')
    for r in frozen['fits']:
        if r['status']!='PASS':continue
        assert sha(ROOT/r['parent']['selected']['checkpoint'])==r['parent']['selected']['sha256']
        for e in r['endpoints']:
            assert sha(ROOT/e['checkpoint'])==e['sha256']
            assert sha((ROOT/e['checkpoint']).parent/f"u{e['step']}-probe.npy")==e['probe_sha256']
    return frozen


def evaluate():
    frozen=verify();freeze=sha(OUT/'frozen.json')
    tasks=[('models',r,freeze) for r in frozen['fits'] if r['status']=='PASS']
    tasks += [('persistence',{'dataset':d,'horizon':h},freeze) for d in DATASETS for h in HORIZONS]
    tasks.sort(key=lambda t:(t[1]['dataset']=='electricity',t[1]['horizon']),reverse=True)
    result={'freeze_sha256':freeze,'models':[],'persistence':[]}
    with concurrent.futures.ProcessPoolExecutor(max_workers=2,mp_context=multiprocessing.get_context('spawn')) as pool:
        for rec in pool.map(test_job,tasks):result[rec['kind']].append(rec['row'])
    dump(OUT/'results.json',result)


if __name__=='__main__':
    torch.set_num_threads(1);p=argparse.ArgumentParser();p.add_argument('mode',choices=('prepare','train','evaluate'))
    args=p.parse_args();globals()[args.mode]()
