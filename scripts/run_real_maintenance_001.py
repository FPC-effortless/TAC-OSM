"""Registered teacher-free external MetroPT sensor study; no repair claims."""
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
import copy
import csv
from datetime import datetime
import hashlib
import zipfile
import multiprocessing
from pathlib import Path
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import numpy as np
import torch
from tac_osm.development.maintenance_forecast import MaintenanceForecast,configurations
from run_learned_stack_dev055 import write,digest
FEATURES=['TP2','TP3','H1','DV_pressure','Reservoirs','Oil_temperature','Motor_current','COMP','DV_eletric','Towers','MPG','LPS','Pressure_switch','Oil_level','Caudal_impulses']
EXPECTED='aab991a970e58210de853bb8078ce0e63abb4d9412fdc5c79792dae3d8e1721a'
PROTOCOL=ROOT/'research/real-maint-001/PROTOCOL.md'
EPOCH=datetime(1970,1,1)


def seconds(text):return int((datetime.fromisoformat(text)-EPOCH).total_seconds())


def windows(times,values):
    result={name:{'x':[],'y':[],'time':[]} for name in ('train','validation','test')};train_end=seconds('2020-02-22');val_end=seconds('2020-03-01')
    for j in range(24,len(times)):
        if times[j]%3600 or not np.all(np.diff(times[j-24:j+1])==600):continue
        name='train' if times[j]<train_end else 'validation' if times[j]<val_end else 'test';result[name]['x'].append(values[j-24:j]);result[name]['y'].append(values[j]);result[name]['time'].append(times[j])
    return {name:{k:torch.tensor(np.array(v),dtype=torch.int64 if k=='time' else torch.float32) for k,v in d.items()} for name,d in result.items()}


def preprocess(csv_path,archive,destination):
    assert digest(archive)==EXPECTED,'Unexpected raw source archive'
    with zipfile.ZipFile(archive) as source:
        with source.open('MetroPT3(AirCompressor).csv') as stream:assert hashlib.file_digest(stream,'sha256').hexdigest()==digest(csv_path),'CSV differs from verified source archive'
    bins={};rows=0;start=time.process_time()
    with Path(csv_path).open(newline='',encoding='utf-8-sig') as stream:
        reader=csv.DictReader(stream);assert reader.fieldnames[2:]==FEATURES,reader.fieldnames
        for row in reader:
            timestamp=seconds(row['timestamp']);end=(timestamp//600+1)*600;value=np.array([float(row[k]) for k in FEATURES],np.float64);assert np.isfinite(value).all();rows+=1
            if end not in bins:bins[end]=[value,1]
            else:bins[end][0]+=value;bins[end][1]+=1
    assert rows==1516948,rows
    times=np.array(sorted(bins),np.int64);values=np.array([bins[t][0]/bins[t][1] for t in times],np.float32);prepared=windows(times,values)
    destination.mkdir(parents=True);torch.save({'times':torch.tensor(times),'values':torch.tensor(values)},destination/'bins.pt')
    for name,data in prepared.items():torch.save(data,destination/f'{name}.pt')
    write(destination/'provenance.json',{'source':'https://archive.ics.uci.edu/static/public/791/metropt+3+dataset.zip','doi':'10.24432/C5VW3R','license':'CC BY4.0','raw_archive_sha256':EXPECTED,'raw_csv_sha256':digest(csv_path),'raw_rows':rows,'bins':len(times),'features':FEATURES,'split_windows':{n:len(d['x']) for n,d in prepared.items()},'preprocessing_cpu_seconds':time.process_time()-start,'time_contract':'Naive source local timestamps; bin features available at right edge. Next-bin forecasts issued before target observations. No future imputation or fault labels.'})


def make(arm,spec,seed,normalization):
    m=MaintenanceForecast(arm,spec['width'],spec['aux'],seed)
    m.mean.copy_(normalization['mean']);m.scale.copy_(normalization['scale']);return m


def loss(m,x,y):
    o=m(x);truth=(y-m.mean)/m.scale
    return (.5*(o['log_variance']+(truth-o['normalized_mean'])**2*o['log_variance'].neg().exp())).mean()


@torch.no_grad()
def evaluate(m,data,include_rows=False):
    dtype=next(m.parameters()).dtype;errors=[];nll=[]
    for start in range(0,len(data['x']),128):
        x=data['x'][start:start+128].to(dtype);y=data['y'][start:start+128].to(dtype);o=m(x);squared=(((y-m.mean)/m.scale)-o['normalized_mean'])**2;errors.extend(squared.mean(-1).tolist());nll.extend((.5*(o['log_variance']+squared*o['log_variance'].neg().exp())).mean(-1).tolist())
    result={'normalized_MSE':float(np.mean(errors)),'Gaussian_NLL':float(np.mean(nll)),'windows':len(errors)}
    if include_rows:result.update(squared_residual=errors,nll=nll,time=data['time'].tolist())
    return result


def group(job):
    seed,lrs,specs,folder,data_path,smoke=job;torch.set_num_threads(1);torch.use_deterministic_algorithms(True);folder=Path(folder);folder.mkdir(parents=True);start=time.process_time();train=torch.load(Path(data_path)/'train.pt',weights_only=True);val=torch.load(Path(data_path)/'validation.pt',weights_only=True);norm={'mean':train['x'].mean((0,1)),'scale':train['x'].std((0,1)).clamp_min(.01)};models={a:make(a,s,seed,norm) for a,s in specs.items()};ema={a:copy.deepcopy(m).eval() for a,m in models.items()};opt={a:torch.optim.AdamW(m.parameters(),lr=lrs[a],weight_decay=1e-4) for a,m in models.items()};curves={a:[] for a in models};rng=torch.Generator().manual_seed(seed);status='RUNNING'
    try:
        for step in range(1,(8 if smoke else 512)+1):
            ids=torch.randint(len(train['x']),(64,),generator=rng)
            for a,m in models.items():
                value=loss(m,train['x'][ids],train['y'][ids]);assert torch.isfinite(value);opt[a].zero_grad();value.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),1.,error_if_nonfinite=True);opt[a].step()
                with torch.no_grad():
                    for e,v in zip(ema[a].parameters(),m.parameters(),strict=True):e.mul_(.99).add_(v,alpha=.01)
                if step==1 or step%64==0 or (smoke and step==8):curves[a].append({'update':step,'training_NLL':float(value.detach()),'validation':evaluate(ema[a],val)})
            if time.process_time()-start>60:status='CAPPED';break
        else:status='COMPLETE'
    except Exception:
        import traceback
        status='FAILED';write(folder/'failure.json',{'error':traceback.format_exc(),'step':step})
    reports=[]
    for a,m in models.items():
        dest=folder/a;dest.mkdir();torch.save(m.state_dict(),dest/'raw.pt');deploy=copy.deepcopy(ema[a]).double().eval();v=evaluate(deploy,val,True);threshold=float(np.quantile(v['squared_residual'],.99));torch.save(deploy.state_dict(),dest/'agent.pt');r={'arm':a,'seed':seed,'spec':specs[a],'lr':lrs[a],'status':status,'updates':step,'curves':curves[a],'validation':v,'anomaly_threshold':threshold,'checkpoint_sha256':digest(dest/'agent.pt'),'source':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()};write(dest/'report.json',r);reports.append(r)
    if time.process_time()-start>60 and status=='COMPLETE':status='CAPPED'
    result={'seed':seed,'status':'SMOKE_ONLY' if smoke else status,'cpu_seconds':time.process_time()-start,'reports':reports};write(folder/'summary.json',result);return result


def run(directory,csv_path=None,archive=None,smoke=False):
    torch.set_num_threads(1);root=Path(directory);root.mkdir(parents=True,exist_ok=False);start=time.process_time();specs=configurations();write(root/'specs.json',specs);ds=root/'_datasets'
    if not smoke:
        assert not subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=ROOT,text=True).strip(),'Commit frozen source'
        preprocess(csv_path,archive,ds)
    else:
        ds.mkdir();rng=torch.Generator().manual_seed(1996301)
        for name in ('train','validation','test'):torch.save({'x':torch.randn(24,24,15,generator=rng),'y':torch.randn(24,15,generator=rng),'time':torch.arange(24)},ds/f'{name}.pt')
    preparation=time.process_time()-start;groups=[];fits=[];pilots=[];seeds=[10092701] if smoke else list(range(92701,92721));offset=10000000*int(smoke)
    with ProcessPoolExecutor(max_workers=2,mp_context=multiprocessing.get_context('spawn')) as executor:
        jobs=[(92700+offset,{a:lr for a in specs},specs,str(root/f'pilot-{i}'),str(ds),smoke) for i,lr in enumerate((.001,) if smoke else (.001,.003))]
        for f in as_completed([executor.submit(group,j) for j in jobs]):
            g=f.result();pilots.append(g);groups.append(g);write(root/'pilots.json',pilots);print('pilot',g['status'],round(g['cpu_seconds'],1),flush=True)
        if smoke or all(g['status']=='COMPLETE' for g in pilots):
            lrs={a:min([r for g in pilots for r in g['reports'] if r['arm']==a],key=lambda r:(r['validation']['Gaussian_NLL'],r['lr']))['lr'] for a in specs};write(root/'selection.json',lrs);jobs=[(s,lrs,specs,str(root/f'group-{s}'),str(ds),smoke) for s in seeds]
            for i in range(0,len(jobs),2):
                if preparation+sum(g['cpu_seconds'] for g in groups)+len(jobs[i:i+2])*60>1200:break
                for f in as_completed([executor.submit(group,j) for j in jobs[i:i+2]]):
                    g=f.result();groups.append(g);fits.append(g);write(root/'groups.json',fits);print('group',g['seed'],g['status'],round(g['cpu_seconds'],1),flush=True)
    complete=len(fits)==len(seeds) and all(g['status']==('SMOKE_ONLY' if smoke else 'COMPLETE') for g in fits);summary={'status':'SMOKE_ONLY' if smoke else 'FITS_COMPLETE' if complete else 'INCOMPLETE','test_opened':False,'fits':len(fits)*4,'pilot_fits':len(pilots)*4,'fitting_cpu_seconds':preparation+sum(g['cpu_seconds'] for g in groups),'protocol_sha256':digest(PROTOCOL)};write(root/'summary.json',summary)
    if complete and not smoke:
        test=torch.load(ds/'test.pt',weights_only=True)
        for g in fits:
            for r in g['reports']:
                folder=root/f"group-{r['seed']}"/r['arm'];m=MaintenanceForecast(r['arm'],r['spec']['width'],r['spec']['aux'],r['seed']).double().eval();m.load_state_dict(torch.load(folder/'agent.pt',weights_only=True));write(folder/'test.json',evaluate(m,test,True))
        summary.update(status='COMPLETE',test_opened=True);write(root/'summary.json',summary)
    write(root/'manifest.json',{str(f.relative_to(root)):digest(f) for f in sorted(root.rglob('*')) if f.is_file() and f.name!='manifest.json'})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--directory',required=True);p.add_argument('--csv');p.add_argument('--source-archive');p.add_argument('--smoke',action='store_true');a=p.parse_args();run(a.directory,a.csv,a.source_archive,a.smoke)
