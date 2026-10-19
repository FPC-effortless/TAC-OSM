"""Registered baseline-first measured forecasting; final EMA, no test selection."""
import argparse
import copy
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import multiprocessing
from pathlib import Path
import resource
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import numpy as np
import torch
from torch import nn
from tac_osm.development.thermal_forecast import ThermalForecast,specifications
from data_real_maintenance_003 import prepare,open_test,write,digest,HORIZONS,ROOMS
PROTOCOL=ROOT/'research/real-maint-003/protocol.json'


def make(spec,seed,normalization):
    model=ThermalForecast(**spec,seed=seed)
    for name in ('location','scale'):getattr(model,name).copy_(normalization[name])
    return model


class SimpleReference(nn.Module):
    def __init__(self,kind):super().__init__();self.kind=kind
    def forward(self,x,minutes):
        last=x[:,-1,None,:len(ROOMS)]
        if self.kind=='persistence':return last.expand(-1,len(HORIZONS),-1)
        if self.kind!='drift':raise ValueError(self.kind)
        rate=(x[:,-1,:len(ROOMS)]-x[:,0,:len(ROOMS)])/(x.shape[1]-1)
        return last+rate[:,None]*x.new_tensor(HORIZONS)[None,:,None]


class UnivariateReference(nn.Module):
    def __init__(self):
        super().__init__();self.register_buffer('scale',torch.ones(10,dtype=torch.float64))
        self.register_buffer('coefficients',torch.zeros(10,8,2,dtype=torch.float64))
    def features(self,x):
        f=((x[:,:-1,:10]-x[:,-1,None,:10])/self.scale).transpose(1,2)
        return torch.cat((f,torch.ones_like(f[:,:,:1])),-1)
    def forward(self,x,minutes):
        return x[:,-1,None,:10]+torch.einsum('brf,rfh->bhr',self.features(x),self.coefficients)*self.scale[None,None]


@torch.no_grad()
def evaluate(model,data,naive_scale,rows=False):
    errors=[];naive=[]
    for start in range(0,len(data['x']),256):
        x=data['x'][start:start+256];y=data['y'][start:start+256]
        prediction=model(x,data['minutes'][start:start+256]);assert torch.isfinite(prediction).all()
        errors.append((prediction-y).abs());naive.append((x[:,-1,None,:10]-y).abs())
    error=torch.cat(errors);baseline=torch.cat(naive);days=data['day'].unique(sorted=True)
    daily=torch.stack([error[data['day']==d].mean(0) for d in days])
    daily_naive=torch.stack([baseline[data['day']==d].mean(0) for d in days])
    mae=daily.mean(0);b=daily_naive.mean(0);ratios=torch.where(b>0,mae/b,torch.where(mae==0,1.,float('nan')))
    finite=bool(torch.isfinite(ratios).all())
    result=dict(windows=len(error),days=len(days),raw_MAE=mae.tolist(),persistence_MAE=b.tolist(),
        MASE=(mae/naive_scale).tolist(),mean_MASE=float((mae/naive_scale).mean()),
        relative_MAE=[[float(v) if torch.isfinite(v) else None for v in row] for row in ratios],
        worst_relative_MAE=float(ratios.max()) if finite else None,
        mean_relative_MAE=float(ratios.mean()) if finite else None,
        all_cells_no_worse=finite and bool((ratios<=1).all()),
        entry_pass=finite and bool((ratios<=1).all()) and float(ratios.mean())<=.95)
    if rows:result.update(daily_absolute_error=daily.tolist(),daily_persistence_error=daily_naive.tolist(),
        day_ids=days.tolist(),absolute_error=error.tolist(),window_day=data['day'].tolist(),
        origin_seconds=data['origin_seconds'].tolist())
    return result


def score(metrics):
    return (metrics['worst_relative_MAE'] if metrics['worst_relative_MAE'] is not None else float('inf'),
            metrics['mean_MASE'])


@torch.no_grad()
def fit_references(train,val,normalization,naive_scale,root):
    references={};reports=[]
    for name in ('persistence','drift'):
        model=SimpleReference(name);references[name]=model
        reports.append(dict(name=name,validation=evaluate(model,val,naive_scale)))
    delta=train['y']-train['x'][:,-1,None,:10]
    for kind in ('univariate','multivariate'):
        for ridge in (.001,.1,10.):
            name=f'{kind}_ridge_{ridge}'
            if kind=='univariate':
                model=UnivariateReference();model.scale.copy_(normalization['scale'][:10])
                x=model.features(train['x']);target=delta/model.scale
                penalty=torch.eye(8,dtype=torch.float64);penalty[-1,-1]=0
                for room in range(10):
                    a=x[:,room];y=target[:,:,room]
                    model.coefficients[room]=torch.linalg.solve(a.T@a/len(a)+ridge*penalty,a.T@y/len(a))
            else:
                model=make(dict(kind='linear',revin=False),0,normalization).double()
                x,scale=model.features(train['x'],train['minutes'])
                x=torch.cat((x,torch.ones_like(x[:,:1])),-1)
                target=(delta/scale[:,None]).flatten(1)
                penalty=torch.eye(x.shape[1],dtype=torch.float64);penalty[-1,-1]=0
                beta=torch.linalg.solve(x.T@x/len(x)+ridge*penalty,x.T@target/len(x))
                model.network.weight.copy_(beta[:-1].T);model.network.bias.copy_(beta[-1])
            references[name]=model
            path=root/'references'/name;path.mkdir(parents=True)
            torch.save(model.state_dict(),path/'agent.pt')
            report=dict(name=name,kind=kind,ridge=ridge,validation=evaluate(model,val,naive_scale),
                checkpoint_sha256=digest(path/'agent.pt'))
            write(path/'report.json',report);reports.append(report)
    # All lambda candidates stay preserved, including those not selected.
    conventional=min(reports,key=lambda r:(r['validation']['mean_MASE'],score(r['validation']),r['name']))['name']
    write(root/'reference-selection.json',dict(selected=conventional,candidates=reports,
        scope='Training-only closed-form fits; validation-only choice. Univariate omits exogenous input, multivariate matches neural input information. No test tuning.'))
    return references,conventional


def group(job):
    seed,lrs,specs,folder,directory,smoke=job;torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    folder=Path(folder);folder.mkdir();start=time.process_time();protocol=json.loads(PROTOCOL.read_text())
    source=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    train=torch.load(Path(directory)/'train.pt',weights_only=True);val=torch.load(Path(directory)/'validation.pt',weights_only=True)
    normalization=torch.load(Path(directory)/'normalization.pt',weights_only=True)
    scale=normalization['naive_scale'];raw={a:make(s,seed,normalization) for a,s in specs.items()}
    ema={a:copy.deepcopy(m).double().eval().requires_grad_(False) for a,m in raw.items()}
    optimizers={a:torch.optim.AdamW(m.parameters(),lr=lrs[a],weight_decay=1e-4) for a,m in raw.items()}
    curves={a:[dict(update=0,validation=evaluate(m,val,scale),eligible_for_selection=False)] for a,m in ema.items()}
    x=train['x'].float();minutes=train['minutes'].float()
    target=(train['y']-train['x'][:,-1,None,:10]).float();loss_scale=scale.float()
    generator=torch.Generator().manual_seed(seed);steps=8 if smoke else protocol['updates'];step=0;status='RUNNING';counts={a:0 for a in specs}
    try:
        for step in range(1,steps+1):
            indices=torch.randint(len(x),(64,),generator=generator)
            for arm,model in raw.items():
                value=((model.change(x[indices],minutes[indices])-target[indices]).abs().mean(0)/loss_scale).mean()
                assert torch.isfinite(value);optimizers[arm].zero_grad();value.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);optimizers[arm].step()
                counts[arm]+=1
                with torch.no_grad():
                    for e,p in zip(ema[arm].parameters(),model.parameters(),strict=True):e.mul_(.995).add_(p.double(),alpha=.005)
                if step%128==0 or step==steps:
                    curves[arm].append(dict(update=step,train_loss=float(value.detach()),
                        validation=evaluate(ema[arm],val,scale),eligible_for_selection=step==steps))
            if time.process_time()-start>protocol['group_cpu_cap_seconds']:raise TimeoutError('Registered per-group CPU cap')
        status='COMPLETE'
    except Exception:
        import traceback
        status='CAPPED' if time.process_time()-start>protocol['group_cpu_cap_seconds'] else 'FAILED'
        write(folder/'failure.json',dict(error=traceback.format_exc()))
    reports=[]
    for arm,model in raw.items():
        path=folder/arm;path.mkdir();torch.save(model.state_dict(),path/'raw.pt');torch.save(ema[arm].state_dict(),path/'agent.pt')
        report=dict(arm=arm,seed=seed,status=status,updates=counts[arm],attempted_iteration=step,spec=specs[arm],learning_rate=lrs[arm],
            parameters=sum(p.numel() for p in model.parameters()),curves=curves[arm],
            validation=evaluate(ema[arm],val,scale,True),checkpoint_sha256=digest(path/'agent.pt'),source=source)
        write(path/'report.json',report);reports.append(report)
    if time.process_time()-start>protocol['group_cpu_cap_seconds']:status='CAPPED'
    result=dict(seed=seed,status='SMOKE_ONLY' if smoke and status=='COMPLETE' else status,
        cpu_seconds=time.process_time()-start,reports=reports)
    write(folder/'summary.json',result);return result


def smoke_data(destination):
    destination.mkdir();generator=torch.Generator().manual_seed(1973001)
    for name in ('train','validation'):
        x=20+torch.randn(64,8,12,generator=generator,dtype=torch.float64)
        torch.save(dict(x=x,y=x[:,-1,None,:10]+.1*torch.randn(64,2,10,generator=generator,dtype=torch.float64),
            minutes=torch.arange(8,dtype=torch.float64)[None].expand(64,-1)*15,
            day=torch.arange(64)//16,origin_seconds=torch.arange(64)),destination/(name+'.pt'))


def run(directory,source,smoke=False):
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    root=Path(directory);root.mkdir(parents=True,exist_ok=False);start=time.process_time();before=resource.getrusage(resource.RUSAGE_CHILDREN)
    protocol=json.loads(PROTOCOL.read_text());specs=specifications();write(root/'specs.json',specs);data=root/'_datasets'
    if not smoke:assert not subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=ROOT,text=True).strip(),'Commit source/registration before real preprocessing or fitting'
    try:
        smoke_data(data) if smoke else prepare(source,data)
        train=torch.load(data/'train.pt',weights_only=True);val=torch.load(data/'validation.pt',weights_only=True)
        normalization=dict(location=train['x'].mean((0,1)),scale=train['x'].std((0,1),unbiased=False))
        normalization['naive_scale']=(train['y']-train['x'][:,-1,None,:10]).abs().mean(0)
        assert all(torch.isfinite(v).all() for v in normalization.values()),'Nonfinite TRAIN normalization'
        assert (normalization['scale']>1e-6).all() and (normalization['naive_scale']>1e-6).all(),'Undefined TRAIN scale: preserve rejection; never drop a target'
        torch.save(normalization,data/'normalization.pt')
        references,conventional=fit_references(train,val,normalization,normalization['naive_scale'],root)
    except Exception:
        import traceback
        write(root/'summary.json',dict(status='DATA_QUALITY_REJECTED',test_opened=False,fits=0,error=traceback.format_exc()))
        raise
    preparation=time.process_time()-start;groups=[];fits=[];pilots=[];candidate=None;offset=10000000*int(smoke)
    with ProcessPoolExecutor(max_workers=2,mp_context=multiprocessing.get_context('spawn')) as executor:
        jobs=[(protocol['pilot_seed']+offset,{a:lr for a in specs},specs,str(root/f'pilot-{i}'),str(data),smoke)
              for i,lr in enumerate(protocol['lrs'][:1] if smoke else protocol['lrs'])]
        for future in as_completed([executor.submit(group,job) for job in jobs]):
            g=future.result();pilots.append(g);groups.append(g);write(root/'pilots.json',pilots)
            print('PILOT',g['status'],round(g['cpu_seconds'],2),flush=True)
        if all(g['status']==('SMOKE_ONLY' if smoke else 'COMPLETE') for g in pilots):
            lrs={a:min([r for g in pilots for r in g['reports'] if r['arm']==a],
                key=lambda r:(score(r['validation']),r['learning_rate']))['learning_rate'] for a in specs}
            selected=[r for g in pilots for r in g['reports'] if r['learning_rate']==lrs[r['arm']]]
            eligible=[r for r in selected if r['validation']['entry_pass']]
            if eligible:candidate=min(eligible,key=lambda r:(score(r['validation']),r['parameters'],r['arm']))['arm']
            write(root/'selection.json',dict(candidate=candidate,lrs=lrs,conventional_reference=conventional,
                decision='SMOKE_ONLY' if smoke else 'PILOT_ELIGIBLE' if candidate else 'VALIDATION_REJECTED_NO_TEST'))
            if candidate is not None or smoke:
                seeds=protocol['seeds'][:1] if smoke else protocol['seeds']
                for begin in range(0,len(seeds),2):
                    selected_seeds=seeds[begin:begin+2]
                    if preparation+sum(g['cpu_seconds'] for g in groups)+len(selected_seeds)*protocol['group_cpu_cap_seconds']>protocol['fit_cpu_cap_seconds']:break
                    jobs=[(s+offset,lrs,specs,str(root/f'group-{s+offset}'),str(data),smoke) for s in selected_seeds]
                    for future in as_completed([executor.submit(group,job) for job in jobs]):
                        g=future.result();groups.append(g);fits.append(g);write(root/'groups.json',fits)
                        print('GROUP',g['seed'],g['status'],round(g['cpu_seconds'],2),flush=True)
    after=resource.getrusage(resource.RUSAGE_CHILDREN)
    cpu=time.process_time()-start+after.ru_utime+after.ru_stime-before.ru_utime-before.ru_stime
    complete=len(fits)==(1 if smoke else 20) and all(g['status']==('SMOKE_ONLY' if smoke else 'COMPLETE') for g in fits) and cpu<=protocol['fit_cpu_cap_seconds']
    valpasses=sum(r['validation']['entry_pass'] for g in fits for r in g['reports'] if r['arm']==candidate)
    eligible=complete and valpasses>=18
    status='SMOKE_ONLY' if smoke and complete else 'VALIDATION_ELIGIBLE_FITS_COMPLETE' if eligible else 'VALIDATION_REJECTED' if complete or not fits and candidate is None and all(g['status']=='COMPLETE' for g in pilots) else 'INCOMPLETE'
    summary=dict(status=status,fits=len(fits)*4,pilot_fits=len(pilots)*4,test_opened=False,
        candidate_validation_passing_seeds=valpasses,fitting_cpu_seconds=cpu,protocol_sha256=digest(PROTOCOL))
    write(root/'summary.json',summary)
    if eligible and not smoke:
        selection=json.loads((root/'selection.json').read_text());selection['checkpoint_hashes']={str(p.relative_to(root)):digest(p) for p in sorted(root.glob('group-*/*/agent.pt'))};write(root/'selection.json',selection)
        try:
            test=open_test(source,root)
            for name,model in references.items():write(root/'references'/name/'test.json',evaluate(model,test,normalization['naive_scale'],True))
            for g in fits:
                for report in g['reports']:
                    path=root/f"group-{report['seed']}"/report['arm'];m=make(report['spec'],report['seed'],normalization).double().eval()
                    m.load_state_dict(torch.load(path/'agent.pt',weights_only=True))
                    write(path/'test.json',evaluate(m,test,normalization['naive_scale'],True))
            summary.update(status='COMPLETE',test_opened=True);write(root/'summary.json',summary)
        except Exception:
            import traceback
            summary.update(status='PARTIAL_TEST_FAILURE',test_opened=(root/'TEST-EXPOSURE.json').exists(),error=traceback.format_exc());write(root/'summary.json',summary)
            raise
    write(root/'manifest.json',{str(p.relative_to(root)):digest(p) for p in sorted(root.rglob('*')) if p.is_file() and p.name!='manifest.json'})
    print('SUMMARY',json.dumps(summary),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--directory',required=True);parser.add_argument('--source');parser.add_argument('--smoke',action='store_true')
    args=parser.parse_args();run(args.directory,args.source,args.smoke)
