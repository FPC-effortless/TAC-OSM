"""Prospectively registered continuation of every ME036 model, plus ridge."""
import math, time, traceback
from pathlib import Path
import numpy as np
import torch
import scripts.me036_research as base
from tac_osm.me036_patch_forecast import build_model

OUT=base.ROOT/'results/me036b'


class Ridge(torch.nn.Module):
    def __init__(self, coef):
        super().__init__();self.register_buffer('weight',torch.as_tensor(coef[:-1].T.copy(),dtype=torch.float32))
        self.register_buffer('bias',torch.as_tensor(coef[-1].copy(),dtype=torch.float32))
    def forward(self,x):
        mean=x.mean(1,keepdim=True);scale=torch.sqrt((x-mean).square().mean(1,keepdim=True)+1e-5)
        z=((x-mean)/scale).transpose(1,2)
        p=torch.nn.functional.linear(z,self.weight,self.bias)
        return p.transpose(1,2)*scale+mean


def ridge_fit(name,h,z,meta):
    folder=OUT/f'{name}-h{h}-ridge';folder.mkdir(parents=True,exist_ok=True)
    if (folder/'fit.json').exists():return base.json.loads((folder/'fit.json').read_text())
    t=time.process_time();train=meta['split_edges'][0]
    aa=np.zeros((97,97));ab=np.zeros((97,h));ix=np.arange(96,train-h+1,24)
    for i in range(0,len(ix),8):
        x,y=base.batch(z,ix[i:i+8],h)
        mean=x.mean(1,keepdim=True);scale=torch.sqrt((x-mean).square().mean(1,keepdim=True)+1e-5)
        a=((x-mean)/scale).transpose(1,2).reshape(-1,96).double().numpy()
        b=((y-mean)/scale).transpose(1,2).reshape(-1,h).double().numpy()
        a=np.c_[a,np.ones(len(a))];aa+=a.T@a;ab+=a.T@b
    aa.flat[::98]+=100.;aa[-1,-1]-=100.
    coef=np.linalg.solve(aa,ab);np.save(folder/'coef.npy',coef)
    rec={'status':'PASS','key':folder.name,'dataset':name,'horizon':h,'arm':'ridge','seed':None,
        'parameters':coef.size,'charged_training_CPU_seconds':time.process_time()-t,
        'training_anchors':len(ix),'training_channel_windows':len(ix)*z.size(1),'stride':24,
        'coefficient':str((folder/'coef.npy').relative_to(base.ROOT)),
        'sha256':base.sha(folder/'coef.npy'),'matched':False}
    base.dump(folder/'fit.json',rec);print('RIDGE',name,h,round(rec['charged_training_CPU_seconds'],2),flush=True)
    return rec


def continue_fit(parent,z,meta):
    key=parent['key'];folder=OUT/key;folder.mkdir(parents=True,exist_ok=True)
    if (folder/'fit.json').exists():return base.json.loads((folder/'fit.json').read_text())
    t=time.process_time();h=parent['horizon'];seed=parent['seed'];arm=parent['arm']
    checkpoint=next(e for e in parent['endpoints'] if e['step']==100)
    assert base.sha(base.ROOT/checkpoint['checkpoint'])==checkpoint['sha256']
    state=torch.load(base.ROOT/checkpoint['checkpoint'],weights_only=False)
    model=build_model(arm,h,seed);model.load_state_dict(state['state_dict'])
    opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001);opt.load_state_dict(state['optimizer'])
    gen=torch.Generator();gen.set_state(state['sampler_rng']);torch.set_rng_state(state['model_rng'])
    endpoints=list(parent['endpoints']);trace=[];charged=time.process_time()-t;train,val,_=meta['split_edges'];step=100
    try:
        for step in range(101,1001):
            t=time.process_time();model.train()
            lr=.0001+.0009*.5*(1+math.cos(math.pi*(step-101)/899))
            for group in opt.param_groups:group['lr']=lr
            anchors=torch.randint(96,train-h+1,(4,),generator=gen)
            x,y=base.batch(z,anchors,h);opt.zero_grad(set_to_none=True)
            loss=(model(x)-y).square().mean()
            if not torch.isfinite(loss):raise ArithmeticError('nonfinite loss')
            loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            if not torch.isfinite(norm):raise ArithmeticError('nonfinite gradient')
            opt.step();dt=time.process_time()-t;charged+=dt
            trace.append({'step':step,'MSE':loss.item(),'CPU_seconds':dt,'learning_rate':lr,'gradient_norm':norm.item()})
            if step in (250,500,1000):
                path=folder/f'u{step}.pt'
                torch.save({'state_dict':model.state_dict(),'optimizer':opt.state_dict(),'sampler_rng':gen.get_state(),
                    'model_rng':torch.get_rng_state(),'step':step,'arm':arm,'horizon':h,'seed':seed},path)
                metric=base.score(model,z,train,val,h,24);probe,_=base.batch(z,[train,train+24],h)
                with torch.no_grad():np.save(folder/f'u{step}-probe.npy',model(probe).numpy())
                endpoints.append({'step':step,'validation':metric,'checkpoint':str(path.relative_to(base.ROOT)),
                    'sha256':base.sha(path),'charged_training_CPU_seconds':parent['charged_training_CPU_seconds']+charged})
                base.dump(folder/'trace.json',trace)
            if charged>500:raise RuntimeError('registered500CPU-second continuation cap')
        rec={**parent,'endpoints':endpoints,'selected':min(endpoints,key=lambda e:(e['validation']['MSE'],e['step'])),
            'updates':step,'examples_seen':step*4,'channel_steps_seen':step*4*96*z.size(1),
            'parent_training_CPU_seconds':parent['charged_training_CPU_seconds'],
            'continuation_training_CPU_seconds':charged,'charged_training_CPU_seconds':parent['charged_training_CPU_seconds']+charged,
            'CPU_cap':620,'process_peak_RSS_KiB':base.resource.getrusage(base.resource.RUSAGE_SELF).ru_maxrss}
    except Exception:
        torch.save({'state_dict':model.state_dict(),'optimizer':opt.state_dict(),'step':step,
            'sampler_rng':gen.get_state(),'model_rng':torch.get_rng_state()},folder/'partial.pt')
        rec={**parent,'status':'FAIL','error':traceback.format_exc(),'updates':step,'endpoints':endpoints}
    base.dump(folder/'trace.json',trace);base.dump(folder/'fit.json',rec)
    print('CONTINUE',key,rec['status'],round(charged,2),rec.get('selected',{}).get('step'),flush=True)
    return rec


def train():
    OUT.mkdir(parents=True,exist_ok=True)
    parent=base.json.loads((base.OUT/'frozen.json').read_text());records=[];ridges=[]
    assert parent['protocol_sha256']==base.sha(base.REPORT/'PROTOCOL.md')
    for name in base.DATASETS:
        z,meta=base.load_data(name)
        for r in parent['fits']:
            if r['dataset']==name:
                records.append(continue_fit(r,z,meta) if r['status']=='PASS' else r)
        for h in base.HORIZONS:ridges.append(ridge_fit(name,h,z,meta))
    base.dump(OUT/'frozen.json',{'protocol_sha256':base.sha(base.REPORT/'PROTOCOL.md'),
        'source_manifest_sha256':base.sha(base.REPORT/'sources.json'),
        'continuation_protocol_sha256':base.sha(base.REPORT/'CONTINUATION.md'),
        'parent_freeze_sha256':base.sha(base.OUT/'frozen.json'),'fits':records,'ridge':ridges,
        'test_scored':False,'hardware':parent['hardware'],'platform':parent['platform'],
        'torch':torch.__version__,'threads':torch.get_num_threads()})


def evaluate():
    # Base evaluator validates all fitted endpoints first, streams exact test contract.
    original=base.OUT;base.OUT=OUT
    frozen=base.json.loads((OUT/'frozen.json').read_text())
    assert frozen['continuation_protocol_sha256']==base.sha(base.REPORT/'CONTINUATION.md')
    for r in frozen['ridge']:assert base.sha(base.ROOT/r['coefficient'])==r['sha256']
    base.evaluate()
    results=base.json.loads((OUT/'results.json').read_text());results['ridge']=[]
    for name in base.DATASETS:
        z,meta=base.load_data(name);_,val,end=meta['split_edges']
        for r in frozen['ridge']:
            if r['dataset']!=name:continue
            net=Ridge(np.load(base.ROOT/r['coefficient']))
            metric=base.score(net,z,val,end,r['horizon'])
            results['ridge'].append({**r,'test':metric})
            base.dump(OUT/'ridge-progress.json',results)
            print('TEST RIDGE',name,r['horizon'],round(metric['MSE'],6),flush=True)
    base.dump(OUT/'results.json',results);base.OUT=original


if __name__=='__main__':
    torch.set_num_threads(1)
    parser=base.argparse.ArgumentParser();parser.add_argument('mode',choices=('train','evaluate','audit'))
    args=parser.parse_args()
    if args.mode=='audit':base.OUT=OUT;base.audit()
    else:globals()[args.mode]()
