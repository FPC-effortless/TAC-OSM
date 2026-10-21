"""Registered ME036 fits, frozen selection, streaming full-sequence evaluation."""
import argparse, csv, hashlib, io, json, platform, resource, time, traceback
from pathlib import Path
import numpy as np
import torch
from tac_osm.me036_patch_forecast import ARMS, build_model

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT/'research/mathematical-engineering/me036'
DATA = ROOT/'results/me036-source'
OUT = ROOT/'results/me036'
DATASETS = ('ETTh1', 'ETTh2', 'ETTm1', 'weather', 'electricity')
HORIZONS = (96, 192, 336, 720)
SEEDS = (36001, 36002)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False)); tmp.replace(path)


def load_data(name):
    meta = json.loads((REPORT/'sources.json').read_text())[name+'.csv']
    path = DATA/(name+'.csv')
    if sha(path) != meta['sha256']: raise ValueError('source checksum mismatch')
    reader = csv.reader(io.StringIO(path.read_text(encoding='utf-8-sig')))
    header = next(reader); rows = list(reader)
    a = np.asarray([[float(v) for v in row[1:]] for row in rows], dtype=np.float32)
    if not np.isfinite(a).all(): raise ValueError('nonfinite real observations')
    n = len(a)
    if name.startswith('ETT'):
        factor = 4 if name == 'ETTm1' else 1
        train, val, end = 8640*factor, 11520*factor, 14400*factor
        assert end <= n
    else:
        train, end = int(n*.7), n
        val = n-int(n*.2)
    mean = a[:train].mean(0, dtype=np.float64).astype(np.float32)
    scale = np.maximum(a[:train].std(0, dtype=np.float64), 1e-5).astype(np.float32)
    z = torch.from_numpy(((a-mean)/scale).copy())
    return z, {'name':name,'source':meta,'rows':n,'columns':header[1:],
        'channels':a.shape[1], 'split_edges':[train,val,end],
        'first_timestamp':rows[0][0], 'last_timestamp':rows[-1][0],
        'timezone':'publisher unspecified; original row order',
        'finite_sentinel_minus9999_count':int((a == -9999).sum()),
        'mean':mean.tolist(),'scale':scale.tolist()}


def batch(z, anchors, horizon):
    anchors = torch.as_tensor(anchors, dtype=torch.long)
    x = z[anchors[:, None]+torch.arange(-96, 0)[None]]
    y = z[anchors[:, None]+torch.arange(horizon)[None]]
    return x, y


@torch.no_grad()
def score(model, z, first, end, h, stride=1):
    model.eval(); start=time.process_time(); ss=sa=0.; count=0
    ix = np.arange(first, end-h+1, stride)
    for i in range(0, len(ix), 16):
        x,y = batch(z, ix[i:i+16], h)
        p = model(x)
        if not torch.isfinite(p).all(): raise ArithmeticError('nonfinite forecast')
        e = (p-y).double()
        ss += e.square().sum().item(); sa += e.abs().sum().item(); count += e.numel()
    return {'MSE':ss/count,'MAE':sa/count,'scalar_targets':count,'windows':len(ix),
        'stride':stride,'CPU_seconds':time.process_time()-start}


def fit(name, h, arm, seed, z, meta):
    key=f'{name}-h{h}-{arm}-s{seed}'; folder=OUT/key; folder.mkdir(parents=True,exist_ok=True)
    if (folder/'fit.json').exists(): return json.loads((folder/'fit.json').read_text())
    started=time.process_time(); model=build_model(arm,h,seed)
    opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
    generator=torch.Generator().manual_seed(seed+1000)
    trace=[]; endpoints=[]; charged=time.process_time()-started
    train, val, _=meta['split_edges']; step=0
    try:
        for step in range(1,101):
            t=time.process_time();model.train()
            anchors=torch.randint(96, train-h+1, (4,), generator=generator)
            x,y=batch(z,anchors,h); opt.zero_grad(set_to_none=True)
            loss=(model(x)-y).square().mean()
            if not torch.isfinite(loss):raise ArithmeticError('nonfinite loss')
            loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            if not torch.isfinite(norm):raise ArithmeticError('nonfinite gradient')
            opt.step();dt=time.process_time()-t;charged+=dt
            trace.append({'step':step,'MSE':loss.item(),'CPU_seconds':dt,'gradient_norm':norm.item()})
            if step in (25,50,100):
                checkpoint=folder/f'u{step}.pt'
                torch.save({'state_dict':model.state_dict(),'optimizer':opt.state_dict(),
                    'sampler_rng':generator.get_state(),'model_rng':torch.get_rng_state(),
                    'step':step,'arm':arm,'horizon':h,'seed':seed},checkpoint)
                metric=score(model,z,train,val,h,24)
                probe,_=batch(z,[train,train+24],h)
                with torch.no_grad():np.save(folder/f'u{step}-probe.npy',model(probe).numpy())
                endpoints.append({'step':step,'validation':metric,'checkpoint':str(checkpoint.relative_to(ROOT)),
                    'sha256':sha(checkpoint),'charged_training_CPU_seconds':charged})
                dump(folder/'trace.json',trace)
            if charged>120:raise RuntimeError('registered120CPU-second fit cap')
        selected=min(endpoints,key=lambda e:(e['validation']['MSE'],e['step']))
        record={'status':'PASS','key':key,'dataset':name,'horizon':h,'arm':arm,'seed':seed,
            'parameters':sum(p.numel() for p in model.parameters()),'updates':step,
            'examples_seen':step*4,'channel_steps_seen':step*4*96*z.size(1),
            'charged_training_CPU_seconds':charged,'CPU_cap':120,'endpoints':endpoints,
            'selected':selected,'process_peak_RSS_KiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
    except Exception:
        torch.save({'state_dict':model.state_dict(),'optimizer':opt.state_dict(),
                    'sampler_rng':generator.get_state(),'model_rng':torch.get_rng_state(),'step':step},folder/'partial.pt')
        dump(folder/'trace.json',trace)
        record={'status':'FAIL','key':key,'dataset':name,'horizon':h,'arm':arm,'seed':seed,
            'updates':step,'charged_training_CPU_seconds':charged,'error':traceback.format_exc(),'endpoints':endpoints}
    dump(folder/'fit.json',record)
    print(key,record['status'],round(charged,2),
          round(record['selected']['validation']['MSE'],6) if record['status']=='PASS' else record['error'],flush=True)
    return record


def train():
    OUT.mkdir(parents=True,exist_ok=True); records=[]; metadata=[]
    for name in DATASETS:
        z,meta=load_data(name);metadata.append(meta);dump(OUT/'data.json',metadata)
        for h in HORIZONS:
            for arm in ARMS:
                for seed in SEEDS:records.append(fit(name,h,arm,seed,z,meta))
        dump(OUT/'fits.json',records)
    dump(OUT/'frozen.json',{'protocol_sha256':sha(REPORT/'PROTOCOL.md'),
        'source_manifest_sha256':sha(REPORT/'sources.json'),'fits':records,
        'hardware':platform.processor(),'platform':platform.platform(),'torch':torch.__version__,
        'threads':torch.get_num_threads(),'test_scored':False})


def evaluate():
    frozen=json.loads((OUT/'frozen.json').read_text())
    assert frozen['protocol_sha256']==sha(REPORT/'PROTOCOL.md')
    assert frozen['source_manifest_sha256']==sha(REPORT/'sources.json')
    # Verify every fitted checkpoint BEFORE interpreting any test targets.
    for r in frozen['fits']:
        for e in r['endpoints']:assert sha(ROOT/e['checkpoint'])==e['sha256']
    result={'freeze_sha256':sha(OUT/'frozen.json'),'models':[],'persistence':[]}
    for name in DATASETS:
        z,meta=load_data(name);_,val,end=meta['split_edges']
        for r in frozen['fits']:
            if r['dataset']!=name or r['status']!='PASS':continue
            model=build_model(r['arm'],r['horizon'],r['seed'])
            model.load_state_dict(torch.load(ROOT/r['selected']['checkpoint'],weights_only=False)['state_dict'])
            metric=score(model,z,val,end,r['horizon'])
            row={k:r[k] for k in ('key','dataset','horizon','arm','seed','parameters','charged_training_CPU_seconds')}
            row.update({'selected_step':r['selected']['step'],'test':metric})
            result['models'].append(row);dump(OUT/'test-progress.json',result)
            print('TEST',r['key'],round(metric['MSE'],6),round(metric['CPU_seconds'],2),flush=True)
        class Persistence(torch.nn.Module):
            def __init__(self,h):super().__init__();self.h=h
            def forward(self,x):return x[:,-1:].expand(-1,self.h,-1)
        for h in HORIZONS:
            result['persistence'].append({'dataset':name,'horizon':h,'test':score(Persistence(h),z,val,end,h)})
    dump(OUT/'results.json',result)


def audit():
    frozen=json.loads((OUT/'frozen.json').read_text()); replays=0; tensors=0
    for name in DATASETS:
        z,meta=load_data(name);train=meta['split_edges'][0]
        for r in frozen['fits']:
            if r['dataset']!=name:continue
            for e in r['endpoints']:
                checkpoint=ROOT/e['checkpoint'];assert sha(checkpoint)==e['sha256']
                net=build_model(r['arm'],r['horizon'],r['seed']).eval()
                state=torch.load(checkpoint,weights_only=False);assert state['step']==e['step']
                net.load_state_dict(state['state_dict']);x,_=batch(z,[train,train+24],r['horizon'])
                with torch.no_grad():p=net(x).numpy()
                old=np.load(checkpoint.parent/f"u{e['step']}-probe.npy")
                assert np.array_equal(p,old),r['key']
                replays+=p.size;tensors+=1
    dump(OUT/'replay.json',{'status':'PASS','bit_exact_scalar_forecasts':replays,
        'checkpoints':tensors,'freeze_sha256':sha(OUT/'frozen.json')})
    print('AUDIT',tensors,replays,flush=True)


if __name__=='__main__':
    torch.set_num_threads(1)
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('train','evaluate','audit'))
    args=p.parse_args();globals()[args.mode]()
