"""Real-data development of successively engineered native TAC forecasters."""
import argparse,csv,hashlib,io,json,platform,resource,shutil,subprocess,time,traceback
from datetime import datetime
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from tac_osm.me035_residual_forecast import ARMS,build_model
from scripts.me028_native_readout_fit import file_sha,write_json
from scripts.me023_predictive_research import atomic_checkpoint

ROOT=Path(__file__).resolve().parents[1]
PROTOCOL=ROOT/'research/mathematical-engineering/me035/PROTOCOL.md'
SOURCE_SHA='f18de3ad269cef59bb07b5438d79bb3042d3be49bdeecf01c1cd6d29695ee066'
COLUMNS=('HUFL','HULL','MUFL','MULL','LUFL','LULL','OT')
SEEDS=(35001,35002,35003)
STEPS=(250,500,750)


def source(root):
    raw=(root/'ETTh1.csv').read_bytes()
    if hashlib.sha256(raw).hexdigest()!=SOURCE_SHA:raise ValueError('official ETTh1 SHA mismatch')
    reader=csv.DictReader(io.StringIO(raw.decode('utf-8-sig')))
    assert tuple(reader.fieldnames)==('date',)+COLUMNS
    rows=[];dates=[]
    for _ in range(11640):
        r=next(reader);d=datetime.fromisoformat(r['date'])
        if dates:assert (d-dates[-1]).total_seconds()==3600
        v=[float(r[c]) for c in COLUMNS]
        if not np.isfinite(v).all():raise ValueError('unexpected missing/nonfinite physical observations')
        rows.append(v);dates.append(d)
    a=np.asarray(rows,dtype=np.float32)
    mean=a[:8760].mean(0,dtype=np.float64).astype(np.float32)
    scale=np.maximum(a[:8760].std(0,dtype=np.float64),.001).astype(np.float32)
    z=(a-mean)/scale
    def pack(first,last,stride):
        ix=np.arange(first,last-24,stride)
        return {'x':torch.from_numpy(z[ix[:,None]+np.arange(-95,1)[None,:]].copy()),
            'y':torch.from_numpy(z[ix[:,None]+np.array([1,6,24])[None,:]].copy()),'indices':ix}
    return pack(95,8760,3),pack(8760,11640,6),{
        'sha256':SOURCE_SHA,'raw_bytes':len(raw),'numeric_rows_parsed':11640,
        'first_timestamp':str(dates[0]),'last_timestamp':str(dates[-1]),
        'mean':mean.tolist(),'scale':scale.tolist(),'columns':list(COLUMNS),
        'license':'CC BY-ND 4.0 (publisher repository LICENSE)',
        'source_url':'https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh1.csv',
        'publisher':'Zhou et al., ETDataset / Informer, AAAI2021; measurements collected with Beijing Guowang Fuda',
        'timestamp_timezone':'publisher-unspecified; consecutive hourly source order',
        'later_rows_numeric_parsed':False,'new_independent_dataset':False}


def metrics(p,y):
    e=np.asarray(p,dtype=np.float64)-np.asarray(y,dtype=np.float64)
    return {'MSE':float(np.mean(e**2)),'MAE':float(np.mean(abs(e))),
        'horizon_MSE':np.mean(e**2,axis=(0,2)).tolist(),
        'channel_MSE':np.mean(e**2,axis=(0,1)).tolist(),'rows':len(e)}


def predict(net,x):
    net.eval()
    with torch.no_grad():return torch.cat([net(x[i:i+256]) for i in range(0,len(x),256)]).numpy()


def ridge_fit(train,dev):
    start=time.process_time();x=train['x'].numpy().reshape(len(train['x']),-1).astype(np.float64)
    x=np.c_[x,np.ones(len(x))];d=(train['y']-train['x'][:,-1:]).numpy().reshape(len(x),21)
    penalty=np.eye(x.shape[1])*100.;penalty[-1,-1]=0
    coef=np.linalg.solve(x.T@x+penalty,x.T@d)
    cpu=time.process_time()-start
    v=np.c_[dev['x'].numpy().reshape(len(dev['x']),-1),np.ones(len(dev['x']))]
    pred=(v@coef).reshape(-1,3,7)+dev['x'][:,-1:].numpy()
    return coef,pred,{'parameters':int(coef.size),'train_rows':len(x),'ridge_alpha':100.,'charged_cpu_seconds':cpu,'matched_exposure':False}


def train_one(arm,seed,data,out):
    start=time.process_time();net=torch.jit.script(build_model(arm,seed));groups=[[],[]]
    for n,p in net.named_parameters():groups[int(n.startswith(('direct.','mix.','output.')))].append(p)
    opt=torch.optim.AdamW([{'params':groups[0],'lr':.001},{'params':groups[1],'lr':.003}],weight_decay=1e-4)
    gen=torch.Generator().manual_seed(seed+50);trace=[];checkpoints=[]
    x,y=data['x'],data['y'];charged=time.process_time()-start;step=0
    def save(label):
        name=f'{arm}-s{seed}-u{label}'
        atomic_checkpoint(out/(name+'.pt'),{'state_dict':net.state_dict(),'optimizer':opt.state_dict(),
            'sampler_rng':gen.get_state(),'model_rng':torch.get_rng_state(),'step':step,'arm':arm,'seed':seed})
        net.save(str(out/(name+'.jit.pt')));write_json(out/(name+'-trace.json'),trace)
        return name
    try:
        for step in range(1,751):
            t=time.process_time();idx=torch.randint(len(x),(64,),generator=gen)
            opt.zero_grad(set_to_none=True);pred=net(x[idx]);loss=F.mse_loss(pred,y[idx])
            if not torch.isfinite(loss):raise ArithmeticError('nonfinite real forecast loss')
            loss.backward();grad=torch.nn.utils.clip_grad_norm_(net.parameters(),1.)
            if not torch.isfinite(grad):raise ArithmeticError('nonfinite gradient')
            memory_grad=sum(float(p.grad.square().sum()) for n,p in net.named_parameters() if n.startswith(('enc.','transition.','gate.','slow.','slow_gate.')) and p.grad is not None)**.5
            opt.step();dt=time.process_time()-t;charged+=dt
            trace.append({'step':step,'MSE':float(loss.detach()),'CPU_seconds':dt,'clipped_gradient_norm':float(grad),'memory_gradient_after_clip':memory_grad})
            if step in STEPS:checkpoints.append({'step':step,'name':save(step),'charged_CPU_seconds':charged})
            if charged>150:raise RuntimeError('registered150CPU-second fit ceiling exceeded')
    except Exception:
        save('partial');raise
    assert any(v['memory_gradient_after_clip']>0 for v in trace)
    return {'arm':arm,'seed':seed,'trainable_parameters':sum(p.numel() for p in net.parameters()),
        'steps':750,'histories':48000,'response_values':1008000,
        'charged_CPU_seconds':charged,'checkpoints':checkpoints,
        'positive_memory_gradient_steps':sum(v['memory_gradient_after_clip']>0 for v in trace)}


def evaluate(out,dev,record):
    rows=[]
    for fit in record['fits']:
        for ck in fit['checkpoints']:
            model=torch.jit.load(str(out/(ck['name']+'.jit.pt')))
            p=predict(model,dev['x']);np.save(out/(ck['name']+'-predictions.npy'),p)
            rows.append({'arm':fit['arm'],'seed':fit['seed'],'step':ck['step'],'metrics':metrics(p,dev['y'].numpy())})
    pooled={a:{str(s):{k:float(np.mean([v['metrics'][k] for v in rows if v['arm']==a and v['step']==s])) for k in ('MSE','MAE')} for s in STEPS} for a in ARMS}
    selected=min(((pooled[a][str(s)]['MSE'],a,s) for a in ARMS for s in STEPS))
    base=[v for v in rows if v['arm']=='native_joint' and v['step']==750]
    chosen=[v for v in rows if v['arm']==selected[1] and v['step']==selected[2]]
    diffs=[next(v['metrics']['MSE'] for v in chosen if v['seed']==s)/next(v['metrics']['MSE'] for v in base if v['seed']==s)-1 for s in SEEDS]
    bh=np.mean([v['metrics']['horizon_MSE'] for v in base],axis=0)
    ch=np.mean([v['metrics']['horizon_MSE'] for v in chosen],axis=0)
    return {'per_model':rows,'mean_metrics':pooled,'selected_development_candidate':{'arm':selected[1],'step':selected[2],'mean_MSE':selected[0]},
        'selected_relative_MSE_vs_native_joint750_per_seed':diffs,
        'selected_horizon_relative_MSE_vs_native_joint750':(ch/bh-1).tolist(),
        'development_improvement_gate':float(np.mean(diffs))<=-.02 and all(d<0 for d in diffs) and bool((ch/bh<=1.05).all()),
        'selection_used_previously_exposed_validation':True,'independent_qualification':False}


def run(out):
    out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(1)
    start=time.process_time();wall=time.perf_counter();train,dev,src=source(ROOT/'results/me035-source')
    for name in ('ETTh1.csv','README.md','LICENSE'):shutil.copyfile(ROOT/'results/me035-source'/name,out/name)
    write_json(out/'source.json',src)
    record={'study':'ME035_RESIDUAL_NATIVE_FORECAST_ARCHITECTURE_DEVELOPMENT','complete':False,
        'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        'protocol_sha256':file_sha(PROTOCOL),'source':src,'fits':[],
        'train_windows':len(train['x']),'development_windows':len(dev['x']),
        'python':platform.python_version(),'torch':torch.__version__,'threads':1,
        'hardware':Path('/proc/cpuinfo').read_text().split('model name')[1].splitlines()[0],
        'KuaiRec_accessed':False,'all_models_frozen_before_scoring':False,
        'parity':{'TAC_parameters':'PASS18924 each','TAC_exposure':'PASS750*64 histories,21 future scalar targets',
            'TAC_selection':'PASS same3endpoints/3seeds; explicit development selection',
            'full_compute':'FAIL actual spend differs, profiler is incomplete','reference_parameters_exposure':'FAIL unmatched persistence/ridge'},
        'architecture_superiority_claim':False,'candidate_promoted':False}
    write_json(out/'results.json',record)
    for seed in SEEDS:
        offset=SEEDS.index(seed);order=ARMS[offset:]+ARMS[:offset]
        for arm in order:
            fit=train_one(arm,seed,train,out);record['fits'].append(fit);write_json(out/'results.json',record)
            print(json.dumps({'completed_fit':arm,'seed':seed,'CPU_seconds':fit['charged_CPU_seconds'],'development_scores_opened':False}),flush=True)
    coef,rp,ref=ridge_fit(train,dev);np.save(out/'ridge-coefficients.npy',coef)
    np.save(out/'ridge-predictions.npy',rp);record['ridge']=ref
    record['all_models_frozen_before_scoring']=True
    write_json(out/'frozen-models.json',{p.name:file_sha(p) for p in out.iterdir() if p.is_file() and p.name!='results.json'})
    record['evaluation']=evaluate(out,dev,record)
    record['evaluation']['references']={'persistence':metrics(dev['x'][:,-1:].expand(-1,3,-1).numpy(),dev['y'].numpy()),'ridge':metrics(rp,dev['y'].numpy())}
    record['complete']=True;record['physical_CPU_seconds']=time.process_time()-start;record['wall_seconds']=time.perf_counter()-wall
    record['process_peak_RSS_KiB']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    write_json(out/'results.json',record);print(json.dumps(record['evaluation']),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',default='results/me035');a=p.parse_args()
    try:run(Path(a.out))
    except Exception:
        path=Path(a.out)
        if path.exists():write_json(path/'failure.json',{'error':traceback.format_exc()})
        raise
