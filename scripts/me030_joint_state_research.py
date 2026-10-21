"""ME030 bounded authentic historical-DEV readout engineering, no new cohort."""
import argparse
import json
import platform
import resource
import subprocess
import time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from tac_osm.me030_joint_readout import make_model, COUNTS
from scripts.me023_predictive_research import load_source, atomic_checkpoint
from scripts.me025_real_transformer_parity import fit_only_popularity
from scripts.me028_native_readout_fit import evaluate, file_sha, write_json, frozen_latent_mean, eval_mean_state
from scripts.me014b_exact_action_memory import batch
from scripts.me022_native_readout import scores

SEEDS=(14001,14002,14003)
ARMS=tuple(COUNTS)
TRACE=(0,1,25,50,100,150,200,250,300)
ROOT=Path(__file__).resolve().parents[1]
PROTOCOL=ROOT/'research/mathematical-engineering/me030/PROTOCOL.md'


def source():
    start=time.process_time()
    data=load_source(ROOT/'results/me022')
    fit,dev=data['train'],data['dev']
    mean=float(fit['duration'].mean())
    scale=max(float(fit['duration'].std(unbiased=False)),1e-5)
    pop=fit_only_popularity(fit)
    return fit,dev,mean,scale,pop,time.process_time()-start


def fit_one(arm,seed,fit,dev,mean,scale,pop,prep,out):
    start=time.process_time();wall=time.perf_counter()
    torch.manual_seed(seed)
    net=torch.jit.script(make_model(arm,mean,scale,pop))
    groups=[[],[]]
    for n,p in net.named_parameters():
        groups[int(n.startswith(('head.','observable_hidden.','observable_output.')))].append(p)
    opt=torch.optim.AdamW([{'params':groups[0],'lr':.001},
                           {'params':groups[1],'lr':.006}],weight_decay=1e-4)
    sampler=torch.Generator().manual_seed(seed+300)
    charged=prep+time.process_time()-start
    weights=torch.tensor([1/6,1/6,1/6,1/2]).reshape(1,4)
    trace=[];clocks=[]
    name=f'{arm}-seed{seed}'
    for step in range(301):
        if step in TRACE:
            p=evaluate(net,dev)
            trace.append({'step':step,'charged_train_cpu_seconds':charged,
                          'metrics':scores(p,dev['y'].numpy())})
            atomic_checkpoint(out/f'{name}.pt',{
                'arm':arm,'seed':seed,'step':step,'state_dict':net.state_dict(),
                'optimizer':opt.state_dict(),'sampler_rng':sampler.get_state(),
                'model_rng':torch.get_rng_state(),'mean':mean,'scale':scale})
            write_json(out/f'{name}-trace.json',{'curve':trace,'per_update':clocks})
            print(json.dumps({'arm':arm,'seed':seed,**trace[-1]}),flush=True)
        if step==300:break
        if charged>=180:raise RuntimeError('registered training CPU cap exceeded; preserve partial')
        t=time.process_time()
        idx=torch.randint(len(fit['y']),(128,),generator=sampler)
        y=torch.cat((fit['metadata'][idx][:,[32,64,96],1],fit['y'][idx]),-1)
        net.train();opt.zero_grad(set_to_none=True)
        logits=net.prefix_logits(*batch(fit,idx))
        loss=(F.binary_cross_entropy_with_logits(logits,y,reduction='none')*weights).sum(-1).mean()
        if not torch.isfinite(loss):raise ArithmeticError('nonfinite FIT loss')
        loss.backward();grad=torch.nn.utils.clip_grad_norm_(net.parameters(),1.)
        if not torch.isfinite(grad):raise ArithmeticError('nonfinite FIT gradient')
        opt.step();elapsed=time.process_time()-t;charged+=elapsed
        clocks.append({'step':step+1,'cpu_seconds':elapsed,'loss':float(loss.detach()),
                       'gradient_norm_before_clip':float(grad)})
    if charged>180:raise RuntimeError('training cap exceeded on final step')
    np.save(out/f'{name}.npy',p)
    mu=frozen_latent_mean(net,fit)
    mean_pred=eval_mean_state(net,dev,mu)
    np.save(out/f'{name}-mean-state.npy',mean_pred)
    net.save(str(out/f'{name}.jit.pt'))
    # Measure single-thread CPU full-window inference: 3 warmups, 9 timings, B128.
    sample=batch(dev,slice(0,128))
    net.eval()
    with torch.no_grad():
        for _ in range(3):net(*sample)
        times=[]
        for _ in range(9):
            t=time.perf_counter();net(*sample);times.append(time.perf_counter()-t)
    files={suffix:file_sha(out/f'{name}{suffix}') for suffix in
           ('.pt','.npy','.jit.pt','-mean-state.npy','-trace.json')}
    row={'arm':arm,'seed':seed,'name':name,'parameters':COUNTS[arm],
         'nonembedding_parameters':COUNTS[arm]-128000,'updates':300,
         'training_rows':38400,'supervised_responses':153600,
         'charged_train_cpu_seconds':charged,'shared_preprocessing_cpu_seconds':prep,
         'whole_fit_cpu_seconds':time.process_time()-start,
         'whole_fit_wall_seconds':time.perf_counter()-wall,
         'process_lifetime_peak_RSS_KiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
         'inference_B128_median_seconds':float(np.median(times)),
         'inference_samples_seconds':times,'metrics':trace[-1]['metrics'],
         'mean_state_replacement_delta_Brier':scores(mean_pred,dev['y'].numpy())['Brier']-trace[-1]['metrics']['Brier'],
         'files_sha256':files,'curve':trace}
    write_json(out/f'{name}-record.json',row)
    return row


def contrasts(out,rows,dev):
    y=dev['y'].numpy().ravel();users,ix=np.unique(dev['users'],return_inverse=True)
    counts=np.bincount(ix);rng=np.random.default_rng(30030);result={}
    by={(r['arm'],r['seed']):r for r in rows}
    for control in ('additive_native','joint_pool'):
        delta=[]
        for seed in SEEDS:
            a=np.load(out/(by['joint_native',seed]['name']+'.npy'))
            b=np.load(out/(by[control,seed]['name']+'.npy'))
            delta.append((a-y)**2-(b-y)**2)
        event=np.mean(delta,axis=0);sums=np.bincount(ix,weights=event)
        samples=[]
        for _ in range(1000):
            draw=rng.integers(len(users),size=len(users))
            samples.append(sums[draw].sum()/counts[draw].sum())
        result[control]={'per_seed_Brier_delta':[float(d.mean()) for d in delta],
                         'mean_Brier_delta':float(event.mean()),
                         'user_clustered_CI95':np.quantile(samples,[.025,.975]).tolist(),
                         'seed':30030,'draws':1000,'users':len(users)}
    return result


def run(out):
    out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(1)
    fit,dev,mean,scale,pop,prep=source()
    record={'study':'ME030_JOINT_NATIVE_STATE_READOUT','protocol_sha256':file_sha(PROTOCOL),
            'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
            'source_snapshot_sha256':file_sha(ROOT/'results/me022/data-snapshot.pt'),
            'python':platform.python_version(),'torch':torch.__version__,'threads':1,
            'cpu':Path('/proc/cpuinfo').read_text().split('model name')[1].splitlines()[0],
            'parameters':COUNTS,'seeds':SEEDS,'rows':[],'complete':False,
            'historical_DEV_reused':True,'reserved_cohorts_opened':False,
            'candidate_promoted':False,'architecture_superiority_allowed':False,
            'parity':{'parameters':'PASS','data_exposure':'PASS','selection':'PASS',
                      'evaluation':'PASS historical DEV only','actual_compute':'FAIL: exposure-matched exploratory lane'}}
    write_json(out/'results.json',record)
    for seed in SEEDS:
        # Rotate execution ordering; identical sampling schedule within seed.
        offset=SEEDS.index(seed)
        for arm in ARMS[offset:]+ARMS[:offset]:
            record['rows'].append(fit_one(arm,seed,fit,dev,mean,scale,pop,prep,out))
            write_json(out/'results.json',record)
    record['contrasts']=contrasts(out,record['rows'],dev)
    ece={arm:float(np.mean([r['metrics']['ECE10'] for r in record['rows'] if r['arm']==arm])) for arm in ARMS}
    gates={}
    for control,c in record['contrasts'].items():
        gates['beats_'+control]=all(d<0 for d in c['per_seed_Brier_delta']) and c['mean_Brier_delta']<=-.0005
    gates['calibration']=all(ece['joint_native']<=ece[c]+.003 for c in ('additive_native','joint_pool'))
    gates['state_diagnostic']=all(r['mean_state_replacement_delta_Brier']>=.0002 for r in record['rows'] if r['arm']=='joint_native')
    gates['engineering_candidate']=all(gates.values())
    record.update(complete=True,gates=gates,mean_ECE10=ece)
    write_json(out/'results.json',record)
    print(json.dumps({'gates':gates,'contrasts':record['contrasts']}),flush=True)


def replay(out):
    record=json.loads((out/'results.json').read_text())
    assert record['complete'] and len(record['rows'])==9
    # Fail closed on every saved payload before loading observations.
    for row in record['rows']:
        for suffix,digest in row['files_sha256'].items():
            if file_sha(out/(row['name']+suffix))!=digest:raise ValueError('artifact hash mismatch')
    torch.set_num_threads(1)
    fit,dev,mean,scale,pop,_=source()
    exact=0
    for row in record['rows']:
        ck=torch.load(out/(row['name']+'.pt'),weights_only=True)
        assert ck['step']==300 and ck['arm']==row['arm'] and ck['seed']==row['seed']
        model=torch.jit.script(make_model(row['arm'],mean,scale,pop))
        model.load_state_dict(ck['state_dict'],strict=True)
        scripted=torch.jit.load(str(out/(row['name']+'.jit.pt')))
        expected=np.load(out/(row['name']+'.npy'))
        for net in (model,scripted):
            actual=evaluate(net,dev)
            np.testing.assert_allclose(actual,expected,rtol=0,atol=1e-7)
            exact+=int(np.array_equal(actual,expected))
        mu=frozen_latent_mean(model,fit)
        np.testing.assert_allclose(eval_mean_state(model,dev,mu),
            np.load(out/(row['name']+'-mean-state.npy')),rtol=0,atol=1e-7)
        assert scores(expected,dev['y'].numpy())==row['metrics']
    actual=contrasts(out,record['rows'],dev)
    assert actual==record['contrasts']
    write_json(out/'replay.json',{'status':'PASS','checkpoint_and_exports':18,
              'forecasts':18*len(dev['y']),'bit_exact_paths':exact,
              'mean_state_paths':9,'source_sha_verified':True,'reserved_cohorts_opened':False})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',default='results/me030');p.add_argument('--replay',action='store_true')
    a=p.parse_args()
    (replay if a.replay else run)(Path(a.out))
