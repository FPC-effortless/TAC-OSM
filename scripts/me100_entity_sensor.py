"""ME100 prospective five-seed capability pilot. No training in replay mode."""
from pathlib import Path
import sys,json,hashlib,time,random,subprocess,argparse
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'src')]
from scripts import me100_sensor_data as data
from scripts.validate_research_parity import check_manifest
from tac_osm.variants.entity_sensor_world import EntitySensorWorld
from tac_osm.parameter_budget import budgeted_optimizer,checked_optimizer_step,save_training_checkpoint,validate_runtime_evidence
OUT=ROOT/'research/replay-bundles/ME100';REG=ROOT/'research/mathematical-engineering/me100/FAIR_COMPARISON.json'
SEEDS=range(100001,100006);STEPS=1536
SOURCES=['scripts/me100_entity_sensor.py','scripts/me100_sensor_data.py','scripts/me100_graph_reference.py','src/tac_osm/variants/entity_sensor_world.py','src/tac_osm/variants/parallel_affine_memory.py','src/tac_osm/variants/retention_controlled_world.py','src/tac_osm/parameter_budget.py']

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
def frozen(paths):
    for p in paths:
        raw=subprocess.check_output(['git','show','HEAD:'+str(p.relative_to(ROOT))],cwd=ROOT,stderr=subprocess.DEVNULL)
        if raw!=p.read_bytes():raise RuntimeError('original file not Git-frozen: '+str(p))
def tensors(d,origins,length):return [torch.from_numpy(x) for x in data.batch(d,origins,length)]

def fit():
    check_manifest(json.loads(REG.read_text()),'preflight')
    if OUT.exists():raise RuntimeError('output exists; originals cannot be overwritten/retrained')
    frozen([REG,data.ARCHIVE,ROOT/'.github/workflows/me100-entity-sensor.yml']+[ROOT/p for p in SOURCES])
    d=data.load('fit');OUT.mkdir(parents=True);rows=[]
    sources={p:sha(ROOT/p) for p in SOURCES}
    for seed in SEEDS:
        torch.manual_seed(seed);np.random.seed(seed);random.seed(seed);rng=np.random.default_rng(seed)
        m=EntitySensorWorld();m.target_mean.copy_(torch.from_numpy(d['mean'][:6]));m.target_scale.copy_(torch.from_numpy(d['std'][:6]))
        folder=OUT/f'seed-{seed}';folder.mkdir()
        opt=budgeted_optimizer(m,torch.optim.AdamW,lr=.0006,weight_decay=.001,budget_manifest=REG,budget_receipt=folder/'live-budget.json',budget_seed=seed,budget_architecture='entity_sensor')
        schedule=[];losses=[];start=time.process_time();wall=time.monotonic();first_grads=None
        for step in range(STEPS):
            length=(8,64,256)[step%3]
            origins=[d['gap'][j] for j in rng.integers(len(d['gap']),size=8)]+[d['general'][j] for j in rng.integers(len(d['general']),size=8)]
            x=tensors(d,origins,length);opt.zero_grad(set_to_none=True);p,_=m(*x[:6]);nll=m.nll(p,x[6],x[7])
            target=(torch.log1p(x[6])-m.target_mean)/m.target_scale
            mse=torch.where(x[7],(p.log_location-target).square(),torch.zeros_like(target)).sum()/x[7].sum().clamp_min(1)
            loss=nll+.1*mse
            if not torch.isfinite(loss):raise RuntimeError('nonfinite loss; no silent retry')
            loss.backward()
            if not all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters()):raise RuntimeError('invalid gradient')
            if step==1:first_grads={n:float(p.grad.norm()) for n,p in m.named_parameters()}
            torch.nn.utils.clip_grad_norm_(m.parameters(),1);checked_optimizer_step(opt)
            schedule.append([length,origins]);losses.append([float(nll.detach()),float(mse.detach())])
            if (step+1)%256==0:print(json.dumps(dict(seed=seed,step=step+1,fit_loss=losses[-1],wall_seconds=time.monotonic()-wall)),flush=True)
        dest=folder/'final.pt'
        save_training_checkpoint(dict(model=m.state_dict(),optimizer=opt.state_dict(),seed=seed,steps=STEPS,schedule=schedule,losses=losses,torch_rng=torch.get_rng_state(),numpy_rng=np.random.get_state(),python_rng=random.getstate(),generator_rng=rng.bit_generator.state,preprocessing=dict(mean=d['mean'],std=d['std']),source_files=sources,source_sha256=data.SHA),dest)
        row=dict(seed=seed,trainable_parameters=sum(p.numel() for p in m.parameters()),deployed_parameters=sum(p.numel() for p in m.parameters()),steps=STEPS,checkpoint_sha256=sha(dest),cpu_seconds=time.process_time()-start,wall_seconds=time.monotonic()-wall,forecast_origin_exposures=STEPS*16,history_event_exposures=512*16*(8+64+256),nonzero_gradient_tensors_step2=sum(v>0 for v in first_grads.values()),gradient_tensors=len(first_grads))
        write(folder/'FIT_RECEIPT.json',row);write(folder/'GRADIENTS.json',first_grads);rows.append(row);print(json.dumps(row),flush=True)
    write(OUT/'FIT_INDEX.json',dict(models=rows,source_files=sources))

def budget_audit():
    frozen([OUT/'FIT_INDEX.json']+[p for seed in SEEDS for p in (OUT/f'seed-{seed}').glob('*')])
    index=json.loads((OUT/'FIT_INDEX.json').read_text())
    for rel,s in index['source_files'].items():
        if sha(ROOT/rel)!=s:raise RuntimeError('training source changed')
    for row in json.loads(REG.read_text())['models']:
        folder=OUT/f"seed-{row['seed']}"
        validate_runtime_evidence(dict(row,_prefit_registration_sha256=sha(REG),_expected_optimizer_steps=STEPS,runtime_evidence=dict(receipt_path=str(folder/'live-budget.json'),receipt_sha256=sha(folder/'live-budget.json'),checkpoint_path=str(folder/'final.pt'),checkpoint_sha256=sha(folder/'final.pt'))))
    return index

def fit_ridge(d):
    # All FIT origins; per-channel availability masks; fixed lambda, no REPORT tuning.
    xx=np.zeros((6,71,71));xy=np.zeros((6,71));orig=d['general']
    for i in range(0,len(orig),2048):
        p=orig[i:i+2048];x=data.ridge_features(d,p);sid=np.array([s for s,t in p]);at=np.array([t for s,t in p]);y=d['raw'][sid,at+6,:6]
        for c in range(6):
            keep=np.isfinite(y[:,c]);a=x[keep];b=(np.log1p(y[keep,c])-d['mean'][c])/d['std'][c]
            xx[c]+=a.T@a;xy[c]+=a.T@b
    return np.stack([np.linalg.solve(xx[c]+100*np.eye(71),xy[c]) for c in range(6)],1)

def metrics(q,y,mask):
    lo,mid,hi=q[:,:,0],q[:,:,1],q[:,:,2];ae=np.abs(mid-y)
    cover=(lo<=y)&(y<=hi);interval=hi-lo+20*np.maximum(lo-y,0)+20*np.maximum(y-hi,0)
    return dict(pm_mae=float(ae[:,0].mean()),mae_by_channel=[float(ae[:,i][mask[:,i]].mean()) for i in range(6)],coverage_by_channel=[float(cover[:,i][mask[:,i]].mean()) for i in range(6)],interval_score_by_channel=[float(interval[:,i][mask[:,i]].mean()) for i in range(6)])

def evaluate(replay=False):
    index=budget_audit();fitdata=data.load('fit');coef=fit_ridge(fitdata);d=data.load('report')
    if not np.array_equal(d['mean'],fitdata['mean']) or not np.array_equal(d['std'],fitdata['std']):raise RuntimeError('normalization changed')
    cohorts={'gap':d['gap'],'general':d['general']};arrays={'ridge_coefficients':coef};results={};rows=[]
    for name,origins in cohorts.items():
        if not origins:raise RuntimeError('empty reserved reporting cohort')
        sid=np.array([s for s,t in origins]);at=np.array([t for s,t in origins]);y=d['raw'][sid,at+6,:6];mask=np.isfinite(y);y=np.nan_to_num(y,nan=0)
        locf=np.maximum(0,np.expm1(d['last'][sid,at,:6]*d['std'][:6]+d['mean'][:6]))
        ridge=np.maximum(0,np.expm1(np.clip(data.ridge_features(d,origins)@coef*d['std'][:6]+d['mean'][:6],-20,20)))
        refs=dict(locf_pm_mae=float(np.abs(locf[:,0]-y[:,0]).mean()),ridge_pm_mae=float(np.abs(ridge[:,0]-y[:,0]).mean()))
        arrays[name+'_origins']=np.asarray(origins);arrays[name+'_truth']=y;arrays[name+'_observed']=mask;arrays[name+'_locf']=locf;arrays[name+'_ridge']=ridge
        results[name]=dict(count=len(origins),calendar_weeks=len(set(at//168)),references=refs,models=[])
    with torch.no_grad():
        for seed in SEEDS:
            cp=torch.load(OUT/f'seed-{seed}'/'final.pt',map_location='cpu',weights_only=False);m=EntitySensorWorld();m.load_state_dict(cp['model']);m.eval();scores={}
            for name,origins in cohorts.items():
                for length in (8,256):
                    qs=[];nlls=[]
                    for i in range(0,len(origins),32):
                        x=tensors(d,origins[i:i+32],length);p,_=m(*x[:6]);qs.append(m.raw_quantiles(p).numpy());nlls.append(float(m.nll(p,x[6],x[7]))*int(x[7].sum()))
                    q=np.concatenate(qs);arrays[f'{name}_{seed}_{length}']=q
                    r=metrics(q,arrays[name+'_truth'],arrays[name+'_observed']);r['raw_density_nll']=sum(nlls)/int(arrays[name+'_observed'].sum());scores[name+str(length)]=r
                    results[name]['models'].append(dict(seed=seed,context=length,**r))
            gate=dict(history=scores['gap256']['pm_mae']<=.95*scores['gap8']['pm_mae'],practical=scores['gap256']['pm_mae']<=.97*min(results['gap']['references'].values()),general_guard=scores['general256']['pm_mae']<=1.05*results['general']['references']['locf_pm_mae'],coverage=scores['gap256']['coverage_by_channel'][0]>=.85)
            rows.append(dict(seed=seed,gates=gate,full_gate=all(gate.values())));print(json.dumps(rows[-1]),flush=True)
    # Pairing by global calendar week preserves cross-station co-movement.
    weeks=arrays['gap_origins'][:,1]//168;unique=np.unique(weeks);rng=np.random.default_rng(100100)
    delta=np.mean([np.abs(arrays[f'gap_{s}_256'][:,0,1]-arrays['gap_truth'][:,0])-np.abs(arrays[f'gap_{s}_8'][:,0,1]-arrays['gap_truth'][:,0]) for s in SEEDS],0)
    sums=np.array([delta[weeks==w].sum() for w in unique]);counts=np.array([(weeks==w).sum() for w in unique]);boots=[]
    for _ in range(2000):
        take=rng.integers(len(unique),size=len(unique));boots.append(float(sums[take].sum()/counts[take].sum()))
    result=dict(status='NEW_SOURCE_2015_REPORT_CAPABILITY_PILOT',cohorts=results,models=rows,all_seed_gate=all(r['full_gate'] for r in rows),history_long_minus_short_clustered_ci95=np.quantile(boots,[.025,.975]).tolist(),bootstrap_clusters=len(unique),independent_architecture_qualification=False,protected_test_scored=False,source_delayed_arrival_verified=False)
    if replay:
        old=np.load(OUT/'FORECASTS.npz')
        if set(old.files)!=set(arrays):raise RuntimeError('forecast keys mismatch')
        for k,v in arrays.items():np.testing.assert_allclose(v,old[k],rtol=1e-5,atol=1e-6)
        def check(a,b):
            if isinstance(a,dict):
                if set(a)!=set(b):raise RuntimeError('summary keys mismatch')
                for k in a:check(a[k],b[k])
            elif isinstance(a,list):
                if len(a)!=len(b):raise RuntimeError('summary length mismatch')
                for v,w in zip(a,b):check(v,w)
            elif isinstance(a,float):np.testing.assert_allclose(a,b,rtol=1e-5,atol=1e-6)
            elif a!=b:raise RuntimeError('categorical summary mismatch')
        check(result,json.loads((OUT/'RESULTS.json').read_text()));print('ME100_ORIGINAL_MODEL_SCORE_REPLAY=PASS',flush=True)
    else:
        if (OUT/'RESULTS.json').exists():raise RuntimeError('original scores exist; no overwrite')
        np.savez_compressed(OUT/'FORECASTS.npz',**arrays);write(OUT/'RESULTS.json',result)
        post=json.loads(REG.read_text())
        for row,receipt in zip(post['models'],index['models']):row['measured']=receipt
        write(REG.with_name('POSTFIT.json'),post)
    print(json.dumps(dict(gate=result['all_seed_gate'],ci=result['history_long_minus_short_clustered_ci95'],counts={k:v['count'] for k,v in results.items()})),flush=True)

if __name__=='__main__':
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['fit','score','replay']);args=parser.parse_args()
    if args.mode=='fit':fit()
    else:evaluate(args.mode=='replay')
