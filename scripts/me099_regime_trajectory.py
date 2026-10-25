"""Frozen real-data multi-horizon capability protocol, no fitting during replay."""
from pathlib import Path
import argparse,hashlib,json,sys,time,random,subprocess
import numpy as np
import torch
import pyarrow.parquet as pq
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'src')]
from scripts import me097_evidence_replication as old
from tac_osm.variants.evidence_retrieval_world import EvidenceRetrievalWorld
from tac_osm.variants.coherent_trajectory_world import CoherentTrajectoryWorld,joint_student_nll,TrajectoryMoments
from tac_osm.variants.regime_trajectory_world import RegimeTrajectoryWorld,RegimeTrajectory,regime_joint_nll,marginal_intervals
from tac_osm.parameter_budget import budgeted_optimizer,checked_optimizer_step,save_training_checkpoint,validate_runtime_evidence
from scripts.validate_research_parity import check_manifest
OUT=ROOT/'research/replay-bundles/ME099';REG=ROOT/'research/mathematical-engineering/me099/FAIR_COMPARISON.json'
SEEDS=(99001,99002,99003,99004,99005);HORIZONS=(1,4,16);TQ=2.13184678632665

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def frozen(files):
    for p in files:
        if not p.exists():raise RuntimeError('missing required frozen file '+str(p))
        try:b=subprocess.check_output(['git','show','HEAD:'+str(p.relative_to(ROOT))],cwd=ROOT,stderr=subprocess.DEVNULL)
        except subprocess.CalledProcessError:raise RuntimeError('file not committed '+str(p))
        if b!=p.read_bytes():raise RuntimeError('working bytes differ from committed original '+str(p))

def prepare(path):
    _,_,prep=old.data(path) # Exact pinned source digest and original FIT-only statistics.
    d=pq.read_table(path).to_pydict();episodes={}
    for e,f,t,x,a in zip(*(d[k] for k in ['episode_index','frame_index','timestamp','observation.state','action'])):
        episodes.setdefault(e,[]).append((f,t,x,a))
    ids=sorted(episodes);fit=ids[:int(.7*len(ids))];dev=ids[int(.7*len(ids)):int(.85*len(ids))]
    def pack(chosen):
        rows=[];index=[]
        for ep in chosen:
            e=episodes[ep]
            if not all(b[0]==a[0]+1 and 0<b[1]-a[1]<.2 for a,b in zip(e,e[1:])):raise RuntimeError('nonconsecutive source')
            starts=list(range(1,len(e)-64-15))
            for start in starts[::max(1,len(starts)//8)][:8]:
                x=torch.tensor([r[2] for r in e[start:start+64]])
                a=torch.tensor([r[3] for r in e[start-1:start+63]])
                y=torch.stack([torch.tensor([r[2] for r in e[start+h:start+h+64]]) for h in HORIZONS],1)
                rows.append(((x-prep['obs_mean'])/prep['obs_scale'],(a-prep['action_mean'])/prep['action_scale'],(y-prep['obs_mean'])/prep['obs_scale']))
                index.append([int(ep),int(start)])
        return tuple(torch.stack([r[k] for r in rows]) for k in range(3)),index
    tr,ti=pack(fit);dv,di=pack(dev)
    prep=dict(prep,trajectory_fit_windows=ti,trajectory_dev_windows=di)
    return tr,dv,prep

def parent(seed):
    seed=98001+(seed-99001);p=ROOT/f'research/replay-bundles/ME098/seed-{seed}/final.pt'
    index=json.loads((ROOT/'research/replay-bundles/ME098/FIT_INDEX.json').read_text())
    row=next(r for r in index['models'] if r['seed']==seed)
    expected=next(r for r in json.loads(REG.read_text())['models'] if r['seed']==seed+1000)['parent_checkpoint_sha256']
    if sha(p)!=row['checkpoint_sha256'] or sha(p)!=expected:raise RuntimeError('original parent SHA mismatch')
    return p,torch.load(p,map_location='cpu',weights_only=False)

def fit(path):
    check_manifest(json.loads(REG.read_text()),'preflight')
    if OUT.exists():raise RuntimeError('original output path exists: no overwrite/retry')
    src=[Path(__file__).relative_to(ROOT).as_posix(),'scripts/me097_evidence_replication.py','src/tac_osm/parameter_budget.py',
      'src/tac_osm/variants/regime_trajectory_world.py','src/tac_osm/variants/coherent_trajectory_world.py','src/tac_osm/variants/evidence_retrieval_world.py',
      'src/tac_osm/variants/parallel_continuous_world.py','src/tac_osm/variants/retention_controlled_world.py',
      'src/tac_osm/variants/parallel_affine_memory.py']
    frozen([ROOT/p for p in src]+[REG,ROOT/'.github/workflows/me099-regime-replay.yml'])
    frozen([ROOT/'research/replay-bundles/ME098/FIT_INDEX.json']+[parent(s)[0] for s in SEEDS])
    tr,_,prep=prepare(path);OUT.mkdir(parents=True)
    norm=(tr[2]-tr[0][:,:,None]).square().mean((0,1)).sqrt().clamp_min(1e-5)
    source={p:sha(ROOT/p) for p in src};receipts=[]
    for seed in SEEDS:
        torch.manual_seed(seed);np.random.seed(seed);random.seed(seed)
        par,checkpoint=parent(seed);oldm=CoherentTrajectoryWorld();oldm.load_state_dict(checkpoint['model'])
        m=RegimeTrajectoryWorld();m.load_coherent_parent(oldm)
        folder=OUT/f'seed-{seed}';folder.mkdir()
        retained=[p for n,p in m.named_parameters() if not n.startswith(('trajectory_head.','regime_gate.')) and n!='raw_broadening']
        shape=list(m.trajectory_head.parameters())
        new=list(m.regime_gate.parameters())+[m.raw_broadening]
        opt=budgeted_optimizer(m,torch.optim.AdamW,parameters=[dict(params=retained,lr=.00005),dict(params=shape,lr=.0002),dict(params=new,lr=.0006)],
          weight_decay=.001,budget_manifest=REG,budget_receipt=folder/'live-budget.json',budget_seed=seed,budget_architecture='regime_trajectory')
        schedule=torch.randint(0,len(tr[0]),(512,8),generator=torch.Generator().manual_seed(seed));losses=[];start=time.process_time()
        weights=norm.new_tensor([.6,.2,.2])[None,None,:,None]
        for idx in schedule:
            opt.zero_grad(set_to_none=True);pred,_=m(tr[0][idx],tr[1][idx])
            mse=(((pred.mean-tr[2][idx])/norm).square()*weights).sum(2).mean()
            nll=regime_joint_nll(pred,tr[2][idx]).mean()/42
            lo,hi=marginal_intervals(pred);truth=tr[2][idx]
            proper_interval=((hi-lo)+20*(lo-truth).clamp_min(0)+20*(truth-hi).clamp_min(0))/norm
            interval=(proper_interval*weights).sum(2).mean()
            loss=mse+.05*nll+.1*interval
            if not torch.isfinite(loss):raise RuntimeError('nonfinite loss, no silent covariance repair')
            loss.backward()
            if not all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters()):raise RuntimeError('invalid active gradients')
            torch.nn.utils.clip_grad_norm_(m.parameters(),1);checked_optimizer_step(opt);losses.append([float(mse.detach()),float(nll.detach()),float(interval.detach())])
        dest=folder/'final.pt'
        save_training_checkpoint(dict(model=m.state_dict(),optimizer=opt.state_dict(),torch_rng=torch.get_rng_state(),numpy_rng=np.random.get_state(),python_rng=random.getstate(),schedule=schedule,seed=seed,preprocessing=prep,loss_normalization=norm,source_files=source,parent_sha256=sha(par),source_sha256=old.SHA,losses=losses,steps=512),dest)
        row=dict(seed=seed,trainable_parameters=sum(p.numel() for p in m.parameters()),deployed_parameters=sum(p.numel() for p in m.parameters()),checkpoint_sha256=sha(dest),parent_sha256=sha(par),optimizer_cpu_seconds=time.process_time()-start,steps=512,history_event_exposures=262144,future_target_exposures=786432)
        (folder/'FIT_RECEIPT.json').write_text(json.dumps(row,indent=2)+'\n');receipts.append(row);print(json.dumps(row),flush=True)
    (OUT/'FIT_INDEX.json').write_text(json.dumps(dict(models=receipts,source_files=source),indent=2)+'\n')

def budget_audit():
    frozen([OUT/'FIT_INDEX.json']+[p for s in SEEDS for p in (OUT/f'seed-{s}').glob('*')])
    for row in json.loads(REG.read_text())['models']:
        d=OUT/f"seed-{row['seed']}";r=dict(row,_prefit_registration_sha256=sha(REG),_expected_optimizer_steps=512,
          runtime_evidence=dict(receipt_path=str(d/'live-budget.json'),receipt_sha256=sha(d/'live-budget.json'),checkpoint_path=str(d/'final.pt'),checkpoint_sha256=sha(d/'final.pt')))
        validate_runtime_evidence(r)
    print('ORIGINAL_LIVE_PARAMETER_RECEIPT_AUDIT=PASS')

def evaluate(path,replay=False):
    budget_audit();tr,dv,prep=prepare(path);index=json.loads((OUT/'FIT_INDEX.json').read_text())
    for rel,digest in index['source_files'].items():
        if sha(ROOT/rel)!=digest:raise RuntimeError('frozen training source changed')
    # Non-neural FIT second-moment Student-t reference, centered on persistence.
    delta=(tr[2]-tr[0][:,:,None]).transpose(-1,-2).double()
    baseS=.5*torch.einsum('btdh,btdk->dhk',delta,delta)/(delta.shape[0]*delta.shape[1])
    baseS=baseS+1e-8*torch.eye(3,dtype=baseS.dtype)
    basem=TrajectoryMoments(dv[0][:,:,None].expand_as(dv[2]),baseS[None,None].expand(len(dv[0]),64,-1,-1,-1))
    thresholds=torch.quantile((tr[2]-tr[0][:,:,None]).square().mean(-1).sqrt().reshape(-1,3),.9,dim=0)
    def score(pred):
        err=pred.mean-dv[2];se=err.square().mean(-1);ae=err.abs().mean(-1)
        if isinstance(pred,RegimeTrajectory):
            lo,hi=marginal_intervals(pred)
            density=float(regime_joint_nll(pred,dv[2]).mean()/14)
        else:
            diag=pred.scale_matrix.diagonal(dim1=-2,dim2=-1).transpose(-1,-2).clamp_min(0).sqrt()
            lo=pred.mean-TQ*diag;hi=pred.mean+TQ*diag
            density=float(joint_student_nll(pred,dv[2]).mean())
        y=dv[2]
        cover=((lo<=y)&(y<=hi)).to(torch.float32).mean(-1)
        interval=(hi-lo)+20*(lo-y).clamp_min(0)+20*(y-hi).clamp_min(0)
        interval=interval.mean(-1);moving=(dv[2]-dv[0][:,:,None]).square().mean(-1).sqrt()>=thresholds
        by=[]
        for j,h in enumerate(HORIZONS):
            by.append(dict(horizon=h,mse=float(se[:,:,j].mean()),mae=float(ae[:,:,j].mean()),late_mse=float(se[:,32:,j].mean()),moving_mse=float(se[:,:,j][moving[:,:,j]].mean()),coverage=float(cover[:,:,j].mean()),interval_score=float(interval[:,:,j].mean())))
        cases=[]
        for ep in sorted(set(e for e,_ in prep['trajectory_dev_windows'])):
            ix=[i for i,(e,_) in enumerate(prep['trajectory_dev_windows']) if e==ep]
            cases.append(dict(episode=ep,mse=se[ix].mean((0,1)).tolist(),mae=ae[ix].mean((0,1)).tolist(),coverage=cover[ix].mean((0,1)).tolist(),interval_score=interval[ix].mean((0,1)).tolist()))
        return dict(horizons=by,joint_nll=density),cases
    baseline,base_cases=score(basem);rows=[];predictions=[];scales=[];case_rows=[];logits=[];broadening=[]
    with torch.no_grad():
        for seed,receipt in zip(SEEDS,index['models']):
            f=OUT/f'seed-{seed}'/'final.pt'
            if sha(f)!=receipt['checkpoint_sha256']:raise RuntimeError('checkpoint mismatch')
            checkpoint=torch.load(f,map_location='cpu',weights_only=False);m=RegimeTrajectoryWorld();m.load_state_dict(checkpoint['model']);m.eval()
            pred,_=m(dv[0],dv[1]);result,cases=score(pred)
            _,par=parent(seed);p=CoherentTrajectoryWorld();p.load_state_dict(par['model']);parent_mse=float((p(dv[0],dv[1])[0].mean[:,:,0]-dv[2][:,:,0]).square().mean())
            gates=dict(forecast=all(r['mse']<=.95*b['mse'] for r,b in zip(result['horizons'],baseline['horizons'])),preserve_one_step=result['horizons'][0]['mse']<=1.05*parent_mse,coverage=all(r['coverage']>=.85 for r in result['horizons']),interval_score=all(r['interval_score']<=1.05*b['interval_score'] for r,b in zip(result['horizons'],baseline['horizons'])))
            rows.append(dict(seed=seed,**result,parent_same_task_h1_mse=parent_mse,gates=gates,full_gate=all(gates.values())))
            case_rows.append(dict(seed=seed,cases=cases));predictions.append(pred.mean.numpy());scales.append(pred.narrow_scale_matrix.numpy());logits.append(pred.regime_logit.numpy());broadening.append(float(pred.broadening))
    result=dict(status='EXPOSED_DEV_ONLY',source_sha256=old.SHA,windows=len(dv[0]),independent_episodes=len(set(e for e,_ in prep['trajectory_dev_windows'])),window_target_positions_per_model=len(dv[0])*64*3,baseline=baseline,models=rows,registered_gate_passed=all(r['full_gate'] for r in rows),independent_qualification=False,architecture_superiority=False)
    arrays=dict(means=np.stack(predictions),narrow_scale_matrices=np.stack(scales),regime_logits=np.stack(logits),broadening=np.asarray(broadening))
    if replay:
        orig=np.load(OUT/'FORECASTS.npz')
        for k,v in arrays.items():np.testing.assert_allclose(v,orig[k],rtol=1e-5,atol=1e-6)
        def check(a,b):
            if isinstance(a,dict):
                if set(a)!=set(b):raise RuntimeError('summary key mismatch')
                for k in a:check(a[k],b[k])
            elif isinstance(a,list):
                if len(a)!=len(b):raise RuntimeError('summary count mismatch')
                for x,y in zip(a,b):check(x,y)
            elif isinstance(a,float):np.testing.assert_allclose(a,b,rtol=1e-5,atol=1e-6)
            elif a!=b:raise RuntimeError('summary categorical mismatch')
        check(result,json.loads((OUT/'RESULTS.json').read_text()))
        check(case_rows,json.loads((OUT/'CASE_SCORES.json').read_text()))
        print('ORIGINAL_REGIME_TRAJECTORY_REPLAY=PASS')
    else:
        if (OUT/'RESULTS.json').exists():raise RuntimeError('original scores exist, no overwrite')
        np.savez_compressed(OUT/'FORECASTS.npz',**arrays)
        (OUT/'RESULTS.json').write_text(json.dumps(result,indent=2)+'\n');(OUT/'CASE_SCORES.json').write_text(json.dumps(case_rows,indent=2)+'\n')
    print(json.dumps(result),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['fit','score','replay','audit']);p.add_argument('--data',default='/tmp/aloha-original.parquet');args=p.parse_args();torch.set_num_threads(1)
    if args.mode=='fit':fit(args.data)
    elif args.mode=='audit':budget_audit()
    else:evaluate(args.data,replay=args.mode=='replay')
