"""One registered five-seed shared-core real-data capability study; no TEST.

Download immutable numeric data; form lagged-action inputs (no same-frame action),
fit only original TRAIN and preserve every original before exposed DEV scoring.
"""
from pathlib import Path
import argparse, hashlib, json, sys, time, urllib.request, random
import numpy as np
import pyarrow.parquet as pq
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT))
from tac_osm.variants.evidence_retrieval_world import EvidenceRetrievalWorld
from tac_osm.parameter_budget import budgeted_optimizer,checked_optimizer_step,save_training_checkpoint
REV='7a752b39f7e69de7e38aee485a6bab07528a061a'
SHA='a3409d3282b2cf9a94d32b301756fbd15dca5862cc1c32973c7316dd8853c57a'
URL=f'https://huggingface.co/datasets/lerobot/aloha_mobile_cabinet/resolve/{REV}/data/chunk-000/file-000.parquet'
REG=ROOT/'research/mathematical-engineering/me097/FAIR_COMPARISON.json'
OUT=ROOT/'research/replay-bundles/ME097'
SEEDS=(97001,97002,97003,97004,97005)

def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def data(path):
    path=Path(path)
    if not path.exists():path.write_bytes(urllib.request.urlopen(URL,timeout=90).read())
    if digest(path)!=SHA:raise RuntimeError('source bytes differ')
    d=pq.read_table(path).to_pydict();episodes={}
    for ep,frame,ts,x,a in zip(*(d[k] for k in ['episode_index','frame_index','timestamp','observation.state','action'])):
        if len(x)!=14 or len(a)!=14 or not np.isfinite(x+a+[ts]).all():raise RuntimeError('invalid observation')
        episodes.setdefault(ep,[]).append((frame,ts,x,a))
    ids=sorted(episodes); fit=ids[:int(.7*len(ids))];dev=ids[int(.7*len(ids)):int(.85*len(ids))]
    def pack(chosen):
        rows=[]; provenance=[]
        for ep in chosen:
            e=episodes[ep]
            if not all(b[0]==a[0]+1 and 0<b[1]-a[1]<.2 for a,b in zip(e,e[1:])):raise RuntimeError('nonconsecutive episode')
            starts=list(range(1,len(e)-64))
            for start in starts[::max(1,len(starts)//8)][:8]:
                rows.append((torch.tensor([r[2] for r in e[start:start+64]]),
                             torch.tensor([r[3] for r in e[start-1:start+63]]),
                             torch.tensor([r[2] for r in e[start+1:start+65]])))
                provenance.append([int(ep),int(start)])
        return tuple(torch.stack([r[i] for r in rows]) for i in range(3)),provenance
    tr,ti=pack(fit);dv,di=pack(dev)
    mean=tr[0].mean((0,1));scale=tr[0].std((0,1)).clamp_min(.0001)
    am=tr[1].mean((0,1));asc=tr[1].std((0,1)).clamp_min(.0001)
    normalize=lambda z:((z[0]-mean)/scale,(z[1]-am)/asc,(z[2]-mean)/scale)
    return normalize(tr),normalize(dv),dict(obs_mean=mean,obs_scale=scale,action_mean=am,action_scale=asc,fit_windows=ti,dev_windows=di)

def fit(path):
    from scripts.validate_research_parity import check_manifest
    check_manifest(json.loads(REG.read_text()),'preflight')
    tr,_,prep=data(path);OUT.mkdir(parents=True,exist_ok=True)
    if any((OUT/f'seed-{s}').exists() for s in SEEDS):raise RuntimeError('original fit directory exists; no retry')
    source_paths=['scripts/me097_evidence_replication.py','src/tac_osm/parameter_budget.py',
       'src/tac_osm/variants/evidence_retrieval_world.py','src/tac_osm/variants/parallel_continuous_world.py',
       'src/tac_osm/variants/retention_controlled_world.py','src/tac_osm/variants/parallel_affine_memory.py']
    source={p:digest(ROOT/p) for p in source_paths}
    rows=[]
    for seed in SEEDS:
        torch.manual_seed(seed);np.random.seed(seed);random.seed(seed)
        m=EvidenceRetrievalWorld();m.set_train_delta_scale((tr[2]-tr[0]).square().mean((0,1)).sqrt())
        folder=OUT/f'seed-{seed}';folder.mkdir()
        opt=budgeted_optimizer(m,torch.optim.AdamW,lr=.0006,weight_decay=.001,
          budget_manifest=REG,budget_receipt=folder/'live-budget.json',budget_seed=seed,budget_architecture='shared_evidence')
        schedule=torch.randint(0,len(tr[0]),(256,8),generator=torch.Generator().manual_seed(seed))
        start=time.process_time();losses=[]
        for step,idx in enumerate(schedule):
            opt.zero_grad(set_to_none=True);pred=m(tr[0][idx],tr[1][idx]);loss=(pred-tr[2][idx]).square().mean()
            if not torch.isfinite(loss):raise RuntimeError('nonfinite training')
            loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),1);checked_optimizer_step(opt)
            losses.append(float(loss.detach()))
        checkpoint=folder/'final.pt'
        save_training_checkpoint(dict(model=m.state_dict(),optimizer=opt.state_dict(),torch_rng=torch.get_rng_state(),numpy_rng=np.random.get_state(),python_rng=random.getstate(),schedule=schedule,preprocessing=prep,source_sha256=SHA,source_files=source,seed=seed,steps=256,losses=losses),checkpoint)
        row=dict(seed=seed,trainable_parameters=sum(p.numel() for p in m.parameters()),deployed_parameters=sum(p.numel() for p in m.parameters()),checkpoint_sha256=digest(checkpoint),optimizer_cpu_seconds=time.process_time()-start,steps=256,examples=2048,supervised_transitions=131072)
        (folder/'FIT_RECEIPT.json').write_text(json.dumps(row,indent=2)+'\n');rows.append(row)
        print(json.dumps(row),flush=True)
    (OUT/'FIT_INDEX.json').write_text(json.dumps(dict(models=rows,source_files=source,source_sha256=SHA),indent=2)+'\n')

def assert_git_frozen():
    import subprocess
    index=OUT/'FIT_INDEX.json'
    if not index.exists():raise RuntimeError('all original fits required before scoring')
    files=[index]
    for seed in SEEDS:
        files.extend((OUT/f'seed-{seed}').glob('*'))
    for p in files:
        rel=str(p.relative_to(ROOT))
        try: committed=subprocess.check_output(['git','show','HEAD:'+rel],cwd=ROOT,stderr=subprocess.DEVNULL)
        except subprocess.CalledProcessError:raise RuntimeError('original not committed before scoring: '+rel)
        if committed!=p.read_bytes():raise RuntimeError('working original differs from committed freeze: '+rel)


def score(path,replay=False):
    assert_git_frozen()
    if not (OUT/'FIT_INDEX.json').exists():raise RuntimeError('all original fits required before scoring')
    tr,dv,prep=data(path);idx=json.loads((OUT/'FIT_INDEX.json').read_text())
    for rel,sha in idx['source_files'].items():
        if digest(ROOT/rel)!=sha:raise RuntimeError('frozen source changed: '+rel)
    delta=(tr[2]-tr[0]).square().mean(-1).sqrt();threshold=torch.quantile(delta,.9)
    moving=(dv[2]-dv[0]).square().mean(-1).sqrt()>=threshold
    def metrics(pred):
        err=(pred-dv[2]).square().mean(-1)
        return dict(mse=float(err.mean()),late_mse=float(err[:,32:].mean()),moving_mse=float(err[moving].mean()))
    persistence=metrics(dv[0]);rows=[];arrays=[]
    with torch.no_grad():
        for seed,row in zip(SEEDS,idx['models']):
            p=OUT/f'seed-{seed}'/'final.pt'
            if digest(p)!=row['checkpoint_sha256']:raise RuntimeError('original checkpoint altered')
            c=torch.load(p,map_location='cpu',weights_only=False)
            m=EvidenceRetrievalWorld();m.load_state_dict(c['model']);m.eval()
            # Loader reconstructs preprocessing from FIT and checks every original statistic.
            for k in ['obs_mean','obs_scale','action_mean','action_scale']:
                torch.testing.assert_close(prep[k],c['preprocessing'][k],rtol=0,atol=0)
            pred,state=m(dv[0],dv[1],return_state=True);arrays.append(pred.numpy())
            scores=metrics(pred)
            passed=all(scores[k]<=.95*persistence[k] for k in ['mse','late_mse']) and scores['moving_mse']<=persistence['moving_mse']
            rows.append(dict(seed=seed,**scores,capability_gate=passed))
    result=dict(status='EXPOSED_DEV_ONLY',source_sha256=SHA,window_count=len(dv[0]),observed_transitions=int(dv[0].shape[0]*64),episode_count=len(set(x[0] for x in prep['dev_windows'])),persistence=persistence,models=rows,registered_gate_passed=all(r['capability_gate'] for r in rows),independent_qualification=False,architecture_superiority=False,causal_control=False)
    predictions=np.stack(arrays)
    if replay:
        original=np.load(OUT/'FORECASTS.npz')['predictions']
        np.testing.assert_allclose(predictions,original,rtol=1e-5,atol=1e-6)
        if result!=json.loads((OUT/'RESULTS.json').read_text()):raise RuntimeError('score replay mismatch')
        print('ORIGINAL_CHECKPOINT_SCORE_REPLAY=PASS')
    else:
        if (OUT/'RESULTS.json').exists():raise RuntimeError('original scoring already exists')
        np.savez_compressed(OUT/'FORECASTS.npz',predictions=predictions)
        (OUT/'RESULTS.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['fit','score','replay']);p.add_argument('--data',default='/tmp/aloha-original.parquet');a=p.parse_args()
    torch.set_num_threads(1)
    if a.mode=='fit':fit(a.data)
    else:score(a.data,replay=a.mode=='replay')
