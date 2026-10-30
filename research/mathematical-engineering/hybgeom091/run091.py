#!/usr/bin/env python3
"""HYBGEOM091 registered original real-source 5-seed × 6-arm 100K CPU FIT/DEV.

Honest exploratory **single_model_training**: same token exposure but extra
geometry/hybrid operations have unmatched FLOPs; NOT architecture-superiority.
Use only authentic repo-backed FIT / exposed DEV; never open protected TEST.
"""
from __future__ import annotations
import argparse,hashlib,io,json,random,sys,time,resource
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
ROOT=Path(__file__).resolve().parents[3]
sys.path[:0]=[str(ROOT),str(ROOT/'src'),str(ROOT/'research/mathematical-engineering/langmem086')]
from scripts.replay_langmem086087 import fit_sources
from scripts.validate_research_parity import check_manifest
from model091 import ARMS,EXPECTED,HybridGeometryTAC091

REG=ROOT/'research/mathematical-engineering/hybgeom091'
MANIFEST=REG/'FAIR_COMPARISON.json'
SEEDS=(1701,1702,1703,1704,1705)
STEPS=96
BATCH=4
LENGTH=64
EVAL_START=32
EVAL_STOP=256
DEV_OWNERS=('Doc/tutorial/controlflow.rst','Doc/tutorial/venv.rst')

def digest(raw): return hashlib.sha256(raw).hexdigest()

def source_docs():
    receipt=json.loads((ROOT/'research/mathematical-engineering/language_source023/SOURCE_RECEIPT.json').read_text())
    if receipt['commit']!='60403a5409ff2c3f3b07dd2ca91a7a3e096839c7': raise ValueError('publisher git commit')
    by={x['path']:x for x in receipt['files']}
    if set(DEV_OWNERS)-set(by): raise ValueError('missing registered DEV owner')
    out={}
    for name in DEV_OWNERS:
        rec=by[name]
        if rec['role']!='DEV': raise ValueError('FIT/DEV crossover')
        raw=(ROOT/'research/source-bundles/LANGUAGE-SOURCE023'/name).read_bytes()
        if len(raw)!=rec['bytes'] or digest(raw)!=rec['sha256'] or hashlib.sha1(f'blob {len(raw)}\0'.encode()+raw).hexdigest()!=rec['git_blob']:
            raise ValueError('invalid original DEV publisher bytes')
        if len(raw)<EVAL_STOP+2: raise ValueError('short DEV document')
        out[name]=raw
    return out

def train_batches(docs,seed):
    rng=np.random.default_rng(seed)
    names=sorted(docs)
    for step in range(STEPS):
        row=[]
        for i in range(BATCH):
            owner=names[int(rng.integers(len(names)))]
            raw=docs[owner]
            pos=int(rng.integers(len(raw)-LENGTH))
            row.append((owner,pos,raw[pos:pos+LENGTH+1]))
        yield step,row

def schedule(docs,seed):
    order=hashlib.sha256();content=hashlib.sha256()
    for _,rows in train_batches(docs,seed):
        for owner,pos,raw in rows:
            order.update(f'{owner}:{pos};'.encode());content.update(raw)
    return {'seed':seed,'steps':STEPS,'batch':BATCH,'context':LENGTH,'targets':STEPS*BATCH*LENGTH,
            'sample_order_sha256':order.hexdigest(),'sample_bytes_sha256':content.hexdigest()}

def state_dict_hash(model):
    h=hashlib.sha256()
    for key,value in sorted(model.state_dict().items()):
        h.update(key.encode()+b'\0'+value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()

def source_hash(): return digest((REG/'model091.py').read_bytes())

def ensure_preflight(arm,seed,model):
    manifest=json.loads(MANIFEST.read_text())
    if check_manifest(manifest,'preflight')['status']!='PASS_PREFIT_SINGLE_MODEL':raise ValueError('prospective gate')
    if manifest.get('implementation_sha256')!=source_hash(): raise ValueError('model source differs from frozen registered graph')
    if arm not in ARMS or seed not in SEEDS: raise ValueError('unregistered arm or seed')
    expected=next(x for x in manifest['models'] if x['name']==arm and x['seed']==seed)
    if model.counts()!={'trainable':expected['trainable_parameters'],'deployed':expected['deployed_parameters']}:
        raise ValueError('actual live parameters do not match registered size')
    if not 95000<=model.counts()['trainable']<=105000:raise ValueError('100k mandate')

@torch.no_grad()
def eval_model(model,docs):
    model.eval();tot=[];times=[]
    for owner,raw in sorted(docs.items()):
        state=model.start((owner,));records=[]
        for t in range(EVAL_STOP):
            token=torch.tensor([[int(raw[t])]],dtype=torch.long)
            before=time.process_time()
            logits,state=model(token,state=state,owners=(owner,))
            times.append(time.process_time()-before)
            if t<EVAL_START:continue
            # Future only becomes a target after the prediction was generated.
            actual=int(raw[t+1]);lp=logits[0,0].log_softmax(-1)
            records.append({'ordinal':t,'nll':float(-lp[actual]),'correct':int(int(lp.argmax())==actual)})
        tot.append({'owner':owner,'source_sha256':digest(raw),'n':len(records),
            'nll':sum(r['nll'] for r in records)/len(records),
            'accuracy':sum(r['correct'] for r in records)/len(records),
            'tokens':records})
    return tot,{'process_seconds':sum(times),'cpu_mean_ms_per_step':1000*sum(times)/len(times)}

def validate_paired_local(model,doc):
    # Future suffix poisoning must NOT influence any earlier token, states, or logits.
    observed=torch.tensor([list(doc[:12])],dtype=torch.long)
    changed=observed.clone();changed[:,8:]=(changed[:,8:]+113)%256
    with torch.no_grad():
        p,_=model(observed,owners=('causal',));q,_=model(changed,owners=('causal',))
        if not torch.equal(p[:,:8],q[:,:8]):raise ValueError('future-target leakage')
        one=model.start(('causal',));chunks=[]
        for t in range(observed.shape[1]):
            logits,one=model(observed[:,t:t+1],state=one,owners=('causal',))
            chunks.append(logits)
        err=(p-torch.cat(chunks,1)).abs().max().item()
        if err>3e-5:raise ValueError('full/chunk logits mismatch')
    return {'future_poison':'PASS', 'incremental_max_abs_error':err}

def run(arm,seed,out):
    if out.exists():raise FileExistsError('output exists: immutable study')
    torch.set_num_threads(2);torch.manual_seed(seed);random.seed(seed);np.random.seed(seed)
    model=HybridGeometryTAC091(arm)
    ensure_preflight(arm,seed,model)
    docs,sha=fit_sources(ROOT)
    manifest=json.loads(MANIFEST.read_text())
    if sha!=manifest['real_world_source']['sha256'] or len(docs)!=14:raise ValueError('original FIT dataset identity')
    sched=schedule(docs,seed)
    # The schedule is frozen by pinned source hash, this versioned generator,
    # and the prespecified seeds; compare independent arm digests after fit.
    if sched['targets']!=STEPS*BATCH*LENGTH: raise ValueError('schedule target count')
    before=state_dict_hash(model);initial=validate_paired_local(model,next(iter(docs.values())))
    optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01)
    losses=[];seen=0;order=hashlib.sha256();content=hashlib.sha256();start=time.process_time()
    all_extra_grads=set()
    for step,rows in train_batches(docs,seed):
        for owner,pos,raw in rows:
            order.update(f'{owner}:{pos};'.encode());content.update(raw)
        x=torch.tensor([list(raw[:-1]) for _,_,raw in rows],dtype=torch.long)
        y=torch.tensor([list(raw[1:]) for _,_,raw in rows],dtype=torch.long)
        optimizer.zero_grad(set_to_none=True)
        logits,_=model(x,owners=tuple(f'fit-{i}' for i in range(BATCH)))
        loss=F.cross_entropy(logits.reshape(-1,256),y.reshape(-1))
        if not torch.isfinite(loss):raise ValueError('nonfinite loss')
        loss.backward()
        for name,p in model.named_parameters():
            if p.grad is None or not torch.isfinite(p.grad).all():raise ValueError('missing or nonfinite gradient '+name)
            if not name.startswith('core.') and float(p.grad.abs().max())>0:all_extra_grads.add(name)
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        optimizer.step()
        losses.append(float(loss.detach()));seen+=y.numel()
        if step in (0,STEPS-1):print(json.dumps({'arm':arm,'seed':seed,'update':step+1,'loss':losses[-1]}),flush=True)
    elapsed=time.process_time()-start
    if order.hexdigest()!=sched['sample_order_sha256'] or content.hexdigest()!=sched['sample_bytes_sha256'] or seen!=sched['targets']:
        raise ValueError('actual sampler/byte exposure mismatch')
    if state_dict_hash(model)==before:raise ValueError('no update')
    # Seal checkpoint bytes before any DEV targets are loaded.
    out.mkdir(parents=True)
    cp={'study':'HYBGEOM091','arm':arm,'seed':seed,'steps':STEPS,
        'source_sha256':sha,'model_source_sha256':source_hash(),
        'state':model.state_dict(),'optimizer':optimizer.state_dict(),
        'torch_rng':torch.get_rng_state(),'numpy_rng':np.random.get_state()[1].tolist(),
        'python_rng':random.getstate(),'schedule':sched,'losses':losses,'counts':model.counts()}
    cp_path=out/'MODEL.pt';torch.save(cp,cp_path)
    checkpoint_sha=digest(cp_path.read_bytes())
    frozen=torch.load(cp_path,map_location='cpu',weights_only=True)
    cold=HybridGeometryTAC091(arm).eval();cold.load_state_dict(frozen['state'],strict=True)
    check=validate_paired_local(cold,next(iter(docs.values())))
    if check['future_poison']!='PASS' or state_dict_hash(cold)!=state_dict_hash(model):raise ValueError('cold checkpoint replay')
    # The score cohort is deliberately *exposed DEV*; never consider untouched TEST.
    dev=source_docs()
    scores,latency=eval_model(cold,dev)
    result={'study_id':'HYBGEOM091','arm':arm,'seed':seed,'checkpoint_sha256':checkpoint_sha,
       'weights_hash':state_dict_hash(cold),'source_sha256':sha,'model_source_sha256':source_hash(),
       'counts':model.counts(),'trained_steps':STEPS,'training_targets':seen,
       'actual_sample_order_sha256':order.hexdigest(),'actual_sample_bytes_sha256':content.hexdigest(),
       'initial_weights_hash':before,'first_fit_nll':losses[0],'last_fit_nll':losses[-1],
       'fit_process_seconds':elapsed,'exposed_DEV':scores,'dev_latency':latency,
       'causal_pretrain':initial,'causal_replay':check,'learned_extra_gradients':sorted(all_extra_grads),
       'independent_TEST_opened':False,'gpu_minutes':0,
       'status':'internal_exploratory_equal_exposure_NOT_training_FLOPs_matched',
       'time_axis':'byte index; NOT real elapsed time','semantic_human_judgments':False,
       'checkpoint_repo_replay':'INCONCLUSIVE_UNTIL_ARCHIVE_COMMITTED_AND_REPLAYED'}
    (out/'RESULTS.json').write_text(json.dumps(result,indent=2,sort_keys=True,allow_nan=False)+'\n')
    print(json.dumps({'arm':arm,'seed':seed,'checkpoint_sha':checkpoint_sha,'dev_nll':[z['nll'] for z in scores]}),flush=True)
    return result

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--arm',choices=ARMS,required=True)
    ap.add_argument('--seed',type=int,choices=SEEDS,required=True)
    ap.add_argument('--out',type=Path,required=True)
    opt=ap.parse_args();run(opt.arm,opt.seed,opt.out)
if __name__=='__main__':main()
