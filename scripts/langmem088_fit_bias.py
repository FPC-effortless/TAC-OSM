#!/usr/bin/env python3
"""LANGMEM088: ONE pretrained slow-read-bias +1 controlled 640-step FIT.

Original compact086 source graph,100521 params,140 original FIT corpus labels.
No new DEV documents can be named/passed into this training runner. The 087
640 original baseline is an immutable separate original Actions artifact.
"""
from __future__ import annotations
import argparse,hashlib,json,random,resource,sys,time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'src'),str(ROOT/'research/mathematical-engineering/langmem086'),str(ROOT/'research/mathematical-engineering/langmem087')]
from scripts import langmem086_run as original
from scripts import langmem087_run as stage_a
from scripts.validate_research_parity import check_manifest
REG=ROOT/'research/mathematical-engineering/langmem088'
SEEDS=(1701,1702,1703)
STEPS=640
BATCH=4
CONTEXT=256
SOURCE_SHA='911981cc63fd5436e0495cb067784e53475971131c811b7026e07d97a886d2d4'
sha=original.digest
weights=original.weight_hash

def fit(seed:int,out:Path):
    if seed not in SEEDS or out.exists():raise ValueError('registered seed or existing artifact')
    torch.set_num_threads(2)
    torch.manual_seed(seed);random.seed(seed);np.random.seed(seed)
    model=original.CompactDualClippedTAC086()
    if model.counts()['trainable']!=100521 or model.counts()['deployed']!=100521:
        raise ValueError('original fully learned param count changed')
    original_start=weights(model)
    original_initial_bias=model.read_mix.bias.detach().cpu().clone()
    with torch.no_grad():
        model.read_mix.bias[1].add_(1.0)
    if abs(float(model.read_mix.bias[1]-original_initial_bias[1])-1.)>1e-7 or not torch.equal(model.read_mix.bias[0],original_initial_bias[0]):
        raise ValueError('bias intervention mismatch')
    manifest=json.loads((REG/'FAIR_COMPARISON.json').read_text())
    if check_manifest(manifest,'preflight')['status']!='PASS_PREFIT_SINGLE_MODEL':
        raise ValueError('registered source/parameters preflight')
    docs,source=original.fit_sources086()
    if source!=SOURCE_SHA:raise ValueError('original unchanged 14 FIT owner SHA')
    reference_plan=json.loads((ROOT/'research/mathematical-engineering/langmem087/FIT_PLAN.json').read_text())
    expected=next(q for q in reference_plan['by_seed'] if q['seed']==seed)['checkpoints'][1]
    if expected['steps']!=STEPS or stage_a.schedule(docs,seed,STEPS)!=expected:
        raise ValueError('640 original FIT sample schedule mismatch')
    fixture=torch.tensor([list(next(iter(docs.values()))[:16])],dtype=torch.long)
    poisoned=fixture.clone();poisoned[:,8:]=(poisoned[:,8:]+113)%256
    with torch.no_grad():
        first,_=model(fixture,owners=('source-FIT-poison',))
        future,_=model(poisoned,owners=('source-FIT-poison',))
    if not torch.equal(first[:,:8],future[:,:8]):
        raise ValueError('future byte poison before fitting')
    optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01)
    initial=weights(model)
    sample_hash=hashlib.sha256()
    actual_byte_hash=hashlib.sha256()
    counts=torch.zeros(256,dtype=torch.int64)
    out.mkdir(parents=True)
    begin=time.process_time()
    losses=[]
    gradient_receipts=[]
    for step,row in stage_a.batches(docs,seed,STEPS):
        x=torch.tensor([list(raw[:-1]) for _,_,raw in row],dtype=torch.long)
        targets=torch.tensor([list(raw[1:]) for _,_,raw in row],dtype=torch.long)
        for owner,start,raw in row:
            sample_hash.update(f'{owner}:{start};'.encode())
            actual_byte_hash.update(raw)
        counts+=torch.bincount(targets.flatten(),minlength=256)
        optimizer.zero_grad(set_to_none=True)
        pred,_=model(x,owners=tuple(f'FIT-row-{i}' for i in range(BATCH)))
        loss=F.cross_entropy(pred.flatten(0,1),targets.flatten())
        loss.backward()
        if not torch.isfinite(loss) or any(p.grad is None or not torch.isfinite(p.grad).all() for p in model.parameters()):
            raise ValueError('actual FIT loss/gradient')
        if step in (0,STEPS-1):
            gradient_receipts.append({'step':step,
                'slow_read_mix_bias_gradient':model.read_mix.bias.grad.detach().cpu().tolist(),
                'slow_read_mix_weight_grad_l2':float(model.read_mix.weight.grad.norm())})
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        optimizer.step()
        losses.append(float(loss.detach()))
        if step%80==0:
            print(json.dumps({'study':'LANGMEM088','seed':seed,'step':step,'FIT_nll':losses[-1]}),flush=True)
    elapsed=time.process_time()-begin
    if (sample_hash.hexdigest()!=expected['sample_order_sha256']
        or actual_byte_hash.hexdigest()!=expected['sample_bytes_sha256']
        or int(counts.sum())!=STEPS*BATCH*CONTEXT
        or len(losses)!=STEPS
        or any(int(s['step'])!=STEPS for s in optimizer.state.values())):
        raise ValueError('FIT sample/step/source mismatch')
    model.eval()
    with torch.no_grad():
        expected_logits,expected_state=model(fixture,owners=('replay',))
    final_hash=weights(model)
    if final_hash==initial:raise ValueError('original model failed to learn')
    nstate=np.random.get_state()
    checkpoint={'study':'LANGMEM088','seed':seed,'steps':STEPS,'intervention':'initial_slow_read_bias_plus_1',
        'model':model.state_dict(),'optimizer':optimizer.state_dict(),'torch_rng':torch.get_rng_state(),
        'python_rng':random.getstate(),'numpy_rng_state':[nstate[0],nstate[1].tolist(),int(nstate[2]),int(nstate[3]),float(nstate[4])],
        'model_counts':model.counts(),'model_source_sha256':original.model_source_hash(),
        'fit_source_sha256':source,'sample_receipt':expected,'FIT_byte_counts':counts.tolist()}
    path=out/'MODEL.pt';torch.save(checkpoint,path)
    checkpoint_hash=sha(path.read_bytes())
    loaded=torch.load(path,map_location='cpu',weights_only=True)
    cold=original.CompactDualClippedTAC086().eval()
    cold.load_state_dict(loaded['model'],strict=True)
    with torch.no_grad():
        got,state=cold(fixture,owners=('replay',))
    if not torch.equal(got,expected_logits) or any(not torch.equal(state.tensors[k],expected_state.tensors[k]) for k in state.tensors):
        raise ValueError('cold logits/state replay failed')
    optim2=torch.optim.AdamW(cold.parameters(),lr=.001,weight_decay=.01)
    optim2.load_state_dict(loaded['optimizer'])
    if any(int(s['step'])!=STEPS for s in optim2.state.values()):raise ValueError('optimizer-step cold replay')
    report={'study':'LANGMEM088','seed':seed,'trained_steps':STEPS,
       'true_FIT_targets':STEPS*BATCH*CONTEXT,
       'fit_source_sha256':source,'source_receipt':expected,
       'original_baseline_initial_weight_sha256':original_start,
       'initial_model_sha256_after_only_bias':initial,'final_trained_model_sha256':final_hash,
       'initial_original_read_mix_bias':original_initial_bias.tolist(),
       'initial_intervention_read_mix_bias':[float(original_initial_bias[0]),float(original_initial_bias[1]+1.)],
       'final_read_mix_bias':model.read_mix.bias.detach().cpu().tolist(),
       'bias_delta_initial':1.0,'no_new_trainable_parameters':True,
       'model_parameters_trainable':100521,'model_parameters_deployed':100521,
       'model_persistent_tensor_state_bytes':1920,'checkpoint_sha256':checkpoint_hash,
       'trained_model_source_sha256':original.model_source_hash(),
       'checkpoint_and_optimizer_cold_replay_bitwise':True,
       'unseen_DEV_accessed_before_fit_seal':False,'original_baseline_checkpoints_modified':False,
       'gradient_receipts':gradient_receipts,'FIT_loss_first':losses[0],'FIT_loss_last':losses[-1],
       'cpu_FIT_process_seconds':elapsed,'peak_RSS_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
       'gpu_minutes':0,'protected_TEST':'NOT_ACCESSED'}
    (out/'FIT_RECEIPT.json').write_text(json.dumps(report,indent=2,sort_keys=True,allow_nan=False)+'\n')
    print('LANGMEM088_BIAS_FIT '+json.dumps({'seed':seed,'checkpoint_sha256':checkpoint_hash,
         'initial_slow_bias':report['initial_intervention_read_mix_bias'][1],
         'final_slow_bias':report['final_read_mix_bias'][1],'FIT_targets':report['true_FIT_targets']}),flush=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int,choices=SEEDS,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    fit(a.seed,a.output)
if __name__=='__main__':main()
