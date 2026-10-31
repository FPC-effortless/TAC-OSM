"""Registered FP32 exact10M pilot, real FIT only; all failures retained."""
import argparse
import math
from pathlib import Path
import subprocess
import sys
import time
import torch
import importlib.util
_spec=importlib.util.spec_from_file_location('langtrain099_common',Path(__file__).with_name('common.py'))
C=importlib.util.module_from_spec(_spec);sys.modules[_spec.name]=C;_spec.loader.exec_module(C)

def replay(args):
    C.setup(args.seed);payload=torch.load(args.checkpoint,map_location='cpu',weights_only=True)
    model=C.make(args.seed).eval();model.load_state_dict(payload['model'],strict=True)
    tokens=torch.tensor(C.read(args.inputs),dtype=torch.long)
    with torch.no_grad():logits,state=model(tokens,owners=tuple('cold-'+str(i) for i in range(len(tokens))))
    import numpy as np
    with args.output.open('xb') as f:np.savez(f,logits=logits.numpy(),**{n:v.numpy() for n,v in state.tensors.items()})

def cold_replay(model,checkpoint,seed,output,tokens):
    import numpy as np
    inputs=output/'COLD_FIT_INPUTS.json'
    C.save(inputs,tokens.tolist());original=output/'COLD_FIT_ORIGINAL.npz'
    with torch.no_grad():y,s=model(tokens,owners=tuple('cold-'+str(i) for i in range(len(tokens))))
    with original.open('xb') as f:np.savez(f,logits=y.numpy(),**{n:v.numpy() for n,v in s.tensors.items()})
    fresh=output/'COLD_FIT_REPLAY.npz'
    subprocess.run([sys.executable,str(Path(__file__)),'--mode','replay','--seed',str(seed),'--checkpoint',str(checkpoint),'--inputs',str(inputs),'--output',str(fresh)],check=True)
    with np.load(original,allow_pickle=False) as a,np.load(fresh,allow_pickle=False) as b:
        if set(a.files)!=set(b.files):raise ValueError('cold original/replay array schema differs')
        if any(not np.isfinite(a[n]).all() or not np.isfinite(b[n]).all() for n in a.files):raise ValueError("nonfinite actual cold replay")
        delta=max(float(np.max(np.abs(a[n]-b[n]))) for n in a.files)
    if delta>1e-6:raise ValueError('actual full model/state cold replay differs')
    return delta

def run(args):
    reg=C.read(C.HERE/'REGISTER.json')
    if reg.get('fit_enabled') is not True:raise ValueError('training is disabled until original readiness evidence is pinned')
    if args.seed not in C.SEEDS:raise ValueError('fixed registered seed required')
    proof=reg['math']['PNDS_scaled_state_proof']
    if not isinstance(proof,dict) or proof.get('status')!='PASS' or not isinstance(proof.get('run_id'),int):
        raise ValueError('original scaled formal proof evidence required before optimizer')
    C.setup(args.seed);source=C.source_hashes(args.source_commit)
    if args.output.exists():raise FileExistsError('retain all original runs')
    args.output.mkdir(parents=True);started=time.perf_counter()
    corpus,windows,episodes,manifest=C.load_fit(args.data,reg)
    index=C.read(args.profiles/'PROFILE_INDEX.json')
    if index.get('optimizer_steps')!=0 or not index.get('fits_budget'):
        raise ValueError('actual zero-update profiles did not fit registered compute budget')
    profile=C.read(args.profiles/f'PROFILE_{args.seed}.json')
    if profile['estimated_training_seconds_including75pct_margin_and120s']>reg['training']['max_seconds']:
        raise ValueError('actual registered seed runtime estimate exceeds budget')
    for p,d in profile['source_hashes'].items():
        if p.endswith('.py') and C.sha(C.ROOT/p)!=d:raise ValueError('code differs from actual no-update FIT profile: '+p)
    initial=args.profiles/profile['initial_archive_file']
    if C.sha(initial)!=profile['initial_archive_sha256']:raise ValueError('original zero-step archive changed')
    payload=torch.load(initial,map_location='cpu',weights_only=True)
    if payload['metadata']['optimizer_steps']!=0:raise ValueError('profile archive is not zero-update')
    model=C.make(args.seed);model.load_state_dict(payload['model'],strict=True)
    if C.state_digest(model)!=profile['model_sha256']:raise ValueError('exact original initialization mismatch')
    steps=reg['training']['steps'];plan=C.schedule(args.seed,steps,windows,episodes);sample_sha=C.digest(plan)
    C.save(args.output/'SAMPLING_JOURNAL.json',{'steps':plan,'sha256':sample_sha})
    optimizer=torch.optim.AdamW(model.parameters(),lr=.0003,weight_decay=.01,betas=(.9,.999),eps=1e-8,foreach=False,fused=False)
    initial_meta={'family':'tac099','seed':args.seed,'optimizer_steps':0,'status':'INITIAL_BEFORE_DEV','source_commit':args.source_commit,'source_hashes':source,'counts':model.counts(),'model_sha256':C.state_digest(model)}
    C.checkpoint(args.output/'INITIAL_MODEL.pt',model,optimizer,initial_meta)
    C.save(args.output/'PREFIT.json',{'counts':model.counts(),'initial_profile_archive_sha256':C.sha(initial),'sample_sha256':sample_sha,'source_hashes':source,'hardware':C.hardware(),'before_first_optimizer_step':True})
    history=[];seen={'observed':0,'supervised':0,'padding':0,'future_targets':0};completed=0;first_gradients={}
    try:
        model.train()
        for step,item in enumerate(plan):
            if time.perf_counter()-started>=reg['training']['max_seconds']:raise TimeoutError('registered training deadline reached')
            warmup=64
            rate=.0003*(step+1)/warmup if step<warmup else .0003*(.1+.9*.5*(1+math.cos(math.pi*(step-warmup)/max(1,steps-warmup-1))))
            for group in optimizer.param_groups:group['lr']=rate
            optimizer.zero_grad(set_to_none=True);values=C.batch(item,corpus,windows,episodes)
            tick=time.perf_counter();loss=C.loss(model,values)
            if not bool(torch.isfinite(loss)):raise ValueError('nonfinite actual FIT loss')
            loss.backward();norms=C.gradients(model)
            if item['mode'] not in first_gradients:first_gradients[item['mode']]=norms
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,foreach=False)
            if not bool(torch.isfinite(norm)):raise ValueError('nonfinite actual gradient norm')
            optimizer.step();completed+=1
            for name,value in values[-1].items():seen[name]+=value
            row={'step':completed,'mode':item['mode'],'loss':float(loss.detach()),'learning_rate':rate,
                 'gradient_norm_before_clip':float(norm),'update_seconds':time.perf_counter()-tick,'exposure':values[-1]}
            history.append(row)
            if completed%32==0:print(C.digest(history[-32:]),completed,row['mode'],round(row['loss'],5),flush=True)
        model.eval();counts=model.counts()
        if counts['trainable']!=10000000 or counts['deployed']!=10000000:raise ValueError('actual postfit exact10M gate failed')
        if any(not bool(torch.isfinite(p).all()) for p in model.parameters()):raise ValueError('nonfinite trained parameter')
        metadata={'family':'tac099','seed':args.seed,'optimizer_steps':completed,'status':'SEALED_BEFORE_DEV','source_commit':args.source_commit,'source_hashes':source,
                  'counts':counts,'model_sha256':C.state_digest(model),'initial_model_sha256':profile['model_sha256'],
                  'sample_sha256':sample_sha,'data_manifest_sha256':manifest['source_manifest_sha256'],'register_sha256':C.sha(C.HERE/'REGISTER.json')}
        weights=args.output/'MODEL.pt';checkpoint_sha=C.checkpoint(weights,model,optimizer,metadata)
        delta=cold_replay(model,weights,args.seed,args.output,C.batch(plan[0],corpus,windows,episodes)[0])
        receipt={'status':'SEALED_BEFORE_DEV','seed':args.seed,'steps':completed,'source_commit':args.source_commit,'source_hashes':source,
                 'trainable_parameters':counts['trainable'],'deployed_parameters':counts['deployed'],'checkpoint_sha256':checkpoint_sha,
                 'initial_checkpoint_sha256':C.sha(args.output/'INITIAL_MODEL.pt'),'initial_profile_archive_sha256':C.sha(initial),
                 'model_sha256':metadata['model_sha256'],'initial_model_sha256':metadata['initial_model_sha256'],
                 'sample_sha256':sample_sha,'source_manifest_sha256':manifest['source_manifest_sha256'],
                 'cold_replay_max_abs_difference':delta,'exposure':seen,'first_actual_gradient_l2':first_gradients,
                 'elapsed_seconds':time.perf_counter()-started,'history':history,'hardware':C.hardware(),'DEV_opened':False,'GPU_minutes':0}
        C.save(args.output/'FIT_RECEIPT.json',receipt)
        print({'status':receipt['status'],'seed':args.seed,'steps':completed,'elapsed_seconds':receipt['elapsed_seconds']},flush=True)
    except BaseException as error:
        C.save(args.output/'FAILED_FIT.json',{'seed':args.seed,'completed_optimizer_steps':completed,'reason':type(error).__name__+': '+str(error),'history':history,'source_commit':args.source_commit,'source_hashes':source})
        C.checkpoint(args.output/'PARTIAL_MODEL.pt',model,optimizer,{'seed':args.seed,'optimizer_steps':completed,'status':'FAILED_RETAINED'})
        raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=('fit','replay'),default='fit');p.add_argument('--seed',type=int,required=True)
    p.add_argument('--data',type=Path);p.add_argument('--profiles',type=Path);p.add_argument('--source-commit');p.add_argument('--output',type=Path,required=True)
    p.add_argument('--checkpoint',type=Path);p.add_argument('--inputs',type=Path);a=p.parse_args();replay(a) if a.mode=='replay' else run(a)
