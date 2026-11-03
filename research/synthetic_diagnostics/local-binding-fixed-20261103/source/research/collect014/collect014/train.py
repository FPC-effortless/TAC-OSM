"""Gated genuine next-byte optimization, immutable checkpoints and exact replay.

Internal learnability only. No parser, role labels, support filter, external
model or sealed TEST enters this runner. Partial budgets cannot claim success.
"""
import argparse,copy,hashlib,json,os,platform,shutil,time
from pathlib import Path
import torch
from torch.nn import functional as F
from .common import ROOT,SPECS,build,canonical,digest,verify_data,source_paths,internal_gate,require

def fingerprint(model):
    h=hashlib.sha256()
    for n,p in model.state_dict().items():h.update(n.encode());h.update(p.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()

def freeze_tree(value):
    if isinstance(value,torch.Tensor):return value.detach().cpu().clone()
    if isinstance(value,dict):return {k:freeze_tree(v) for k,v in value.items()}
    if isinstance(value,list):return [freeze_tree(v) for v in value]
    if isinstance(value,tuple):return tuple(freeze_tree(v) for v in value)
    return value

def batch(data,generator,batch_size,length,device):
    indices=torch.randint(0,len(data)-length,(batch_size,),generator=generator)
    slab=torch.tensor([list(data[i:i+length+1]) for i in indices.tolist()],dtype=torch.long,device=device)
    return slab[:,:-1],slab[:,1:],indices.tolist()

def evaluate(model,data,device,windows=64):
    prior=model.training;model.eval();total=0.;count=0
    g=torch.Generator().manual_seed(120099)
    with torch.no_grad():
        for _ in range(windows//4):
            x,y,_=batch(data,g,4,128,device);logits,_=model(x)
            total+=float(F.cross_entropy(logits.flatten(0,1),y.flatten(),reduction='sum'));count+=y.numel()
    model.train(prior)
    return dict(nll=total/count,targets=count)

def fit(args):
    torch.set_num_threads(1)
    path=ROOT/'SMOKE_REGISTRATION.json' if args.smoke else ROOT/'registrations'/args.arm/'FAIR_COMPARISON.json'
    raw=path.read_bytes();manifest=json.loads(raw);gate=internal_gate(manifest)
    require(args.arm in SPECS,'unknown graph');fit=manifest['fit'];steps=fit['steps']
    registered=[m for m in manifest['models'] if m['architecture']==args.arm and m['seed']==args.seed]
    require(len(registered)==1,'unregistered graph/seed')
    require(args.smoke==(steps==8),'short/full registration mismatch')
    source={p:digest(f) for p,f in source_paths().items()}
    if not args.smoke:require(source==manifest['source_hashes'],'runtime source differs from frozen registration')
    if not args.smoke:
        require(all(digest(ROOT/p)==h for p,h in manifest['training_source_hashes'].items()),'training source changed after registration')
    data,split=verify_data();device=torch.device(args.device)
    if device.type=='cuda':
        require(torch.cuda.device_count()==2,'exactly two observed GPUs required')
        require(all('T4' in torch.cuda.get_device_name(i) for i in range(2)),'exact T4 devices required')
    model=build(args.arm,args.seed).to(device);initial=fingerprint(model)
    require(sum(p.numel() for p in model.parameters())==registered[0]['deployed_parameters'],'live parameter count differs')
    optimizer=torch.optim.AdamW(model.parameters(),lr=fit['learning_rate'],betas=tuple(fit['betas']),
        weight_decay=fit['weight_decay'],foreach=False)
    generator=torch.Generator().manual_seed(args.seed+100000)
    dest=args.output;dest.mkdir(parents=True,exist_ok=True)
    require(not (dest/'journal.jsonl').exists(),'fresh run directory required; use a separate reviewed resume path')
    protocol_sha=hashlib.sha256(raw).hexdigest();head='0'*64;done=0;attempted=0;losses=[];times=[];last=None
    if args.resume is not None:
        saved=torch.load(args.resume,map_location='cpu',weights_only=True)
        require(saved['protocol_sha256']==protocol_sha and saved['source_hashes']==source,'resume source/protocol mismatch')
        require(saved['data_sha256']==split['parts']['train']['sha256'] and saved['arm']==args.arm and saved['seed']==args.seed,'resume data/graph/seed mismatch')
        done=int(saved['steps']);require(0<=done<steps,'resume must continue an incomplete fit or original initialization')
        prefix=[];previous='0'*64
        for line in (args.resume.parent/'journal.jsonl').read_text().splitlines()[:done]:
            row=json.loads(line);sha=row.pop('sha256')
            require(row['previous']==previous and hashlib.sha256(canonical(row)).hexdigest()==sha,'resume journal changed')
            row['sha256']=sha;previous=sha;prefix.append(row)
        require(len(prefix)==done and previous==saved['journal_head'],'resume journal/checkpoint progress mismatch')
        model.load_state_dict(saved['model'],strict=True);optimizer.load_state_dict(saved['optimizer'])
        torch.set_rng_state(saved['torch_rng']);generator.set_state(saved['batch_rng'])
        if device.type=='cuda' and saved['cuda_rng'] is not None:torch.cuda.set_rng_state(saved['cuda_rng'],device)
        head=previous;attempted=done;losses=[r['nll'] for r in prefix];times=[r['seconds'] for r in prefix]
        (dest/'journal.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in prefix))
        shutil.copy2(args.resume,dest/args.resume.name)
        (dest/'RESUME_PROVENANCE.json').write_text(json.dumps(dict(checkpoint_sha256=digest(args.resume),start_step=done),indent=2))
    started=time.perf_counter();deadline=started+fit['worker_seconds_cap']
    receipt=dict(status='STARTING_INTERNAL_FIT',arm=args.arm,seed=args.seed,smoke=args.smoke,
        planned_steps=steps,optimizer_updates=0,attempted_updates=0,qualified=False,protected_test_read=False,
        data_sha256=split['parts']['train']['sha256'],protocol_sha256=protocol_sha,source_hashes=source,
        torch=torch.__version__,python=platform.python_version(),device=str(device),initial_parameter_sha256=initial,
        trainable_parameters=sum(p.numel() for p in model.parameters()),independent_attestation=False,
        full_training_flops='NOT_MEASURED_NOT_A_COMPUTE_MATCHED_COMPARISON')
    def report(status,error=None):
        receipt.update(status=status,optimizer_updates=done,attempted_updates=attempted,
            elapsed_seconds=time.perf_counter()-started,journal_head=head,loss_first=losses[0] if losses else None,
            loss_last=losses[-1] if losses else None,error=error,total_recorded_optimizer_seconds=sum(times))
        (dest/'FIT_RECEIPT.json').write_text(json.dumps(receipt,indent=2))
    def save(step):
        checkpoint=dest/f'checkpoint_{step:04}.pt';require(not checkpoint.exists(),'immutable checkpoint exists')
        payload=dict(schema='collect014.training.v1',arm=args.arm,seed=args.seed,steps=step,
            model=freeze_tree(model.state_dict()),optimizer=freeze_tree(optimizer.state_dict()),
            torch_rng=torch.get_rng_state(),batch_rng=generator.get_state(),
            cuda_rng=torch.cuda.get_rng_state(device) if device.type=='cuda' else None,
            protocol_sha256=protocol_sha,data_sha256=receipt['data_sha256'],source_hashes=source,journal_head=head)
        torch.save(payload,checkpoint)
        (dest/f'checkpoint_{step:04}.sha256').write_text(digest(checkpoint)+'\n')
        return checkpoint
    try:
        if args.resume is None:save(0)
        report('RUNNING_INTERNAL_FIT')
        for step in range(done+1,steps+1):
            require(time.perf_counter()<deadline,'allocation ended before complete fit')
            require(path.read_bytes()==raw,'registration changed during execution')
            x,y,indices=batch(data['train'],generator,fit['batch_size'],fit['sequence_length'],device)
            tic=time.perf_counter();model.train();optimizer.zero_grad(set_to_none=True)
            logits,_=model(x);lm_loss=F.cross_entropy(logits.flatten(0,1),y.flatten())
            balance_loss,bank_counts=model.balancing()
            loss=lm_loss+model.balance_coefficient*balance_loss if model.balance_coefficient else lm_loss
            require(bool(torch.isfinite(loss)),'nonfinite language loss');loss.backward()
            require(all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in model.parameters()),'missing/nonfinite gradients')
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),fit['clip_norm'])
            attempted=step;optimizer.step();done=step
            require(all(bool(torch.isfinite(p).all()) for p in model.parameters()),'nonfinite updated weights')
            seconds=time.perf_counter()-tic;value=float(lm_loss.detach());losses.append(value);times.append(seconds)
            row=dict(step=step,nll=value,objective=float(loss.detach()),balance_loss=float(balance_loss.detach()),bank_counts=bank_counts.tolist(),gradient_norm=float(norm),indices=indices,seconds=seconds,previous=head)
            row['sha256']=hashlib.sha256(canonical(row)).hexdigest();head=row['sha256']
            with (dest/'journal.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
            if step in fit['checkpoint_steps']:save(step)
            last=(x.detach(),y.detach())
            if step%16==0 or step==steps:
                report('RUNNING_INTERNAL_FIT');print(json.dumps(dict(arm=args.arm,seed=args.seed,step=step,nll=value)),flush=True)
        require(done==steps,'incomplete fit')
        require({p:digest(f) for p,f in source_paths().items()}==source,'runtime source changed during fit')
        if not args.smoke:
            require(all(digest(ROOT/p)==h for p,h in manifest['training_source_hashes'].items()),'training source changed during fit')
        require(fingerprint(model)!=initial,'parameters did not change')
        checkpoint=dest/f'checkpoint_{done:04}.pt';saved=torch.load(checkpoint,map_location='cpu',weights_only=True)
        clone=build(args.arm,args.seed).to(device);clone.load_state_dict(saved['model'],strict=True)
        model.eval();clone.eval()
        with torch.no_grad():
            a,_=model(last[0]);b,_=clone(last[0])
        require(torch.equal(a,b),'saved checkpoint replay differs')
        torch.save(dict(tokens=last[0].cpu(),targets=last[1].cpu(),logits=a.cpu()),dest/'REPLAY.pt')
        require(all(int(state['step'])==done for state in optimizer.state.values()),'AdamW update counters differ')
        measured=copy.deepcopy(manifest)
        for m in measured['models']:
            if m['architecture']==args.arm and m['seed']==args.seed:
                m['measured']=dict(trainable_parameters=registered[0]['trainable_parameters'],
                    deployed_parameters=registered[0]['deployed_parameters'],checkpoint_sha256=digest(checkpoint))
        # Full multi-seed final admission is performed by the collector, never
        # forged here with receipts for unfinished sibling seeds.
        receipt.update(checkpoint_sha256=digest(checkpoint),checkpoint=str(checkpoint.name),
            parameter_changed=True,serialization_replay_exact=True,mean_step_seconds=sum(times)/len(times),
            actual_tokens=done*fit['batch_size']*fit['sequence_length'],adamw_step_counters=done)
        if not args.smoke:
            receipt['exposed_dev']=evaluate(model,data['dev'],device,fit['dev_windows'])
        (dest/'FINAL_PARTIAL_MANIFEST.json').write_text(json.dumps(measured,indent=2))
        report('SMOKE_COMPLETE_NOT_FULL_TRAINING' if args.smoke else 'FULL_FIT_COMPLETE_INTERNAL_UNQUALIFIED')
    except BaseException as exc:
        # Exact attempted/completed counters and immutable originals survive.
        report('FAILED_RETAINED_INTERNAL',type(exc).__name__+': '+str(exc))
        raise
    return receipt

def main():
    p=argparse.ArgumentParser();p.add_argument('--arm',choices=list(SPECS),required=True)
    p.add_argument('--seed',type=int,required=True);p.add_argument('--device',default='cpu')
    p.add_argument('--smoke',action='store_true');p.add_argument('--output',type=Path,required=True)
    p.add_argument('--resume',type=Path)
    print(json.dumps(fit(p.parse_args()),indent=2))

if __name__=='__main__':main()
