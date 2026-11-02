"""Preregistered 384-byte source-grounded joint LM/copy training; all fits retained."""
import argparse,copy,hashlib,json,platform,time
from pathlib import Path
import torch
from torch.nn import functional as F
from .common import ROOT,ARMS,build,canonical,digest,require,verify_data,internal_gate,source_paths,fingerprint,freeze_tree
from .data import pools,training_batch
from .model import run_chunks
def objective(logits,y,answer_mask,balance):
    ce=F.cross_entropy(logits.flatten(0,1),y.flatten(),reduction='none').reshape_as(y)
    lm=ce[0].mean();task=ce[1].mean();answer=ce[answer_mask].mean()
    return lm+.25*task+answer+.01*balance,dict(natural_nll=float(lm.detach()),task_nll=float(task.detach()),answer_nll=float(answer.detach()),balance_loss=float(balance.detach()))
def fit(a):
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    path=ROOT/'SMOKE_REGISTRATION.json' if a.smoke else ROOT/'registrations'/a.arm/'FAIR_COMPARISON.json'
    raw=path.read_bytes();manifest=json.loads(raw);gate=internal_gate(manifest);f=manifest['fit']
    require(any(x['architecture']==a.arm and x['seed']==a.seed for x in manifest['models']),'unregistered arm/seed')
    source={p:digest(v) for p,v in source_paths().items()}
    require(source==manifest['source_hashes'],'source changed after registration')
    data,split=verify_data();pool=pools(data['train']);m=build(a.arm,a.seed);initial=fingerprint(m)
    opt=torch.optim.AdamW(m.parameters(),lr=f['learning_rate'],betas=tuple(f['betas']),weight_decay=f['weight_decay'],foreach=False)
    dest=a.output;dest.mkdir(parents=True,exist_ok=True);require(not (dest/'FIT_RECEIPT.json').exists(),'immutable original run already exists')
    head='0'*64;done=0;attempted=0;seconds=[];last=None;t0=time.perf_counter();psha=hashlib.sha256(raw).hexdigest()
    receipt=dict(arm=a.arm,seed=a.seed,planned_steps=f['steps'],optimizer_updates=0,attempted_updates=0,device='cpu',
        torch=torch.__version__,python=platform.python_version(),initial_parameter_sha256=initial,
        trainable_parameters=sum(p.numel() for p in m.parameters()),source_hashes=source,protocol_sha256=psha,
        data_sha256=split['parts']['train']['sha256'],status='STARTING',smoke=a.smoke,gpu_minutes=0,
        independently_qualified=False,protected_test_read=False,full_training_flops='NOT_MEASURED_NO_ARCHITECTURE_SUPERIORITY')
    def report(status,error=None):
        receipt.update(status=status,optimizer_updates=done,attempted_updates=attempted,journal_head=head,
            elapsed_seconds=time.perf_counter()-t0,total_recorded_optimizer_seconds=sum(seconds),error=error)
        (dest/'FIT_RECEIPT.json').write_text(json.dumps(receipt,indent=2))
    def save(step):
        ck=dest/f'checkpoint_{step:04}.pt';require(not ck.exists(),'immutable checkpoint exists')
        torch.save(dict(schema='collect016.training.v1',arm=a.arm,seed=a.seed,steps=step,
            model=freeze_tree(m.state_dict()),optimizer=freeze_tree(opt.state_dict()),torch_rng=torch.get_rng_state(),
            protocol_sha256=psha,source_hashes=source,data_sha256=receipt['data_sha256'],journal_head=head),ck)
        (ck.with_suffix('.sha256')).write_text(digest(ck)+'\n');return ck
    try:
        save(0);report('RUNNING_INTERNAL_FIT')
        for step in range(1,f['steps']+1):
            require(time.perf_counter()-t0<f['worker_seconds_cap'],'allocation ended before complete budget')
            require(path.read_bytes()==raw,'registration changed during fit')
            x,y,mask,meta=training_batch(data['train'],pool,seed=a.seed,step=step,length=f['sequence_length'])
            opt.zero_grad(set_to_none=True);m.train();tic=time.perf_counter()
            logits,_,bal,counts=run_chunks(m,x,detach=a.arm=='detach_128')
            loss,metrics=objective(logits,y,mask,bal);require(bool(torch.isfinite(loss)),'nonfinite objective');loss.backward()
            require(all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in m.parameters()),'missing/nonfinite gradient')
            gn=torch.nn.utils.clip_grad_norm_(m.parameters(),f['clip_norm']);attempted=step;opt.step();done=step
            require(all(bool(torch.isfinite(p).all()) for p in m.parameters()),'nonfinite weights')
            dt=time.perf_counter()-tic;seconds.append(dt)
            row=dict(step=step,objective=float(loss.detach()),gradient_norm=float(gn),seconds=dt,bank_counts=counts.tolist(),
                exposure=meta,previous=head,**metrics)
            row['sha256']=hashlib.sha256(canonical(row)).hexdigest();head=row['sha256']
            with (dest/'journal.jsonl').open('a') as out:out.write(json.dumps(row)+'\n')
            if step in f['checkpoint_steps']:save(step)
            last=(x,y,mask)
            if step%16==0 or step==f['steps']:
                report('RUNNING_INTERNAL_FIT');print(json.dumps(dict(arm=a.arm,seed=a.seed,step=step,**metrics)),flush=True)
        require({p:digest(v) for p,v in source_paths().items()}==source,'source changed during fit')
        ck=dest/f'checkpoint_{done:04}.pt';saved=torch.load(ck,weights_only=True,map_location='cpu');clone=build(a.arm,a.seed);clone.load_state_dict(saved['model']);m.eval();clone.eval()
        with torch.no_grad():
            z,_,_,_=run_chunks(m,last[0]);zz,_,_,_=run_chunks(clone,last[0])
        require(torch.equal(z,zz),'saved checkpoint replay differs')
        torch.save(dict(tokens=last[0],targets=last[1],answer_mask=last[2],logits=z),dest/'REPLAY.pt')
        require(all(int(s['step'])==done for s in opt.state.values()),'AdamW counters disagree')
        require(fingerprint(m)!=initial,'weights unchanged')
        receipt.update(checkpoint=ck.name,checkpoint_sha256=digest(ck),parameter_changed=True,serialization_replay_exact=True,
            presented_bytes=done*2*f['sequence_length'],answer_targets=done*4,mean_step_seconds=sum(seconds)/len(seconds),adamw_step_counters=done)
        final=copy.deepcopy(manifest)
        for row in final['models']:
            if row['architecture']==a.arm and row['seed']==a.seed:
                row['measured']=dict(trainable_parameters=receipt['trainable_parameters'],deployed_parameters=receipt['trainable_parameters'],checkpoint_sha256=digest(ck))
        (dest/'FINAL_PARTIAL_MANIFEST.json').write_text(json.dumps(final,indent=2))
        report('SMOKE_COMPLETE_NOT_FULL_TRAINING' if a.smoke else 'FULL_FIT_COMPLETE_INTERNAL_UNQUALIFIED')
    except BaseException as exc:
        report('FAILED_RETAINED_INTERNAL',type(exc).__name__+': '+str(exc));raise
    return receipt
def main():
    p=argparse.ArgumentParser();p.add_argument('--arm',choices=ARMS,required=True);p.add_argument('--seed',type=int,required=True)
    p.add_argument('--smoke',action='store_true');p.add_argument('--output',type=Path,required=True)
    print(json.dumps(fit(p.parse_args()),indent=2))
if __name__=='__main__':main()
