"""Fixed-budget two-objective local acquisition campaign. Never opens TEST."""
import argparse,copy,hashlib,json,platform,time
from pathlib import Path
import torch
from torch.nn import functional as F
from .common import ROOT,ARMS,build,canonical,digest,source_paths,verify_data,internal_gate,require,fingerprint,freeze_tree
from .model import run_chunks
from .data import pools,training_batch

def objective(logits,y,mask,first,balance,focus):
    ce=F.cross_entropy(logits.flatten(0,1),y.flatten(),reduction='none').reshape_as(y)
    natural=ce[0].mean();task=ce[1:].mean();answer=ce[mask].mean();firstloss=ce[first].mean()
    loss=natural+.25*task+answer+3*focus*firstloss+.01*balance
    metrics=dict(natural_nll=float(natural.detach()),task_nll=float(task.detach()),answer_nll=float(answer.detach()),first_nll=float(firstloss.detach()),
        first_accuracy=float((logits.argmax(-1)[first]==y[first]).float().mean()),balance_loss=float(balance.detach()))
    return loss,metrics

def fit(a):
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    path=ROOT/'smoke_registrations'/a.arm/'FAIR_COMPARISON.json' if a.smoke else ROOT/'registrations'/a.arm/'FAIR_COMPARISON.json'
    raw=path.read_bytes();mfest=json.loads(raw);gate=internal_gate(mfest);f=mfest['fit'];focus=a.arm.endswith('_first')
    require(any(r['architecture']==a.arm and r['seed']==a.seed for r in mfest['models']),'unregistered model')
    source={p:digest(v) for p,v in source_paths().items()};require(source==mfest['source_hashes'],'source differs from prefit registration')
    data,split=verify_data();pool=pools(data['train']);m=build(a.arm,a.seed);initial=fingerprint(m)
    optimizer=torch.optim.AdamW(m.parameters(),lr=f['learning_rate'],betas=tuple(f['betas']),weight_decay=f['weight_decay'],foreach=False)
    dest=a.output;dest.mkdir(parents=True,exist_ok=True);require(not (dest/'FIT_RECEIPT.json').exists(),'original run exists')
    head='0'*64;done=0;attempted=0;seconds=[];start=time.perf_counter();psha=digest(path);last=None
    receipt=dict(arm=a.arm,seed=a.seed,planned_steps=f['steps'],optimizer_updates=0,attempted_updates=0,device='cpu',torch=torch.__version__,python=platform.python_version(),
        initial_parameter_sha256=initial,trainable_parameters=sum(p.numel() for p in m.parameters()),deployed_parameters=sum(p.numel() for p in m.parameters()),
        source_hashes=source,protocol_sha256=psha,data_sha256=split['parts']['train']['sha256'],smoke=a.smoke,gpu_minutes=0,independently_qualified=False,protected_test_read=False,
        full_training_flops='NOT_MEASURED_NO_ARCHITECTURE_SUPERIORITY',objective_profile='first_plus_3' if focus else 'original_joint')
    def report(status,error=None):
        receipt.update(status=status,optimizer_updates=done,attempted_updates=attempted,journal_head=head,elapsed_seconds=time.perf_counter()-start,total_recorded_optimizer_seconds=sum(seconds),error=error)
        (dest/'FIT_RECEIPT.json').write_text(json.dumps(receipt,indent=2))
    def save(step):
        p=dest/f'checkpoint_{step:04}.pt';require(not p.exists(),'snapshot already exists')
        torch.save(dict(schema='collect018.training.v1',arm=a.arm,seed=a.seed,steps=step,model=freeze_tree(m.state_dict()),optimizer=freeze_tree(optimizer.state_dict()),torch_rng=torch.get_rng_state(),
            source_hashes=source,protocol_sha256=psha,data_sha256=receipt['data_sha256'],journal_head=head),p);p.with_suffix('.sha256').write_text(digest(p)+'\n');return p
    try:
        save(0);report('RUNNING_INTERNAL_FIT')
        for step in range(1,f['steps']+1):
            require(time.perf_counter()-start<f['worker_seconds_cap'],'allocation ended before complete budget')
            require(path.read_bytes()==raw,'registration changed during fit')
            x,y,mask,first,meta=training_batch(data['train'],pool,seed=a.seed,step=step)
            m.train();optimizer.zero_grad(set_to_none=True);tic=time.perf_counter();z,_,bal,counts=run_chunks(m,x,chunk=f['chunk_size'])
            loss,metrics=objective(z,y,mask,first,bal,focus);require(bool(torch.isfinite(loss)),'nonfinite objective');loss.backward()
            require(all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in m.parameters()),'missing/nonfinite gradient')
            gn=torch.nn.utils.clip_grad_norm_(m.parameters(),f['clip_norm']);attempted=step;optimizer.step();done=step
            require(all(bool(torch.isfinite(p).all()) for p in m.parameters()),'nonfinite parameters')
            dt=time.perf_counter()-tic;seconds.append(dt);row=dict(step=step,objective=float(loss.detach()),seconds=dt,gradient_norm=float(gn),exposure=meta,bank_counts=counts.tolist(),previous=head,**metrics)
            row['sha256']=hashlib.sha256(canonical(row)).hexdigest();head=row['sha256']
            with (dest/'journal.jsonl').open('a') as out:out.write(json.dumps(row)+'\n')
            if step in f['checkpoint_steps']:save(step)
            last=(x,y,mask,first)
            if step%32==0 or step==f['steps']:report('RUNNING_INTERNAL_FIT');print(json.dumps(dict(arm=a.arm,seed=a.seed,step=step,**metrics)),flush=True)
        require(source=={p:digest(v) for p,v in source_paths().items()},'source changed during fit')
        ck=dest/f'checkpoint_{done:04}.pt';saved=torch.load(ck,weights_only=True,map_location='cpu');clone=build(a.arm,a.seed);clone.load_state_dict(saved['model']);clone.eval();m.eval()
        with torch.no_grad():z,_,_,_=run_chunks(m,last[0],chunk=f['chunk_size']);zz,_,_,_=run_chunks(clone,last[0],chunk=f['chunk_size'])
        require(torch.equal(z,zz),'serialized replay differs');torch.save(dict(tokens=last[0],targets=last[1],answer_mask=last[2],first_mask=last[3],logits=z),dest/'REPLAY.pt')
        require(all(int(r['step'])==done for r in optimizer.state.values()),'optimizer counters differ');require(fingerprint(m)!=initial,'parameters unchanged')
        receipt.update(checkpoint=ck.name,checkpoint_sha256=digest(ck),parameter_changed=True,serialization_replay_exact=True,presented_bytes=done*512,answer_targets=done*12,first_targets=done*3,
            mean_step_seconds=sum(seconds)/len(seconds),adamw_step_counters=done)
        final=copy.deepcopy(mfest)
        for r in final['models']:
            if r['architecture']==a.arm and r['seed']==a.seed:r['measured']=dict(trainable_parameters=receipt['trainable_parameters'],deployed_parameters=receipt['deployed_parameters'],checkpoint_sha256=digest(ck))
        (dest/'FINAL_PARTIAL_MANIFEST.json').write_text(json.dumps(final,indent=2));report('SMOKE_COMPLETE_NOT_FULL_TRAINING' if a.smoke else 'FULL_FIT_COMPLETE_INTERNAL_UNQUALIFIED')
    except BaseException as e:report('FAILED_RETAINED_INTERNAL',type(e).__name__+': '+str(e));raise
    return receipt
def main():
    p=argparse.ArgumentParser();p.add_argument('--arm',choices=ARMS,required=True);p.add_argument('--seed',type=int,required=True);p.add_argument('--smoke',action='store_true');p.add_argument('--output',type=Path,required=True)
    print(json.dumps(fit(p.parse_args()),indent=2))
if __name__=='__main__':main()
