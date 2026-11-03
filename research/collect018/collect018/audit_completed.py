"""Independent immutable-source/exposure and trained-checkpoint audit; no optimizer."""
import argparse,copy,hashlib,json
from pathlib import Path
import torch
from .common import ROOT,build,canonical,digest,source_paths,verify_data,require,fingerprint
from .model import run_chunks,tensors
from .reference import whole_reference
from .evaluate import state_hash

def reconstruct(data,meta):
    j=meta['natural_offset'];rows=[data[j:j+129]]
    for r in meta['records']:
        latest={};records=[]
        for fact in r['facts']:
            j=fact['source_offset'];quote=data[j:j+8]
            require(quote.decode('ascii')==fact['quote'] and quote[:4].hex()==fact['answer_hex'],'source quotation/label changed')
            records.append(b'@'+fact['key'].encode()+b': '+quote+b'\n');latest[fact['key']]=quote[:4]
        answer=latest.get(r['key'],b'????');require(answer.hex()==r['answer_hex'] and r['absent']==(r['key'] not in latest),'latest/absent target differs')
        l=r['leading_offset'];n=r['leading_bytes'];f=r['filler_offset'];g=r['filler_bytes'];t=r['tail_offset'];tail=r['tail_bytes']
        prefix=data[l:l+n]+b''.join(records)+b'\n'+data[f:f+g]+b'Copy '+r['key'].encode()+b': '
        require(len(prefix)-1==r['first_position'],'target timing differs')
        episode=prefix+answer+data[t:t+tail];require(len(episode)==129,'reconstructed length differs');rows.append(episode)
    x=torch.tensor([list(row) for row in rows],dtype=torch.long)
    require(x.shape==(4,129) and hashlib.sha256(x.numpy().tobytes()).hexdigest()==meta['batch_sha256'],'independent original exposure differs')

def audit(folder,output):
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    r=json.loads((folder/'FIT_RECEIPT.json').read_text());require(r['status']=='FULL_FIT_COMPLETE_INTERNAL_UNQUALIFIED','original fit incomplete')
    arm=r['arm'];seed=r['seed'];path=ROOT/'registrations'/arm/'FAIR_COMPARISON.json';manifest=json.loads(path.read_text())
    require(digest(path)==r['protocol_sha256'],'registration differs');actual={p:digest(v) for p,v in source_paths().items()};require(actual==r['source_hashes']==manifest['source_hashes'],'source differs')
    data,split=verify_data();require(r['data_sha256']==split['parts']['train']['sha256'],'training partition differs')
    rows=[json.loads(x) for x in (folder/'journal.jsonl').read_text().splitlines()];require(len(rows)==r['optimizer_updates']==1024,'update budget differs')
    head='0'*64;exposures=[];label_counts={};positions=set()
    for step,row in enumerate(rows,1):
        check=dict(row);sha=check.pop('sha256');require(check['previous']==head and check['step']==step and hashlib.sha256(canonical(check)).hexdigest()==sha,'journal chain differs');head=sha
        reconstruct(data['train'],row['exposure']);exposures.append(row['exposure']['batch_sha256'])
        focus=3*row['first_nll'] if arm.endswith('_first') else 0.
        loss=row['natural_nll']+.25*row['task_nll']+row['answer_nll']+.01*row['balance_loss']+focus;require(abs(loss-row['objective'])<=2e-5,'loss arithmetic differs')
        expected=0 if arm.startswith('gru_') else 4*4*128;require(sum(sum(z) for z in row['bank_counts'])==expected,'routing exposure differs')
        for q in row['exposure']['records']:
            c=bytes.fromhex(q['answer_hex'])[0];label_counts[c]=label_counts.get(c,0)+1;positions.add(q['first_position'])
    require(head==r['journal_head'],'final journal head differs');snapshots=[]
    for step in (0,256,512,1024):
        p=folder/f'checkpoint_{step:04}.pt';require(digest(p)==p.with_suffix('.sha256').read_text().strip(),'checkpoint hash differs')
        s=torch.load(p,weights_only=True,map_location='cpu');require(s['steps']==step and s['source_hashes']==actual and s['protocol_sha256']==r['protocol_sha256'],'snapshot source/step differs')
        require(s['journal_head']==('0'*64 if step==0 else rows[step-1]['sha256']),'snapshot journal differs');require(all(bool(torch.isfinite(t).all()) for t in s['model'].values()),'nonfinite snapshot')
        if step:require(all(int(t['step'])==step and torch.isfinite(t['exp_avg']).all() and torch.isfinite(t['exp_avg_sq']).all() for t in s['optimizer']['state'].values()),'AdamW counters/accumulators differ')
        snapshots.append(dict(step=step,sha256=digest(p)))
    initial=torch.load(folder/'checkpoint_0000.pt',weights_only=True,map_location='cpu');m=build(arm,seed)
    require(fingerprint(m)==r['initial_parameter_sha256'],'initializer differs')
    for name,p in m.state_dict().items():require(torch.equal(p,initial['model'][name]),'saved initializer differs')
    final=torch.load(folder/'checkpoint_1024.pt',weights_only=True,map_location='cpu');m.load_state_dict(final['model']);m.eval();before=fingerprint(m)
    count=sum(p.numel() for p in m.parameters());require(count==r['trainable_parameters']==r['deployed_parameters'],'live final counts differ')
    require(len(final['optimizer']['state'])==len(list(m.parameters())),'optimizer omitted active parameters')
    replay=torch.load(folder/'REPLAY.pt',weights_only=True,map_location='cpu')
    with torch.no_grad():z,_,_,_=run_chunks(m,replay['tokens'],chunk=128)
    replay_error=float((z-replay['logits']).abs().max());require(replay_error<=4e-5,'cross-runtime original replay exceeds frozen tolerance')
    references=[]
    for dtype in (torch.float32,torch.float64):
        a=build(arm,seed).to(dtype);a.load_state_dict(final['model']);b=copy.deepcopy(a);x=replay['tokens'][:2,:9]
        y,s=a(x);zz,t=whole_reference(b,x);tol=4e-5 if dtype==torch.float32 else 3e-9
        torch.testing.assert_close(y,zz,atol=tol,rtol=tol);state_error=0.
        for name,p in tensors(s).items():
            q=tensors(t)[name];torch.testing.assert_close(p,q,atol=tol,rtol=tol)
            if p.numel():state_error=max(state_error,float((p-q).abs().max().detach()))
        y.square().sum().backward();zz.square().sum().backward();gradient_error=0.
        for (name,p),(nn,q) in zip(a.named_parameters(),b.named_parameters()):
            require(name==nn,'parameter identities differ');torch.testing.assert_close(p.grad,q.grad,atol=tol*10,rtol=tol*10);gradient_error=max(gradient_error,float((p.grad-q.grad).abs().max()))
        references.append(dict(dtype=str(dtype),logits_max_abs=float((y-zz).abs().max().detach()),state_max_abs=state_error,surrogate_gradient_max_abs=gradient_error))
    x=replay['tokens'][:2,:35].clone();future=x.clone();future[:,19:]=(future[:,19:]+73)%256;other=x.clone();other[1]=(other[1]+31)%256
    with torch.no_grad():
        a,s=m(x,owners=('audit-a','audit-b'));b,_=m(future,owners=('audit-a','audit-b'));c,_=m(other,owners=('audit-a','audit-b'))
        require(torch.equal(a[:,:19],b[:,:19]),'future poison changed earlier output');require(torch.equal(a[0],c[0]),'another owner changed output')
        old=state_hash(s);m(x[:,:3],state=s,owners=s.owners);require(old==state_hash(s),'branch mutated caller state')
        rejected=False
        try:m(x[:,:3],state=s,owners=tuple(reversed(s.owners)))
        except ValueError:rejected=True
        require(rejected,'owner mismatch accepted')
    m.zero_grad(set_to_none=True);captures=[]
    def hook(module,args,out):out.retain_grad();captures.append(out)
    h=m.core.embedding.register_forward_hook(hook);z,_,_,_=run_chunks(m,replay['tokens'],chunk=128);loss=torch.nn.functional.cross_entropy(z[replay['first_mask']],replay['targets'][replay['first_mask']]);loss.backward();h.remove()
    g=captures[0].grad;require(float(g[0].abs().sum())==0.,'natural owner received task-only activation gradient')
    profile=g[1:].abs().sum(-1);require(before==fingerprint(m),'audit changed weights')
    result=dict(status='PASS_TRAINED_SOURCE_AUDIT',arm=arm,seed=seed,checkpoints=snapshots,updates=1024,parameters=count,replay_max_abs=replay_error,
        trained_reference=references,source_exposure_verified=1024,exposure_sequence_sha256=hashlib.sha256(canonical(exposures)).hexdigest(),
        first_target_counts=label_counts,answer_positions=sorted(positions),first_answer_gradient=dict(nll=float(loss.detach()),per_task_position_l1=profile.tolist(),natural_owner_gradient_l1=0.),
        causal_future_poison=True,fixed_batch_owner_isolation=True,prefix_branch_immutable=True,owner_mismatch_rejected=True,optimizer_updates_in_audit=0,
        torch=torch.__version__,independent_qualification=False,full_Lean_source_IEEE_AD='OPEN')
    output.write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k!='first_answer_gradient'},indent=2));return result
def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();audit(a.run,a.output)
if __name__=='__main__':main()
