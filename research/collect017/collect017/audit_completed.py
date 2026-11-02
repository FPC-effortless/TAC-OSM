"""Read-only original/checkpoint/journal and trained-source boundary audit."""
import argparse,copy,hashlib,json
from pathlib import Path
import torch
from .common import ROOT,build,canonical,digest,source_paths,verify_data,require,fingerprint,internal_gate
from collect016.data import pools,training_batch
from .model import run_chunks
from collect016.reference import whole_reference
from .evaluate import state_hash
from collect011.model import named_state_tensors
def independent_exposure(data,row):
    meta=row['exposure'];latest={};records=[]
    for fact in meta['facts']:
        i=fact['source_offset'];q=data[i:i+8];require(q.decode('ascii')==fact['quote'],'source quote changed')
        require(q[:4].hex()==fact['answer_hex'],'source target differs');latest[fact['key']]=q[:4]
        records.append(b'@'+fact['key'].encode()+b': '+q+b'\n')
    label=latest.get(meta['key'],b'????');require(label.hex()==meta['answer_hex'],'latest source label differs')
    require(meta['absent']==(meta['key'] not in latest),'absence metadata differs')
    j=meta['filler_offset'];n=meta['filler_bytes']
    episode=b''.join(records)+b'\n'+data[j:j+n]+b'Copy '+meta['key'].encode()+b': '+label
    i=meta['natural_offset'];slab=torch.tensor([list(data[i:i+385]),list(episode)],dtype=torch.long)
    require(slab.shape==(2,385),'reconstructed exposure shape differs')
    require(hashlib.sha256(slab.numpy().tobytes()).hexdigest()==meta['batch_sha256'],'independent source batch differs')
def audit(folder,out):
    torch.set_num_threads(1);r=json.loads((folder/'FIT_RECEIPT.json').read_text());require(r['status']=='FULL_FIT_COMPLETE_INTERNAL_UNQUALIFIED','incomplete original fit')
    manifest=json.loads((ROOT/'registrations'/r['arm']/'FAIR_COMPARISON.json').read_text())
    require(digest(ROOT/'registrations'/r['arm']/'FAIR_COMPARISON.json')==r['protocol_sha256'],'protocol differs')
    sources={p:digest(v) for p,v in source_paths().items()};require(sources==r['source_hashes']==manifest['source_hashes'],'source differs')
    data,split=verify_data();require(r['data_sha256']==split['parts']['train']['sha256'],'source partition differs')
    rows=[json.loads(z) for z in (folder/'journal.jsonl').read_text().splitlines()];require(len(rows)==r['optimizer_updates']==manifest['fit']['steps']==512,'update budget differs')
    head='0'*64;exposures=[]
    for i,row in enumerate(rows,1):
        check=dict(row);sha=check.pop('sha256');require(check['step']==i and check['previous']==head and hashlib.sha256(canonical(check)).hexdigest()==sha,'journal chain differs')
        head=sha;independent_exposure(data['train'],row);exposures.append(row['exposure']['batch_sha256'])
        goal=row['natural_nll']+.25*row['task_nll']+row['answer_nll']+.01*row['balance_loss']
        require(abs(row['objective']-goal)<5e-6,'declared objective arithmetic differs')
        require(sum(sum(v) for v in row['bank_counts'])==0,'attention control must have no phase-router counts')
    require(head==r['journal_head'],'final journal head differs')
    snapshots=[]
    for step in (0,128,256,512):
        p=folder/f'checkpoint_{step:04}.pt';require(digest(p)==p.with_suffix('.sha256').read_text().strip(),'immutable checkpoint hash differs')
        saved=torch.load(p,weights_only=True,map_location='cpu')
        require(saved['steps']==step and saved['source_hashes']==sources and saved['protocol_sha256']==r['protocol_sha256'],'snapshot metadata differs')
        expected='0'*64 if step==0 else rows[step-1]['sha256'];require(saved['journal_head']==expected,'snapshot progress differs')
        require(all(bool(torch.isfinite(v).all()) for v in saved['model'].values()),'nonfinite saved parameters')
        if step:
            require(all(int(v['step'])==step and torch.isfinite(v['exp_avg']).all() and torch.isfinite(v['exp_avg_sq']).all() for v in saved['optimizer']['state'].values()),'AdamW state differs')
        snapshots.append(dict(step=step,sha256=digest(p)))
    initial=torch.load(folder/'checkpoint_0000.pt',weights_only=True,map_location='cpu');m=build(r['arm'],r['seed'])
    require(fingerprint(m)==r['initial_parameter_sha256'],'live initializer differs')
    for n,v in m.state_dict().items():require(torch.equal(v,initial['model'][n]),'saved initializer differs')
    final=torch.load(folder/'checkpoint_0512.pt',weights_only=True,map_location='cpu');m.load_state_dict(final['model']);m.eval()
    require(sum(v.numel() for v in m.parameters())==r['trainable_parameters'],'actual final parameter count differs')
    require(len(final['optimizer']['state'])==len(list(m.parameters())),'optimizer omitted counted tensors')
    replay=torch.load(folder/'REPLAY.pt',weights_only=True,map_location='cpu')
    with torch.no_grad():z,_,_,_=run_chunks(m,replay['tokens'])
    error=float((z-replay['logits']).abs().max());require(error<=2e-5,'cross-runtime replay differs')
    refs=[]
    for dtype in (torch.float32,torch.float64):
        a=build(r['arm'],r['seed']).to(dtype);a.load_state_dict(final['model']);b=copy.deepcopy(a)
        x=replay['tokens'][:,:9];y,s=a(x);z,t=whole_reference(b,x)
        tolerance=3e-5 if dtype==torch.float32 else 2e-9
        torch.testing.assert_close(y,z,atol=tolerance,rtol=tolerance)
        state_error=0.
        for n,v in named_state_tensors(s).items():
            w=named_state_tensors(t)[n];torch.testing.assert_close(v,w,atol=tolerance,rtol=tolerance)
            if v.numel():state_error=max(state_error,float((v-w).abs().max().detach()))
        y.square().mean().backward();z.square().mean().backward();gradient_error=0.
        for (n,p),(nn,q) in zip(a.named_parameters(),b.named_parameters()):
            require(n==nn,'parameter naming differs');torch.testing.assert_close(p.grad,q.grad,atol=tolerance,rtol=tolerance)
            gradient_error=max(gradient_error,float((p.grad-q.grad).abs().max()))
        refs.append(dict(dtype=str(dtype),logits_max_abs=float((y-z).abs().max().detach()),state_max_abs=state_error,surrogate_gradient_max_abs=gradient_error))
    x=replay['tokens'][:,:35].clone();future=x.clone();future[:,19:]=(future[:,19:]+97)%256;other=x.clone();other[1]=(other[1]+57)%256
    with torch.no_grad():
        a,s=m(x,owners=('audit-0','audit-1'));b,_=m(future,owners=('audit-0','audit-1'));c,_=m(other,owners=('audit-0','audit-1'))
        require(torch.equal(a[:,:19],b[:,:19]),'future poison changed earlier output');require(torch.equal(a[0],c[0]),'another owner changed output')
        identity=state_hash(s);m(x[:,:3],state=s,owners=s.owners);require(state_hash(s)==identity,'prefix branch mutated state')
        rejected=False
        try:m(x[:,:3],state=s,owners=tuple(reversed(s.owners)))
        except ValueError:rejected=True
        require(rejected,'owner mismatch accepted')
    # Actual final answer loss credit into first-chunk activations and state.
    m.zero_grad(set_to_none=True);activations=[];states=[];state=None;owners=('credit-0','credit-1')
    def hook(module,args,value):value.retain_grad();activations.append(value)
    handle=m.core.embedding.register_forward_hook(hook);parts=[]
    for lo in range(0,384,128):
        a,state=m(replay['tokens'][:,lo:lo+128],state=state,owners=owners);parts.append(a)
        if lo==0:
            for st in state.phase:
                if st.memory.requires_grad:st.memory.retain_grad()
            states=list(state.phase)
        if r['arm']=='detach_128' and lo<256:state=state.detach()
    loss=torch.nn.functional.cross_entropy(torch.cat(parts,1)[replay['answer_mask']],replay['targets'][replay['answer_mask']]);loss.backward();handle.remove()
    earlier=activations[0].grad;credit=dict(first_chunk_embedding_gradient_l1=0. if earlier is None else float(earlier.abs().sum()),
        phase_state_gradient_l1=[None if p.memory.grad is None else float(p.memory.grad.abs().sum()) for p in states])
    if r['arm']=='detach_128':require(credit['first_chunk_embedding_gradient_l1']==0.,'declared detach retained early credit')
    m.zero_grad(set_to_none=True);m.eval()
    result=dict(status='PASS_TRAINED_SOURCE_AUDIT',arm=r['arm'],seed=r['seed'],checkpoints=snapshots,updates=512,
        replay_max_abs=error,trained_reference=refs,cross_boundary_answer_credit=credit,
        source_exposure_verified=512,exposure_sequence_sha256=hashlib.sha256(canonical(exposures)).hexdigest(),
        causal_future_poison=True,owner_isolation=True,owner_mismatch_rejected=True,prefix_branch_immutable=True,
        optimizer_updates_in_audit=0,independent_qualification=False,full_Lean_IEEE_AD='OPEN')
    out.write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2));return result
def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();audit(a.run,a.output)
if __name__=='__main__':main()
