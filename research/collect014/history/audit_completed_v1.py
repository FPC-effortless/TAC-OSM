"""Additional read-only trained-source checks in the local recorded runtime.

This neither changes the frozen fit/evaluator nor retrains/selects checkpoints.
Cross-runtime replay is measured with explicit tolerance; exact same-runtime
replay is separately retained in the original remote evaluator.
"""
import argparse,hashlib,json
from pathlib import Path
import torch
from .common import ROOT,build,verify_data,require,digest
from .reference import whole_reference
from .train import batch,fingerprint
from collect011.model import named_state_tensors

def audit(folder):
    torch.set_num_threads(1)
    receipt=json.loads((folder/'FIT_RECEIPT.json').read_text())
    require(receipt['status']=='FULL_FIT_COMPLETE_INTERNAL_UNQUALIFIED','partial fit cannot pass completed audit')
    checkpoint=folder/receipt['checkpoint'];require(digest(checkpoint)==receipt['checkpoint_sha256'],'checkpoint changed')
    saved=torch.load(checkpoint,map_location='cpu',weights_only=True)
    require(saved['steps']==receipt['optimizer_updates']==1024,'fixed budget incomplete')
    registry=json.loads((ROOT/'REUSED_CONTROLS.json').read_text())
    expected=next(r for r in registry['controls'] if r['seed']==receipt['seed'])
    require(receipt['initial_parameter_sha256']==expected['initial_parameter_sha256'],'initial weights differ from reused control')
    manifest=json.loads((ROOT/'registrations'/receipt['arm']/'FAIR_COMPARISON.json').read_text())
    require(receipt['source_hashes']==manifest['source_hashes'],'actual fit source differs')
    require(receipt['protocol_sha256']==digest(ROOT/'registrations'/receipt['arm']/'FAIR_COMPARISON.json'),'fit registration differs')
    previous='0'*64;rows=[]
    for line in (folder/'journal.jsonl').read_text().splitlines():
        r=json.loads(line);sha=r.pop('sha256')
        require(r['previous']==previous and hashlib.sha256(json.dumps(r,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()==sha,'loss journal changed')
        previous=sha;rows.append(r)
    require(len(rows)==1024 and previous==saved['journal_head'],'journal progress differs from checkpoint')
    states=saved['optimizer']['state'];require(bool(states),'AdamW state missing')
    require(all(int(s['step'])==1024 and bool(torch.isfinite(s['exp_avg']).all()) and bool(torch.isfinite(s['exp_avg_sq']).all()) for s in states.values()),'AdamW moments/counters invalid')
    for p in folder.glob('checkpoint_*.pt'):
        require(digest(p)==p.with_suffix('.sha256').read_text().strip(),'immutable snapshot hash changed')
    m=build(receipt['arm'],receipt['seed']);m.load_state_dict(saved['model']);m.eval();before=fingerprint(m)
    replay=torch.load(folder/'REPLAY.pt',weights_only=True,map_location='cpu')
    with torch.no_grad():actual,_=m(replay['tokens'])
    cross_error=float((actual-replay['logits']).abs().max())
    torch.testing.assert_close(actual,replay['logits'],atol=2e-5,rtol=2e-5)
    x=replay['tokens'][:2,:16].clone()
    with torch.no_grad():
        y,_=m(x);changed=x.clone();changed[:,8:]=(changed[:,8:]+81)%256
        poison,_=m(changed);require(torch.equal(y[:,:8],poison[:,:8]),'future-byte causality failed')
        changed=x.clone();changed[1]=(changed[1]+77)%256
        other,_=m(changed);require(torch.equal(y[0],other[0]),'owner isolation failed')
    reference_results=[]
    for dtype in (torch.float32,torch.float64):
        a=build(receipt['arm'],receipt['seed']).to(dtype);a.load_state_dict(saved['model'])
        xx=x[:,:8];mask=torch.ones_like(xx,dtype=torch.bool);mask[:,3]=False
        y,s=a(xx,write_mask=mask);ref,rs=whole_reference(a,xx,write_mask=mask)
        tol=8e-5 if dtype==torch.float32 else 1e-9
        torch.testing.assert_close(y,ref,atol=tol,rtol=tol)
        state_error=0.
        for n,t in named_state_tensors(s).items():
            tt=named_state_tensors(rs)[n]
            torch.testing.assert_close(t,tt,atol=tol,rtol=tol)
            if t.numel():state_error=max(state_error,float((t-tt).abs().max().detach()))
        params=tuple(a.parameters());g=torch.autograd.grad(y.square().mean(),params,retain_graph=True,allow_unused=True)
        gg=torch.autograd.grad(ref.square().mean(),params,allow_unused=True)
        grad_error=0.
        for (n,_),v,w in zip(a.named_parameters(),g,gg):
            require((v is None)==(w is None),'reference gradient connectivity differs: '+n)
            if v is not None:
                torch.testing.assert_close(v,w,atol=tol,rtol=5*tol,msg=n)
                grad_error=max(grad_error,float((v-w).abs().max()))
        reference_results.append(dict(dtype=str(dtype),maximum_logit_error=float((y-ref).abs().max().detach()),
            maximum_state_error=state_error,maximum_gradient_error=grad_error,absolute_tolerance=tol,status='PASS_FINITE_SOURCE_FIXTURES'))
    routing=[]
    if receipt['arm']!='attention':
        data,_=verify_data();captured={};handles=[]
        for i,b in enumerate(m.core.blocks):
            if hasattr(b.phase_memory,'router'):
                def record(module,args,out,i=i):
                    c=torch.bincount(out.argmax(-1).flatten(),minlength=out.shape[-1]).cpu()
                    captured[i]=captured.get(i,torch.zeros_like(c))+c
                handles.append(b.phase_memory.router.register_forward_hook(record))
        gen=torch.Generator().manual_seed(120099)
        with torch.inference_mode():
            for _ in range(16):
                xx,_,_=batch(data['dev'],gen,4,128,'cpu');m(xx)
        for h in handles:h.remove()
        routing=[dict(layer=i,observed_counts=c.tolist(),total=int(c.sum()),
            active_banks=int((c>0).sum()),qualification='observed hard routing on 64 exposed-DEV windows, not a capacity or semantic-specialization proof') for i,c in sorted(captured.items())]
    require(fingerprint(m)==before,'read-only audit changed weights')
    return dict(arm=receipt['arm'],seed=receipt['seed'],status='PASS_READ_ONLY_TRAINED_SOURCE_AUDIT',torch=torch.__version__,
        original_checkpoint_sha256=receipt['checkpoint_sha256'],completed_updates=1024,
        same_runtime_remote_replay_exact=json.loads((folder/'EVALUATION.json').read_text())['cold_replay_exact'],
        local_cross_runtime_maximum_logit_error=cross_error,cross_runtime_absolute_tolerance=2e-5,
        future_poison='PASS',owner_isolation='PASS',whole_graph_reference=reference_results,hard_bank_usage=routing,
        parameter_weights_unchanged=True,optimizer_updates=0,independently_qualified=False)

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    result=audit(a.run);a.output.parent.mkdir(exist_ok=True,parents=True);a.output.write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))

if __name__=='__main__':main()
