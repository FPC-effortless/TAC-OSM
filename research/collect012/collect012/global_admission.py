"""Executed mathematical counterexamples and actual trained-router gradients."""
import hashlib,json
from pathlib import Path
import torch
from .common import ROOT,build
from .train import fingerprint
from collect011.memory import write_step

def ridge(k,v,lam):
    a=v.T@k;g=k.T@k+lam*torch.eye(k.shape[1],dtype=k.dtype)
    return torch.linalg.solve(g,a.T).T

def rls(k,v,lam):
    d=k.shape[1];s=k.new_zeros(v.shape[1],d);p=torch.eye(d,dtype=k.dtype)/lam;out=[]
    for key,value in zip(k,v):
        z=p@key;gain=z/(1+key@z)
        s=s+(value-s@key)[:,None]*gain[None,:]
        p=p-gain[:,None]*(key@p)[None,:];out.append(s.clone())
    return torch.stack(out)

def normalized(k,v):
    s=k.new_zeros(v.shape[1],k.shape[1]);p=torch.eye(k.shape[1],dtype=k.dtype)/.2
    for key,value in zip(k,v):s,p=write_step(s,p,key,value,k.new_tensor(.5))
    return s

def main():
    torch.set_num_threads(1);rows=[]
    protocol=json.loads((ROOT/'COLLECT013_PROTOCOL.json').read_text())
    for dtype in (torch.float32,torch.float64):
        for seed in protocol['seeds']:
            g=torch.Generator().manual_seed(seed);theta=torch.randn(16,4,generator=g,dtype=dtype)
            k=torch.stack((theta.cos(),theta.sin()),-1).flatten(-2)/2
            v=torch.randn(16,5,generator=g,dtype=dtype);lam=.2
            states=rls(k,v,lam);refs=torch.stack([ridge(k[:i],v[:i],lam) for i in range(1,17)])
            err=float((states-refs).abs().max());tol=2e-5 if dtype==torch.float32 else 1e-11
            assert err<tol
            perm=torch.randperm(16,generator=g);permuted=rls(k[perm],v[perm],lam)[-1]
            perr=float((states[-1]-permuted).abs().max());assert perr<tol
            norm=normalized(k,v);mismatch=float((norm-states[-1]).abs().max());assert mismatch>1e-3
            q=k[0];vv=v.clone();vv[8:]+=7
            causal_change=float(((rls(k,vv,lam)[3]-states[3])@q).abs().max());assert causal_change==0
            global_change=float(((ridge(k,vv,lam)-ridge(k,v,lam))@q).abs().max());assert global_change>1e-3
            orth=torch.eye(8,dtype=dtype);val=torch.randn(8,5,generator=g,dtype=dtype)
            shrink=float((ridge(orth,val,lam)@orth.T-val.T).abs().max());assert shrink>1e-3
            many=torch.randn(12,4,generator=g,dtype=dtype);true=torch.randn(5,4,generator=g,dtype=dtype);compatible=many@true.T
            fit=torch.linalg.lstsq(many,compatible).solution
            compatible_error=float((many@fit-compatible).abs().max());assert compatible_error<tol
            logits=torch.randn(200,4,generator=g,dtype=dtype)
            unchanged=all(torch.equal(logits.argmax(-1),(logits/t).softmax(-1).argmax(-1)) for t in (1.,2.,5.,10.));assert unchanged
            rows.append(dict(dtype=str(dtype),seed=seed,rls_prefix_batch_max_error=err,rls_permutation_max_error=perr,
                normalized_operator_ridge_mismatch=mismatch,causal_prefix_future_change=causal_change,global_future_change=global_change,
                positive_ridge_orthogonal_recall_error=shrink,compatible_N12_d4_reconstruction_error=compatible_error,
                temperature_only_hard_choices_unchanged=unchanged))
    folder=ROOT.parent/'collect012_downloads/hybrid_hard_p_12003'
    saved=torch.load(folder/'checkpoint_1024.pt',weights_only=True,map_location='cpu')
    replay=torch.load(folder/'REPLAY.pt',weights_only=True,map_location='cpu')
    model=build('hybrid_hard_p',12003);model.load_state_dict(saved['model']);before=fingerprint(model)
    router=[];handles=[]
    for b in model.core.blocks:
        handles.append(b.phase_memory.router.register_forward_hook(lambda module,args,out:router.append(out)))
    model(replay['tokens']);losses=[];usage=[]
    for logits in router:
        probs=logits.softmax(-1).reshape(-1,4);f=torch.nn.functional.one_hot(probs.argmax(-1),4).to(probs.dtype).mean(0)
        p=probs.mean(0);loss=4*(f*p).sum();losses.append(loss);usage.append(f.tolist())
    total=torch.stack(losses).mean();grads=torch.autograd.grad(total,[b.phase_memory.router.weight for b in model.core.blocks])
    for h in handles:h.remove()
    assert all(torch.isfinite(g).all() and g.abs().sum()>0 for g in grads);assert fingerprint(model)==before
    result=dict(protocol=protocol,status='PASS_EXECUTED_ADMISSION_FIXTURES_NOT_TRAINED_REPAIR',rows=rows,
        actual_trained_router=dict(checkpoint_sha256=hashlib.sha256((folder/'checkpoint_1024.pt').read_bytes()).hexdigest(),
            mean_unscaled_switch_loss=float(total.detach()),per_layer_assignment_fractions=usage,
            router_weight_gradient_l1=[float(g.abs().sum()) for g in grads],weights_unchanged=True),optimizer_updates=0)
    out=ROOT/'results/COLLECT013_ADMISSION.json';out.write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))

if __name__=='__main__':main()
