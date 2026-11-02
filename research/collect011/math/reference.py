"""Independent scalar-loop mathematical reference, not an optimized kernel."""
import torch

def normalized_preconditioned_write(s,p,k,v,g):
    # Finite SPD domain, positive leverage, no clamp branch.
    vdim,kdim=s.shape
    z=torch.stack([sum(p[i,j]*k[j] for j in range(kdim)) for i in range(kdim)])
    leverage=sum(k[i]*z[i] for i in range(kdim))
    error=torch.stack([v[i]-sum(s[i,j]*k[j] for j in range(kdim)) for i in range(vdim)])
    ss=torch.stack([torch.stack([s[i,j]+g*error[i]*z[j]/leverage for j in range(kdim)]) for i in range(vdim)])
    pp=torch.stack([torch.stack([p[i,j]-g*z[i]*z[j]/(1+g*leverage) for j in range(kdim)]) for i in range(kdim)])
    return ss,(pp+pp.T)/2

def check():
    from collect011.memory import write_step
    torch.manual_seed(91);rows=[]
    for dtype in (torch.float32,torch.float64):
        s=torch.randn(3,6,dtype=dtype,requires_grad=True)
        a=torch.randn(6,6,dtype=dtype);p=(a@a.T+torch.eye(6,dtype=dtype)).requires_grad_()
        k=torch.randn(6,dtype=dtype,requires_grad=True);v=torch.randn(3,dtype=dtype,requires_grad=True)
        g=torch.tensor(.31,dtype=dtype,requires_grad=True)
        y,q=write_step(s,p,k,v,g);yr,qr=normalized_preconditioned_write(s,p,k,v,g)
        grad=torch.autograd.grad(y.square().sum()+q.square().sum(),(s,p,k,v,g),retain_graph=True)
        ref=torch.autograd.grad(yr.square().sum()+qr.square().sum(),(s,p,k,v,g))
        tol=1e-4 if dtype==torch.float32 else 1e-10
        torch.testing.assert_close(y,yr,atol=tol,rtol=tol);torch.testing.assert_close(q,qr,atol=tol,rtol=tol)
        for aa,bb in zip(grad,ref):torch.testing.assert_close(aa,bb,atol=tol,rtol=tol)
        rows.append(dict(dtype=str(dtype),forward_max_error=max(float((y-yr).abs().max().detach()),float((q-qr).abs().max().detach())),
            backward_max_error=max(float((aa-bb).abs().max()) for aa,bb in zip(grad,ref)),status='PASS_COMPONENT_ONLY'))
    return rows

if __name__=='__main__':
    import json
    from pathlib import Path
    x=check();Path('results/MATH_REFERENCE.json').write_text(json.dumps(x,indent=2));print(json.dumps(x,indent=2))
