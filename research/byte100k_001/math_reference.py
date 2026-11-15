"""Independent partial arithmetic/objective reference, not native graph proof."""
import numpy as np


def parameter_count():
    d,e,r,f,l,v=48,8,4,96,4,256
    block=2*d+3*d+4*d*e+r*e+d*d+2*r*e+3*d*f+2+d*r+r
    attention=d+3*d*d+1  # q/out plus two grouped KV projections total3d²
    return v*d+l*block+2*attention+d


def mean_byte_ce_and_logit_gradient(logits,targets):
    x=np.asarray(logits,dtype=np.float64)
    y=np.asarray(targets)
    if x.ndim!=3 or x.shape[-1]!=256 or y.shape!=x.shape[:-1] or not np.issubdtype(y.dtype,np.integer):
        raise ValueError('byte objective layout')
    if not x.size or not np.isfinite(x).all() or np.any(y<0) or np.any(y>255):
        raise ValueError('byte objective values')
    centered=x-x.max(axis=-1,keepdims=True)
    exp=np.exp(centered);prob=exp/exp.sum(axis=-1,keepdims=True)
    flat=y.reshape(-1)
    nll=(np.log(exp.sum(axis=-1))-np.take_along_axis(centered,y[...,None],axis=-1)[...,0]).mean()
    grad=prob.copy().reshape(-1,256)
    grad[np.arange(len(flat)),flat]-=1
    return float(nll),grad.reshape(x.shape)/len(flat)
