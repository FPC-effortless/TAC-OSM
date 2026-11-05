"""Independent NumPy float64 CSL-125-R1 recurrence and boundedness reference.

This is not a trained-architecture equivalence proof; it checks the exact
weighted numerator/mass state law and normalization used in the new variant.
"""
from __future__ import annotations
import numpy as np


def serial_weighted_mean_memory(values, addresses, rates, initial=None, epsilon=1e-8):
    """values[B,T,E], addresses[B,T,R], rates[R,E] > 0.

    Returns read[B,T,E] and state[B,R,E,2] (numerator, nonnegative mass).
    The read is sum_j address_j * numerator_j/(mass_j + eps) and is
    always in [-1,1] when input values are in [-1,1] and state is empty.
    """
    values=np.asarray(values,dtype=np.float64)
    addresses=np.asarray(addresses,dtype=np.float64)
    rates=np.asarray(rates,dtype=np.float64)
    if values.ndim!=3 or addresses.shape[:2]!=values.shape[:2] or rates.shape!=(addresses.shape[-1],values.shape[-1]):
        raise ValueError('reference input dimensions mismatch')
    if (not np.isfinite(values).all() or not np.isfinite(addresses).all() or
        not np.isfinite(rates).all() or np.any(rates <= 0) or
        np.any(addresses<0) or np.any(np.abs(addresses.sum(axis=-1)-1)>2e-7)):
        raise ValueError('reference requires positive rates and normalized finite addresses')
    if not np.isfinite(epsilon) or epsilon<=0:raise ValueError('epsilon must be positive')
    b,t,e=values.shape;r=addresses.shape[-1]
    state=(np.zeros((b,r,e,2),dtype=np.float64) if initial is None
           else np.array(initial,dtype=np.float64,copy=True))
    if state.shape!=(b,r,e,2):raise ValueError('invalid initial state')
    decay=np.exp(-rates)[None,:,:]
    write=np.expm1(-rates)*(-1.0)
    reads=[]; history=[]
    for i in range(t):
        a=addresses[:,i,:,None]
        v=values[:,i,None,:]
        state[...,0]=decay*state[...,0]+write[None]*a*v
        state[...,1]=decay*state[...,1]+write[None]*a
        reads.append((addresses[:,i,:,None]*(state[...,0]/(state[...,1]+epsilon))).sum(axis=1))
        history.append(state.copy())
    return np.stack(reads,axis=1),np.stack(history,axis=1),state
