"""LANGMEM084 deterministic memory-channel operators; no learned new parameters."""
import hashlib
import itertools
import struct
import torch
from torch.nn import functional as F

VALUES=('native','bytebits')
KEYS=('native','context_hash')
UPDATES=('source','overwrite')
ARMS=tuple('_'.join(x) for x in itertools.product(VALUES,KEYS,UPDATES))
AGES=(8,32,128,256)

def arm_parts(arm):
    for values,key,update in itertools.product(VALUES,KEYS,UPDATES):
        if arm=='_'.join((values,key,update)):return values,key,update
    raise ValueError('unregistered arm')

def context_key(context):
    if type(context) is not bytes or len(context)!=4:raise ValueError('four observed previous bytes')
    digest=hashlib.sha256(b'LANGMEM084-key-v1\0'+context).digest()
    values=[((x+.5)/2**32)*2-1 for x in struct.unpack('<8I',digest)]
    return F.normalize(torch.tensor(values,dtype=torch.float32),dim=0)

def byte_book(dtype=torch.float32):
    bits=((torch.arange(256)[:,None]>>torch.arange(8)[None,:])&1).to(dtype)
    return (2*bits-1)/(8**.5)

def update_matrix(matrix,key,value,*,mode,rho,beta,epsilon):
    if matrix.shape!=(8,8) or key.shape!=(8,) or value.shape!=(8,):raise ValueError('native dimensions')
    if mode=='source':
        P=rho*(torch.eye(8,dtype=matrix.dtype)-beta*key[:,None]*key[None,:])
        result=matrix@P+epsilon*value[:,None]*key[None,:]
    elif mode=='overwrite':
        base=rho*matrix
        result=base+(value-base@key)[:,None]*key[None,:]
    else:raise ValueError('update mode')
    if not torch.isfinite(result).all():raise ValueError('nonfinite memory')
    return result

def decode(matrix,key,book):
    vector=matrix@key
    sims=book@F.normalize(vector,dim=0,eps=1e-12)
    return int(sims.argmax()),vector

def select_controls(prefix,keys,a,t,owner):
    """Past observed contexts only. No historical target or future-byte argument."""
    if len(prefix)!=t+1 or not 4<=a<t:raise ValueError('observed boundary')
    context=prefix[a-4:a]
    candidates=[b for b in range(4,t) if prefix[b-4:b]!=context]
    if not candidates:raise ValueError('no wrong context')
    stack=torch.stack([keys[b] for b in candidates]);q=keys[a]
    cosines=stack@q
    # Stable latest-clock tie rule, independent of source target.
    top=max(range(len(candidates)),key=lambda i:(float(cosines[i]),candidates[i]))
    digest=hashlib.sha256(f'LANGMEM084|{owner}|{t}|{a}'.encode()).digest()
    random_clock=candidates[int.from_bytes(digest[:8],'big')%len(candidates)]
    return candidates[top],random_clock,float(cosines[top])
