"""CODEVALUE003 fixed source schedule and inherited, unchanged model constructors."""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE.parent / 'codevalue002'))
import train as historical

REG = json.loads((HERE/'REGISTER.json').read_text())
ARMS = ('tac','gru','ssm')
SEEDS = (1701,1702,1703)
MILESTONES = (320,1280,2560,5120,10240)

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def read_sources(folder, partitions):
    folder = Path(folder)
    receipt = json.loads((folder/'SOURCE_RECEIPT.json').read_text())
    rows=[]
    for r in receipt['files']:
        if r['partition'] not in partitions: continue
        raw=(folder/r['file']).read_bytes()
        if hashlib.sha256(raw).hexdigest()!=r['sha256'] or len(raw)!=r['size']:
            raise ValueError('immutable source mismatch: '+r['file'])
        if historical.gitblob(raw)!=r['git_blob_sha1']:
            raise ValueError('publisher blob mismatch')
        rows.append((r,raw))
    if not rows: raise ValueError('missing source partition')
    return receipt,rows

def sampler(docs,seed,steps=10240):
    rng=np.random.default_rng(seed)
    names=sorted(docs)
    for step in range(steps):
        rows=[]
        for _ in range(8):
            owner=names[int(rng.integers(len(names)))]
            off=int(rng.integers(len(docs[owner])-129+1))
            rows.append((owner,off,docs[owner][off:off+129]))
        yield step,rows,rng.bit_generator.state

def update_digest(h,rows):
    for owner,off,raw in rows:
        h.update(owner.encode()+b'\0'+str(off).encode()+b'\0'+raw)

def planned_digests(docs,seed):
    h=hashlib.sha256(); result={}
    for step,rows,_ in sampler(docs,seed):
        update_digest(h,rows)
        if step+1 in MILESTONES: result[step+1]=h.hexdigest()
    return result

def make(arm,device):
    return historical.make(arm).to(device)

def cpu_tree(value):
    if isinstance(value,torch.Tensor):return value.detach().cpu()
    if isinstance(value,dict):return {k:cpu_tree(v) for k,v in value.items()}
    if isinstance(value,list):return [cpu_tree(v) for v in value]
    if isinstance(value,tuple):return tuple(cpu_tree(v) for v in value)
    return value

def checkpoint_path(out,arm,seed,step):
    return Path(out)/f'{arm}-{seed}'/f'MODEL-{step:06d}.pt'

def atomic_json(path,value):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n')
    tmp.replace(path)
