"""Fixed CODEVALUE004 protocol: unchanged models, factorial data/schedule study."""
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys

import numpy as np
import torch

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
sys.path.insert(0,str(HERE.parent/'codevalue002'))
import train as original

SEEDS=(1701,1702,1703)
ARMS=('tac','gru')
CELLS=('small_constant','small_cosine','broad_constant','broad_cosine')
MILESTONES=(320,640,1280,2560,5120)
STEPS=5120
PARAMS={'tac':499623,'gru':500165}

def load_scoring():
    spec=importlib.util.spec_from_file_location('codevalue003_scoring_004',HERE.parent/'codevalue003/scoring.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module

scoring=load_scoring()

def digest(raw):return hashlib.sha256(raw).hexdigest()
def sha(path):return digest(Path(path).read_bytes())

def atomic_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n');tmp.replace(path)

def make(arm,device):
    if arm not in ARMS:raise ValueError('unregistered architecture')
    m=original.make(arm).to(device)
    if sum(p.numel() for p in m.parameters() if p.requires_grad)!=PARAMS[arm]:raise ValueError('actual parameter mismatch')
    return m

def cpu_tree(v):
    if isinstance(v,torch.Tensor):return v.detach().cpu()
    if isinstance(v,dict):return {k:cpu_tree(x) for k,x in v.items()}
    if isinstance(v,list):return [cpu_tree(x) for x in v]
    if isinstance(v,tuple):return tuple(cpu_tree(x) for x in v)
    return v

def read_sources(folder,partitions):
    folder=Path(folder);r=json.loads((folder/'SOURCE_RECEIPT.json').read_text());rows=[]
    for row in r['files']:
        if row['partition'] not in partitions:continue
        raw=(folder/row['file']).read_bytes()
        if sha(folder/row['file'])!=row['sha256'] or len(raw)!=row['size'] or original.gitblob(raw)!=row['git_blob_sha1']:
            raise ValueError('publisher bytes changed: '+row['file'])
        rows.append((row,raw))
    if not rows:raise ValueError('missing source partition')
    return r,rows

def corpus_docs(data,cell):
    if cell not in CELLS:raise ValueError('unregistered factorial cell')
    parts={'BASE_FIT'}|({'EXPANSION_FIT'} if cell.startswith('broad_') else set())
    receipt,rows=read_sources(data,parts)
    return receipt,{r['repo']+':'+r['path']:b for r,b in rows}

def batches(docs,seed,steps=STEPS):
    rng=np.random.default_rng(seed);names=sorted(docs)
    for i in range(steps):
        rows=[]
        for _ in range(8):
            owner=names[int(rng.integers(len(names)))];raw=docs[owner]
            off=int(rng.integers(len(raw)-129+1));rows.append((owner,off,raw[off:off+129]))
        yield i,rows,rng.bit_generator.state

def update_hash(h,rows):
    for owner,off,raw in rows:h.update(owner.encode()+b'\0'+str(off).encode()+b'\0'+raw)

def planned_hashes(docs,seed):
    h=hashlib.sha256();result={}
    for i,rows,state in batches(docs,seed):
        update_hash(h,rows)
        if i+1 in MILESTONES:result[i+1]=h.hexdigest()
    return result

def learning_rate(cell,step):
    if cell not in CELLS:raise ValueError("unregistered schedule")
    if not 0<=step<STEPS:raise ValueError('update index out of range')
    return .001 if cell.endswith('_constant') else .001*(.1+.9*.5*(1+math.cos(math.pi*step/(STEPS-1))))

def select_checkpoint(curve):
    if [r['step'] for r in curve]!=list(MILESTONES):raise ValueError('missing selection candidate')
    if any(not math.isfinite(r['nll']) for r in curve):raise ValueError('nonfinite DEV selection')
    return min(curve,key=lambda r:(r['nll'],r['step']))['step']

def folder_name(arm,cell,seed):return f'{arm}-{cell}-{seed}'

def path_for(out,arm,cell,seed,step):return Path(out)/folder_name(arm,cell,seed)/f'MODEL-{step:06d}.pt'
