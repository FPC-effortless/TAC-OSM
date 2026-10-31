"""Source/data/count/checkpoint helpers; no optimizer or dataset execution."""
import hashlib
import importlib.util
import json
from pathlib import Path
import random
import math
import subprocess
import sys
import time
import resource
import numpy as np
import torch

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
SEEDS=(9901,9902,9903)
FAMILIES=('tac099',)
CONDITIONS=('carry','reset','donor','corrupt')

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()
def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def read(path):return json.loads(Path(path).read_text())
def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as f:json.dump(value,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n')
read_json=read
write_json=save
sha256=sha
def bytes_list(value,name,maximum):
    if not isinstance(value,list) or not 1<=len(value)<=maximum or any(type(x) is not int or not 0<=x<=255 for x in value):
        raise ValueError('legal bounded observed byte list required: '+name)
def cpu_tree(value):
    if isinstance(value,torch.Tensor):return value.detach().cpu()
    if isinstance(value,dict):return {k:cpu_tree(v) for k,v in value.items()}
    if isinstance(value,(tuple,list)):return type(value)(cpu_tree(v) for v in value)
    return value
def native_digest(states):
    h=hashlib.sha256()
    for state in states:
        h.update(json.dumps([state.arm,state.owners,state.clock]).encode())
        for name,t in sorted(state.tensors.items()):
            h.update(name.encode());h.update(str(tuple(t.shape)).encode());h.update(t.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()
def load(name):
    key='langtrain099_'+name
    if key in sys.modules:return sys.modules[key]
    spec=importlib.util.spec_from_file_location(key,HERE/(name+'.py'));m=importlib.util.module_from_spec(spec);sys.modules[key]=m;spec.loader.exec_module(m);return m
def make(seed):
    if seed not in SEEDS:raise ValueError('registered original seed required')
    with torch.random.fork_rng():
        torch.manual_seed(seed);m=load('model').ScaledContextualAssociativeTAC099()
    if m.counts()['trainable']!=10000000 or m.counts()['deployed']!=10000000:
        raise ValueError('actual exact10M scale gate failed')
    return m
def state_digest(m):
    h=hashlib.sha256()
    for n,p in sorted(m.state_dict().items()):
        h.update(n.encode());h.update(str(tuple(p.shape)).encode());h.update(p.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()
def source_hashes(commit):
    if not isinstance(commit,str) or len(commit)!=40:raise ValueError('exact implementation commit required')
    files=list(HERE.glob('*.py'))+[HERE/'REGISTER.json',HERE/'FAIR_COMPARISON.json',HERE/'PREREGISTRATION.md',
        HERE/'SCALE_AUTHORIZATION.md',ROOT/'scripts/validate_research_parity.py',ROOT/'research/language-scaling/lang10m/tokenizer/FIT_MANIFEST.json',
        ROOT/'research/mathematical-engineering/langacquire097/data.py',ROOT/'research/mathematical-engineering/langacquire097/SOURCE_PROVENANCE.json']
    result={}
    for p in files:
        name=p.relative_to(ROOT).as_posix();original=subprocess.check_output(['git','show',commit+':'+name],cwd=ROOT)
        if original!=p.read_bytes():raise ValueError('original source differs: '+name)
        result[name]=sha(p)
    return result
def hardware():
    import platform
    cpu=next((s.split(':',1)[1].strip() for s in Path('/proc/cpuinfo').read_text().splitlines() if s.startswith('model name')),'unknown')
    return {'cpu':cpu,'platform':platform.platform(),'torch':torch.__version__,'threads':torch.get_num_threads(),
            'peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024}
def setup(seed):
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True);torch.manual_seed(seed);np.random.seed(seed);random.seed(seed)

def load_fit(path,register):
    path=Path(path);manifest=read(path/'FIT_MANIFEST.json')
    if sha(path/'FIT_MANIFEST.json')!=register['data']['FIT_MANIFEST.json']:raise ValueError('frozen FIT manifest changed')
    if sha(path/'FIT_EPISODES.json')!=manifest['episodes_sha256']:raise ValueError('original episodic FIT bytes changed')
    corpus=[];windows=[]
    for r in manifest['sources']:
        p=path/r['file']
        if p.is_symlink() or sha(p)!=r['sha256'] or p.stat().st_size!=r['bytes']:raise ValueError('original FIT corpus bytes changed')
        raw=p.read_bytes();corpus.append(raw)
        for start in range(0,len(raw)-128,128):windows.append((len(corpus)-1,start,r['id']))
    episodes=read(path/'FIT_EPISODES.json')
    if any(e['partition']!='FIT' for e in episodes):raise ValueError('nonFIT episode in training package')
    if any(p.name.startswith('DEV') or p.name=='EVAL_MANIFEST.json' for p in path.rglob('*')):
        raise ValueError('evaluation bytes were delivered to fit worker')
    return corpus,windows,episodes,manifest

def schedule(seed,steps,windows,episodes):
    rng=np.random.default_rng(seed);lex=[i for i,e in enumerate(episodes) if e['domain']=='lexical'];arc=[i for i,e in enumerate(episodes) if e['domain']=='arc'];plan=[]
    for step in range(steps):
        if step%4==3:plan.append({'mode':'episodic','indices':[int(i) for i in np.concatenate((rng.choice(lex,2,replace=False),rng.choice(arc,2,replace=False)))]})
        else:plan.append({'mode':'language','indices':[int(i) for i in rng.integers(0,len(windows),4)]})
    return plan

def batch(item,corpus,windows,episodes):
    if item['mode']=='language':
        raw=[corpus[windows[i][0]][windows[i][1]:windows[i][1]+129] for i in item['indices']]
        values=torch.tensor([list(r) for r in raw],dtype=torch.long)
        return values[:,:-1],values[:,1:],None,None,None,{'observed':512,'supervised':512,'padding':0}
    rows=[episodes[i] for i in item['indices']];x=torch.zeros(4,272,dtype=torch.long);y=torch.zeros_like(x);valid=torch.zeros_like(x,dtype=torch.bool);future=torch.zeros_like(valid);domains=[]
    for i,r in enumerate(rows):
        a,q,t=r['acquisition_bytes'],r['query_bytes'],r['target_bytes'];raw=a+q+t;n=len(raw)-1
        if not 0<n<=272:raise ValueError('real episode exceeds frozen compute slots')
        x[i,:n]=torch.tensor(raw[:-1]);y[i,:n]=torch.tensor(raw[1:]);valid[i,:n]=True;future[i,len(a)+len(q)-1:n]=True
        domains.append('code' if r['domain']=='lexical' else 'arc')
    return x,y,future,valid,tuple(domains),{'observed':int(valid.sum()),'supervised':int(valid.sum()),'future_targets':int(future.sum()),'padding':int((~valid).sum())}

def loss(model,values):
    x,y,future,valid,domains,_=values;logits,_=model(x,owners=tuple('fit-'+str(i) for i in range(4)))
    if future is None:return torch.nn.functional.cross_entropy(logits.flatten(0,1),y.flatten())
    return load('objective').controlled_loss(logits,y,future,valid,domains,prefix_weight=.5)

def gradients(model):
    result={}
    for n,p in model.named_parameters():
        if p.grad is None or not bool(torch.isfinite(p.grad).all()):raise ValueError('missing/nonfinite gradient: '+n)
        result[n]=float(torch.linalg.vector_norm(p.grad))
        if not math.isfinite(result[n]):raise ValueError("nonfinite parameter gradient norm: "+n)
    return result

def checkpoint(path,model,optimizer,metadata):
    path=Path(path)
    if path.exists():raise FileExistsError('never overwrite original weights')
    torch.save({'schema':'langtrain099.checkpoint.v1','metadata':metadata,'model':{n:p.detach().cpu() for n,p in model.state_dict().items()},
                'optimizer':optimizer.state_dict() if optimizer else None,'rng':{'torch':torch.random.get_rng_state(),'python':random.getstate(),'numpy':{'algorithm':np.random.get_state()[0],'keys':torch.tensor(np.random.get_state()[1].astype(np.int64)),'position':int(np.random.get_state()[2]),'has_gauss':int(np.random.get_state()[3]),'cached_gaussian':float(np.random.get_state()[4])}}},path)
    return sha(path)
