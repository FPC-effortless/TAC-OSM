"""Original-source and provenance helpers; never executes a model on import."""
import hashlib,importlib.util,json,os,random,subprocess,sys
from pathlib import Path
import numpy as np
import torch
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
SEEDS=(9901,9902,9903);POLICIES=('native','recompute');PARENT='068ed9bd5bd68c19b0fd98dcff9a42197c1e4d21'

def load(name,path=None):
    key='langrecomp100_'+name
    if key in sys.modules:return sys.modules[key]
    spec=importlib.util.spec_from_file_location(key,path or HERE/(name+'.py'));m=importlib.util.module_from_spec(spec);sys.modules[key]=m;spec.loader.exec_module(m);return m
C=load('ancestor_common',ROOT/'research/mathematical-engineering/langtrain099/common.py')
read=C.read;save=C.save;sha=C.sha;digest=C.digest

def source_hashes(commit):
    paths=list(HERE.glob('*.py'))+[HERE/n for n in ('REGISTER.json','PREREGISTRATION.md','FAIR_COMPARISON.json','SCALE_AUTHORIZATION.md')]+[ROOT/'scripts/validate_research_parity.py']
    result={}
    for p in paths:
        name=p.relative_to(ROOT).as_posix()
        if subprocess.check_output(['git','show',commit+':'+name],cwd=ROOT)!=p.read_bytes():raise ValueError('actual100 source differs: '+name)
        result[name]=sha(p)
    return result

def verify_original_sources(metadata):
    for name,expected in metadata['source_hashes'].items():
        original=subprocess.check_output(['git','show',PARENT+':'+name],cwd=ROOT)
        if hashlib.sha256(original).hexdigest()!=expected:raise ValueError('original099 checkpoint/source binding differs')
        # The owner-directed100 count amendment changes this shared validator;
        # original model, loss, data and recipe bytes must remain identical.
        if name!='scripts/validate_research_parity.py' and (ROOT/name).read_bytes()!=original:
            raise ValueError('original099 source was altered: '+name)

def tensor_digest(tensors):
    h=hashlib.sha256()
    for name,tensor in sorted(tensors.items()):
        value=tensor.detach().cpu().contiguous();h.update(name.encode());h.update(str(tuple(value.shape)).encode());h.update(str(value.dtype).encode());h.update(memoryview(value.numpy()).cast('B'))
    return h.hexdigest()

def optimizer_digest(optimizer,model):
    rows={}
    for name,p in model.named_parameters():
        for field,t in optimizer.state[p].items():
            if not isinstance(t,torch.Tensor):raise ValueError('original AdamW scalar/tensor schema changed')
            rows[name+':'+field]=t
    return tensor_digest(rows)

def restore_rng(payload):
    r=payload['rng'];torch.random.set_rng_state(r['torch']);random.setstate(r['python'])
    n=r['numpy'];np.random.set_state((n['algorithm'],n['keys'].numpy().astype(np.uint32),n['position'],n['has_gauss'],n['cached_gaussian']))

def gradients(model):
    values={}
    for n,p in model.named_parameters():
        if p.grad is None or not bool(torch.isfinite(p.grad).all()):raise ValueError('missing/nonfinite complete gradient: '+n)
        values[n]=p.grad
    return tensor_digest(values),{n:float(p.norm()) for n,p in values.items()}

def loss(model,values,policy,counter=None):
    x,y,future,valid,domains,_=values;yhat,state=load('recompute').forward(model,x,owners=tuple('fit-'+str(i) for i in range(len(x))),policy=policy,counter=counter)
    value=torch.nn.functional.cross_entropy(yhat.flatten(0,1),y.flatten()) if future is None else C.load('objective').controlled_loss(yhat,y,future,valid,domains,prefix_weight=.5)
    return value,yhat,state

def parent_artifact(seed):
    index=read(ROOT/'research/mathematical-engineering/langtrain099/results/ORIGINAL_ARTIFACTS.json')
    return next(r for r in index['artifacts'] if r['name']=='langtrain099-original-seed-'+str(seed))

def load_parent(folder,seed):
    folder=Path(folder);receipt=read(folder/'FIT_RECEIPT.json');checkpoint=folder/'MODEL.pt'
    expected=read(HERE/'REGISTER.json')['parents'][str(seed)]
    if any(receipt[k]!=v for k,v in expected.items()):raise ValueError('original parent differs from frozen099 receipt')
    if sha(checkpoint)!=receipt['checkpoint_sha256']:raise ValueError('original trained099 archive changed')
    payload=torch.load(checkpoint,map_location='cpu',weights_only=True);meta=payload['metadata']
    if (meta['source_commit']!=PARENT or meta['seed']!=seed or meta['optimizer_steps']!=2048 or meta['status']!='SEALED_BEFORE_DEV'
        or meta['model_sha256']!=receipt['model_sha256']):raise ValueError('original trained099 identity differs')
    verify_original_sources(meta);C.setup(seed);model=C.make(seed);model.load_state_dict(payload['model'],strict=True)
    if C.state_digest(model)!=meta['model_sha256']:raise ValueError('actual original weights differ')
    optimizer=torch.optim.AdamW(model.parameters(),lr=3e-5,foreach=False,fused=False)
    optimizer.load_state_dict(payload['optimizer'])
    if len(optimizer.state)!=17 or any(float(s['step'])!=2048 for s in optimizer.state.values()):raise ValueError('all17 original2048-step moments required')
    for parameter in model.parameters():
        for field in ('exp_avg','exp_avg_sq'):
            moment=optimizer.state[parameter][field]
            if moment.shape!=parameter.shape or not bool(torch.isfinite(moment).all()):raise ValueError('actual original AdamW tensor differs')
    for group in optimizer.param_groups:
        group.update(lr=3e-5,foreach=False,fused=False)
    restore_rng(payload);return model,optimizer,payload,receipt
