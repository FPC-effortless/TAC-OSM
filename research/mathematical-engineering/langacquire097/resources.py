"""Create-once receipts, source checks and finite CPU budgets for LANGACQUIRE097."""
from __future__ import annotations
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import random
import re
import subprocess
import sys
import time

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
SEEDS=(1701,1702,1703,1704,1705)
FAMILIES=('tac','gru','ssm','transformer')
OBJECTIVES=('full','query')
CONDITIONS=('carry','reset','donor','corrupt')


def load_local(name):
    qualified='langacquire097_'+name
    path=HERE/(name+'.py')
    if qualified in sys.modules:
        module=sys.modules[qualified]
        if Path(module.__file__).resolve()!=path.resolve(): raise ValueError('local source collision')
        return module
    spec=importlib.util.spec_from_file_location(qualified,path)
    module=importlib.util.module_from_spec(spec);sys.modules[qualified]=module
    try:spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(qualified,None);raise
    return module


def sha256(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda:handle.read(1024*1024),b''):digest.update(chunk)
    return digest.hexdigest()


def json_digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def read_json(path,max_bytes=8*1024*1024):
    path=Path(path)
    if path.stat().st_size>max_bytes:raise ValueError('oversized research JSON')
    return json.loads(path.read_text())


def write_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as handle:json.dump(value,handle,indent=2,sort_keys=True,allow_nan=False);handle.write('\n')


def bytes_list(value,label,maximum=4096):
    if not isinstance(value,list) or not 1<=len(value)<=maximum or any(type(v) is not int or not 0<=v<=255 for v in value):
        raise ValueError(label+' requires bounded nonempty original bytes')
    return value


def load_register(path):
    register=read_json(path)
    if (tuple(register.get('seeds',()))!=SEEDS or tuple(register.get('families',()))!=FAMILIES
            or tuple(register.get('objectives',()))!=OBJECTIVES):raise ValueError('frozen forty-model matrix differs')
    training=register.get('training',{})
    for key in ('steps','batch_size','context','max_arm_seconds'):
        if type(training.get(key)) is not int or training[key]<=0:raise ValueError('invalid training budget: '+key)
    leg_seconds=training.get('max_leg_seconds',training.get('max_seed_seconds'))
    if type(leg_seconds) is not int or not 1<=leg_seconds<=2400:
        raise ValueError('each objective leg requires a bounded CPU budget')
    by_family=training.get('max_arm_seconds_by_family')
    if (not isinstance(by_family,dict) or set(by_family)!=set(FAMILIES)
            or any(type(value) is not int or not 1<=value<=2400 for value in by_family.values())):
        raise ValueError('all four actual model-family CPU ceilings are required')
    if training['batch_size']!=4 or training['context']!=272:
        raise ValueError('registered bounded CPU geometry differs')
    for key in ('learning_rate','weight_decay','grad_clip'):
        value=training.get(key)
        if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) or value<0 or (key!='weight_decay' and value==0):raise ValueError('invalid optimizer '+key)
    if register.get('generation_limits')!={'lexical':4,'arc':40}:raise ValueError('fixed generation budgets required')
    if set(register.get('manifest_paths',{}))!=set(FAMILIES):raise ValueError('each family needs its own parity manifest')
    return register


def verify_source(commit,register_path):
    if re.fullmatch('[0-9a-f]{40}',commit or '') is None:raise ValueError('complete prospective implementation commit required')
    models=load_local('models')
    paths=set(HERE.glob('*.py'))|{Path(register_path).resolve(),HERE/'PREREGISTRATION.md'}
    paths.update(ROOT/p for p in models.source_digests())
    paths.update({ROOT/'src/tac_osm/research_objective_budget.py',
                  ROOT/'scripts/validate_research_parity.py'})
    register=load_register(register_path)
    paths.update(ROOT/p for p in register['manifest_paths'].values())
    receipts={}
    for path in sorted(paths):
        relative=path.resolve().relative_to(ROOT).as_posix()
        result=subprocess.run(['git','show',commit+':'+relative],cwd=ROOT,capture_output=True,check=True)
        if result.stdout!=path.read_bytes():raise ValueError('source changed since prospective commit: '+relative)
        receipts[relative]=sha256(path)
    return receipts


def assert_counts(model,expected=None):
    counts=model.counts()
    if not 95000<=counts['trainable']<=105000 or not 95000<=counts['deployed']<=105000:
        raise ValueError('actual model outside prospective 100K band')
    if expected is not None and (counts['trainable']!=expected or counts['deployed']!=expected):
        raise ValueError('actual model differs from frozen active/deployed parameter count')
    return counts


def model_digest(model):
    digest=hashlib.sha256()
    for name,tensor in sorted(model.state_dict().items()):
        digest.update(name.encode());digest.update(str(tuple(tensor.shape)).encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def state_digest(states):
    digest=hashlib.sha256()
    for state in states:
        digest.update(json.dumps([state.arm,state.owners,state.clock]).encode())
        for name,tensor in sorted(state.tensors.items()):
            digest.update(name.encode());digest.update(str(tuple(tensor.shape)).encode())
            digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def cpu_tree(value):
    import torch
    if isinstance(value,torch.Tensor):return value.detach().cpu()
    if isinstance(value,dict):return {k:cpu_tree(v) for k,v in value.items()}
    if isinstance(value,(tuple,list)):return type(value)(cpu_tree(v) for v in value)
    return value


def capture_rng():
    import numpy as np
    import torch
    current=np.random.get_state()
    return {'python':random.getstate(),'numpy':(current[0],current[1].tolist(),current[2],current[3],current[4]),
            'torch':torch.get_rng_state()}


def runtime_hardware():
    import platform
    import torch
    model='UNKNOWN'
    cpu=Path('/proc/cpuinfo')
    if cpu.exists():
        for line in cpu.read_text().splitlines():
            if line.startswith('model name'):
                model=line.split(':',1)[-1].strip();break
    return {'platform':platform.platform(),'cpu_model':model,'torch_version':torch.__version__,
            'torch_threads':torch.get_num_threads(),'cpu_devices':os.cpu_count(),'cuda_used':False}


def peak_rss_bytes():
    # Do not call this module resource.py: it would shadow the stdlib extension.
    import resource
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024


class Deadline:
    def __init__(self,seconds):self.started=time.perf_counter();self.expires=self.started+seconds
    def check(self):
        if time.perf_counter()>self.expires:raise TimeoutError('registered finite CPU budget exhausted')
    def remaining(self):return max(0.,self.expires-time.perf_counter())
