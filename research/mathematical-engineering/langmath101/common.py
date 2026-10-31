"""Shared exact-source LANGMATH101 internal pilot contracts; no auto fitting."""
from dataclasses import replace
import hashlib
import importlib.util
import json
from pathlib import Path
import platform
import random
import resource
import sys
import time
import numpy as np
import torch
from torch.nn import functional as F

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2] if HERE.parent.name=='mathematical-engineering' else HERE.parent
SEEDS=(10101,10102,10103)
RECIPE={'steps_parameter_lane':512,'batch_size':4,'dense_context':272,'real_language_context':128,
 'optimizer':'AdamW','learning_rate':.001,'betas':[.9,.999],'eps':1e-8,'weight_decay':.01,
 'grad_clip':1.,'prefix_weight':.5,'threads':2,'tuning_trials':0,'max_arm_seconds':2700,
 'counted_lane_min_steps':200,'counted_lane_max_steps':4096,'dtype':'FP32',
 'learning_rate_policy':'constant; same optimizer policy for variable-length counted runs'}

def load_module(name,path,package=False):
 spec=importlib.util.spec_from_file_location(name,path,submodule_search_locations=[str(path.parent)] if package else None)
 m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m
def local(name):
 key='langmath101_'+name
 if key not in sys.modules:load_module(key,HERE/(name+'.py'))
 return sys.modules[key]
def read(path):return json.loads(Path(path).read_text())
def save(path,value):
 p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('x') as f:json.dump(value,f,indent=2,allow_nan=False,sort_keys=True);f.write('\n')
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def source_hashes():
 paths=list(HERE.rglob('*.py'))+[HERE/'PROTOCOL.json']
 return {str(p.relative_to(ROOT)):sha(p) for p in sorted(paths) if not any(x in p.parts for x in ('results','profiles','tests','__pycache__')) and not p.name.startswith('test_')}
def state_digest(model):
 h=hashlib.sha256()
 for n,p in sorted(model.state_dict().items()):
  h.update(n.encode());h.update(str(tuple(p.shape)).encode());h.update(p.detach().cpu().contiguous().numpy().tobytes())
 return h.hexdigest()
def setup(seed):
 torch.set_num_threads(2);torch.manual_seed(seed);np.random.seed(seed);random.seed(seed);torch.use_deterministic_algorithms(True)
def factory():
 key='langmath101_candidate_family'
 if key not in sys.modules:load_module(key,HERE/'candidates/__init__.py',True)
 return sys.modules[key].models
def adaptive():
 key='langmath101_adaptive_family'
 if key not in sys.modules:load_module(key,HERE/'adaptive/__init__.py',True)
 return sys.modules[key].models
def arm_names():return [x for x in factory().catalog() if x!='factorized098']+list(adaptive().MODES)
def make_model(arm,tokenizer_receipt=None):
 if arm.startswith('a5_'):
  tok=adaptive().FrozenBPE.from_receipt(tokenizer_receipt) if arm=='a5_fixed_bpe' else None
  return adaptive().make_model(arm,tokenizer=tok)
 return factory().make_model(arm)
def load_model(payload):
 meta=payload['metadata'];m=make_model(meta['arm'],meta.get('tokenizer_receipt'));m.load_state_dict(payload['model'],strict=True)
 if meta.get('model_sha256'):assert state_digest(m)==meta['model_sha256'],'loaded weights identity differs'
 return m.eval()
def padded_batch(bundle,item):
 v=local('data').batch(bundle,item);x,y=v['tokens'],v['targets']
 if x.shape[1]==272:return v
 assert x.shape==(4,128)
 out={**v};out['tokens']=F.pad(x,(0,144));out['targets']=F.pad(y,(0,144))
 valid=torch.zeros((4,272),dtype=torch.bool);valid[:,:128]=True
 out['valid']=valid;out['future']=valid.clone();out['domains']=('language',)*4
 out['counts']={**v['counts'],'padding':576};return out
def loss(model,v):
 logits,_=model(v['tokens'],owners=tuple('FIT-'+str(i) for i in range(4)))
 ce=F.cross_entropy(logits.flatten(0,1),v['targets'].flatten(),reduction='none').reshape(4,272)
 valid,future=v['valid'],v['future'];prefix=valid&~future
 per=(ce*future).sum(1)/future.sum(1).clamp_min(1)
 per=per+RECIPE['prefix_weight']*(ce*prefix).sum(1)/prefix.sum(1).clamp_min(1)
 result=per.mean()
 if not bool(torch.isfinite(result)):raise ValueError('nonfinite original-source loss')
 return result
def exposures(bundle,plan):
 counts={'steps':len(plan),'examples_seen':len(plan)*4,'tokens_seen':0,'dense_compute_tokens':len(plan)*4*272}
 for item in plan:counts['tokens_seen']+=padded_batch(bundle,item)['counts']['observed']
 return counts
def hardware():
 cpu=next((s.split(':',1)[1].strip() for s in Path('/proc/cpuinfo').read_text().splitlines() if s.startswith('model name')),'UNKNOWN')
 return {'cpu':cpu,'platform':platform.platform(),'torch':torch.__version__,'threads':torch.get_num_threads(),
  'peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,'energy':'NOT_MEASURED'}
def checkpoint(path,model,optimizer,metadata):
 p=Path(path)
 if p.exists():raise FileExistsError('Never overwrite original checkpoint')
 torch.save({'model':{n:v.detach().cpu() for n,v in model.state_dict().items()},
  'optimizer':optimizer.state_dict() if optimizer else None,'metadata':metadata,
  'rng':{'torch':torch.random.get_rng_state(),'python':random.getstate(),
         'numpy_keys':torch.tensor(np.random.get_state()[1].astype(np.int64))}},p)
 return sha(p)
