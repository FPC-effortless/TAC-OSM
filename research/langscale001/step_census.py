"""Entire recipe operation census on constructed inputs; model weights never update.

Counts official Torch formula-covered FLOPs and lists every uncovered packet.
Uncovered packets never silently become full-compute qualification.
AdamW runs on disposable copied parameters, with copied fixture gradients.
"""
import argparse,collections,hashlib,json,sys
from pathlib import Path
import torch
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils.flop_counter import FlopCounterMode,flop_registry
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE));import models

def tensors(value):
 if isinstance(value,torch.Tensor):yield value
 elif isinstance(value,(list,tuple)):
  for x in value:yield from tensors(x)
 elif isinstance(value,dict):
  for x in value.values():yield from tensors(x)

class Census(TorchDispatchMode):
 def __init__(self):super().__init__();self.stage='unset';self.records={}
 def __torch_dispatch__(self,func,types,args=(),kwargs=None):
  kwargs=kwargs or {};out=func(*args,**kwargs)
  key=(self.stage,str(func));r=self.records.setdefault(key,{'calls':0,'floating_input_elements':0,'floating_output_elements':0,'official_formula_registered':func._overloadpacket in flop_registry})
  r['calls']+=1;r['floating_input_elements']+=sum(x.numel() for x in tensors((args,kwargs)) if x.is_floating_point())
  r['floating_output_elements']+=sum(x.numel() for x in tensors(out) if x.is_floating_point())
  return out

def run(output,only_scale=None):
 torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
 x=(torch.arange(512).reshape(4,128)*13+5)%256;y=(x+7)%256
 report={'schema':'langscale001.recipe-operation-census.v1','classification':'CONSTRUCTED_UNIT_MECHANICS_NOT_TRAINED_CAPABILITY',
  'torch':torch.__version__,'batch':4,'sequence_length':128,'real_data_access':False,'dev_test_access':False,
  'original_model_optimizer_updates':0,'disposable_tensor_optimizer_updates':0,'full_flop_qualification':False,
  'limits':['Torch counts shape/formula-defined theoretical FLOPs, not hardware instruction/energy counters',
    'Ordinary operators lacking official formulas contribute zero; every uncovered packet is retained',
    'First and steady AdamW moments are unit fixtures; not a trained optimizer trajectory',
    'Python scalar arithmetic/control/host bookkeeping is outside Torch dispatch coverage',
    'CPU traces do not qualify CUDA fused kernels or dual-T4 execution'], 'configurations':[]}
 for row in models.configurations():
  if only_scale and row['scale']!=only_scale:continue
  torch.manual_seed(11201);m=models.build(row).float().train();before=models.preflight.parameter_digest(m)
  c=Census();stage_flops={};previous=0
  with FlopCounterMode(display=False) as flops,c:
   c.stage='model_forward';z=m(x);z=z[0] if isinstance(z,tuple) else z
   current=flops.get_total_flops();stage_flops[c.stage]=current-previous;previous=current
   c.stage='cross_entropy';loss=torch.nn.functional.cross_entropy(z.reshape(-1,256),y.reshape(-1))
   assert torch.isfinite(loss)
   current=flops.get_total_flops();stage_flops[c.stage]=current-previous;previous=current
   c.stage='backward';loss.backward()
   current=flops.get_total_flops();stage_flops[c.stage]=current-previous;previous=current
   c.stage='gradient_finiteness_checks';assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters())
   current=flops.get_total_flops();stage_flops[c.stage]=current-previous;previous=current
  # Cloning and moment fixture setup are not charged as optimizer arithmetic.
  for mode in ('first_step','steady_step'):
   copies=[torch.nn.Parameter(p.detach().clone()) for p in m.parameters()]
   for q,p in zip(copies,m.parameters()):q.grad=p.grad.detach().clone()
   optimizer=torch.optim.AdamW(copies,lr=.0003,betas=(.9,.95),eps=1e-8,weight_decay=.1,foreach=False)
   if mode=='steady_step':
    for q in copies:optimizer.state[q]={'step':torch.tensor(127.),'exp_avg':torch.full_like(q,.001),'exp_avg_sq':torch.full_like(q,.01)}
   c.stage='adamw_disposable_'+mode
   with FlopCounterMode(display=False) as f,c:optimizer.step()
   stage_flops[c.stage]=f.get_total_flops()
   assert any(not torch.equal(q,p) for q,p in zip(copies,m.parameters()))
   report['disposable_tensor_optimizer_updates']+=1
   del copies,optimizer
  assert models.preflight.parameter_digest(m)==before
  records=[{'stage':stage,'operator':op,**r} for (stage,op),r in sorted(c.records.items())]
  uncovered=[r for r in records if not r['official_formula_registered'] and r['floating_input_elements']]
  report['configurations'].append({'configuration':row,'formula_covered_flops_by_stage':stage_flops,'operator_census':records,
   'uncovered_floating_packets':uncovered,'model_weights_unchanged':True,'full_flop_qualification':False})
  print(row['scale'],row['arm'],json.dumps(stage_flops),flush=True)
 report['status']='COMPLETE_RECIPE_TRACE_INCOMPLETE_FLOP_COVERAGE'
 output.write_text(json.dumps(report,indent=2)+'\n')
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--scale',choices=('small','medium'));a=p.parse_args();run(a.out,a.scale)
