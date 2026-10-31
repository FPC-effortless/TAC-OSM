"""Read-only original continuation archive audit; NO neural forward or optimizer."""
from pathlib import Path
import argparse,hashlib,json,math
import numpy as np
import torch
p=argparse.ArgumentParser();p.add_argument('--artifacts-root',type=Path,required=True);p.add_argument('--run',type=int,required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
root=args.artifacts_root/str(args.run)
def digest(tensors):
 h=hashlib.sha256()
 for n,t in sorted(tensors.items()):
  v=t.detach().cpu().contiguous();h.update(n.encode());h.update(str(tuple(v.shape)).encode());h.update(str(v.dtype).encode());h.update(memoryview(v.numpy()).cast('B'))
 return h.hexdigest()
def file_sha(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for block in iter(lambda:f.read(1048576),b''):h.update(block)
 return h.hexdigest()
records=[];pairs=[]
for seed in (9901,9902,9903):
 loaded={};receipts={}
 for policy in ('native','recompute'):
  folder=root/('langrecomp100-original-seed-'+str(seed))/policy;r=json.loads((folder/'FIT_RECEIPT.json').read_text());checkpoint=folder/'MODEL.pt'
  assert file_sha(checkpoint)==r['checkpoint_sha256']
  payload=torch.load(checkpoint,map_location='cpu',weights_only=True);meta=payload['metadata'];model=payload['model'];opt=payload['optimizer']
  assert meta['optimizer_steps']==2064 and r['steps']==meta['new_optimizer_steps']==16 and meta['seed']==seed and meta['policy']==policy
  names=[n for n in model if n not in ('bank_min','bank_max')];assert len(names)==17 and sum(model[n].numel() for n in names)==10000000
  assert all(torch.isfinite(t).all() for t in model.values()) and digest(model)==r['history'][-1]['model_sha256']
  identifiers=[i for g in opt['param_groups'] for i in g['params']];assert len(identifiers)==len(opt['state'])==17
  moments={}
  for name,i in zip(names,identifiers):
   slot=opt['state'][i];assert float(slot['step'])==2064
   for field,value in slot.items():
    assert torch.isfinite(value).all()
    if field!='step':assert value.shape==model[name].shape
    moments[name+':'+field]=value
  assert digest(moments)==r['history'][-1]['optimizer_sha256']
  with np.load(folder/'COLD_ORIGINAL.npz',allow_pickle=False) as a,np.load(folder/'COLD_REPLAY.npz',allow_pickle=False) as b:
   assert set(a.files)==set(b.files) and all(np.array_equal(a[n],b[n]) for n in a.files)
  assert r['cold_replay_bitwise'] and r['DEV_opened'] is False and r['GPU_minutes']==0
  for step in r['history']:
   assert len(step['parameter_gradient_norms'])==17 and all(math.isfinite(v) for v in step['parameter_gradient_norms'].values())
  records.append({'seed':seed,'policy':policy,'status':'PASS_READ_ONLY_ORIGINAL_ARCHIVE','actual_parameters':10000000,'actual_optimizer_slots':17,'absolute_optimizer_steps':2064,'new_steps':16,'checkpoint_sha256':r['checkpoint_sha256'],'cold_actualFIT_replay':'BITWISE_IDENTICAL','source_commit':r['source_commit']})
  loaded[policy]=payload;receipts[policy]=r
 a,b=loaded['native'],loaded['recompute'];ra,rb=receipts['native'],receipts['recompute']
 assert a['optimizer']['param_groups']==b['optimizer']['param_groups']
 assert set(a['model'])==set(b['model']) and all(torch.equal(a['model'][n],b['model'][n]) for n in a['model'])
 for i in a['optimizer']['state']:
  assert all(torch.equal(t,b['optimizer']['state'][i][n]) for n,t in a['optimizer']['state'][i].items())
 assert a['rng']['python']==b['rng']['python'] and torch.equal(a['rng']['torch'],b['rng']['torch'])
 for key,value in a['rng']['numpy'].items():
  other=b['rng']['numpy'][key]
  assert torch.equal(value,other) if isinstance(value,torch.Tensor) else value==other
 for x,y in zip(ra['history'],rb['history']):
  for field in ('step','absolute_step','mode','loss','outputs_state_sha256','raw_all_parameter_gradient_sha256','parameter_gradient_norms','gradient_norm_before_clip','model_sha256','optimizer_sha256','torch_rng_sha256','exposure'):assert x[field]==y[field],(seed,x['step'],field)
 pairs.append({'seed':seed,'all16_original_full_gradient_digests_equal':True,'actual_final_all10M_parameter_tensor_elements_equal':True,'actual_final_all17_AdamW_slots_equal':True,'actual_final_python_numpy_torch_RNG_equal':True})
 del a,b,loaded
result={'status':'PASS_READ_ONLY_ALL6_ORIGINAL_CHECKPOINTS_AND3_PAIRS','run_id':args.run,'neural_forward_calls':0,'new_optimizer_steps':0,'records':records,'pairs':pairs,'qualification':'Internal original-artifact audit, not independent scientific qualification'}
args.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
