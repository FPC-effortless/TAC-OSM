"""Read-only engineering probes. No data, targets, optimizer or capability scoring."""
import sys,json,hashlib,math,statistics,argparse
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'research/langcap103'))
import preflight
OUT=Path(__file__).parent

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def run(originals, output):
 torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
 p=json.loads((ROOT/'research/langcap105/OWNER_PROTOCOL.json').read_text())
 audit=json.loads((ROOT/'research/langcap105_verification/results/FINAL_CHECKPOINT_AUDIT.json').read_text())
 checks={s:sha(ROOT/s)==h for s,h in audit['verified_original_source_sha256'].items()}
 assert all(checks.values()),checks
 report={'classification':'ENGINEERING_ONLY_NO_CAPABILITY_EVIDENCE','torch':torch.__version__,'optimizer_updates':0,'real_data_access':False,'dev_test_access':False,'source_hash_checks':checks,'models':[]}
 for scale,arm,dirname in [('small','tac_associative_stack','small-tac-10301'),('medium','tac_associative_stack','medium-tac-10301'),('medium','stacked_gru','medium-gru-10301')]:
  spec=next(x for x in p['models'][scale]['configurations'] if x['arm']==arm)
  torch.manual_seed(10301)
  m=preflight.build({k:v for k,v in spec.items() if k!='trainable_parameters'},p['models'][scale]['target_parameters'],preflight.load_native(),preflight.comparator_classes(torch)).eval()
  checkpoint=originals/dirname/'TRAINED.pt';receipt=json.loads((checkpoint.parent/'FIT_RECEIPT.json').read_text())
  original_sha=sha(checkpoint)
  expected=next(r for r in audit['models'] if r['scale']==scale and r['arm']==arm and r['seed']==10301)
  assert original_sha==receipt['checkpoint_sha256']==expected['checkpoint_sha256']
  payload=torch.load(checkpoint,map_location='cpu',weights_only=True);m.load_state_dict(payload['model'],strict=True)
  counts={}
  for name,param in m.named_parameters():counts[name.split('.')[0]]=counts.get(name.split('.')[0],0)+param.numel()
  row={'scale':scale,'arm':arm,'checkpoint_sha256':original_sha,'parameter_allocation':counts,'total_parameters':sum(counts.values()),'checkpoint_unchanged':None}
  if arm=='tac_associative_stack':
   row['head_width']=m.head.width;row['persistent_shapes']=m.shapes
   row['state_floats']=m.start(('audit',)).tensor_bytes()//4
   row['nonlexical_state_floats']=row['state_floats']-3*m.d
   row['block_parameter_allocation']={name:param.numel() for name,param in m.blocks[0].named_parameters()}
   row['trained_context_horizons']=[[float(z) for z in (-torch.log(torch.sigmoid(-b.raw_decay))).reciprocal().detach()] for b in m.blocks]
  loss=receipt['fit_loss_series']
  row['original_loss_blocks_256']=[statistics.mean(loss[i:i+256]) for i in range(0,len(loss),256)]
  # Deterministic fixture tests inspect actual source but are not language scores.
  x=torch.arange(256,dtype=torch.long).reshape(2,128)
  def logits(x,**kw):
   y=m(x,**kw);return y[0] if isinstance(y,tuple) else y
  before=preflight.parameter_digest(m)
  with torch.no_grad():
   y=logits(x);poison=x.clone();poison[:,64:]=(poison[:,64:]+71)%256
   row['future_prefix_max_abs']=float((y[:,:64]-logits(poison)[:,:64]).abs().max())
   row['entity_max_abs']=float((y-torch.cat([logits(x[i:i+1]) for i in range(2)])).abs().max())
   if arm=='tac_associative_stack':
    row['parallel_serial_max_abs']=float((y-logits(x,parallel=False)).abs().max())
    a,s=m(x[:,:47],owners=('a','b'));b,s=m(x[:,47:],state=s,owners=('a','b'))
    row['chunk_carry_max_abs']=float((y-torch.cat((a,b),1)).abs().max())
  capture=[]
  def hook(module,inputs,out):capture.append(out)
  handle=m.embedding.register_forward_hook(hook)
  y=logits(x);objective=(y[:,-1]*torch.linspace(-1,1,256)).mean()
  grad=torch.autograd.grad(objective,capture[0])[0];handle.remove()
  row['fixture_embedding_gradient_norm_by_lag']={str(lag):float(grad[:,127-lag].norm()) for lag in (0,1,3,4,8,16,32,64,127)}
  if arm=='tac_associative_stack':
   m.zero_grad();yp=logits(x);yp.square().mean().backward();gp={n:v.grad.clone() for n,v in m.named_parameters()}
   m.zero_grad();ys=logits(x,parallel=False);ys.square().mean().backward()
   row['parallel_serial_gradient_relative_l2']=float(torch.sqrt(sum((v.grad-gp[n]).square().sum() for n,v in m.named_parameters())/sum(g.square().sum() for g in gp.values())))
   row['fixture_all_gradients_finite']=all(v.grad is not None and bool(torch.isfinite(v.grad).all()) for v in m.parameters())
  row['checkpoint_unchanged']=sha(checkpoint)==original_sha and preflight.parameter_digest(m)==before
  assert row['checkpoint_unchanged'] and row['future_prefix_max_abs']<1e-5 and row['entity_max_abs']<1e-4
  if arm=='tac_associative_stack':assert row['parallel_serial_max_abs']<1e-4 and row['chunk_carry_max_abs']<1e-4 and row['parallel_serial_gradient_relative_l2']<1e-4
  report['models'].append(row)
 output.write_text(json.dumps(report,indent=2)+'\n')
 print(json.dumps(report,indent=2))
if __name__=='__main__':
 parser=argparse.ArgumentParser(description=__doc__)
 parser.add_argument('--originals',type=Path,required=True)
 parser.add_argument('--out',type=Path,required=True)
 args=parser.parse_args();run(args.originals,args.out)
