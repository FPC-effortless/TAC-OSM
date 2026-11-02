"""Prepared FIT-only runner. Pending explicit owner exception; default is DENY.

This does not weaken or impersonate canonical scientific preflight. Qualified
training uses the separate signed gate. An owner-exploratory exception, if
explicitly authorized, permits FIT optimization only, never qualification.
"""
import argparse,hashlib,json,os,random,signal,sys,time,shutil
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
sys.path.insert(0,str(HERE))
SCHEMA='langscale001.explicit-owner-exploration.v1'
PHRASE='AUTHORIZE_LANGSCALE001_EXPLORATORY_FIT_ONLY_WITHOUT_INDEPENDENT_SIGNOFF'

def require(v,msg):
 if not v:raise ValueError(msg)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def authorized(path):
 a=json.loads(path.read_text())
 require(a.get('schema')==SCHEMA,'wrong owner-exploration schema')
 require(a.get('owner_authorized') is True and a.get('explicit_exception')==PHRASE,
         'BLOCKED: explicit owner exception for exploratory FIT-only training is not recorded')
 require(isinstance(a.get('authorization_record'),str) and len(a['authorization_record'])>=20,
         'explicit user authorization record is absent')
 require(a.get('protocol_sha256')==sha(HERE/'PROTOCOL.json'),'frozen protocol changed')
 require(a.get('independent_scientific_qualification') is False and a.get('dev_access') is False
         and a.get('test_access') is False,'qualification/evaluation boundary changed')
 for path,expected in a.get('code_sha256',{}).items():
  require(sha(ROOT/path)==expected,'frozen execution code changed: '+path)
 require(bool(a.get('code_sha256')),'execution source pins missing')
 require(a.get('optimizer_updates_per_fit')==2560 and a.get('max_parallel_jobs')==3,'budget changed')
 require(a.get('milestones')==[256,512,1024,2048,2560],'milestone retention changed')
 return a


def retain_resume_milestones(resume,output,completed,milestones):
 old=json.loads((resume.parent/'FIT_RECEIPT.json').read_text())
 copied=[]
 for cp in old['checkpoints']:
  if cp['steps'] not in milestones or cp['steps']>completed or not cp['path'].startswith('STEP_'):continue
  require(cp['path']=='STEP_'+str(cp['steps']).zfill(4)+'.pt','noncanonical resume milestone path')
  source=resume.parent/cp['path'];target=output/cp['path']
  require(sha(source)==cp['sha256'],'prior milestone digest differs')
  require(not target.exists(),'prior milestone target already exists')
  shutil.copy2(source,target);require(sha(target)==cp['sha256'],'prior milestone copy differs')
  copied.append(cp)
 require(sorted(c['steps'] for c in copied)==[v for v in milestones if v<=completed],'prior resume milestones missing')
 return copied

def fit(args):
 # This check precedes any dataset read, model construction, output creation or optimizer.
 a=authorized(args.authorization)
 import models
 sys.path.insert(0,str(ROOT/'research/langcap105'));import train_owner as original
 require(args.seed in (10301,10302,10303),'unregistered seed')
 require(not args.output.exists(),'refuse overwrite of prior evidence')
 protocol=json.loads((HERE/'PROTOCOL.json').read_text())
 rows=[r for r in protocol['arms'] if r['scale']==args.scale and r['arm']==args.arm]
 require(len(rows)==1,'unregistered model geometry')
 registered=rows[0];keys={'scale','target','arm','width','depth','ffn','d','c','k','v'}
 row={k:v for k,v in registered.items() if k in keys}
 require(row in models.configurations(),'registered geometry differs from actual factory')
 legacy=json.loads((ROOT/'research/langcap105/OWNER_PROTOCOL.json').read_text())
 data=original.verify_source(args.source_receipt,args.fit_archive,legacy)
 import torch
 require(torch.__version__=='2.15.0+cpu','use the frozen CPU Torch version')
 torch.set_num_threads(2);torch.manual_seed(args.seed);random.seed(args.seed)
 torch.use_deterministic_algorithms(True)
 m=models.build(row).float().cpu().train()
 require(sum(p.numel() for p in m.parameters())==registered['trainable_parameters'],'live parameter count changed')
 initial_digest=models.preflight.parameter_digest(m)
 optimizer=torch.optim.AdamW(m.parameters(),lr=.0003,betas=(.9,.95),eps=1e-8,weight_decay=.1,foreach=False)
 blocks=original.blocks_for(data,128);stream=original.training_stream(blocks,args.seed)
 args.output.mkdir(parents=True)
 started=time.monotonic();completed=0;losses=[];receipts=[];abort=[False]
 resume_receipt=None
 if args.resume is not None:
  require(args.resume_sha256 is not None and sha(args.resume)==args.resume_sha256,'resume checkpoint digest differs')
  saved=torch.load(args.resume,map_location='cpu',weights_only=True)
  require(all(saved[k]==v for k,v in [('arm',args.arm),('scale',args.scale),('seed',args.seed),('protocol_sha256',a['protocol_sha256']),('authorization_sha256',sha(args.authorization))]),'resume identity differs')
  require(saved['source_sha256']==legacy['source']['source_sha256'],'resume data differs')
  completed=saved['steps'];losses=saved['fit_loss_series'];require(0<completed<2560 and len(losses)==completed,'resume step/loss history invalid')
  require(saved['sample_windows_consumed']==completed*4,'resume order position invalid')
  m.load_state_dict(saved['model'],strict=True);optimizer.load_state_dict(saved['optimizer'])
  require(all(state['step'].item()==completed for state in optimizer.state.values()),'resume AdamW update counters differ')
  for _ in range(completed*4):next(stream)
  torch.set_rng_state(saved['rng_cpu']);random.setstate(saved['rng_python'])
  initial_digest=saved['initial_parameter_sha256'];resume_receipt={'sha256':args.resume_sha256,'steps':completed}
  receipts=retain_resume_milestones(args.resume,args.output,completed,a['milestones'])
 for sig in (signal.SIGTERM,signal.SIGINT):signal.signal(sig,lambda *_:abort.__setitem__(0,True))
 def checkpoint(name,status):
  target=args.output/name;temp=args.output/(name+'.tmp')
  require(not target.exists(),'refuse checkpoint overwrite')
  torch.save({'model':m.state_dict(),'optimizer':optimizer.state_dict(),'rng_cpu':torch.get_rng_state(),
   'rng_python':random.getstate(),'sample_windows_consumed':completed*4,'scale':args.scale,'arm':args.arm,'seed':args.seed,
   'steps':completed,'source_sha256':legacy['source']['source_sha256'],'protocol_sha256':a['protocol_sha256'],
   'fit_loss_series':losses,'initial_parameter_sha256':initial_digest,
   'authorization_sha256':sha(args.authorization),'scientific_status':'OWNER_AUTHORIZED_EXPLORATORY_FIT_NOT_INDEPENDENTLY_QUALIFIED'},temp)
  temp.replace(target);receipts.append({'path':target.name,'sha256':sha(target),'steps':completed,'status':status})
 def report(status):
  value={'status':status,'scale':args.scale,'arm':args.arm,'seed':args.seed,'actual_steps':completed,'fit_loss_series':losses,
   'source_sha256':legacy['source']['source_sha256'],'protocol_sha256':a['protocol_sha256'],'authorization_sha256':sha(args.authorization),
   'torch_version':torch.__version__,'cpu_threads':torch.get_num_threads(),
   'initial_parameter_sha256':initial_digest,'resume_checkpoint':resume_receipt,'checkpoints':receipts,'elapsed_seconds':time.monotonic()-started,
   'observed_source_byte_presentations':completed*512,'dev_access':False,'test_access':False,
   'full_training_flops':'NOT_QUALIFIED','independent_scientific_qualification':False,'architecture_superiority_claim':False}
  (args.output/'FIT_RECEIPT.json').write_text(json.dumps(value,indent=2)+'\n')
 try:
  for step in range(completed+1,2561):
   if time.monotonic()-started>=40*60:abort[0]=True
   if abort[0]:
    checkpoint('INTERRUPTED.pt','INTERRUPTED');report('INTERRUPTED_RETAINED');raise SystemExit(75)
   coordinates=[next(stream) for _ in range(4)]
   pieces=[data[n][offset:offset+129] for n,offset in coordinates]
   require(all(len(b)==129 for b in pieces),'cross-file target')
   x=torch.tensor([list(b[:-1]) for b in pieces]);y=torch.tensor([list(b[1:]) for b in pieces])
   optimizer.zero_grad(set_to_none=True);z=m(x);z=z[0] if isinstance(z,tuple) else z
   require(z.shape==(4,128,256),'unexpected logits')
   loss=torch.nn.functional.cross_entropy(z.reshape(-1,256),y.reshape(-1))
   require(torch.isfinite(loss).item(),'nonfinite FIT loss');loss.backward()
   require(all(p.grad is not None and torch.isfinite(p.grad).all().item() for p in m.parameters()),'missing/nonfinite parameter gradients')
   optimizer.step();completed=step;losses.append(float(loss.detach()))
   if step in a['milestones']:
    checkpoint('STEP_'+str(step).zfill(4)+'.pt','COMPLETED_MILESTONE');report('IN_PROGRESS')
   if step in (1,64,128,256,512,1024,2048,2560):print(json.dumps({'scale':args.scale,'arm':args.arm,'seed':args.seed,'step':step,'fit_loss':losses[-1]}),flush=True)
  report('FIT_COMPLETED_UNQUALIFIED')
 except Exception:
  checkpoint('FAILED.pt','FAILED_PARTIAL_STEP_POSSIBLE');report('FIT_FAILED_RETAINED');raise

def cli():
 p=argparse.ArgumentParser();p.add_argument('--authorization',type=Path,default=HERE/'EXPLORATORY_AUTHORIZATION.json')
 p.add_argument('--scale',choices=('small','medium'),required=True);p.add_argument('--arm',required=True);p.add_argument('--seed',type=int,required=True)
 p.add_argument('--resume',type=Path);p.add_argument('--resume-sha256')
 p.add_argument('--source-receipt',type=Path,required=True);p.add_argument('--fit-archive',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
 return p.parse_args()
if __name__=='__main__':fit(cli())
