"""Prepared all-30 FIT receipt/checkpoint auditor. Never trains or opens DEV/TEST."""
import argparse,json,hashlib,sys
from pathlib import Path
import torch
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE));import models,reference
from exploratory_fit import authorized,sha,require

def audit(inputs,out,authorization):
 a=authorized(authorization);protocol=json.loads((HERE/'PROTOCOL.json').read_text());torch.set_num_threads(2)
 expected={(r['scale'],r['arm'],seed):r for r in protocol['arms'] for seed in protocol['seeds']}
 found={};reports=[]
 for path in sorted(inputs.rglob('FIT_RECEIPT.json')):
  r=json.loads(path.read_text());key=(r['scale'],r['arm'],r['seed'])
  require(key in expected and key not in found,'unexpected or duplicate model receipt')
  require(r['status']=='FIT_COMPLETED_UNQUALIFIED' and r['actual_steps']==2560,'fit did not complete')
  require(len(r['fit_loss_series'])==2560 and all(__import__('math').isfinite(x) for x in r['fit_loss_series']),'invalid loss series')
  require(r['protocol_sha256']==a['protocol_sha256'] and r['authorization_sha256']==sha(authorization),'receipt registration differs')
  require(r['observed_source_byte_presentations']==1310720 and not r['dev_access'] and not r['test_access'],'data boundary differs')
  spec=expected[key];geom={k:v for k,v in spec.items() if k in {'scale','target','arm','depth','width','ffn','d','c','k','v'}}
  m=models.build(geom).float().eval();count=sum(p.numel() for p in m.parameters());require(count==spec['trainable_parameters'],'live count differs')
  final=None;verified=[]
  for cp in r['checkpoints']:
   file=path.parent/cp['path'];require(sha(file)==cp['sha256'],'checkpoint hash differs')
   payload=torch.load(file,map_location='cpu',weights_only=True)
   require(payload['steps']==cp['steps'] and payload['steps'] in a['milestones'],'checkpoint milestone differs')
   require(payload['sample_windows_consumed']==payload['steps']*4,'checkpoint data position differs')
   require(payload['authorization_sha256']==sha(authorization) and payload['protocol_sha256']==a['protocol_sha256'],'checkpoint registration differs')
   m.load_state_dict(payload['model'],strict=True)
   require(all(p.dtype==torch.float32 and torch.isfinite(p).all().item() for p in m.parameters()),'invalid weights')
   opt=payload['optimizer'];require(len(opt['state'])==len(list(m.parameters())),'AdamW parameter coverage incomplete')
   for state in opt['state'].values():
    require(state['step'].item()==payload['steps'],'AdamW step count differs')
    require(torch.isfinite(state['exp_avg']).all().item() and torch.isfinite(state['exp_avg_sq']).all().item(),'nonfinite moments')
   verified.append({'steps':payload['steps'],'sha256':cp['sha256']})
   if payload['steps']==2560:final=payload
  require(sorted(x['steps'] for x in verified)==a['milestones'],'missing milestone checkpoints')
  require(final is not None,'final weights missing')
  m.load_state_dict(final['model']);before=models.preflight.parameter_digest(m)
  require(before!=r['initial_parameter_sha256'],'final equals seeded initial weights')
  tokens=(torch.arange(18).reshape(2,9)*17+3)%256
  actual=m(tokens);actual=actual[0] if isinstance(actual,tuple) else actual
  arm='tac_associative_stack' if key[1].startswith('tac') else key[1]
  predicted,_=reference.forward(arm,tokens,dict(m.named_parameters()),depth=spec['depth'])
  require(torch.allclose(actual,predicted,atol=2e-5,rtol=2e-5),'trained graph/reference mismatch')
  require(before==models.preflight.parameter_digest(m),'audit changed weights')
  reports.append({'scale':key[0],'arm':key[1],'seed':key[2],'parameters':count,'checkpoints':verified,
    'bounded_constructed_trained_graph_check':'PASS','original_first_fit_loss':r['fit_loss_series'][0],
    'last_128_changing_model_fit_loss_mean':sum(r['fit_loss_series'][-128:])/128})
  found[key]=True
 require(set(found)==set(expected),'campaign incomplete; missing fits: '+str(sorted(set(expected)-set(found))))
 out.write_text(json.dumps({'status':'ALL_30_EXPLORATORY_FITS_AUDITED_NOT_QUALIFIED','models':reports,
   'model_count':len(reports),'optimizer_updates_during_audit':0,'dev_test_access':False,
   'full_compute_parity':'OPEN','independent_scientific_qualification':False,'architecture_superiority_claim':False},indent=2)+'\n')
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--inputs',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
 p.add_argument('--authorization',type=Path,default=HERE/'EXPLORATORY_AUTHORIZATION.json');a=p.parse_args();audit(a.inputs,a.out,a.authorization)
