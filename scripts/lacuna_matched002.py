#!/usr/bin/env python3
"""Frozen FIT repeated-context attribution; future labels never select the cohort."""
import argparse,json,time,resource,sys
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.lacuna_survival001 import ROOT,OWNERS,patch,nll,decode
from scripts.replay_langmem086087 import fit_sources,load_source,original_checkpoints,weights
BINS=((8,32),(32,128),(128,256),(256,513))

def select_history(history,context,clock,lo,hi):
 return max((j for j in history.get(context,[]) if lo<=clock-j<hi),default=None)

@torch.no_grad()
def run(out):
 if out.exists():raise FileExistsError('original002result exists')
 torch.set_num_threads(2);manifest=json.loads((ROOT/'research/replication/langmem086087/MANIFEST.json').read_text());docs,sha=fit_sources(ROOT)
 if sha!=manifest['fit_source_sha256']:raise ValueError('original FIT source')
 cls=load_source(ROOT,manifest);rows=[];availability=[];checks=[];started=time.process_time()
 for checkpoint,cp,fit in original_checkpoints(ROOT,manifest):
  if cp['steps']!=640:continue
  seed=cp['seed'];model=cls().eval();model.load_state_dict(cp['model']);before=weights(model)
  for owner in OWNERS:
   source=docs[owner];state=model.start((owner,));history={};keys={};pool=torch.empty((0,8,8),dtype=torch.float64);initial_norm={};maxerror=0.
   for t in range(2048):
    token=torch.tensor([[source[t]]]);prior=state;logits,state,tr=model(token,state=state,owners=(owner,),return_trace=True);f=tr[0]
    e=(model.embedding(token)+model.position(model.positions(t,1,1,torch.float32,token.device))).tanh();eta=float((.01*model.slow_gate(e).sigmoid())[0,0,0]);rho=float(.995+.004*model.slow_decay.sigmoid());key=f['writer'][0].double();value=f['value'][0].double();pending=[]
    if t>=128 and t%8==0:
     context=source[t-3:t+1]
     for lo,hi in BINS:
      j=select_history(history,context,t,lo,hi);availability.append({'seed':seed,'owner':owner,'clock':t,'bin_begin':lo,'bin_end_exclusive':hi,'available':j is not None})
      if j is None:continue
      choices=[b for b in range(max(4,j-64),min(t,j+65)) if 5<=abs(b-j)<=64 and not b-4<=j<b and source[b-4:b]!=context]
      if not choices:
       rows.append({'seed':seed,'owner':owner,'clock':t,'bin_begin':lo,'historical_clock':j,'status':'INCONCLUSIVE_NO_CLEAN_WRONG_EVENT'});continue
      event=pool[j-1];norm=float(event.norm());wrong=min(choices,key=lambda b:abs(float(pool[b-1].norm())-norm));control=pool[wrong-1]
      if min(norm,float(control.norm()))<1e-12:raise ValueError('recorded event norm underflow')
      control=control*(norm/float(control.norm()))
      changed=model(token,state=patch(prior,prior.tensors['slow_memory']-event.float()[None]),owners=(owner,))[0];wronglog=model(token,state=patch(prior,prior.tensors['slow_memory']-control.float()[None]),owners=(owner,))[0]
      transported=rho*(event-eta*(event@key)[:,None]*key[None,:]);keyerror=float((state.tensors['previous_writer'][0].double()-keys[j]).abs().max())
      pending.append(({'seed':seed,'owner':owner,'clock':t,'bin_begin':lo,'bin_end_exclusive':hi,'historical_clock':j,'age':t-j,'wrong_clock':wrong,'status':'MEASURED','writer_parity_error':keyerror,'current_query_dot_matching_writer':float(f['query'][0].double()@keys[j]),'norm_survival':float(transported.norm())/initial_norm[j],'fullmatrix_historical_byte_correct':int(decode(state.tensors['slow_memory'][0].double()@keys[j],model.byte_codes.double())==source[j])},changed,wronglog))
    if t>0:
     pool=rho*(pool-eta*(pool@key)[:,:,None]*key[None,None,:]);new=eta*value[:,None]*key[None,:];pool=torch.cat((pool,new[None]),0);keys[t]=key.clone();initial_norm[t]=float(new.norm())
    if t>=4:history.setdefault(source[t-4:t],[]).append(t)
    maxerror=max(maxerror,float((pool.sum(0)-state.tensors['slow_memory'][0].double()).abs().max()))
    if maxerror>1e-5:raise ValueError('original path transport')
    # Target first enters scorer after ALL context/control selection and prediction.
    y=source[t+1]
    for row,changed,wronglog in pending:
     row.update(historical_continuation_matches_nextbyte=int(source[row['historical_clock']]==y),old_event_removal_minus_full_nll=nll(changed,y)-nll(logits,y),wrong_event_removal_minus_full_nll=nll(wronglog,y)-nll(logits,y),full_nll=nll(logits,y));rows.append(row)
   checks.append({'seed':seed,'owner':owner,'source_sum_error':maxerror})
  if weights(model)!=before:raise ValueError('frozen weights changed')
 summary=[];passed=[]
 for lo,hi in BINS:
  allrows=[x for x in rows if x['bin_begin']==lo];rr=[x for x in allrows if x['status']=='MEASURED'];cells={(x['owner'],x['clock']) for x in rr}
  m={k:sum(x[k] for x in rr)/len(rr) if rr else None for k in ('historical_continuation_matches_nextbyte','old_event_removal_minus_full_nll','wrong_event_removal_minus_full_nll','writer_parity_error','current_query_dot_matching_writer','norm_survival','fullmatrix_historical_byte_correct')}
  means=[sum(x['old_event_removal_minus_full_nll'] for x in rr if x[k]==v)/sum(x[k]==v for x in rr) for k,vals in [('seed',(1701,1702,1703)),('owner',OWNERS)] for v in vals if any(x[k]==v for x in rr)]
  gate='INCONCLUSIVE_LOW_SUPPORT' if len(cells)<20 else 'NOT_PASSED'
  if gate!='INCONCLUSIVE_LOW_SUPPORT' and m['old_event_removal_minus_full_nll']>=.001 and m['old_event_removal_minus_full_nll']>m['wrong_event_removal_minus_full_nll'] and len(means)==5 and min(means)>0:gate='PROVISIONAL_POSITIVE_FIT_ONLY';passed.append(lo)
  summary.append({'bin_begin':lo,'bin_end_exclusive':hi,'measurement_count':len(rr),'distinct_source_query_cells':len(cells),'unavailable_or_no_controls':sum(x['bin_begin']==lo for x in availability)-len(rr),'metrics':m,'utility_screen':gate})
 result={'study':'LACUNA-MATCHED002','status':'PASS_FROZEN_FIT_COHORT_SCREEN','cohort_selected_from_observed_prefix_only':True,'same_source_cells_reused_across_seeds':True,'summary':summary,'rows':rows,'availability':availability,'source_checks':checks,'new_training_steps':0,'GPU_minutes':0,'DEV_scoring':False,'protected_TEST':False,'cpu_seconds':time.process_time()-started,'peak_RSS_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,'provisional_positive_bins':passed,'original001_gate_unchanged':True,'independently_qualified':False}
 out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(result,indent=2,sort_keys=True,allow_nan=False)+'\n');print(json.dumps({'summary':summary,'cpu_seconds':result['cpu_seconds'],'positive_bins':passed}))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();run(a.output)
