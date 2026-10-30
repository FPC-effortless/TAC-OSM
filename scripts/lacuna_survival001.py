#!/usr/bin/env python3
"""Frozen authentic FIT mechanism screens; no fitting or held-out scoring."""
from __future__ import annotations
import argparse,hashlib,json,math,struct,sys,time,importlib.util,resource
from dataclasses import replace
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.replay_langmem086087 import fit_sources,load_source,original_checkpoints,weights,digest
from scripts.langmem086_run import PackedRing086
REG=ROOT/'research/mathematical-engineering/lacuna_survival001'
if digest((REG/'pnds087_reference.py').read_bytes())!='ee342c6c444c7729c7187ec95922e0a0a3fdaa393e023b6820ad3fa80a949653':raise ValueError('pinned independent PNDS reference changed')
spec=importlib.util.spec_from_file_location('pnds087_reference',REG/'pnds087_reference.py');pnds=importlib.util.module_from_spec(spec);spec.loader.exec_module(pnds)
AGES=(8,32,128,256,512);OWNERS=('Doc/tutorial/appendix.rst','Doc/tutorial/classes.rst')

class PackedContexts:
 """Exactly256B:24Bheader +25*9Bslots +7Bchargedunused; no external history."""
 def __init__(self,policy,seed):
  if policy not in ('fifo','reservoir','diversity'):raise ValueError('policy')
  self.policy=policy;self.data=bytearray(256);struct.pack_into('<IHHQ4sI',self.data,0,0,0,0,seed|1,b'\0'*4,0)
 def slots(self,count):return [struct.unpack_from('<IBI',self.data,24+i*9) for i in range(count)]
 def append(self,value):
  clock,cursor,count,rng,context,_=struct.unpack_from('<IHHQ4sI',self.data);idx=None
  if clock>=4:
   key=int.from_bytes(context,'little');entries=self.slots(count)
   matching=[i for i,(k,v,t) in enumerate(entries) if k==key]
   if self.policy=='diversity' and matching:idx=matching[0]
   elif count<25:idx=count;count+=1
   elif self.policy=='reservoir':
    rng^=(rng<<13)&((1<<64)-1);rng^=rng>>7;rng^=(rng<<17)&((1<<64)-1);rng&=(1<<64)-1
    draw=rng%(clock-3);idx=draw if draw<25 else None
   else:idx=cursor;cursor=(cursor+1)%25
   if idx is not None:struct.pack_into('<IBI',self.data,24+idx*9,key,value,clock)
  struct.pack_into('<IHHQ4sI',self.data,0,clock+1,cursor,count,rng,context[1:]+bytes([value]),0)
 def predict(self,prior):
  clock,cursor,count,rng,context,_=struct.unpack_from('<IHHQ4sI',self.data);key=int.from_bytes(context,'little');counts=np.zeros(256)
  entries=self.slots(count)
  for k,v,t in entries:
   if k==key:counts[v]+=1
  return (counts+8*prior)/(counts.sum()+8),int(counts.sum()),[clock-1-t for k,v,t in entries]


def patch(state,matrix):return replace(state,tensors={**state.tensors,'slow_memory':matrix})
def nll(logits,y):return -float(logits[0,0].log_softmax(-1)[y])
def decode(vector,codes):return int((codes@vector).argmax())

@torch.no_grad()
def run(out):
 if out.exists():raise FileExistsError('original screen output exists')
 torch.set_num_threads(2);mfest=json.loads((ROOT/'research/replication/langmem086087/MANIFEST.json').read_text());docs,sha=fit_sources(ROOT)
 if sha!=mfest['fit_source_sha256']:raise ValueError('original FIT source')
 cls=load_source(ROOT,mfest);events=[];policy_rows=[];source_checks=[];ad_checks=[];started=time.process_time()
 cps=[x for x in original_checkpoints(ROOT,mfest) if x[0]['steps']==640]
 if len(cps)!=3:raise ValueError('three original640 checkpoints')
 for checkpoint,cp,fit in cps:
  seed=checkpoint['seed'];model=cls().eval();model.load_state_dict(cp['model'],strict=True);before=weights(model)
  if model.counts()['trainable']!=100521:raise ValueError('100521 original parameters')
  prior=(np.array(cp['fit_byte_counts'],dtype=np.float64)+.5)/(640*1024+128);modal=int(prior.argmax())
  for owner in OWNERS:
   raw=docs[owner];state=model.start((owner,));ring=PackedRing086(1928);caches={p:PackedContexts(p,seed) for p in ('fifo','reservoir','diversity')}
   pool=torch.empty((0,8,8),dtype=torch.float64);vals={};keys={};contexts={};flags=bytearray(2);sum_error=0.;current_term_error=0.;normal_wall=0.;reference_error=0.;cov=np.eye(8);kalman=np.zeros((8,8));min_eigen=1.;min_denom=1.;offdiag_max=0.;calls={p:0 for p in ('always','random','entropy','survival_proxy')};look_wall={p:0. for p in calls};contradictions=0;erased_rows=[]
   for t in range(1024):
    token=torch.tensor([[raw[t]]]);oldstate=state;tic=time.perf_counter();logits,state,tr=model(token,state=state,owners=(owner,),return_trace=True);normal_wall+=time.perf_counter()-tic;f=tr[0]
    e=(model.embedding(token)+model.position(model.positions(t,1,1,torch.float32,token.device))).tanh();eta=float((.01*model.slow_gate(e).sigmoid())[0,0,0]);rho=float(.995+.004*model.slow_decay.sigmoid());k=f['writer'][0].double();v=f['value'][0].double();q=f['query'][0].double()
    kk=k.numpy();vv=v.numpy();pred=cov+1e-4*np.eye(8);pk=pred@kk;denom=.1+float(kk@pk);gain=pk/denom;kalman=rho*kalman+np.outer(vv-rho*kalman@kk,gain);cov=pred-np.outer(pk,pk)/denom;cov=.5*(cov+cov.T);min_eigen=min(min_eigen,float(np.linalg.eigvalsh(cov).min()));min_denom=min(min_denom,denom);offdiag_max=max(offdiag_max,float(np.linalg.norm(cov-np.diag(np.diag(cov)))))
    if min_eigen< -1e-10 or min_denom<=0:raise ValueError('actual-key Kalman PSD/denominator')
    ring.append(raw[t])
    for c in caches.values():c.append(raw[t])
    staged_events=[]
    if t>0 and t%128==0:
     hh,cc=pnds.split_slow(oldstate.tensors['slow_memory'][0].double().tolist(),k.tolist(),v.tolist(),rho,eta)
     expected=torch.tensor(hh,dtype=torch.float64)+torch.tensor(cc,dtype=torch.float64);reference_error=max(reference_error,float((expected-state.tensors['slow_memory'][0].double()).abs().max()))
     if reference_error>1e-6:raise ValueError('independent PNDS/source slow step mismatch')
    # pool[j-1] is contribution from observed event j in ORIGINAL previous S.
    if t>=576 and t%64==0:
     for age in AGES:
      j=t-age;prev=pool[j-1].clone();choices=[b for b in range(max(1,j-16),min(t,j+17)) if 5<=abs(b-j)<=16 and not b-4<=j<b]
      if not choices:raise ValueError('registered wrong-event controls absent')
      norm=float(torch.linalg.vector_norm(prev));wrong=min(choices,key=lambda b:abs(float(torch.linalg.vector_norm(pool[b-1]))-norm));control=pool[wrong-1].clone();cn=float(torch.linalg.vector_norm(control))
      if min(norm,cn)<1e-12:raise ValueError('INCONCLUSIVE zero event norm; do not silently omit')
      control*=norm/cn
      changed=model(token,state=patch(oldstate,oldstate.tensors['slow_memory']-prev.float()[None]),owners=(owner,))[0]
      wronglog=model(token,state=patch(oldstate,oldstate.tensors['slow_memory']-control.float()[None]),owners=(owner,))[0]
      now=rho*(prev-eta*(prev@k)[:,None]*k[None,:]);selfkey=keys[j];source_value=vals[j];initial=source_value[:,None]*selfkey[None,:]
      # Read of whole compressed matrix at supplied historical key is DIAGNOSTIC ONLY.
      fullread=state.tensors['slow_memory'][0].double()@selfkey;isolated=now@selfkey
      staged_events.append(({'seed':seed,'owner':owner,'clock':t,'age':age,'historical_clock':j,'wrong_clock':wrong,'target_byte':raw[j],
       'survival_ratio':float(torch.linalg.vector_norm(now))/float(torch.linalg.vector_norm(initial)),'query_projection_norm':float(torch.linalg.vector_norm(now@q)),
       'historical_key_projection_norm':float(torch.linalg.vector_norm(isolated)),'full_matrix_byte_correct':int(decode(fullread,model.byte_codes.double())==raw[j]),
       'isolated_event_byte_correct_ORACLE':int(decode(isolated,model.byte_codes.double())==raw[j]),'FIT_modal_correct':int(raw[j]==modal),
       'max_logit_change':float((changed-logits).abs().max())},changed,wronglog))
    if t>0:
     pool=rho*(pool-eta*(pool@k)[:,:,None]*k[None,None,:]);addition=eta*v[:,None]*k[None,:];pool=torch.cat((pool,addition[None]),0);keys[t]=k.clone();vals[t]=eta*v.clone()
    actual=state.tensors['slow_memory'][0].double();sum_error=max(sum_error,float((pool.sum(0)-actual).abs().max()) if t else float(actual.abs().max()))
    if sum_error>1e-5:raise ValueError('source exact-path event decomposition failed')
    if t==576:
     # Actual trained full-source local AD, without any weight update/backward.
     with torch.enable_grad():
      label=int(logits[0,0].argmax())
      leaf=oldstate.tensors['slow_memory'].detach().clone().requires_grad_(True);loss=model(token,state=patch(oldstate,leaf),owners=(owner,))[0].log_softmax(-1)[0,0,label]
      grad=torch.autograd.grad(loss,leaf)[0];direction=torch.ones_like(leaf)/8;analytic=float((grad*direction).sum())
     h=.001;a=model(token,state=patch(oldstate,leaf.detach()+h*direction),owners=(owner,))[0];b=model(token,state=patch(oldstate,leaf.detach()-h*direction),owners=(owner,))[0]
     fd=float((a.log_softmax(-1)[0,0,label]-b.log_softmax(-1)[0,0,label])/(2*h));err=abs(fd-analytic)
     if err>2e-3 or not torch.isfinite(grad).all():raise ValueError('actual source local VJP mismatch')
     ad_checks.append({'seed':seed,'owner':owner,'error':err,'slow_state_loss_gradient_norm':float(grad.norm())})
    # Genuine observed continuation disagreement, NOT a factual contradiction.
    context=raw[t-4:t] if t>=4 else None;seen=contexts.get(context,set()) if context else set();different=bool(seen and raw[t] not in seen)
    erasures={}
    if t>=128 and different:
     contradictions+=1
     for name,erasekey in [('context',k.float()),('wrong',keys[max(1,t-17)].float())]:
      S=oldstate.tensors['slow_memory'];erased=S-.25*(S@erasekey[None,:,None])*erasekey[None,None,:]
      erasures[name]=model(token,state=patch(oldstate,erased),owners=(owner,))[0]
    if context:contexts.setdefault(context,set()).add(raw[t])
    if t<128:continue
    lp=logits[0,0].log_softmax(-1);p=lp.exp().numpy().astype(np.float64);entropy=-float((lp.exp()*lp).sum());overlap=abs(float(q@k));block=t//4
    if t%4==0:flags=bytearray(2)
    entropy_trigger=not flags[0] and (entropy>=3. or t%4==3);survival_trigger=not flags[1] and (entropy+.2*(1-overlap)>=3.1 or t%4==3)
    if entropy_trigger:flags[0]=True
    if survival_trigger:flags[1]=True
    random_trigger=t%4==(hashlib.sha256(f'{seed}:{owner}:{block}'.encode()).digest()[0]&3)
    probs={'never':p};matches={}
    for name,trigger in [('always',True),('random',random_trigger),('entropy',entropy_trigger),('survival_proxy',survival_trigger)]:
     if trigger:
      tic=time.perf_counter();rp,n=ring.predict(torch.tensor(prior));look_wall[name]+=time.perf_counter()-tic;calls[name]+=1;probs[name]=.5*p+.5*rp.numpy() if n>=2 else p;matches[name]=n
     else:probs[name]=p
    cache_predictions={name:c.predict(prior) for name,c in caches.items()}
    # Reveal the actual future target only after ALL policy choices/predictions.
    y=raw[t+1]
    for event,changed,wronglog in staged_events:
     event.update(old_event_removal_minus_full_nll=nll(changed,y)-nll(logits,y),wrong_normmatched_removal_minus_full_nll=nll(wronglog,y)-nll(logits,y));events.append(event)
    row={'seed':seed,'owner':owner,'clock':t,'full_nll':-math.log(p[y]),'entropy':entropy,'trigger_entropy':int(entropy_trigger),'trigger_survival_proxy':int(survival_trigger),'trigger_random':int(random_trigger)}
    for name,pp in probs.items():row[name+'_nll']=-math.log(pp[y])
    for name,(pp,n,ages) in cache_predictions.items():row[name+'_nll']=-math.log(pp[y]);row[name+'_hit']=int(n>0);row[name+'_mean_retained_age']=sum(ages)/len(ages) if ages else 0
    if erasures:
     row['disagreement_context_erasure_minus_full_nll']=nll(erasures['context'],y)-nll(logits,y);row['disagreement_wrong_erasure_minus_full_nll']=nll(erasures['wrong'],y)-nll(logits,y)
    policy_rows.append(row)
   if any(calls[n]!=224 for n in ('random','entropy','survival_proxy')) or calls['always']!=896:raise ValueError('exact query budgets')
   source_checks.append({'seed':seed,'owner':owner,'event_sum_error':sum_error,'independent_PNDS_step_error':reference_error,'lookup_calls':calls,'lookup_wall_seconds':look_wall,'native_forward_wall_seconds':normal_wall,'disagreement_events_NOT_stale_facts':contradictions,'cache_bytes_each':{k:len(c.data) for k,c in caches.items()},'lookback_tier_bytes':3858,'Kalman_actual_keys_min_cov_eigenvalue':min_eigen,'Kalman_min_innovation_denominator':min_denom,'Kalman_discarded_offdiagonal_maxnorm':offdiag_max,'Kalman_extra_cov_bytes':512})
  if weights(model)!=before:raise ValueError('original weights changed')
 summary=[]
 for age in AGES:
  rows=[x for x in events if x['age']==age];fields=('survival_ratio','query_projection_norm','historical_key_projection_norm','full_matrix_byte_correct','isolated_event_byte_correct_ORACLE','FIT_modal_correct','old_event_removal_minus_full_nll','wrong_normmatched_removal_minus_full_nll','max_logit_change')
  summary.append({'age':age,'events':len(rows),**{k:sum(x[k] for x in rows)/len(rows) for k in fields}})
 eligible=[]
 for a in summary:
  if a['age']<128:continue
  rr=[x for x in events if x['age']==a['age']];means=[sum(x['old_event_removal_minus_full_nll'] for x in rr if x[field]==value)/sum(x[field]==value for x in rr) for field,values in [('seed',(1701,1702,1703)),('owner',OWNERS)] for value in values]
  if a['old_event_removal_minus_full_nll']>=.001 and a['old_event_removal_minus_full_nll']>a['wrong_normmatched_removal_minus_full_nll'] and min(means)>0:eligible.append(a['age'])
 policies={k:sum(x[k] for x in policy_rows)/len(policy_rows) for k in policy_rows[0] if k.endswith(('_nll','_hit','_mean_retained_age')) and not k.startswith('disagreement_')}
 # Per-cell erasure fields have a different denominator and are kept separately.
 er=[x for x in policy_rows if 'disagreement_context_erasure_minus_full_nll' in x]
 erase_summary={k:sum(x[k] for x in er)/len(er) if er else None for k in ('disagreement_context_erasure_minus_full_nll','disagreement_wrong_erasure_minus_full_nll')}
 structural=[]
 for name,source in docs.items():
  stack=[];maximum=0;overflow=0;unmatched=0;bounded=[]
  pairs={41:40,93:91,125:123}
  for byte in source:
   if byte in (40,91,123):
    stack.append(byte);maximum=max(maximum,len(stack))
    if len(bounded)<8:bounded.append(byte)
    else:overflow+=1
   elif byte in pairs:
    if stack and stack[-1]==pairs[byte]:stack.pop()
    else:unmatched+=1
    if bounded and bounded[-1]==pairs[byte]:bounded.pop()
  structural.append({'owner':name,'maximum_literal_bracket_depth':maximum,'bounded8_overflow_events':overflow,'unmatched_literal_closers':unmatched,'syntax_parser':False,'oracle_labels_fed_to_model':False})
 result={'study':'LACUNA-SURVIVAL001','status':'PASS_FROZEN_FIT_SCREENS','source_model_sha256':mfest['model_source_sha256'],'checkpoint_sha256s':[x[0]['checkpoint_sha256'] for x in cps],'FIT_targets_per_seed':1792,'distinct_source_positions':1792,'repeated_across_seeds':True,'events':events,'policy_rows':policy_rows,'age_summary':summary,'policy_mean':policies,'source_checks':source_checks,'local_AD_checks':ad_checks,'structural_stack_prefix_audit':structural,'targeted_unlearning':{'status':'MECHANICS_ONLY_EFFICACY_BLOCKED_NO_AUTHENTICATED_SUPERSESSIONS','disagreement_rows':len(er),**erase_summary},'candidate_fit_gate':{'passed_ages':eligible,'status':'PASS_SCREEN_ONLY_STILL_REQUIRES_MATCHED_REGISTRATION' if eligible else 'STOP_NO_CONSISTENT_NONTRIVIAL_OLD_CONTENT_UTILITY'},'cpu_seconds':time.process_time()-started,'peak_RSS_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,'new_training_steps':0,'DEV_scoring':False,'GPU_minutes':0,'protected_TEST':False,'independently_qualified':False}
 out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(result,indent=2,sort_keys=True,allow_nan=False)+'\n');print(json.dumps({k:result[k] for k in ('age_summary','policy_mean','candidate_fit_gate','targeted_unlearning','cpu_seconds')}))

def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();run(a.output)
if __name__=='__main__':main()
