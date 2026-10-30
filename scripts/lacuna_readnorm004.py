#!/usr/bin/env python3
"""Paired exposed-DEV frozen read control; no fit or autonomous-memory claim."""
from __future__ import annotations
import argparse
import importlib.util
import json
import math
from pathlib import Path
import resource
import subprocess
import sys
import time

import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'src')]
from scripts import lacuna_retrieval003 as base
from scripts.replay_langmem086087 import digest,load_source,original_checkpoints,weights

REG=ROOT/'research/mathematical-engineering/lacuna_readnorm004'
EPS=1e-12
INHERITED_SOURCE_SHA256='54fc433972583ec524307a0fada385a1900a09cdbe554e4bcdfaaf7e603febca'
REFERENCE_SHA256='c57d486e647526ead537c1f9d853c69276ae206def732fac5473c8a0b58e1855'


def dual_norm(candidate,history,current):
    """Closest candidate direction on the prescribed sphere/hyperplane intersection."""
    for vector in (candidate,history,current):
        if not isinstance(vector,torch.Tensor) or vector.shape!=(8,) or not vector.is_floating_point() or not torch.isfinite(vector).all():
            raise ValueError('finite floating [8] tensors required')
    candidate,history,current=(v.double() for v in (candidate,history,current))
    cn=current.norm()
    if float(cn)==0:
        vn=candidate.norm()
        if float(vn)<=EPS:return history.clone(),'INCONCLUSIVE_ZERO_CANDIDATE'
        return candidate*(history.norm()/vn),'REGULAR_ZERO_CURRENT'
    if float(cn)<=EPS:return history.clone(),'INCONCLUSIVE_NEAR_ZERO_CURRENT'
    u=current/cn;a=history@u;hp=history-a*u;r=hp.norm()
    if float(r)<=EPS:return history.clone(),'NO_ORTHOGONAL_FREEDOM'
    vp=candidate-(candidate@u)*u;pn=vp.norm()
    if float(pn.detach())<=EPS:return history.clone(),'INCONCLUSIVE_CANDIDATE_PARALLEL'
    return a*u+r*vp/pn,'REGULAR'


def controlled_reads(reads,current,trace):
    """Prefix-only preparation; no source-document, target or score input."""
    h=reads['A'];c=current@trace['query'][0].double()
    prepared={}
    mix=trace['mix'][0].double();rf=trace['read_fast'][0].double()
    original_mixed=mix[0]*rf+mix[1]*trace['read_slow'][0].double()
    for name,v in reads.items():
        transformed,status=dual_norm(v,h,c)
        methods={}
        for method,historical in [('historical_only',base.magnitude_match(v,h)),('dual_norm',transformed)]:
            total=(historical+c).float()
            effective_history=total.double()-c
            lp=base.byte_read(historical,trace['_codes'])
            methods[method]={'read':total,'past_lp':lp,'historical':historical,
                'status':status if method=='dual_norm' else 'REGISTERED003',
                'history_norm_error64':abs(float(historical.norm()-h.norm())),
                'total_norm_error64':abs(float((historical+c).norm()-(h+c).norm())),
                'projection_error64':abs(float(historical@c-h@c)),
                'history_norm_error_native':abs(float(effective_history.norm()-h.norm())),
                'total_norm_error_native':abs(float(total.double().norm()-(h+c).norm())),
                'mixed_norm_change_native':float((mix[0]*rf+mix[1]*total.double()).norm()-original_mixed.norm()),
                'rotation_cosine':float(torch.nn.functional.cosine_similarity(historical,h,dim=0)),
                'vectors':{'history':h.tolist(),'current':c.tolist(),'candidate':v.tolist(),
                           'transformed_history':historical.tolist(),'native_total':total.tolist()}}
        prepared[name]=methods
    return prepared


def summarize(rows):
    out=[]
    for lo,hi in base.BINS:
        rr=[r for r in rows if r['bin_begin']==lo]
        summary={'bin_begin':lo,'bin_end_exclusive':hi,'measurements':len(rr),
                 'distinct_source_query_cells':len({(r['owner'],r['clock']) for r in rr}),
                 'arms':{},'provisional_diagnostic_signals':[]}
        for name in base.ARMS:
            methods={}
            for method in ('historical_only','dual_norm'):
                fields=('next_byte_nll','original_minus_arm_nll','past_byte_nll','past_byte_correct',
                        'history_norm_error64','total_norm_error64','projection_error64',
                        'history_norm_error_native','total_norm_error_native','mixed_norm_change_native','rotation_cosine')
                metrics={f:sum(r['arms'][name][method][f] for r in rr)/len(rr) for f in fields}
                metrics['fallback_count']=sum(not r['arms'][name][method]['status'].startswith('REGULAR') for r in rr) if method=='dual_norm' else 0
                methods[method]=metrics
            methods['dual_minus_historical_nll']=methods['dual_norm']['next_byte_nll']-methods['historical_only']['next_byte_nll']
            summary['arms'][name]=methods
        for name in ('A','B','C','D'):
            wrong='matrix_wrong' if name in ('A','B') else 'raw_wrong'
            controls=(name,wrong,name+'_shuffle')
            fallback=any(summary['arms'][a]['dual_norm']['fallback_count'] for a in controls)
            means={str(value):sum(r['arms'][name]['dual_norm']['original_minus_arm_nll'] for r in rr if r[field]==value)/sum(r[field]==value for r in rr)
                   for field,vs in [('seed',(1701,1702,1703)),('owner',sorted({r['owner'] for r in rows}))]
                   for value in vs if any(r[field]==value for r in rr)}
            gain=summary['arms'][name]['dual_norm']['original_minus_arm_nll']
            gate='INCONCLUSIVE_FALLBACK' if fallback else 'NOT_PASSED'
            if not fallback and summary['distinct_source_query_cells']>=20 and len(means)==5 and min(means.values())>0 and gain>=.001 and all(gain>summary['arms'][a]['dual_norm']['original_minus_arm_nll'] for a in controls[1:]):
                gate='PROVISIONAL_EXPOSED_DEV_DIRECTION_SIGNAL';summary['provisional_diagnostic_signals'].append(name)
            summary['arms'][name]['gate']=gate;summary['arms'][name]['seed_owner_gains']=means
        out.append(summary)
    return out


@torch.no_grad()
def run(output):
    if output.exists():raise FileExistsError('preserve original004output')
    if digest((ROOT/'scripts/lacuna_retrieval003.py').read_bytes())!=INHERITED_SOURCE_SHA256:raise ValueError('inherited003source changed')
    torch.set_num_threads(2)
    docs=base.source_docs()
    if digest((REG/'pnds_reference.py').read_bytes())!=REFERENCE_SHA256:raise ValueError('independent reference changed')
    original=json.loads((ROOT/'research/evidence/lacuna_retrieval003/RESULTS.json').read_text())
    expected={(r['seed'],r['owner'],r['clock'],r['bin_begin']):r for r in original['rows']}
    spec=importlib.util.spec_from_file_location('readnorm004_reference',REG/'pnds_reference.py');ref=importlib.util.module_from_spec(spec);spec.loader.exec_module(ref)
    manifest=json.loads((ROOT/'research/replication/langmem086087/MANIFEST.json').read_text());cls=load_source(ROOT,manifest)
    rows=[];checks=[];seen=set();started=time.process_time()
    for checkpoint,cp,fit in original_checkpoints(ROOT,manifest):
        if cp['steps']!=640:continue
        model=cls().eval();model.load_state_dict(cp['model'],strict=True);before=weights(model)
        if model.counts()['trainable']!=100521:raise ValueError('original parameter count')
        for owner,raw in docs.items():
            state=model.start((owner,));ring=base.RawRing()
            coef=torch.zeros((4096,8),dtype=torch.float64);vals=torch.zeros_like(coef);keys=torch.zeros_like(coef)
            errors={name:0. for name in ('transport','reconstructed_key','reconstructed_value','original003_nll','identity_logits','zero_logits','independent_projection','future_poison','history64','total64','history_native','total_native')}
            gradient=None
            for t in range(4096):
                token=torch.tensor([[raw[t]]]);prior=state
                logits,state,traces=model(token,state=prior,owners=(owner,),return_trace=True);tr=traces[0];tr['_codes']=model.byte_codes.double()
                if state.tensor_bytes()!=1920:raise ValueError('original state bytes')
                ring.append(raw[t]);n=max(0,t-1)
                e=(model.embedding(token)+model.position(model.positions(t,1,1,torch.float32,token.device))).tanh()
                eta=float((.01*model.slow_gate(e).sigmoid())[0,0,0]);rho=float(.995+.004*model.slow_decay.sigmoid())
                k,v=tr['writer'][0].double(),tr['value'][0].double();pending=[]
                if t:
                    coef[:n]=rho*(coef[:n]-eta*(coef[:n]@k)[:,None]*k)
                    current=eta*v[:,None]*k[None,:];old=state.tensors['slow_memory'][0].double()-current
                if t>=256 and t%16==0:
                    indices=ring.candidates();rk,rv=base.reconstruct(model,ring,indices)
                    errors['reconstructed_key']=max(errors['reconstructed_key'],float((rk-keys[indices]).abs().max()))
                    errors['reconstructed_value']=max(errors['reconstructed_value'],float((rv-vals[torch.tensor(indices)-1]).abs().max()))
                    for lo,hi in base.BINS:
                        matching=[j for j in indices if lo<=t-j<hi and ring.context(j)==raw[t-3:t+1]];j=max(matching,default=None)
                        if j is None:continue
                        key=(cp['seed'],owner,t,lo)
                        oldrow=expected[key]
                        prepared=base.prepare_reads(model,ring,tr,old,current,keys,vals[:n],coef[:n],indices,j)
                        if prepared is None:raise ValueError('original controls now missing')
                        reads,_,info,_,_=prepared
                        if j!=oldrow['historical_clock'] or info['wrong_clock']!=oldrow['wrong_clock']:raise ValueError('paired cohort/control drift')
                        controls=controlled_reads(reads,current,tr);preds={}
                        for name,methods in controls.items():
                            preds[name]={}
                            for method,item in methods.items():
                                preds[name][method]=base.patched_logits(model,token,prior,owner,tr,item['read'])
                                if method=='dual_norm':
                                    for f,metric in [('history64','history_norm_error64'),('total64','total_norm_error64'),('history_native','history_norm_error_native'),('total_native','total_norm_error_native')]:
                                        errors[f]=max(errors[f],item[metric])
                        for method in ('historical_only','dual_norm'):
                            errors['identity_logits']=max(errors['identity_logits'],float((preds['A'][method]-logits[0,0]).abs().max()))
                        if gradient is None:
                            h=reads['A'];c=current@tr['query'][0].double();candidate=reads['D']
                            reference,status=ref.dual_norm(candidate.tolist(),h.tolist(),c.tolist())
                            actual,actual_status=dual_norm(candidate,h,c)
                            if actual_status!=status:raise ValueError('independent degeneracy classification')
                            errors['independent_projection']=max(abs(a-b) for a,b in zip(reference,actual.tolist()))
                            zero=base.patched_logits(model,token,prior,owner,tr,torch.zeros(8))
                            source_zero=model(token,state=prior,owners=(owner,),lesion='slow_zero')[0][0,0]
                            errors['zero_logits']=float((zero-source_zero).abs().max())
                            # Actual prefix reconstruction after changing every future byte.
                            poison=bytearray(raw[:t+17]);poison[t+1:]=bytes((b+113)%256 for b in poison[t+1:])
                            pl,ps,pt=model(torch.tensor([list(poison[:t+1])]),owners=(owner,),return_trace=True)
                            pring=base.RawRing()
                            for b in poison[:t+1]:pring.append(b)
                            ptr=pt[-1];ptr['_codes']=model.byte_codes.double()
                            pp=base.prepare_reads(model,pring,ptr,ps.tensors['slow_memory'][0].double()-current,current,keys,vals[:n],coef[:n],indices,j)
                            poisoned=controlled_reads(pp[0],current,ptr)
                            errors['future_poison']=max(float((pl[0,-1]-logits[0,0]).abs().max()),max(float((poisoned[a][m]['read']-controls[a][m]['read']).abs().max()) for a in base.ARMS for m in ('historical_only','dual_norm')))
                            label=int(logits[0,0].argmax());direction=torch.ones(8,dtype=torch.float64)/math.sqrt(8)
                            with torch.enable_grad():
                                leaf=candidate.detach().clone().requires_grad_(True);projection,_=dual_norm(leaf,h,c)
                                loss=base.patched_logits(model,token,prior,owner,tr,(projection+c).float()).log_softmax(0)[label]
                                grad=torch.autograd.grad(loss,leaf)[0]
                            step=.001
                            plus,_=dual_norm(candidate+step*direction,h,c);minus,_=dual_norm(candidate-step*direction,h,c)
                            lp=base.patched_logits(model,token,prior,owner,tr,(plus+c).float()).log_softmax(0)[label]
                            lm=base.patched_logits(model,token,prior,owner,tr,(minus+c).float()).log_softmax(0)[label]
                            error=abs(float(grad@direction)-float((lp-lm)/(2*step)))
                            if error>.002 or not torch.isfinite(grad).all():raise ValueError('composed projection/source VJP')
                            gradient={'error':error,'gradient_norm':float(grad.norm()),'status':actual_status,'label':'prefix_argmax_not_target'}
                        pending.append((key,oldrow,j,info,controls,preds))
                if t:
                    coef[t-1]=eta*k;vals[t-1]=v;keys[t]=k
                errors['transport']=max(errors['transport'],float((vals[:t].T@coef[:t]-state.tensors['slow_memory'][0].double()).abs().max()))
                # Future target first enters004scoring after all predictions.
                y=raw[t+1];full=-float(logits[0,0].log_softmax(0)[y])
                for key,oldrow,j,info,controls,preds in pending:
                    errors['original003_nll']=max(errors['original003_nll'],abs(full-oldrow['original_nll']))
                    row={'seed':cp['seed'],'owner':owner,'clock':t,'bin_begin':key[3],'historical_clock':j,
                         'wrong_clock':info['wrong_clock'],'next_byte':y,'target_byte':raw[j],'original_nll':full,'arms':{}}
                    for name,methods in controls.items():
                        row['arms'][name]={}
                        for method,item in methods.items():
                            nll=-float(preds[name][method].log_softmax(0)[y]);historical_lp=item['past_lp']
                            result={f:value for f,value in item.items() if f not in ('read','past_lp','historical')}
                            result.update(next_byte_nll=nll,original_minus_arm_nll=full-nll,
                                          past_byte_nll=-float(historical_lp[raw[j]]),past_byte_correct=int(int(historical_lp.argmax())==raw[j]))
                            row['arms'][name][method]=result
                            if method=='historical_only':errors['original003_nll']=max(errors['original003_nll'],abs(nll-oldrow['arms'][name]['next_byte_nll']))
                    rows.append(row);seen.add(key)
            limits={'transport':1e-5,'independent_projection':1e-10,'history64':1e-10,'total64':1e-10,'history_native':3e-6,'total_native':3e-6}
            if any(value>limits.get(name,3e-5) for name,value in errors.items()):raise ValueError('integrity tolerance: '+str(errors))
            checks.append({'seed':cp['seed'],'owner':owner,'errors':errors,'local_composed_VJP':gradient,
                           'state_tensor_bytes':1920,'logical_clock_bytes':8,'extra_raw_ring_bytes':256,
                           'decomposition_bytes':786432,'unimplemented':['full_parameter_AD','IEEE_proof','independent_qualification']})
        if weights(model)!=before:raise ValueError('frozen weights changed')
    if seen!=set(expected):raise ValueError('missing original003cohort cells')
    result={'study':'LACUNA-READNORM004','status':'PASS_PAIRED_EXPOSED_DEV_INTEGRITY','summary':summarize(rows),'rows':rows,'source_checks':checks,
            'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
            'fresh_evaluation':False,'new_training_steps':0,'GPU_minutes':0,'protected_TEST':False,'prior_failed_gates_unchanged':True,
            'training_authorized_by_result':False,'independently_qualified':False,'full_neural_IEEE_AD':'OPEN',
            'main_patched_decoder_calls':len(rows)*len(base.ARMS)*2,'cpu_seconds':time.process_time()-started,
            'peak_RSS_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024}
    output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(result,indent=2,sort_keys=True,allow_nan=False)+'\n')
    print(json.dumps({'summary':result['summary'],'cpu_seconds':result['cpu_seconds']}))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);run(p.parse_args().output)
