#!/usr/bin/env python3
"""Fresh-source, frozen-model address-specificity diagnostic. No fitting."""
from __future__ import annotations
import argparse,hashlib,importlib.util,json,math,resource,subprocess,sys,time
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'src')]
from scripts import lacuna_retrieval003 as base
from scripts.replay_langmem086087 import digest,load_source,original_checkpoints,weights
REG=ROOT/'research/mathematical-engineering/lacuna_address005'
EPS=1e-12
REF_SHA='f8fdd5893d09e71a54ed345a62605a1fe22eb02215916afa3954cc8d8e5a9620'
ARMS=('A','B','C','D','matrix_wrong','raw_wrong','A_shuffle','B_shuffle','C_shuffle','D_shuffle','matrix_irrelevant','raw_irrelevant','random0','random1')
FROZEN_BLOBS={
 'cpython:Doc/library/decimal.rst':('916f17cadfaa7ed560f76789fb049aa966d03d57',81974),
 'cpython:Doc/library/itertools.rst':('43e665c3f0d5e377c70b32642f1a2a4e23e39053',64405),
 'cpython:Doc/library/statistics.rst':('614f5b905a4a2ebfb70a71958de1e908eb7e5780',43667),
 'rust-book:src/ch13-01-closures.md':('ecda1acff5cfecdcc955d1256c9f185e6b99dea4',20936),
 'rust-book:src/ch15-05-interior-mutability.md':('2bc04a770cbe1f4c5102dc1777c8064d5d5aa704',18548),
 'rust-book:src/ch17-01-futures-and-syntax.md':('4ac55c467a6174d2564abd5fd2e51423de65ef19',19958)}

def project(candidate,history,current,fast):
    for v in (candidate,history,current,fast):
        if not isinstance(v,torch.Tensor) or v.shape!=(8,) or not v.is_floating_point() or not torch.isfinite(v).all():raise ValueError('finite floating [8] vectors')
    candidate,history,current,fast=(v.double() for v in (candidate,history,current,fast));basis=[]
    def perpendicular(v):
        out=v
        for _ in range(2):
            for q in basis:out=out-(out@q)*q
        return out
    for axis in (current,fast):
        n=float(axis.norm().detach())
        if n==0:continue
        if n<=EPS:return history.clone(),'INCONCLUSIVE_NEAR_ZERO_AXIS',len(basis)
        residual=perpendicular(axis);rn=residual.norm()
        if float(rn.detach())>EPS:basis.append(residual/rn)
    hp=perpendicular(history);r=hp.norm()
    if float(r.detach())<=EPS:return history.clone(),'NO_DIRECTION_RADIUS',len(basis)
    vp=perpendicular(candidate);pn=vp.norm()
    if float(pn.detach())<=EPS:return history.clone(),'NO_DIRECTION_CANDIDATE',len(basis)
    return history-hp+r*vp/pn,'REGULAR',len(basis)

def random_vector(owner,clock,draw):
    raw=hashlib.sha256(f'ADDRESS005:{owner}:{clock}:{draw}'.encode()).digest()
    return torch.tensor([(int.from_bytes(raw[i:i+4],'little')+.5)/2**31-1 for i in range(0,32,4)],dtype=torch.float64)

def choice(owner,clock,salt,items):
    if not items:return None
    n=int.from_bytes(hashlib.sha256(f'ADDRESS005:{owner}:{clock}:{salt}'.encode()).digest()[:8],'little')
    return items[n%len(items)]

def source_docs():
    receipt=json.loads((REG/'SOURCE_RECEIPT.json').read_text());docs={};groups={}
    for r in receipt['sources']:
        raw=(ROOT/r['local_path']).read_bytes()
        if len(raw)!=r['bytes'] or digest(raw)!=r['sha256'] or hashlib.sha1(f'blob {len(raw)}\0'.encode()+raw).hexdigest()!=r['git_blob']:raise ValueError('publisher blob mismatch')
        if r['role']=='NEW_DEV_DOCUMENT':
            owner=r['publisher']+':'+r['path'];docs[owner]=raw;groups[owner]=r['publisher']
            if (r['git_blob'],len(raw))!=FROZEN_BLOBS.get(owner):raise ValueError('preregistered source identity')
    if len(docs)!=6 or sorted(groups.values())!=['cpython']*3+['rust-book']*3:raise ValueError('six frozen source owners')
    if len({digest(raw[:4097]) for raw in docs.values()})!=6:raise ValueError('duplicate scoring prefix')
    return docs,groups

def read_control(candidate,h,c,tr,status='AVAILABLE'):
    out,flag,rank=project(candidate,h,c,tr['read_fast'][0])
    if status!='AVAILABLE':flag=status
    total=(out+c).float();mix=tr['mix'][0];fast=tr['read_fast'][0]
    original_mixed=mix[0]*fast+mix[1]*tr['read_slow'][0]
    mixed=original_mixed+mix[1]*(total-tr['read_slow'][0])
    mf=tr['mix'][0].double();f=fast.double()
    expected_mix=mf[0]*f+mf[1]*(h+c);float64mix=mf[0]*f+mf[1]*(out+c)
    errors={'history64':abs(float(out.norm()-h.norm())),
            'total64':abs(float((out+c).norm()-(h+c).norm())),
            'mixed64':abs(float(float64mix.norm()-expected_mix.norm())),
            'history_native':abs(float((total.double()-c).norm()-h.norm())),
            'total_native':abs(float(total.double().norm()-(h+c).norm())),
            'mixed_native':abs(float(mixed.double().norm()-original_mixed.double().norm()))}
    return total,out,{'status':flag,'rank':rank,'errors':errors,'candidate':candidate.tolist(),
                      'history_out':out.tolist(),'slow_native':total.tolist(),'mixed_native':mixed.tolist()}

def group_means(rows,values,owners):
    means={o:sum(v for r,v in zip(rows,values) if r['owner']==o)/sum(r['owner']==o for r in rows) for o in owners if any(r['owner']==o for r in rows)}
    publishers=sorted({o.split(':')[0] for o in owners})
    pub={p:sum(v for o,v in means.items() if o.split(':')[0]==p)/sum(o.split(':')[0]==p for o in means) for p in publishers if any(o.split(':')[0]==p for o in means)}
    seed={}
    for s in (1701,1702,1703):
        om=[sum(v for r,v in zip(rows,values) if r['seed']==s and r['owner']==o)/sum(r['seed']==s and r['owner']==o for r in rows) for o in owners if any(r['seed']==s and r['owner']==o for r in rows)]
        if om:seed[str(s)]=sum(om)/len(om)
    return {'macro':sum(means.values())/len(means) if means else None,'owners':means,'publishers':pub,'seeds':seed}

def summarize(rows,all_rows,owners):
    bins=[]
    for lo,hi in base.BINS:
        rr=[r for r in rows if r['bin_begin']==lo]
        support={o:len({r['clock'] for r in rr if r['owner']==o}) for o in owners}
        result={'bin_begin':lo,'bin_end_exclusive':hi,'measurement_count':len(rr),'source_support':support,'arms':{},'gates':{}}
        for arm in ARMS:
            gain=[r['original_nll']-r['arms'][arm]['nll'] for r in rr]
            result['arms'][arm]={'original_minus_arm':group_means(rr,gain,owners),'fallback_or_missing':sum(r['arms'][arm]['status']!='REGULAR' for r in rr)}
        for arm,wrong,irrelevant in [('B','matrix_wrong','matrix_irrelevant'),('D','raw_wrong','raw_irrelevant')]:
            comparisons={}
            for label,control in [('wrong',wrong),('shuffled',arm+'_shuffle'),('irrelevant',irrelevant),('random_mean',None)]:
                delta=[(r['arms'][control]['nll'] if control else .5*(r['arms']['random0']['nll']+r['arms']['random1']['nll']))-r['arms'][arm]['nll'] for r in rr]
                comparisons[label]=group_means(rr,delta,owners)
            relevant=(arm,wrong,irrelevant,arm+'_shuffle','random0','random1')
            bad=any(result['arms'][a]['fallback_or_missing'] for a in relevant)
            sufficient=all(n>=20 for n in support.values()) and len(support)==6
            d=comparisons['wrong'];primary=d['macro'] is not None and d['macro']>=.001 and len(d['owners'])==6 and min(d['owners'].values())>0 and len(d['seeds'])==3 and min(d['seeds'].values())>0 and len(d['publishers'])==2 and min(d['publishers'].values())>=.001
            additional=all(comparisons[k]['macro'] is not None and comparisons[k]['macro']>=.001 and len(comparisons[k]['publishers'])==2 and min(comparisons[k]['publishers'].values())>0 for k in ('shuffled','irrelevant','random_mean'))
            gate='INCONCLUSIVE_SUPPORT_OR_CONTROL' if bad or not sufficient else ('PASS_FRESH_CONDITIONAL_SPECIFICITY' if primary and additional else 'NOT_PASSED')
            utility=result['arms'][arm]['original_minus_arm'];practical=utility['macro'] is not None and utility['macro']>=.001 and len(utility['owners'])==6 and min(utility['owners'].values())>0 and len(utility['seeds'])==3 and min(utility['seeds'].values())>0
            result['gates'][arm]={'specificity':gate,'practical_prediction_utility':bool(practical),'comparisons':comparisons}
        bins.append(result)
    hits=group_means(all_rows,[r['query_context_hit']-r['random_context_hit'] for r in all_rows],owners)
    select=hits['macro'] is not None and hits['macro']>=.01 and len(hits['owners'])==6 and min(hits['owners'].values())>0 and len(hits['seeds'])==3 and min(hits['seeds'].values())>0
    native={}
    for label in ('original','random_mean'):
        delta=[(r['original_nll'] if label=='original' else .5*(r['scores']['random0']+r['scores']['random1']))-r['scores']['C'] for r in all_rows]
        native[label]=group_means(all_rows,delta,owners)
    good=lambda d:d['macro'] is not None and d['macro']>=.001 and len(d['owners'])==6 and min(d['owners'].values())>0 and len(d['publishers'])==2 and min(d['publishers'].values())>0 and len(d['seeds'])==3 and min(d['seeds'].values())>0
    nofallback=all(all(r['statuses'][a]=='REGULAR' for a in ('C','random0','random1')) for r in all_rows)
    native_pass=bool(nofallback and all(good(d) for d in native.values()))
    return {'bins':bins,'primary':bins[0]['gates']['D'],'query_selection':{'hit_advantage':hits,'passed':bool(select)},
            'all_query_native':{'comparisons':native,'passed':native_pass},'autonomous_conditional_gate':bool(select and native_pass),
            'all_query_measurements':len(all_rows),'all_query_source_positions':len({(r['owner'],r['clock']) for r in all_rows})}

@torch.no_grad()
def run(output):
    if output.exists():raise FileExistsError('preserve original005output')
    if digest((ROOT/'scripts/lacuna_retrieval003.py').read_bytes())!='54fc433972583ec524307a0fada385a1900a09cdbe554e4bcdfaaf7e603febca':raise ValueError('source helper changed')
    if digest((REG/'pnds_reference.py').read_bytes())!=REF_SHA:raise ValueError('reference source changed')
    spec=importlib.util.spec_from_file_location('independent005',REG/'pnds_reference.py');ref=importlib.util.module_from_spec(spec);spec.loader.exec_module(ref)
    torch.set_num_threads(2);docs,groups=source_docs();owners=sorted(docs)
    manifest=json.loads((ROOT/'research/replication/langmem086087/MANIFEST.json').read_text());cls=load_source(ROOT,manifest)
    rows=[];all_rows=[];checks=[];availability=[];started=time.process_time();attempts=0;decoder_calls=0
    for checkpoint,cp,fit in original_checkpoints(ROOT,manifest):
        if cp['steps']!=640:continue
        model=cls().eval();model.load_state_dict(cp['model'],strict=True);before=weights(model)
        if model.counts()['trainable']!=100521:raise ValueError('source count')
        for owner in owners:
            raw=docs[owner];state=model.start((owner,));ring=base.RawRing();coef=torch.zeros((4096,8),dtype=torch.float64);vals=torch.zeros_like(coef);keys=torch.zeros_like(coef)
            errors={k:0. for k in ('transport','reconstructed_key','reconstructed_value','identity','zero','reference','poison','history64','total64','mixed64','history_native','total_native','mixed_native')};gradient=None
            counts={lo:{'queries':0,'context_available':0} for lo,hi in base.BINS}
            for t in range(4096):
                token=torch.tensor([[raw[t]]]);prior=state;logits,state,traces=model(token,state=prior,owners=(owner,),return_trace=True);tr=traces[0]
                if state.tensor_bytes()!=1920:raise ValueError('state budget')
                ring.append(raw[t]);n=max(0,t-1);e=(model.embedding(token)+model.position(model.positions(t,1,1,torch.float32,token.device))).tanh()
                eta=float((.01*model.slow_gate(e).sigmoid())[0,0,0]);rho=float(.995+.004*model.slow_decay.sigmoid());k,v=tr['writer'][0].double(),tr['value'][0].double();pending=[];all_pending=None
                if t:
                    coef[:n]=rho*(coef[:n]-eta*(coef[:n]@k)[:,None]*k);current=eta*v[:,None]*k[None,:];old=state.tensors['slow_memory'][0].double()-current
                if t>=256 and t%4==0:
                    indices=ring.candidates();rk,rv=base.reconstruct(model,ring,indices);q=tr['query'][0].double();h=old@q;c=current@q;f=tr['read_fast'][0].double()
                    errors['reconstructed_key']=max(errors['reconstructed_key'],float((rk-keys[indices]).abs().max()));errors['reconstructed_value']=max(errors['reconstructed_value'],float((rv-vals[torch.tensor(indices)-1]).abs().max()))
                    cache={};prepared={};rawq=base.attention(rk,rv,q)[0];shuffled=vals[:n].roll(1,0).T@coef[:n]
                    common={'A':h,'C':rawq,'A_shuffle':shuffled@q,'C_shuffle':base.attention(rk,rv.roll(1,0),q)[0],
                            'random0':random_vector(owner,t,0),'random1':random_vector(owner,t,1)}
                    def forecast(name,candidate,status='AVAILABLE'):
                        nonlocal attempts,decoder_calls
                        attempts+=1;total,out,record=read_control(candidate,h,c,tr,status)
                        for field,value in record['errors'].items():errors[field]=max(errors[field],value)
                        fingerprint=total.numpy().tobytes()
                        if fingerprint not in cache:
                            cache[fingerprint]=base.patched_logits(model,token,prior,owner,tr,total);decoder_calls+=1
                        return record,cache[fingerprint],base.byte_read(out,model.byte_codes.double())
                    for a,value in common.items():prepared[a]=forecast(a,value)
                    errors['identity']=max(errors['identity'],float((prepared['A'][1]-logits[0,0]).abs().max()))
                    selected=int((rk@q).argmax());random_index=choice(owner,t,'uniform-key',list(range(len(indices))))
                    context=raw[t-3:t+1];all_pending={'seed':cp['seed'],'owner':owner,'publisher':groups[owner],'clock':t,
                        'query_index':indices[selected],'random_index':indices[random_index],
                        'query_context_hit':int(ring.context(indices[selected])==context),'random_context_hit':int(ring.context(indices[random_index])==context),
                        'statuses':{a:prepared[a][0]['status'] for a in ('C','random0','random1')},'preds':{a:prepared[a][1] for a in ('C','random0','random1')}}
                    for lo,hi in base.BINS:
                        counts[lo]['queries']+=1;matches=[j for j in indices if lo<=t-j<hi and ring.context(j)==context];j=max(matches,default=None)
                        if j is None:continue
                        counts[lo]['context_available']+=1;a=rk[indices.index(j)]
                        wrongs=[i for i,b in enumerate(indices) if abs(b-j)>=5 and not b-4<=j<b and ring.context(b)!=context]
                        wi=max(wrongs,key=lambda i:(float(rk[i]@a),-indices[i])) if wrongs else None
                        irrelevants=[i for i in wrongs if lo<=t-indices[i]<hi];ii=choice(owner,t,f'irrelevant-{lo}',irrelevants)
                        wrong=rk[wi] if wi is not None else q;irrelevant=rk[ii] if ii is not None else q
                        extra={'B':old@a,'D':base.attention(rk,rv,a)[0],
                               'matrix_wrong':old@wrong,'raw_wrong':base.attention(rk,rv,wrong)[0],
                               'B_shuffle':shuffled@a,'D_shuffle':base.attention(rk,rv.roll(1,0),a)[0],
                               'matrix_irrelevant':old@irrelevant,'raw_irrelevant':base.attention(rk,rv,irrelevant)[0]}
                        pp=dict(prepared)
                        for name,vector in extra.items():
                            missing=('_wrong' in name and wi is None) or ('_irrelevant' in name and ii is None)
                            pp[name]=forecast(name,vector,'INCONCLUSIVE_MISSING_CONTROL' if missing else 'AVAILABLE')
                        original_mixed=tr['mix'][0,0]*tr['read_fast'][0]+tr['mix'][0,1]*tr['read_slow'][0]
                        pending.append(({'seed':cp['seed'],'owner':owner,'publisher':groups[owner],'clock':t,'bin_begin':lo,'bin_end_exclusive':hi,'historical_clock':j,
                                         'wrong_clock':indices[wi] if wi is not None else None,'irrelevant_clock':indices[ii] if ii is not None else None,
                                         'geometry':{'history':h.tolist(),'current':c.tolist(),'fast':f.tolist(),'mix':tr['mix'][0].tolist(),'original_mixed_native':original_mixed.tolist()}},pp))
                    if gradient is None:
                        candidate=common['random0'];actual,status,rank=project(candidate,h,c,f);expected,es,er=ref.project(candidate.tolist(),h.tolist(),c.tolist(),f.tolist())
                        if (status,rank)!=(es,er):raise ValueError('reference branch parity')
                        errors['reference']=max(abs(x-y) for x,y in zip(actual.tolist(),expected))
                        z=base.patched_logits(model,token,prior,owner,tr,torch.zeros(8));sz=model(token,state=prior,owners=(owner,),lesion='slow_zero')[0][0,0];errors['zero']=float((z-sz).abs().max())
                        poisoned=bytearray(raw[:t+17]);poisoned[t+1:]=bytes((b+113)%256 for b in poisoned[t+1:])
                        pl,ps,pt=model(torch.tensor([list(poisoned[:t+1])]),owners=(owner,),return_trace=True);ptr=pt[-1];pq=ptr['query'][0].double();ph=(ps.tensors['slow_memory'][0].double()-current)@pq;pc=current@pq
                        p_ring=base.RawRing()
                        for byte in poisoned[:t+1]:p_ring.append(byte)
                        pk,pv=base.reconstruct(model,p_ring,indices);pcommon={'C':base.attention(pk,pv,pq)[0],'random0':common['random0'],'random1':common['random1']}
                        errors['poison']=max(float((pl[0,-1]-logits[0,0]).abs().max()),max(float((read_control(vv,ph,pc,ptr)[0]-(project(common[aa],h,c,f)[0]+c).float()).abs().max()) for aa,vv in pcommon.items()))
                        label=int(logits[0,0].argmax());direction=torch.ones(8,dtype=torch.float64)/math.sqrt(8)
                        with torch.enable_grad():
                            leaf=candidate.clone().requires_grad_(True);out,_,_=project(leaf,h,c,f);loss=base.patched_logits(model,token,prior,owner,tr,(out+c).float()).log_softmax(0)[label];grad=torch.autograd.grad(loss,leaf)[0]
                        step=.001;plus,_,_=project(candidate+step*direction,h,c,f);minus,_,_=project(candidate-step*direction,h,c,f)
                        lp=base.patched_logits(model,token,prior,owner,tr,(plus+c).float()).log_softmax(0)[label];lm=base.patched_logits(model,token,prior,owner,tr,(minus+c).float()).log_softmax(0)[label]
                        err=abs(float(grad@direction)-float((lp-lm)/(2*step)))
                        if err>.002 or not torch.isfinite(grad).all():raise ValueError('composed source VJP')
                        gradient={'error':err,'norm':float(grad.norm()),'prefix_argmax_label':label,'status':status}
                if t:coef[t-1]=eta*k;vals[t-1]=v;keys[t]=k
                errors['transport']=max(errors['transport'],float((vals[:t].T@coef[:t]-state.tensors['slow_memory'][0].double()).abs().max()))
                # First future-label use, after all scheduled selections/predictions.
                y=raw[t+1];full=-float(logits[0,0].log_softmax(0)[y])
                if all_pending is not None:
                    ap=all_pending.pop('preds');all_pending.update(next_byte=y,original_nll=full,scores={a:-float(p.log_softmax(0)[y]) for a,p in ap.items()});all_rows.append(all_pending)
                for row,pp in pending:
                    target=raw[row['historical_clock']];row.update(target_byte=target,next_byte=y,original_nll=full,arms={})
                    for a,(record,pred,lp) in pp.items():row['arms'][a]={**record,'nll':-float(pred.log_softmax(0)[y]),'past_nll':-float(lp[target]),'past_correct':int(int(lp.argmax())==target)}
                    rows.append(row)
            limits={'transport':1e-5,'history64':1e-10,'total64':1e-10,'mixed64':1e-10,'reference':1e-10,'history_native':3e-6,'total_native':3e-6,'mixed_native':3e-6}
            if any(v>limits.get(k,3e-5) for k,v in errors.items()):raise ValueError('registered source tolerance: '+str(errors))
            checks.append({'seed':cp['seed'],'owner':owner,'errors':errors,'composed_VJP':gradient,'model_parameters':100521,'state_tensor_bytes':1920,'clock_bytes':8,'extra_ring_bytes':256,'decomposition_bytes':786432})
            availability.append({'seed':cp['seed'],'owner':owner,'counts':counts})
        if weights(model)!=before:raise ValueError('frozen weights changed')
    summary=summarize(rows,all_rows,owners)
    result={'study':'LACUNA-ADDRESS005','status':'PASS_FRESH_SOURCE_FROZEN_INTEGRITY','summary':summary,'rows':rows,'all_query_rows':all_rows,'availability':availability,'source_checks':checks,
            'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'main_decoder_attempts':attempts,'actual_cached_decoder_calls':decoder_calls,
            'new_training_steps':0,'GPU_minutes':0,'protected_TEST':False,'old_gates_reopened':False,'training_authorized_by_result':False,'independently_qualified':False,'full_neural_IEEE_AD':'OPEN',
            'cpu_seconds':time.process_time()-started,'peak_RSS_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024}
    output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(result,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n')
    print(json.dumps({'primary':summary['primary'],'query_selection':summary['query_selection'],'all_query_native':summary['all_query_native'],'cpu_seconds':result['cpu_seconds']}))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);run(p.parse_args().output)
