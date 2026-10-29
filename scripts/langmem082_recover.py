#!/usr/bin/env python3
"""Frozen-source memory read diagnostic. Supplied old keys are evaluator privileges.

Nearest-codebook decoding is an untrained diagnostic, not the model decoder.
An isolated tagged event uses a history side table and is only a ceiling.
"""
from __future__ import annotations
import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import struct
import time
import torch
from torch.nn import functional as F
from scripts.langmem081_context_address import (ROOT, DualNativeStreamingTAC079,
    restore_original, source_factors, original_source, fit_sources, score)
from context_transition import phi_position
from candidate import transform

AGES=(8,32,128,256)
SEEDS=(1701,1702,1703)

class RawRing:
    """Persistent payload is exactly budget bytes; metadata lives in the header.

    CPython object/allocator overhead and transient scan workspace excluded,
    as for native tensor payload. No hidden persistent history or cursor.
    """
    def __init__(self,budget):
        if budget not in (256,4160):raise ValueError('unregistered budget')
        self.data=bytearray(budget)
    def append(self,byte):
        if type(byte) is not int or not 0<=byte<256:raise ValueError('byte')
        clock,count=struct.unpack_from('<QQ',self.data)
        cap=len(self.data)-16
        self.data[16+clock%cap]=byte
        struct.pack_into('<QQ',self.data,0,clock+1,min(count+1,cap))
    def get(self,a):
        clock,count=struct.unpack_from('<QQ',self.data)
        if not clock-count<=a<clock:return None
        return self.data[16+a%(len(self.data)-16)]
    def predict(self,L):
        clock,count=struct.unpack_from('<QQ',self.data)
        t=clock-1;start=clock-count
        context=bytes(self.get(i) for i in range(t-L+1,t+1))
        counts=torch.zeros(256,dtype=torch.float64)
        for a in range(max(start+L,t-256,4),t):
            if bytes(self.get(i) for i in range(a-L,a))==context:
                counts[self.get(a)]+=1
        return (counts+8/256)/(counts.sum()+8),int(counts.sum())

@torch.no_grad()
def codebook(model,clock):
    tokens=torch.arange(256,dtype=torch.long)
    tt=torch.full((256,),clock,dtype=torch.long)
    embed=model.embedding(tokens)
    e=(embed+model.pos(transform(phi_position(tt,embed.dtype),'cap_all'))).tanh()
    return F.normalize(model.native_value(e).tanh(),dim=-1,eps=1e-12)

def decode(read,book):
    v=read.reshape(8)
    if not torch.isfinite(v).all() or not torch.isfinite(book).all():raise ValueError('nonfinite read')
    # Deterministic tie rule; a zero read always returns byte0. Report separately.
    sims=book@F.normalize(v,dim=0,eps=1e-12)
    return int(sims.argmax())

def read(matrix,key):
    return (matrix@key.unsqueeze(-1)).reshape(8)

def wrong_writer(prefix,keys,a,t):
    # Keys are external evaluator storage; selector never consumes byte at a.
    choices=[b for b in range(4,t) if prefix[b-4:b]!=prefix[a-4:a]]
    if not choices:return None
    return max(choices,key=lambda b:(float((keys[b]*keys[a]).sum()),b))

def average(rows,key):return sum(x[key] for x in rows)/len(rows)

@torch.no_grad()
def evaluate(seed,checkpoint_dir,original_aggregate):
    torch.set_num_threads(2)
    fit=fit_sources();_,dev,_=original_source(ROOT)
    if len(fit)!=14 or len(dev)!=4 or set(fit)&set(dev):raise ValueError('source owners')
    m,receipt=restore_original(seed,checkpoint_dir,original_aggregate)
    engine=DualNativeStreamingTAC079(m,'cap_all')
    weight_hash=engine._weight_hash()
    books={t:codebook(m,t) for t in range(320)}
    started=time.process_time();history=[];future=[];target_hash=hashlib.sha256()
    hist_hash=hashlib.sha256();max_error=0.;prefix_error=0.;poison=0;shams=0
    timers={'ring_read_seconds':0.,'matrix_read_decode_seconds':0.}
    for owner in sorted(dev):
        raw=dev[owner]
        if len(raw)<321:raise ValueError('short source')
        state=engine.start((owner,));keys={};events={};values={};rings=[RawRing(256),RawRing(4160)]
        for t in range(320):
            if state.tensor_bytes()!=4160 or state.clock!=t:raise ValueError('original state')
            token=torch.tensor([raw[t]],dtype=torch.long)
            f=source_factors(m,state,token)
            logits,nxt=engine.step(token,state,owners=(owner,))
            expected=state.tensors['slow_native']@f['slow']['p']+f['slow']['q']
            err=float((expected-nxt.tensors['slow_native']).abs().max());max_error=max(max_error,err)
            if err>1e-5:raise ValueError('actual source affine mismatch')
            events={a:c@f['slow']['p'] for a,c in events.items()}
            if t>=4:
                events[t]=f['slow']['q'].clone();keys[t]=f['write_key'].clone();values[t]=f['value'].clone()
            for ring in rings:ring.append(int(raw[t]))
            if t==127:
                prefix=torch.tensor(list(raw[:128]),dtype=torch.long).view(1,-1)
                prefix_error=max(prefix_error,float((m(prefix)[:,-1]-logits).abs().max()))
                if prefix_error>3e-5:raise ValueError('original graph parity')
            if t<128:state=nxt;continue
            # All candidate predictions and memory interventions complete before raw[t+1].
            prefix=bytes(raw[:t+1]);M=nxt.tensors['slow_native'];q=f['query']
            native_pred=decode(read(M,q),books[t])
            null_pred=decode(torch.zeros(8),books[t])
            wrong_pred=decode(read(M.roll(1,dims=1),q),books[t])
            zero=replace(state,tensors={**state.tensors,'slow_native':torch.zeros_like(M)})
            perm=replace(state,tensors={**state.tensors,'slow_native':state.tensors['slow_native'].roll(1,dims=1)})
            zero_logits,_=engine.step(token,zero,owners=(owner,))
            perm_logits,_=engine.step(token,perm,owners=(owner,))
            if t==128:
                sham=replace(state,tensors={**state.tensors,'slow_native':state.tensors['slow_native'].clone()})
                sham_logits,_=engine.step(token,sham,owners=(owner,))
                if not torch.equal(sham_logits,logits):raise ValueError('sham changed prediction')
                shams+=1
                # Actual-source future poison: both source-to-token producers observe only <=t.
                altered=bytearray(raw);altered[t+1:]=bytes((x+113)%256 for x in altered[t+1:])
                poisoned_token=torch.tensor([altered[t]],dtype=torch.long)
                plogits,pstate=engine.step(poisoned_token,state,owners=(owner,))
                pf=source_factors(m,state,poisoned_token)
                if not torch.equal(plogits,logits) or any(not torch.equal(pstate.tensors[k],nxt.tensors[k]) for k in nxt.tensors) or any(not torch.equal(pf[k],f[k]) for k in ('query','write_key','value')):
                    raise ValueError('actual source future noninterference')
                poison+=1
            st=time.perf_counter();ring_choices={(len(r.data),L):r.predict(L) for r in rings for L in (2,3,4)}
            timers['ring_read_seconds']+=time.perf_counter()-st
            pending=[]
            for age in AGES:
                a=t-age
                if a<4:continue
                key=keys[a];book=books[a]
                st=time.perf_counter()
                recovered=decode(read(M,key),book)
                wrong_a=wrong_writer(prefix,keys,a,t)
                bad=decode(read(M,keys[wrong_a]),book) if wrong_a is not None else None
                scrambled=decode(read(M.roll(1,dims=1),key),book)
                isolated=decode(read(events[a],key),book)
                original=decode(f['value'].reshape(8),books[t])
                timers['matrix_read_decode_seconds']+=time.perf_counter()-st
                # Direction fidelity to actual original latent historical value;
                # evaluator gets the label only after decoding all alternatives.
                removed=read(M-events[a],key)
                # Privileged intervention: same old event key/time/transport/norm,
                # but source-derived wrong historical byte content. Not deployed.
                donors=[b for b in range(4,a) if prefix[b]!=prefix[a]]
                donor=donors[-1] if donors else None
                wrong_event_correct=None;wrong_event_nll=None;event_removed_nll=None
                event_norm_error=None
                if donor is not None:
                    v=values[a]
                    bad_v=book[prefix[donor]].reshape(1,8)*torch.linalg.vector_norm(v)
                    coefficient=(v.unsqueeze(1)@events[a])/(v*v).sum()
                    wrong_event=bad_v.unsqueeze(-1)*coefficient
                    event_norm_error=abs(float(torch.linalg.vector_norm(wrong_event))-float(torch.linalg.vector_norm(events[a])))
                    if event_norm_error>1e-6:raise ValueError('norm-matched wrong event')
                    wrong_event_correct=decode(read(M-events[a]+wrong_event,key),book)
                    # Before current write, undo its P to obtain the transported
                    # past-event contribution in prior state directly from cache.
                    before=torch.linalg.solve(f['slow']['p'].transpose(-1,-2),events[a].transpose(-1,-2)).transpose(-1,-2)
                    wrong_before=torch.linalg.solve(f['slow']['p'].transpose(-1,-2),wrong_event.transpose(-1,-2)).transpose(-1,-2)
                    without=replace(state,tensors={**state.tensors,'slow_native':state.tensors['slow_native']-before})
                    wrong_state=replace(state,tensors={**state.tensors,'slow_native':state.tensors['slow_native']-before+wrong_before})
                    removed_logits,_=engine.step(token,without,owners=(owner,))
                    wrong_event_logits,wrong_next=engine.step(token,wrong_state,owners=(owner,))
                    if float((wrong_next.tensors['slow_native']-(M-events[a]+wrong_event)).abs().max())>1e-5:
                        raise ValueError('wrong-history actual-source transport mismatch')
                    event_removed_nll=removed_logits
                    wrong_event_nll=wrong_event_logits
                pending.append((a,age,wrong_a,recovered,bad,scrambled,isolated,decode(removed,book),original,
                    {len(r.data):r.get(a) for r in rings},read(M,key),donor,wrong_event_correct,wrong_event_nll,event_removed_nll,event_norm_error))
            matches=[a for a in range(max(4,t-256),t) if prefix[a-4:a]==prefix[t-3:t+1]]
            matching_byte=prefix[matches[-1]] if matches else None
            y=int(raw[t+1]);target_hash.update(owner.encode()+b':'+str(t).encode()+b':'+bytes([y])+b';')
            row={'owner':owner,'clock':t,'target':y,'full_nll':score(logits,y)['nll'],
                 'zero_slow_nll':score(zero_logits,y)['nll'],'permuted_slow_nll':score(perm_logits,y)['nll'],
                 'native_codebook_correct':int(native_pred==y),'wrong_content_codebook_correct':int(wrong_pred==y),
                 'zero_codebook_correct':int(null_pred==y),'current_byte_correct':int(raw[t]==y),
                 'exact_context_available':bool(matches),'native_matching_history_correct':int(native_pred==matching_byte) if matches else None,
                 'exact_history_nextbyte_correct':int(matching_byte==y) if matches else None}
            for (budget,L),(probs,n) in ring_choices.items():
                row[f'ring{budget}_L{L}_nll']=-float(probs[y].log())
                row[f'ring{budget}_L{L}_matches']=n
            future.append(row)
            for a,age,wrong_a,recovered,bad,scrambled,isolated,removed,current,stored,vector,donor,wrong_event_correct,wrong_event_logits,removed_logits,norm_error in pending:
                truth=int(raw[a]);historical_book=books[a]
                hist_hash.update(owner.encode()+b':'+str(t).encode()+b':'+str(a).encode()+b':'+bytes([truth])+b';')
                history.append({'owner':owner,'clock':t,'anchor':a,'age':age,'historical_byte':truth,
                    'wrong_content_donor':donor,'norm_matched_wrong_event_correct':int(wrong_event_correct==truth) if donor is not None else None,
                    'wrong_event_minus_full_nextbyte_nll':score(wrong_event_logits,y)['nll']-row['full_nll'] if donor is not None else None,
                    'event_removed_minus_full_nextbyte_nll':score(removed_logits,y)['nll']-row['full_nll'] if donor is not None else None,
                    'matched_event_norm_abs_error':norm_error,'matrix_correct':int(recovered==truth),'wrong_address_correct':int(bad==truth) if bad is not None else None,
                    'wrong_address_anchor':wrong_a,'permuted_content_correct':int(scrambled==truth),
                    'zero_correct':int(null_pred==truth),'isolated_event_correct':int(isolated==truth),
                    'event_removed_correct':int(removed==truth),'current_value_correct':int(current==truth),
                    'value_cosine':float((F.normalize(vector,dim=0,eps=1e-12)*historical_book[truth]).sum()),
                    'ring256_available':stored[256] is not None,'ring256_correct':int(stored[256]==truth),
                    'ring4160_available':stored[4160] is not None,'ring4160_correct':int(stored[4160]==truth)})
            state=nxt
    if len(future)!=768 or len(history)!=2528:raise ValueError(f'cohort grid {len(future)}/{len(history)}')
    if engine._weight_hash()!=weight_hash:raise ValueError('weights mutated')
    fields=('norm_matched_wrong_event_correct','wrong_event_minus_full_nextbyte_nll','event_removed_minus_full_nextbyte_nll','matched_event_norm_abs_error','matrix_correct','wrong_address_correct','permuted_content_correct','zero_correct',
            'isolated_event_correct','event_removed_correct','current_value_correct','value_cosine',
            'ring256_available','ring256_correct','ring4160_available','ring4160_correct')
    metrics={str(age):{k:sum(x[k] for x in history if x['age']==age and x[k] is not None)/sum(1 for x in history if x['age']==age and x[k] is not None) for k in fields} for age in AGES}
    return {'study':'LANGMEM082','seed':seed,'checkpoint_sha256':receipt['checkpoint_sha256'],
        'source_fit_order_sha256':receipt['source_sample_order_sha256'],'no_new_fits':True,'gpu_minutes':0,
        'protected_TEST':'NOT_ACCESSED','independently_qualified':False,'parameters':499627,
        'native_state_tensor_bytes':4160,'slow_tensor_bytes':256,'ring_total_bytes':[256,4160],
        'ring_payload_bytes':[240,4144],'ring_header_bytes':16,'history_writer_side_table':'EVALUATOR_ONLY_NOT_NATIVE_STATE',
        'isolated_events':'EVALUATOR_ONLY_NOT_NATIVE_STATE','decoder':'UNTRAINED_FIXED_COSINE_CODEBOOK_NOT_NATIVE_DECODER',
        'actual_source_nextbyte_sha256':target_hash.hexdigest(),'historical_targets_sha256':hist_hash.hexdigest(),
        'max_actual_source_affine_error':max_error,'original_prefix_error':prefix_error,
        'actual_source_future_poison_pairs':poison,'bitwise_sham_pairs':shams,'checkpoint_unchanged':True,
        'history_metrics':metrics,'future_metrics':{k:sum(x[k] for x in future if x[k] is not None)/sum(1 for x in future if x[k] is not None) for k in future[0] if k not in ('owner','clock','target')},
        'cpu_process_seconds':time.process_time()-started,'timing':timers,'history_rows':history,'future_rows':future,
        'limits':['Previously exposed DEV, no independent generalization','Original 499627p historical checkpoint; no fit',
                  'Old writer key and clock supplied by evaluator; not autonomous old-event selection',
                  'Codebook nearest direction failure does not prove information-theoretic impossibility',
                  'Permutation preserves matrix norm but is not a naturally occurring wrong document',
                  'Native payload excludes owner/clock metadata; ring includes16B cursors; Python allocator excluded for both',
                  'Raw ring lookup is byte storage, not learned semantics; full4160B ring unsaturated on320-byte streams',
                  'One-step lesions hold other prior state fixed; full historical interventions OPEN',
                  'Whole-model IEEE/AD formal proof and general information-flow isolation OPEN']}

def main():
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int,choices=SEEDS,required=True)
    p.add_argument('--checkpoint-dir',type=Path,required=True);p.add_argument('--original-aggregate',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise FileExistsError('immutable output exists')
    result=evaluate(a.seed,a.checkpoint_dir,a.original_aggregate)
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2,sort_keys=True,allow_nan=False)+'\n')
    print(json.dumps({'seed':a.seed,'history_metrics':result['history_metrics'],'future_metrics':result['future_metrics']}))
if __name__=='__main__':main()
