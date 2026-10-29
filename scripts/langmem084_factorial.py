#!/usr/bin/env python3
"""No-fit source-value/key/update factorial; NOT an integrated trained TAC change."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import torch
from torch.nn import functional as F
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'src'),str(ROOT/'research/mathematical-engineering/langmem084')]
from operators import ARMS,AGES,arm_parts,context_key,byte_book,update_matrix,decode,select_controls
from scripts.langmem082_recover import codebook,RawRing
from scripts.langmem081_context_address import restore_original,source_factors,DualNativeStreamingTAC079
from scripts.langmem083_fit_eval import read_unseen_publisher_docs

SEEDS=(1701,1702,1703)
CLOCKS=range(128,640,4)


def summarize(rows):
    result={}
    for arm in ARMS:
        result[arm]={}
        for age in AGES:
            data=[r for r in rows if r['arm']==arm and r['age']==age]
            metrics={k:sum(r[k] for r in data)/len(data) for k in (
                'correct','wrong_hard_correct','wrong_uniform_correct','zero_correct','true_value_cosine',
                'wrong_hard_cosine','ring256_available','ring256_correct')}
            metrics['count']=len(data)
            for label,selector in [('unique_context',lambda r:r['context_occurrences']==1),
                                   ('conflicting_context',lambda r:r['same_context_distinct_bytes']>1)]:
                subset=[r for r in data if selector(r)]
                metrics[label]={'count':len(subset),**{k:(sum(r[k] for r in subset)/len(subset) if subset else None) for k in ('correct','wrong_hard_correct','wrong_uniform_correct')}}
            result[arm][str(age)]=metrics
    return result

@torch.no_grad()
def evaluate(seed,checkpoint_dir,verified,documents):
    torch.set_num_threads(2)
    m,receipt=restore_original(seed,checkpoint_dir,verified)
    for p in m.parameters():p.requires_grad_(False)
    engine=DualNativeStreamingTAC079(m,'cap_all');weight_hash=engine._weight_hash()
    books={a:codebook(m,a) for a in range(640)};bits=byte_book()
    docs=read_unseen_publisher_docs(documents) # These are EXPOSED by LANGMEM083.
    rows=[];cohort=hashlib.sha256();source_error=0.;prefix_error=0.;source_steps=0
    poison_pairs=0;write_pairs=0;shams=0;started=time.process_time()
    geometry=[];costs={a:0. for a in ARMS}
    for owner,raw in sorted(docs.items()):
        if len(raw)<641:raise ValueError('registered source length')
        state=engine.start((owner,));bank={a:torch.zeros(8,8) for a in ARMS}
        keys={'native':{},'context_hash':{}};contexts={};ring=RawRing(256)
        for t in range(640):
            token=torch.tensor([raw[t]],dtype=torch.long)
            f=source_factors(m,state,token)
            logits,nxt=engine.step(token,state,owners=(owner,))
            observed=raw[:t+1]
            old_context=raw[max(0,t-4):t].rjust(4,b'\0')
            current_keys={'native':f['write_key'].reshape(8),'context_hash':context_key(old_context)}
            before_bank={k:v.clone() for k,v in bank.items()} if t==128 else None
            if t>0:
                for arm,M in bank.items():
                    value_kind,key_kind,mode=arm_parts(arm)
                    value=f['value'].reshape(8) if value_kind=='native' else bits[int(raw[t])]
                    start=time.perf_counter()
                    bank[arm]=update_matrix(M,current_keys[key_kind],value,mode=mode,
                        rho=f['rho_slow'],beta=f['beta_slow'],epsilon=f['write_slow'])
                    costs[arm]+=time.perf_counter()-start
            err=float((bank['native_native_source']-nxt.tensors['slow_native'][0]).abs().max())
            source_error=max(source_error,err);source_steps+=1
            if err>1e-5:raise ValueError('unaltered native baseline not actual source')
            if t>=4:
                for k in keys:keys[k][t]=current_keys[k].clone()
                contexts.setdefault(old_context,[]).append(t)
            ring.append(int(raw[t]))
            if t==127:
                original=m(torch.tensor(list(raw[:128]),dtype=torch.long).view(1,-1))[:,-1]
                prefix_error=max(prefix_error,float((original-logits).abs().max()))
                if prefix_error>3e-5:raise ValueError('original graph prefix mismatch')
            if t==128:
                # Actual source pipeline poison: prefix, token, factors and next state.
                changed=bytearray(raw);changed[t+1:]=bytes((b+113)%256 for b in changed[t+1:])
                poisoned=torch.tensor([changed[t]],dtype=torch.long)
                plogits,pstate=engine.step(poisoned,state,owners=(owner,))
                pf=source_factors(m,state,poisoned)
                if not torch.equal(plogits,logits) or any(not torch.equal(pstate.tensors[k],nxt.tensors[k]) for k in nxt.tensors):raise ValueError('source future poison')
                if any(not torch.equal(pf[k],f[k]) for k in ('query','write_key','value')):raise ValueError('source future factors')
                hash_poison=context_key(bytes(changed[max(0,t-4):t]))
                if not torch.equal(hash_poison,current_keys['context_hash']):raise ValueError('hash future poison')
                for arm,prior_matrix in before_bank.items():
                    vk,kk,mode=arm_parts(arm)
                    pv=pf['value'].reshape(8) if vk=='native' else bits[changed[t]]
                    pk=pf['write_key'].reshape(8) if kk=='native' else hash_poison
                    altered_bank=update_matrix(prior_matrix,pk,pv,mode=mode,rho=pf['rho_slow'],beta=pf['beta_slow'],epsilon=pf['write_slow'])
                    if not torch.equal(altered_bank,bank[arm]):raise ValueError('actual bank future poison')
                poison_pairs+=1
                alt=source_factors(m,state,torch.tensor([raw[t]^1],dtype=torch.long))
                if not torch.equal(alt['write_key'],f['write_key']):raise ValueError('current-byte write-key leakage')
                if torch.equal(bits[raw[t]],bits[raw[t]^1]):raise ValueError('byte code not injective')
                write_pairs+=1
                sham_logits,_=engine.step(token,state,owners=(owner,))
                if not torch.equal(sham_logits,logits):raise ValueError('source sham')
                shams+=1
            if t not in CLOCKS:state=nxt;continue
            for age in AGES:
                a=t-age
                if a<4:continue
                selectors={kind:select_controls(observed,keys[kind],a,t,owner) for kind in keys}
                # Freeze ALL decoder/control predictions before scoring raw[a].
                predictions={}
                for arm,M in bank.items():
                    vk,kk,_=arm_parts(arm);book=books[a] if vk=='native' else bits
                    hard,uniform,cos=selectors[kk];key=keys[kk][a]
                    pred,vector=decode(M,key,book)
                    wrong,_=decode(M,keys[kk][hard],book);random,_=decode(M,keys[kk][uniform],book)
                    predictions[arm]=(pred,wrong,random,vector,cos,hard,uniform)
                stored=ring.get(a)
                context=observed[a-4:a];same=contexts[context]
                # Past-content evaluator metadata and labels are now revealed.
                y=int(raw[a]);distinct=len({raw[b] for b in same})
                cohort.update(owner.encode()+b':'+str(t).encode()+b':'+str(a).encode()+b':'+bytes([y])+b';')
                for arm,(pred,wrong,random,vector,cos,hard,uniform) in predictions.items():
                    vk,kk,mode=arm_parts(arm);book=books[a] if vk=='native' else bits
                    rows.append({'owner':owner,'clock':t,'anchor':a,'age':age,'arm':arm,'historical_byte':y,
                        'correct':int(pred==y),'zero_correct':int(y==0),'wrong_hard_correct':int(wrong==y),'wrong_uniform_correct':int(random==y),
                        'predicted_byte':pred,'wrong_hard_predicted_byte':wrong,'wrong_uniform_predicted_byte':random,
                        'true_value_cosine':float((F.normalize(vector,dim=0,eps=1e-12)*book[y]).sum()),
                        'wrong_hard_cosine':cos,'wrong_hard_anchor':hard,'wrong_uniform_anchor':uniform,
                        'context_occurrences':len(same),'same_context_distinct_bytes':distinct,'latest_context_event':same[-1],
                        'ring256_available':stored is not None,'ring256_correct':int(stored==y)})
            state=nxt
        # Source address geometry is descriptive, not a fitted key transform.
        for kind in keys:
            stack=torch.stack([v for a,v in keys[kind].items() if a>=4]).double()
            spectrum=torch.linalg.eigvalsh(stack.T@stack).clamp_min(0)
            effective_rank=float(spectrum.sum()**2/(spectrum*spectrum).sum())
            geometry.append({'owner':owner,'key_kind':kind,'uncentered_participation_rank':effective_rank,
                             'mean_key_norm':float(torch.linalg.vector_norm(stack.mean(0)))})
    if len(rows)!=956*8 or source_steps!=1280:raise ValueError('original registered cohort incomplete')
    if engine._weight_hash()!=weight_hash:raise ValueError('frozen source weights changed')
    if any(M.numel()*M.element_size()!=256 for M in bank.values()):raise ValueError('bank payload')
    return {'study':'LANGMEM084','seed':seed,'original_checkpoint_sha256':receipt['checkpoint_sha256'],
        'source_fit_sample_order_sha256':receipt['source_sample_order_sha256'],'frozen_weights_sha256':weight_hash,
        'new_trainable_parameters':0,'optimizer_updates':0,'no_new_fits':True,'weights_unchanged':True,
        'original_parameters':499627,'original_state_tensor_bytes':4160,'bank_tensor_bytes_each':256,
        'ring_total_bytes':256,'ring_payload_bytes':240,'evaluator_history':'EXTERNAL_OLD_KEYS_CONTEXTS_NOT_BANK_STATE',
        'full_arm_deployment':'NOT_INTEGRATED_4160BYTE_TAC; shared original4160B producer plus diagnostic bank256B',
        'source_role':'LANGMEM083_DOCUMENTS_NOW_PREVIOUSLY_EXPOSED_DEV','source_documents_sha256':{k:hashlib.sha256(v).hexdigest() for k,v in docs.items()},
        'source_historical_targets_sha256':cohort.hexdigest(),'historical_cells_per_seed':956,'repeated_arm_rows':len(rows),
        'max_actual_native_bank_error':source_error,'original_prefix_error':prefix_error,
        'actual_source_future_poison_pairs':poison_pairs,'source_write_key_current_byte_invariance_pairs':write_pairs,
        'bitwise_source_sham_pairs':shams,'key_geometry':geometry,'bank_update_wall_seconds':costs,
        'cpu_process_seconds':time.process_time()-started,'metrics':summarize(rows),'rows':rows,
        'gpu_minutes':0,'protected_TEST':'NOT_ACCESSED','independently_qualified':False,
        'limits':['No-fit deterministic mechanism interventions; not matched trained-architecture evidence',
                  'Evaluator supplies old keys and clocks; no autonomous historical selection',
                  'Shared frozen source producer state4160B and evaluator history excluded from256B mechanism bank',
                  'Same PSF documents now exposed by083; no unseen or independent generalization claim',
                  'Source and unit-overwrite updates differ in gain, erase and operation count; fixed factorial not an optimized algorithm contest',
                  'Repeated contexts can alias distinct old continuation bytes; stratified outcomes retained',
                  'Whole model IEEE/AD, general leakage clearance, task-level usefulness and integrated deployment OPEN']}

def main():
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int,choices=SEEDS,required=True)
    p.add_argument('--checkpoint-dir',type=Path,required=True);p.add_argument('--original-aggregate',type=Path,required=True)
    p.add_argument('--documents',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise FileExistsError('immutable evidence output exists')
    result=evaluate(a.seed,a.checkpoint_dir,a.original_aggregate,a.documents)
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2,sort_keys=True,allow_nan=False)+'\n')
    print(json.dumps({k:result[k] for k in ('study','seed','metrics','key_geometry','source_historical_targets_sha256')}))
if __name__=='__main__':main()
