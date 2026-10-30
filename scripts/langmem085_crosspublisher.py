#!/usr/bin/env python3
"""LANGMEM085 first new Rust publisher no-fit causal 256B write-policy trial.

Only trained original 078 checkpoint as frozen source producer; this is NOT
a trained TAC decoder or a deployed autonomous historical address index.
"""
import argparse,hashlib,json,math,sys,time
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1]
for n in ("langmem085","langmem084","langmem083","langmem082","langmem081","langmem080","langmem079","langmem078","langscale075","langscale070","context026","langmemory042","langbase024"):
    p=str(ROOT/"research/mathematical-engineering"/n)
    if p not in sys.path:sys.path.insert(0,p)
sys.path[:0]=[str(ROOT/"research/mathematical-engineering/langmem085"),str(ROOT),str(ROOT/"src")]
from operators085 import ARMS,AGES,write,prospective_wrong_keys,load_frozen_new_rust,fixed_fit_prior,context_key,byte_book,decode
from scripts.langmem081_context_address import restore_original,source_factors,DualNativeStreamingTAC079
from scripts.langmem082_recover import RawRing
from scripts.langlayer035_sampling import fit_sources
SEEDS=(1701,1702,1703)

def summarize(rows):
    result={}
    for arm in ARMS:
        result[arm]={}
        for age in AGES:
            group=[r for r in rows if r['arm']==arm and r['age']==age]
            if not group:raise ValueError('no age source cell')
            n=len(group)
            result[arm][str(age)]={
              'count':n,
              'correct':sum(r['correct'] for r in group)/n,
              'hard_wrong':sum(r['hard_wrong_correct'] for r in group)/n,
              'uniform_wrong':sum(r['uniform_wrong_correct'] for r in group)/n,
              'FIT_prior':sum(r['FIT_prior_correct'] for r in group)/n,
              'raw_ring_available':sum(r['raw_ring_available'] for r in group)/n,
              'raw_ring_correct':sum(r['raw_ring_correct'] for r in group)/n,
              'eta_mean':sum(r['eta'] for r in group)/n,
              'eta_full_fraction':sum(r['eta']>=.999 for r in group)/n,
              'unique_count':sum(r['context_occurrences']==1 for r in group),
              'unique_correct':sum(r['correct'] for r in group if r['context_occurrences']==1),
              'conflicting_count':sum(r['context_distinct_continuations']>1 for r in group),
              'conflicting_correct':sum(r['correct'] for r in group if r['context_distinct_continuations']>1),
              'paired_correct_minus_hard':sum(r['correct']-r['hard_wrong_correct'] for r in group)/n}
    return result

@torch.no_grad()
def evaluate(seed,ckpt,verified,source_dir):
    if seed not in SEEDS:raise ValueError('unregistered original checkpoint')
    torch.set_num_threads(2)
    m,receipt=restore_original(seed,ckpt,verified)
    for p in m.parameters():p.requires_grad_(False)
    engine=DualNativeStreamingTAC079(m,'cap_all')
    weight_sha=engine._weight_hash()
    docs=load_frozen_new_rust(source_dir,ROOT/'research/mathematical-engineering/langmem085/NEW_PUBLISHER_SOURCE.json')
    prior=fixed_fit_prior(fit_sources())
    modes=prior.argmax(dim=1).tolist()
    book=byte_book()
    rows=[];cohort=hashlib.sha256();cost={arm:0. for arm in ARMS}
    eta_all={arm:[] for arm in ARMS}
    start=time.process_time();max_prefix_error=0.;steps=0
    for owner,raw in sorted(docs.items()):
        state=engine.start((owner,))
        banks={arm:torch.zeros((8,8),dtype=torch.float32) for arm in ARMS}
        keys={};contexts={};ring=RawRing(256)
        for t in range(640):
            if state.clock!=t or state.tensor_bytes()!=4160:raise ValueError('original real TAC clock or state changed')
            x=torch.tensor([raw[t]],dtype=torch.long)
            f=source_factors(m,state,x)
            logits,nxt=engine.step(x,state,owners=(owner,))
            context=raw[max(0,t-4):t].rjust(4,bytes([0]))
            k=context_key(context)
            eta_latest={}
            if t>0:
                for arm,M in banks.items():
                    tic=time.perf_counter()
                    new,eta,_=write(M,k,book[int(raw[t])],arm,rho=f['rho_slow'],beta=f['beta_slow'],epsilon=f['write_slow'])
                    cost[arm]+=time.perf_counter()-tic
                    banks[arm]=new;eta_latest[arm]=eta;eta_all[arm].append(eta)
            if t>=4:
                keys[t]=k.clone();contexts.setdefault(context,[]).append(t)
            ring.append(int(raw[t]))
            if t==127:
                prefix=m(torch.tensor(list(raw[:128]),dtype=torch.long).view(1,-1))[:,-1]
                max_prefix_error=max(max_prefix_error,float((prefix-logits).abs().max()))
                if max_prefix_error>3e-5:raise ValueError('genuine original model prefix mismatch')
            steps+=1
            if t in range(128,640,4):
                observed=raw[:t+1]
                for age in AGES:
                    a=t-age
                    if a<4:continue
                    hard,uniform,pool=prospective_wrong_keys(observed,keys,a,t,owner)
                    predictions={}
                    for arm,M in banks.items():
                        pred,_=decode(M,keys[a],book)
                        hardpred,_=decode(M,keys[hard],book)
                        uniformpred,_=decode(M,keys[uniform],book)
                        predictions[arm]=(pred,hardpred,uniformpred)
                    retained=ring.get(a)
                    match=contexts[observed[a-4:a]]
                    # Only now reveal real source historical target to scoring.
                    y=int(raw[a]);different=len({raw[z] for z in match})
                    cohort.update(owner.encode()+b':'+str(t).encode()+b':'+str(a).encode()+b':'+bytes([y])+b';')
                    for arm,(pred,bad,rand) in predictions.items():
                        rows.append({'owner':owner,'clock':t,'anchor':a,'age':age,'arm':arm,
                            'historical_byte':y,'correct':int(pred==y),
                            'hard_wrong_correct':int(bad==y),'uniform_wrong_correct':int(rand==y),
                            'predicted_byte':pred,'hard_wrong_predicted_byte':bad,
                            'uniform_wrong_predicted_byte':rand,'wrong_hard_anchor':hard,
                            'wrong_uniform_anchor':uniform,'wrong_pool_size':pool,
                            'decoy_clean':int(not(a<hard<=a+4 or a<uniform<=a+4)),
                            'hard_wrong_age_diff':abs(hard-a),'uniform_wrong_age_diff':abs(uniform-a),
                            'FIT_prior_correct':int(y==modes[AGES.index(age)]),
                            'raw_ring_available':int(retained is not None),
                            'raw_ring_correct':int(retained==y),'eta':eta_latest[arm],
                            'context_occurrences':len(match),'context_distinct_continuations':different})
            state=nxt
        if state.clock!=640:raise ValueError('original source owner truncated')
    if len(rows)!=956*len(ARMS) or steps!=1280:raise ValueError('registered genuine source cohort wrong')
    if engine._weight_hash()!=weight_sha:raise ValueError('original weights changed')
    if any(M.numel()*M.element_size()!=256 for M in banks.values()):raise ValueError('operator matrix exceeds256B')
    return {'study':'LANGMEM085','seed':seed,'result_status':'NEW_RUST_PUBLISHER_CAUSAL_SELECTIVE_WRITE_DIAGNOSTIC',
       'source_role':'FIRST_NEW_RUST_PUBLISHER_DEV','original_checkpoint_sha256':receipt['checkpoint_sha256'],
       'original_sample_order_sha256':receipt['source_sample_order_sha256'],
       'frozen_model_weight_hash':weight_sha,'no_new_fits':True,'optimizer_updates':0,
       'original_pretrained_parameters':499627,'original_producer_FP32_state_bytes':4160,
       'bank_FP32_bytes':256,'proposed_tensors_total_FP32_bytes':4416,
       'external_writer_history':'NOT_INCLUDED_IN_BANK_OR_PROPOSED_4416B',
       'historical_cells_per_seed':956,'arm_rows_per_seed':len(rows),
       'source_steps':steps,'original_prefix_error':max_prefix_error,
       'strict_decoy_no_overlap_count':sum(r['decoy_clean'] for r in rows),
       'source_documents_sha256':{k:hashlib.sha256(v).hexdigest() for k,v in docs.items()},
       'historical_byte_cohort_sha256':cohort.hexdigest(),
       'FIT_original_modal_byte_per_age':{str(a):modes[i] for i,a in enumerate(AGES)},
       'eta_full_write_fraction':{arm:sum(x>=.999 for x in etas)/len(etas) for arm,etas in eta_all.items()},
       'operator_update_wall_seconds':cost,'cpu_process_seconds':time.process_time()-start,
       'metrics':summarize(rows),'rows':rows,
       'gpu_minutes':0,'protected_TEST':'NOT_ACCESSED','independently_qualified':False,
       'limits':['Rust publisher independent of pretrained CPython publisher but two chapters only',
         'Same fixed byte code and SHA-key across original seeds; seeds not independent model replications',
         'External evaluator writer key index and actual clock not bank-stored',
         'Bank payload256B excludes shared original4160B, history index and model weights',
         'Unmatched residual gate/source update compute/FLOPs; no trained architecture superiority',
         'Historical byte not next-byte model prediction or autonomous factual retrieval',
         'Repeated historical byte outcomes across arms/seeds, no protected TEST/merge']}

def main():
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int,choices=SEEDS,required=True)
    p.add_argument('--checkpoint-dir',type=Path,required=True);p.add_argument('--original-aggregate',type=Path,required=True)
    p.add_argument('--rust-sources',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise FileExistsError('source result overwrite forbidden')
    r=evaluate(a.seed,a.checkpoint_dir,a.original_aggregate,a.rust_sources)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(r,indent=2,sort_keys=True,allow_nan=False)+'\n')
    print('LANGMEM085_RUST_SOURCE '+json.dumps({'seed':r['seed'],'targets':r['historical_cells_per_seed'],'source_sha':r['historical_byte_cohort_sha256'],'metrics':{a:{d:{'correct':v['correct'],'hard':v['hard_wrong'],'prior':v['FIT_prior'],'eta':v['eta_mean']} for d,v in x.items()} for a,x in r['metrics'].items()}},sort_keys=True))
if __name__=='__main__':main()
