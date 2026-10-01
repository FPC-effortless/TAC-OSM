#!/usr/bin/env python3
"""Run preregistered TACOSM-C5-PROTOTYPE-BEAM-001."""

from __future__ import annotations
import json, math, statistics, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from tac_osm import Query
from tac_osm.c5_end_to_end import EndToEndExecutor, EndToEndTask, build_population, build_task, prepare_state
from tac_osm.contract import load_contract
from tac_osm.learned_prototype_state import LearnedPrototypeStateIndex, PrototypeStateIndexConfig
from tac_osm.learned_state_index import LearnedSemanticStateIndex, LearnedStateIndexConfig
from tac_osm.noisy_state_tasks import CODEBOOK

SEEDS=(0,1,2,3,4); H_LEVELS=(64,128,256); H_ANCHOR=64; M=64; R=4; K=8
EPOCHS=512; EVAL_STEPS=100; NEGATIVE_COUNT=8; LATENT_DIM=16
TRAIN_CODES=tuple(CODEBOOK[16:64]); EVAL_CODES=tuple(CODEBOOK[:16])
QUERY_PROJECTION_MACS=10*LATENT_DIM; EXHAUSTIVE_STATE_SCORE_MACS=M*LATENT_DIM
PROTOTYPE_SCORE_MACS=16*LATENT_DIM; MAX_RERANK_MACS=K*LATENT_DIM

def _model(seed):
    return LearnedSemanticStateIndex(LearnedStateIndexConfig(input_dim=10,latent_dim=LATENT_DIM,learning_rate=0.02,margin=0.25,epochs=EPOCHS,bucket_bits=8,probe_radius=1,shortlist_k=1,seed=seed))

def _task(seed,h,step):
    base=build_task(seed,H_ANCHOR,step); candidates=build_population(seed,h)
    relevant=frozenset(i for i,c in enumerate(candidates) if tuple(c.descriptor)==base.target_value)
    if len(relevant)!=R: raise AssertionError('registered task must have four relevant programs')
    expected,work,calls=EndToEndExecutor().execute_population(tuple(candidates[i] for i in sorted(relevant)),base.target_value)
    assert work==R*12 and calls==R
    return EndToEndTask(query=base.query,state_updates=base.state_updates,candidates=candidates,target_address=base.target_address,target_value=base.target_value,expected_output=expected,relevant_indices=relevant,step=step)

def _pool(state):
    out=[]
    for address in state.addresses():
        read=state.read(Query(text='\t'+address,step=state.current_step,provenance='c5_prototype_beam_pool'))
        if read.keys: out.append((address,tuple(int(x) for x in read.values[0])))
    if len(out)!=M: raise AssertionError('all 64 state items must be readable')
    return tuple(out)

def _norm(v):
    n=math.sqrt(sum(x*x for x in v))
    if n<=1e-8: raise AssertionError('zero embedding')
    return tuple(x/n for x in v)

def _cos(a,b): return sum(x*y for x,y in zip(a,b))

def run_cell(seed,h):
    first=_task(seed,h,0); state=prepare_state(first); pool=_pool(state)
    model=_model(seed)
    updates=model.train_with_negative_coverage(TRAIN_CODES,negative_count=NEGATIVE_COUNT,aggregation='mean',positive_views=1)
    if updates!=EPOCHS*len(TRAIN_CODES): raise AssertionError('update count changed')
    exhaustive_embeddings=tuple(_norm(model.encode_state(value)) for _,value in pool)
    selective=LearnedPrototypeStateIndex(model,PrototypeStateIndexConfig(prototype_count=16,bucket_capacity=4,kmeans_iterations=8))
    build=selective.build(state,TRAIN_CODES)
    executor=EndToEndExecutor()
    ex_target=[]; prop_target=[]; sel_target=[]; ex_success=[]; sel_success=[]
    sizes=[]; proto_macs=[]; rerank_macs=[]; ex_work=[]; sel_work=[]
    for step in range(EVAL_STEPS):
        task=_task(seed,h,step); q=_norm(model.encode_query(task.query))
        scores=[_cos(q,emb) for emb in exhaustive_embeddings]
        ex_i=max(range(M),key=lambda i:(scores[i],-i)); ex_addr=pool[ex_i][0]; ex_value=pool[ex_i][1]
        ex_target.append(int(ex_addr==task.target_address))
        ex_rel=tuple(i for i,c in enumerate(task.candidates) if tuple(c.descriptor)==tuple(ex_value))
        ex_out,ew,_=executor.execute_population(tuple(task.candidates[i] for i in ex_rel),ex_value)
        ex_success.append(int(ex_out==task.expected_output)); ex_work.append(ew)
        hit=selective.lookup_beam(task.query,beam_width=2)
        prop_target.append(int(task.target_address in hit.candidate_addresses)); sizes.append(len(hit.candidate_addresses))
        proto_macs.append(hit.prototype_score_macs); rerank_macs.append(hit.state_rerank_macs)
        if hit.address is None:
            sel_target.append(0); sel_success.append(0); sel_work.append(0); continue
        value=next(v for a,v in pool if a==hit.address); sel_target.append(int(hit.address==task.target_address))
        rel=tuple(i for i,c in enumerate(task.candidates) if tuple(c.descriptor)==tuple(value))
        out,sw,_=executor.execute_population(tuple(task.candidates[i] for i in rel),value)
        sel_success.append(int(out==task.expected_output)); sel_work.append(sw)
    return {
      'seed':seed,'H':h,'M':M,'R':R,'K':K,'queries':EVAL_STEPS,
      'exhaustive_target_recall':statistics.fmean(ex_target),
      'proposal_actual_target_retention':statistics.fmean(prop_target),
      'selective_actual_target_recall':statistics.fmean(sel_target),
      'exhaustive_end_to_end_success':statistics.fmean(ex_success),
      'selective_actual_end_to_end_success':statistics.fmean(sel_success),
      'shortlist_size_mean':statistics.fmean(sizes),
      'prototype_score_macs_mean':statistics.fmean(proto_macs),
      'state_rerank_macs_mean':statistics.fmean(rerank_macs),
      'selective_query_macs_mean':QUERY_PROJECTION_MACS+statistics.fmean(proto_macs)+statistics.fmean(rerank_macs),
      'exhaustive_query_macs':QUERY_PROJECTION_MACS+EXHAUSTIVE_STATE_SCORE_MACS,
      'build_macs_total':build.total_build_macs,
      'exhaustive_executor_work_mean':statistics.fmean(ex_work),
      'selective_executor_work_mean':statistics.fmean(sel_work),
    }

def main():
    contract=load_contract('TACOSM-C5-PROTOTYPE-BEAM-001')
    contract.require_levels(H_LEVELS); contract.require_seeds(SEEDS); contract.require_steps(EPOCHS); contract.require_eval_steps(EVAL_STEPS); contract.require_k_levels([K]); contract.require_arms(['exhaustive_cosine','prototype_beam'])
    if set(TRAIN_CODES)&set(EVAL_CODES): raise AssertionError('training/evaluation semantic codes overlap')
    cells=[run_cell(seed,h) for h in H_LEVELS for seed in SEEDS]
    def f(name): return statistics.fmean(c[name] for c in cells)
    ex_t=f('exhaustive_target_recall'); sel_t=f('selective_actual_target_recall'); ex_s=f('exhaustive_end_to_end_success'); sel_s=f('selective_actual_end_to_end_success')
    max_short=max(c['shortlist_size_mean'] for c in cells)
    result={'protocol':{'name':'TACOSM-C5-PROTOTYPE-BEAM-001','seeds':list(SEEDS),'H_levels':list(H_LEVELS),'M':M,'R':R,'K':K,'latent_dim':LATENT_DIM,'epochs':EPOCHS,'negative_count':NEGATIVE_COUNT,'positive_views':1,'prototype_count':16,'bucket_capacity':4,'prototype_beam':2,'queries_per_seed_per_H':EVAL_STEPS},'overall':{'exhaustive_target_recall_mean':ex_t,'selective_actual_target_recall_mean':sel_t,'exhaustive_end_to_end_success_mean':ex_s,'selective_actual_end_to_end_success_mean':sel_s,'state_target_gap':ex_t-sel_t,'end_to_end_gap':ex_s-sel_s,'max_shortlist_size_mean':max_short,'capability_rule_pass':sel_t>=ex_t-0.05 and sel_s>=ex_s-0.05 and max_short<=K,'exhaustive_query_macs':f('exhaustive_query_macs'),'selective_query_macs_mean':f('selective_query_macs_mean'),'build_macs_total':f('build_macs_total')},'by_H':{},'cells':cells}
    for h in H_LEVELS:
        g=[c for c in cells if c['H']==h]
        result['by_H'][str(h)]={'exhaustive_target_recall_mean':statistics.fmean(c['exhaustive_target_recall'] for c in g),'proposal_target_retention_mean':statistics.fmean(c['proposal_actual_target_retention'] for c in g),'selective_target_recall_mean':statistics.fmean(c['selective_actual_target_recall'] for c in g),'exhaustive_end_to_end_success_mean':statistics.fmean(c['exhaustive_end_to_end_success'] for c in g),'selective_end_to_end_success_mean':statistics.fmean(c['selective_actual_end_to_end_success'] for c in g),'shortlist_size_mean':statistics.fmean(c['shortlist_size_mean'] for c in g)}
    out=Path('artifacts/TACOSM-C5-PROTOTYPE-BEAM-001.json'); out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n'); print(json.dumps(result,indent=2,sort_keys=True))

if __name__=='__main__': main()