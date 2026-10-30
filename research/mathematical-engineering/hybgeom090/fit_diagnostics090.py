#!/usr/bin/env python3
"""HYBGEOM090: authentic original publisher FIT, original Git-backed checkpoints, NO FIT.

This is an intentionally nonqualifying mechanism comparison. Text has no true
interaction timestamps or context-specific human semantic judgments, so this
cannot establish continuous-time adaptation or semantic usefulness.
"""
from __future__ import annotations
import argparse,hashlib,importlib.util,json,sys,time
from pathlib import Path
from collections import defaultdict
import torch
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from scripts.replay_langmem086087 import fit_sources,load_source,original_checkpoints,weights

OP=Path(__file__).with_name('operators090.py')
spec=importlib.util.spec_from_file_location('hybgeom090_operators',OP)
op=importlib.util.module_from_spec(spec);sys.modules[spec.name]=op;spec.loader.exec_module(op)
ARMS=('original','fixed_euclidean','contextual_euclidean','contextual_spd','hybrid','hybrid_contextual_spd')
SEEDS=(1701,1702,1703)
START=64;STOP=320;RATES=(.015,.15);RESETS=(1.,.9);ALPHA=.5;GAMMA=.15;STRENGTH=.01


def digest(x):return hashlib.sha256(x).hexdigest()

@torch.inference_mode()
def evaluate(model,owner,raw):
    st=model.start((owner,))
    hs=op.start(model,(owner,))
    scores=defaultdict(lambda:[0.,0.,0])
    max_upper=0.; latencies=defaultdict(float)
    for t in range(STOP):
        byte=int(raw[t]);observed=torch.tensor([[byte]],dtype=torch.long)
        t0=time.process_time();logits,st=model(observed,state=st,owners=(owner,))
        latencies['original']+=time.process_time()-t0
        t0=time.process_time();hlog,hs=op.hybrid_token(model,observed,hs,float(t),rates=RATES,resets=RESETS)
        latencies['hybrid']+=time.process_time()-t0
        max_upper=max(max_upper,float(st.tensors['upper'].norm()),float(hs.core.tensors['upper'].norm()))
        if t<START: continue
        base=logits[:,0]; hybrid=hlog[:,0];last=observed[:,0]
        t0=time.process_time()
        fixed=op.geometry_logits(base,model.embedding.weight,last,st.tensors['upper'],contextual=False,alpha=0.,strength=STRENGTH)
        euro=op.geometry_logits(base,model.embedding.weight,last,st.tensors['upper'],contextual=True,alpha=0.,strength=STRENGTH)
        spd=op.geometry_logits(base,model.embedding.weight,last,st.tensors['upper'],contextual=True,alpha=ALPHA,strength=STRENGTH)
        hybrid_spd=op.geometry_logits(hybrid,model.embedding.weight,last,hs.core.tensors['upper'],contextual=True,alpha=ALPHA,strength=STRENGTH)
        latencies['geometry_4_arms']+=time.process_time()-t0
        candidate={'original':base,'fixed_euclidean':fixed,'contextual_euclidean':euro,
                   'contextual_spd':spd,'hybrid':hybrid,'hybrid_contextual_spd':hybrid_spd}
        # Only after all predictions are computed may the observed continuation be scored.
        y=int(raw[t+1])
        for name,arr in candidate.items():
            lp=arr[0].log_softmax(-1)
            values=scores[name]
            values[0]+=float(-lp[y]);values[1]+=int(int(arr[0].argmax())==y);values[2]+=1
    return {a:{'nll':scores[a][0]/scores[a][2], 'accuracy':scores[a][1]/scores[a][2],
               'n':int(scores[a][2])} for a in ARMS},latencies,max_upper


def run():
    torch.set_num_threads(2)
    manifest=json.loads((ROOT/'research/replication/langmem086087/MANIFEST.json').read_text())
    if manifest['DEV_scoring'] or manifest['new_fits'] or manifest['protected_TEST']:
        raise ValueError('original archive scope changed')
    docs,source_sha=fit_sources(ROOT)
    if source_sha!=manifest['fit_source_sha256'] or len(docs)!=14:
        raise ValueError('authentic original FIT source bytes mismatch')
    cls=load_source(ROOT,manifest)
    # The helper checks every original archive SHA, checkpoint SHA, and parent chain.
    originals=original_checkpoints(ROOT,manifest)
    checkpoints={r['seed']:(r,cp,fr) for r,cp,fr in originals if r['steps']==1280}
    if tuple(sorted(checkpoints))!=SEEDS: raise ValueError('3 original1280 checkpoints required')
    owners=(sorted(docs)[0],sorted(docs)[-1]);cases=[]
    for owner in owners:
        if len(docs[owner])<=STOP: raise ValueError('source too short')
    for seed in SEEDS:
        info,cp,fit=checkpoints[seed]
        model=cls().eval();model.load_state_dict(cp['model'],strict=True)
        if model.counts()['trainable']!=100521 or weights(model)!=fit['trained_model_hash']:
            raise ValueError('actual original checkpoint count/hash mismatch')
        original_hash=weights(model)
        for owner in owners:
            metrics,timing,upper=evaluate(model,owner,docs[owner])
            if weights(model)!=original_hash: raise ValueError('original weights changed during frozen inference')
            cases.append({'seed':seed,'owner':owner,'original_checkpoint_sha256':info['checkpoint_sha256'],
                 'source_owner_sha256':digest(docs[owner]),'observed_prefix_bytes':STOP,
                 'scored_next_bytes':STOP-START,'score_start_ordinal':START,'score_end_ordinal_exclusive':STOP,
                 'scores':metrics,'cpu_process_seconds':dict(timing),'max_upper_norm':upper})
    total={}
    for a in ARMS:
        vals=[x['scores'][a] for x in cases]
        n=sum(v['n'] for v in vals)
        total[a]={'n':n,'nll':sum(v['n']*v['nll'] for v in vals)/n,
                  'accuracy':sum(v['n']*v['accuracy'] for v in vals)/n}
    for a in ARMS[1:]:total[a]['delta_nll_vs_original']=total[a]['nll']-total['original']['nll']
    return {'study_id':'HYBGEOM090','level':'internal_no_fit_original_FIT_diagnostic_not_capability',
            'source':'publisher-authenticated original CPython 14 FIT docs, first and last sorted owners',
            'source_sha256':source_sha,'seeds':SEEDS,'original_training_steps':1280,
            'checkpoint_archive_verification':'all original archives/checkpoints SHA256 and ancestry verified',
            'new_fit_steps':0,'gpu_minutes':0,'DEV_scored':False,'TEST_opened':False,
            'time_axis':'byte ordinal t only; NOT real elapsed/interaction time',
            'geometry':'frozen original embedding and upper-state context; no human semantic judgments',
            'strength':STRENGTH,'context_gamma':GAMMA,'metric_alpha':ALPHA,
            'flow_rates':RATES,'mode_reset_factors':RESETS,
            'cases':cases,'aggregate':total,
            'meaning':'NLL is source FIT and not a semantic/adaptation/generalization or architecture superiority result',
            'unknown':['real time responsiveness','held-out context-annotated semantic quality',
                       'trained time-aware geometry','full hybrid stability including TAC core resets',
                       'physical multi-agent macro sufficiency','independent qualification']}


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True,type=Path);a=p.parse_args()
    if a.output.exists():raise FileExistsError('never overwrite evidence')
    result=run();a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,indent=2,sort_keys=True,allow_nan=False)+'\n')
    print(json.dumps({'study':result['study_id'],'nll':{k:v['nll'] for k,v in result['aggregate'].items()},
                      'count':result['aggregate']['original']['n'],'gpu_minutes':0}))
if __name__=='__main__':main()
