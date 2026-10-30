#!/usr/bin/env python3
"""HYBGEOM092; read-only authentic frozen HYBGEOM091 checkpoint counterfactuals.

Uses actual original CPython exposed DEV documents and native next-byte predictions.
Results are correlated across only two publishers documents and five training seeds.
"""
from __future__ import annotations
import argparse,hashlib,json,math,statistics,sys,time
from collections import defaultdict
from pathlib import Path
import torch
from torch.nn import functional as F
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'research/mathematical-engineering/hybgeom091'))
from model091 import HybridGeometryTAC091,EXPECTED
from run091 import source_docs,source_hash,state_dict_hash,SEEDS,EVAL_START,EVAL_STOP
from math092 import corrections,assert_equivalence
REG=ROOT/'research/mathematical-engineering/hybgeom092'
ARCHIVE=ROOT/'research/replay-bundles/HYBGEOM091'
ARMS=('original','fixed_euclidean')
METRICS=('original','fixed_full','fixed_algebraic','fixed_core_only','fixed_dot_only','fixed_norm_only','fixed_shuffled_dot')


def sha(raw:bytes)->str:return hashlib.sha256(raw).hexdigest()


def verified_model(arm:str,seed:int):
    location=ARCHIVE/f'{arm}-{seed}'
    results=json.loads((location/'RESULTS.json').read_text())
    raw=(location/'MODEL.pt').read_bytes()
    if sha(raw)!=results['checkpoint_sha256'] or results['arm']!=arm or results['seed']!=seed:
        raise ValueError('archive identity')
    saved=torch.load(location/'MODEL.pt',map_location='cpu',weights_only=True)
    if saved['study']!='HYBGEOM091' or saved['seed']!=seed or saved['arm']!=arm or saved['steps']!=96:
        raise ValueError('original fitted study')
    if saved['model_source_sha256']!=source_hash() or saved['source_sha256']!='911981cc63fd5436e0495cb067784e53475971131c811b7026e07d97a886d2d4':
        raise ValueError('actual original checkpoint source')
    model=HybridGeometryTAC091(arm).eval();model.load_state_dict(saved['state'],strict=True)
    if model.counts()['trainable']!=EXPECTED[arm] or state_dict_hash(model)!=results['weights_hash']:
        raise ValueError('weights/size identities')
    if len(saved['optimizer']['state'])!=len(list(model.parameters())) or any(int(v['step'])!=96 for v in saved['optimizer']['state'].values()):
        raise ValueError('original optimizer step count')
    return model,results,sha(raw)


@torch.no_grad()
def replay_doc(original,model,doc,*,owner):
    original_state=original.start((owner,))
    fixed_core_state=model.core.start((owner,))
    full_fixed_state=model.start((owner,))
    losses=defaultdict(list);correct=defaultdict(list)
    torch.set_num_threads(2)
    strength=.02*torch.sigmoid(model.raw_strength)
    max_logit_error=0.;max_identity_error=0.
    for t in range(EVAL_STOP):
        token=torch.tensor([[doc[t]]],dtype=torch.long)
        base,original_state=original(token,state=original_state,owners=(owner,))
        core_logits,fixed_core_state=model.core(token,state=fixed_core_state,owners=(owner,))
        full_logits,full_fixed_state=model(token,state=full_fixed_state,owners=(owner,))
        parts=corrections(model.core.embedding.weight,token[:,0],strength)
        max_identity_error=max(max_identity_error,assert_equivalence(parts))
        test=core_logits[:,0]+parts['original_distance']
        max_logit_error=max(max_logit_error,float((test-full_logits[:,0]).abs().max()))
        if t<EVAL_START:continue
        # The actual future byte is used only as a scoring target below.
        y=int(doc[t+1]);candidate={
            'original':base[:,0],
            'fixed_full':full_logits[:,0],
            'fixed_algebraic':core_logits[:,0]+parts['algebraic'],
            'fixed_core_only':core_logits[:,0],
            'fixed_dot_only':core_logits[:,0]+parts['similarity'],
            'fixed_norm_only':core_logits[:,0]+parts['vocab_norm'],
            'fixed_shuffled_dot':core_logits[:,0]+parts['shuffled_similarity'],
        }
        for k,logits in candidate.items():
            lp=F.log_softmax(logits,-1)
            losses[k].append(-float(lp[0,y]))
            correct[k].append(int(int(lp.argmax())==y))
    if max_logit_error>3e-5:raise ValueError('original fixed FULL logits not reconstructed')
    if any(len(v)!=224 for v in losses.values()):raise ValueError('incorrect original target count')
    return {k:{'nll':statistics.mean(losses[k]),'accuracy':statistics.mean(correct[k]),
               'token_nll':losses[k],'token_correct':correct[k]} for k in METRICS},max_logit_error,max_identity_error


def run():
    torch.set_num_threads(2)
    docs=source_docs() # original publisher SHA+Git blob verified
    allrows=[];seeds={}
    for seed in SEEDS:
        original,ref,original_sha=verified_model('original',seed)
        fixed,fixed_ref,fixed_sha=verified_model('fixed_euclidean',seed)
        case={'seed':seed,'checkpoint_sha256':{'original':original_sha,'fixed_euclidean':fixed_sha},'documents':{}}
        for owner,doc in sorted(docs.items()):
            predicted,logit_error,identity_error=replay_doc(original,fixed,doc,owner=owner)
            record={ 'owner':owner,'source_sha256':sha(doc),'scores':{k:{q:v for q,v in val.items() if not q.startswith('token_')} for k,val in predicted.items()},
                     'max_original_model_reconstruction_error':logit_error,
                     'max_euclidean_tied_identity_error':identity_error }
            old=[x for x in ref['exposed_DEV'] if x['owner']==owner][0]
            sibling=[x for x in fixed_ref['exposed_DEV'] if x['owner']==owner][0]
            if record['source_sha256'] != old['source_sha256'] or record['source_sha256']!=sibling['source_sha256']:
                raise ValueError('publisher DEV identity mismatch')
            for k,expected in [('original',old),('fixed_full',sibling)]:
                if abs(record['scores'][k]['nll']-expected['nll'])>3e-5 or any(abs(a-b['nll'])>3e-5 for a,b in zip(predicted[k]['token_nll'],expected['tokens'])):
                    raise ValueError('historical trained DEV predictions not identical')
            case['documents'][owner]=record
        case['mean']={k:{metric:statistics.mean(case['documents'][o]['scores'][k][metric] for o in case['documents']) for metric in ('nll','accuracy')} for k in METRICS}
        seeds[seed]=case
        print(json.dumps({'seed':seed,'means':{k:round(v['nll'],6) for k,v in case['mean'].items()}}),flush=True)
    summary={};baselines=['original','fixed_core_only']
    for key in METRICS:
        vals=[seeds[s]['mean'][key]['nll'] for s in SEEDS]
        deltas={}
        for parent in baselines:
            diff=[seeds[s]['mean'][key]['nll']-seeds[s]['mean'][parent]['nll'] for s in SEEDS]
            mu=statistics.mean(diff);half=2.776445*statistics.stdev(diff)/math.sqrt(len(diff))
            deltas[parent]={'mean':mu,'seed_values':diff,'seed_95pct_t_interval':[mu-half,mu+half],'wins':sum(v<0 for v in diff)}
        summary[key]={'mean_nll':statistics.mean(vals),'mean_accuracy':statistics.mean(seeds[s]['mean'][key]['accuracy'] for s in SEEDS),'paired':deltas}
    return {'study':'HYBGEOM092','status':'INTERNAL_READ_ONLY_EXPOSED_DEV_DIAGNOSTIC',
            'fit_updates_executed':0,'weights_modified':False,'cpu_only':True,'gpu_minutes':0,
            'original_trained_checkpoint_count':10,'original_source_owners':sorted(docs),
            'targets_per_arm':len(SEEDS)*len(docs)*224,
            'independent_source_document_count':len(docs),'independent_seed_count':len(SEEDS),
            'protected_TEST_opened':False,'semantic_human_judgments':False,
            'comparison_note':'postfit lesions are NOT independently trained equal-budget architecture alternatives; original core and Euclidean core were independently fitted',
            'per_seed':seeds,'summary':summary}


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise FileExistsError('immutable no-fit evidence path')
    result=run();a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,indent=2,sort_keys=True,allow_nan=False)+'\n')
    print('HYBGEOM092_SUMMARY='+json.dumps({k:{'nll':round(v['mean_nll'],8),'delta_vs_original':round(v['paired']['original']['mean'],8),'delta_vs_core':round(v['paired']['fixed_core_only']['mean'],8)} for k,v in result['summary'].items()},sort_keys=True),flush=True)
if __name__=='__main__':main()
