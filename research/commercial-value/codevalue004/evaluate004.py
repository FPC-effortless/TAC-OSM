#!/usr/bin/env python3
"""Freeze all DEV-selected checkpoints, then score the new transfer cohort once."""
import argparse
import gc
import json
from pathlib import Path
import resource
import time

import numpy as np
import torch

from protocol004 import (ARMS,CELLS,HERE,MILESTONES,PARAMS,ROOT,SEEDS,atomic_json,corpus_docs,
    folder_name,make,path_for,planned_hashes,read_sources,scoring,select_checkpoint,sha)


def seal(out,data):
    from scripts.validate_research_parity import check_manifest
    manifest=json.loads((ROOT/'research/mathematical-engineering/codevalue004/FAIR_COMPARISON.json').read_text())
    records={};selections={};finals={}
    for cell in CELLS:
        receipt,docs=corpus_docs(data,cell)
        for seed in SEEDS:
            expected=planned_hashes(docs,seed)
            for arm in ARMS:
                folder=out/folder_name(arm,cell,seed)
                chosen=json.loads((folder/'SELECTION.json').read_text())
                if select_checkpoint(chosen['candidate_curve'])!=chosen['selected_step']:raise ValueError('DEV selection changed')
                selected_path=path_for(out,arm,cell,seed,chosen['selected_step'])
                if sha(selected_path)!=chosen['selected_checkpoint_sha256']:raise ValueError('selected weights changed')
                for step in MILESTONES:
                    path=path_for(out,arm,cell,seed,step)
                    rec=json.loads(path.with_suffix('.json').read_text())
                    if sha(path)!=rec['checkpoint_sha256'] or rec['sample_sha256']!=expected[step]:
                        raise ValueError('original weight or paired data receipt mismatch')
                    if (rec['arm'],rec['cell'],rec['seed'],rec['steps'])!=(arm,cell,seed,step):raise ValueError('wrong run identity')
                    if rec['trainable_parameters']!=PARAMS[arm] or rec['deployed_parameters']!=PARAMS[arm]:raise ValueError('count mismatch')
                    if rec['source_cohort_sha256']!=receipt['cohort_sha256']:raise ValueError('cohort mismatch')
                    if rec['cold_replay_max_abs_difference']>1e-6 or rec['future_poison_max_abs_difference']>1e-5:raise ValueError('replay/causality failed')
                    records[str(path.relative_to(out))]=rec['checkpoint_sha256']
                    if step==5120:finals[arm,cell,seed]={**rec,'selected_step':chosen['selected_step'],
                        'selected_checkpoint_sha256':chosen['selected_checkpoint_sha256']}
                selections[folder.name]=chosen
    for row in manifest['models']:row['measured']=finals[row['architecture'],row['cell'],row['seed']]
    result=check_manifest(manifest,'final')
    atomic_json(out/'FINAL_MANIFEST.json',manifest);atomic_json(out/'FINAL_RESOURCE_GATE.json',result)
    atomic_json(out/'SELECTION_SEAL.json',{'study':'CODEVALUE004','checkpoints':records,
        'source_cohort_sha256':receipt['cohort_sha256'],'selections':selections,
        'status':'120_CHECKPOINTS_AND24_DEV_SELECTIONS_SEALED_BEFORE_TRANSFER_SCORING',
        'protected_TEST':'NOT_OPENED','independently_qualified':False})
    return selections

@torch.inference_mode()
def suggestions(model,raw,device,latency=False):
    positions=scoring.offers(raw);rows=[]
    for lo in range(0,len(positions),64):
        pos=positions[lo:lo+64]
        x=torch.tensor([list(raw[p-128:p]) for p in pos],device=device,dtype=torch.long)
        generated=[];confidence=torch.zeros(len(pos),device=device)
        for step in range(8):
            logits=model(x)[:,-1];nxt=logits.argmax(-1)
            if step<4:confidence+=torch.log_softmax(logits,-1).gather(-1,nxt[:,None]).squeeze(-1)
            generated.append(nxt);x=torch.cat((x,nxt[:,None]),1)[:,-128:]
        prediction=torch.stack(generated,1).cpu().tolist();log_probs=confidence.cpu().tolist()
        for p,g,c in zip(pos,prediction,log_probs):
            row=scoring.suggestion_row(p,g,raw[p:p+8])
            row['exact']={str(n):int(row['longest_correct_prefix']>=n) for n in (2,4,8)}
            row['generated4_log_probability']=float(c);rows.append(row)
    timing={str(n):[] for n in (4,8)}
    if latency:
        for j,p in enumerate(positions[:24]):
            x=torch.tensor([list(raw[p-128:p])],device=device,dtype=torch.long)
            if device.startswith('cuda'):torch.cuda.synchronize()
            start=time.perf_counter()
            for step in range(8):
                nxt=model(x)[:,-1].argmax(-1);x=torch.cat((x,nxt[:,None]),1)[:,-128:]
                if device.startswith('cuda'):torch.cuda.synchronize()
                if j>=8 and step+1 in (4,8):timing[str(step+1)].append((time.perf_counter()-start)*1000)
    times={n:{'p50':float(np.median(v)),'p95':float(np.quantile(v,.95)),'samples':len(v)}
           for n,v in timing.items() if v}
    return rows,times


def run(out,data,device='cuda:0'):
    if (out/'SELECTION_SEAL.json').exists():raise FileExistsError('refuse repeated new-transfer scoring')
    chosen=seal(out,data)
    _,transfer=read_sources(data,{'TRANSFER_DEV'})
    for corpus,cell in (('small','small_constant'),('broad','broad_constant')):
        receipt,docs=corpus_docs(data,cell)
        for name,order in (('p3',3),('wb6',6)):
            count=scoring.Counts(list(docs.values()),order)
            fit_counts=(sum(len(ct) for ct in count.counts),sum(sum(v.values()) for ct in count.counts for v in ct.values()))
            for row,raw in transfer:
                score=count.score(raw)
                for offer in score['offers']:
                    offer['generated_hex']=offer['generated_hex'][:16]
                    offer['longest_correct_prefix']=min(8,offer['longest_correct_prefix'])
                    offer['exact']={k:v for k,v in offer['exact'].items() if k in ('2','4','8')}
                score.update(source=row,predictor=name,corpus=corpus,
                    peak_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
                atomic_json(out/'scores'/f'{name}-{corpus}-{row["file"]}.json',score)
            after=(sum(len(ct) for ct in count.counts),sum(sum(v.values()) for ct in count.counts for v in ct.values()))
            if after!=fit_counts:raise ValueError('count predictor mutated during transfer evaluation')
            count.probs.cache_clear();del count;gc.collect()
    for cell in CELLS:
        for seed in SEEDS:
            for arm in ARMS:
                sel=chosen[folder_name(arm,cell,seed)]
                path=path_for(out,arm,cell,seed,sel['selected_step'])
                cp=torch.load(path,map_location='cpu',weights_only=True)
                if cp['arm']!=arm or cp['cell']!=cell or cp['seed']!=seed:raise ValueError('selected checkpoint identity mismatch')
                model=make(arm,device).eval();model.load_state_dict(cp['model'],strict=True)
                if device.startswith('cuda'):torch.cuda.reset_peak_memory_stats()
                for i,(row,raw) in enumerate(transfer):
                    score=scoring.neural_file(model,raw,device,completion=False)
                    gen,times=suggestions(model,raw,device,latency=i==0)
                    score.update(offers=gen,batch1_suggestion_ms=times,source=row,predictor=arm,
                        cell=cell,corpus=cell.split('_')[0],seed=seed,selected_step=sel['selected_step'],
                        checkpoint_sha256=sha(path),
                        weights_bytes=sum(v.numel()*v.element_size() for v in model.state_dict().values()),
                        gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated() if device.startswith('cuda') else 0,
                        gpu_peak_reserved_bytes=torch.cuda.max_memory_reserved() if device.startswith('cuda') else 0,
                        peak_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
                    atomic_json(out/'scores'/f'{arm}-{cell}-{seed}-{row["file"]}.json',score)
                print(json.dumps({'stage':'selected_transfer_scored','arm':arm,'cell':cell,
                                  'seed':seed,'selected_step':sel['selected_step']}),flush=True)
                del model,cp
                if device.startswith('cuda'):torch.cuda.empty_cache()
    summarize(out)


def summarize(out,repetitions=10000):
    rows=[json.loads(p.read_text()) for p in sorted((out/'scores').glob('*.json'))]
    if len(rows)!=280:raise ValueError('expected24 neural selections and4 count predictors times10 transfer files')
    grouped={}
    for r in rows:
        key=(r['predictor'],r.get('cell',r['corpus']),r.get('seed'))
        grouped.setdefault(key,[]).append(r)
    summary=[]
    for (predictor,cell,seed),files in grouped.items():
        if len(files)!=10:raise ValueError('incomplete transfer files for one predictor')
        targets=sum(r['targets'] for r in files);offers=sum(len(r['offers']) for r in files)
        exact=sum(o['exact']['4'] for r in files for o in r['offers'])
        summary.append({'predictor':predictor,'cell':cell,'seed':seed,'targets':targets,
            'offers':offers,'nll':sum(r['nll_sum'] for r in files)/targets,
            'accuracy':sum(r['top1'] for r in files)/targets,'exact4':exact,
            'exact4_rate':exact/offers,'rejection_cost4_net_keys':int(exact*3-(offers-exact)*4)})
    # Paired bootstrap: resample seed, publisher, then file. Same draws for both cells.
    neural={(r['predictor'],r['cell'],r['seed'],r['source']['file']):r for r in rows if r['predictor'] in ARMS}
    publishers={}
    for r in rows:
        if r['predictor']=='tac' and r['cell']=='small_constant' and r['seed']==1701:
            publishers.setdefault(r['source']['group'],[]).append(r['source']['file'])
    files=sorted({f for g in publishers.values() for f in g})
    targets=np.array([neural['tac','small_constant',1701,f]['targets'] for f in files],dtype=float)
    index={f:i for i,f in enumerate(files)}
    groups=[np.array([index[f] for f in fs]) for _,fs in sorted(publishers.items())]
    comparisons={}
    for arm in ARMS:
        losses={cell:np.array([[neural[arm,cell,seed,f]['nll_sum']/neural[arm,cell,seed,f]['targets']
                              for f in files] for seed in SEEDS]) for cell in CELLS}
        effects={
            'broad_cosine_vs_small_constant_gain':losses['small_constant']-losses['broad_cosine'],
            'corpus_expansion_gain':.5*((losses['small_constant']-losses['broad_constant'])+(losses['small_cosine']-losses['broad_cosine'])),
            'cosine_schedule_gain':.5*((losses['small_constant']-losses['small_cosine'])+(losses['broad_constant']-losses['broad_cosine'])),
            'interaction_gain':(losses['broad_constant']-losses['broad_cosine'])-(losses['small_constant']-losses['small_cosine'])}
        arm_results={}
        for effect,differences in effects.items():
            rng=np.random.default_rng(940004);samples=[]
            for _ in range(repetitions):
                seeds=rng.integers(0,3,3)
                picked=[]
                for group_idx in rng.integers(0,len(groups),len(groups)):
                    g=groups[int(group_idx)];picked.extend(rng.choice(g,size=len(g),replace=True).tolist())
                w=targets[picked]
                samples.append(float(np.mean((differences[seeds][:,picked]*w).sum(1)/w.sum())))
            lo,hi=np.quantile(samples,[.025,.975])
            groupwins=int(sum(float(np.mean((differences[:,g]*targets[g]).sum(1)/targets[g].sum()))>0 for g in groups))
            seedwins=int(sum(float((v*targets).sum()/targets.sum())>0 for v in differences))
            arm_results[effect]={'mean_nll_gain':float(np.mean((differences*targets).sum(1)/targets.sum())),
                'paired_seed_publisher_file95ci':[float(lo),float(hi)],'publisher_wins':groupwins,'seed_wins':seedwins}
        comparisons[arm]=arm_results
    utility={}
    count_rows={(r['predictor'],r['corpus'],r['source']['file']):r for r in rows if r['predictor'] in ('p3','wb6')}
    def exact_rate(row):return sum(o['exact']['4'] for o in row['offers'])/len(row['offers'])
    for cell in CELLS:
        cell_report={};passed=True
        for baseline in ('p3','wb6'):
            differences=np.array([[exact_rate(neural['tac',cell,seed,f])-exact_rate(count_rows[baseline,cell.split('_')[0],f])
                                   for f in files] for seed in SEEDS])
            rng=np.random.default_rng(940004);samples=[]
            for _ in range(repetitions):
                seeds=rng.integers(0,3,3);picked=[]
                for gi in rng.integers(0,len(groups),len(groups)):
                    g=groups[int(gi)];picked.extend(rng.choice(g,size=len(g),replace=True).tolist())
                samples.append(float(np.mean(differences[seeds][:,picked])))
            lo,hi=np.quantile(samples,[.025,.975])
            groupwins=int(sum(float(np.mean(differences[:,g]))>0 for g in groups))
            seedwins=int(sum(float(np.mean(d))>0 for d in differences))
            cell_report[baseline]={'mean_exact4_rate_gain':float(np.mean(differences)),
                'paired_seed_publisher_file95ci':[float(lo),float(hi)],'publisher_wins':groupwins,'seed_wins':seedwins}
            passed=passed and float(lo)>0 and groupwins>=4 and seedwins>=2
        net=float(np.mean([r['rejection_cost4_net_keys'] for r in summary if r['predictor']=='tac' and r['cell']==cell]))
        cell_report['mean_rejection_cost4_net_keys']=net
        cell_report['utility_screen']='PASS_CONSIDER_HUMAN_PILOT' if passed and net>0 else 'FAIL'
        utility[cell]=cell_report
    primary=comparisons['tac']['broad_cosine_vs_small_constant_gain']
    advance=primary['paired_seed_publisher_file95ci'][0]>0 and primary['publisher_wins']>=4 and primary['seed_wins']>=2
    atomic_json(out/'RESULTS.json',{'study':'CODEVALUE004','status':'COMPLETE_INTERNAL_RESEARCH',
        'summary':summary,'factorial_effects':comparisons,'tac_utility_screens':utility,
        'tac_primary_generalization_gate':'PASS_INTERNAL_LANGUAGE_SIGNAL' if advance else 'FAIL',
        'commercial_viability_proved':False,'architecture_superiority_eligible':False,
        'independently_qualified':False,'protected_TEST':'NOT_OPENED',
        'limitations':['Three seeds and five publishers; bootstrap uncertainty remains limited',
            'Corpus expansion changes size and content together, not diversity alone',
            'Full training FLOPs not matched; no architectural superiority claim',
            'No actual users, costs, revenue, Android, English or code execution assessment',
            'New transfer source audit is not access-separated blind TEST']})
    print(json.dumps({'stage':'complete','tac_generalization_gate':'PASS' if advance else 'FAIL'}),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True)
    p.add_argument('--data',type=Path,required=True);p.add_argument('--device',default='cuda:0')
    a=p.parse_args();run(a.out,a.data,a.device)
