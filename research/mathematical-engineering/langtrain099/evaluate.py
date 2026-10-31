"""Seal all three exact10M fits, then independent fresh-process DEV actors.

Language prediction is on real held-out code/docs. Lexical reuse and authored
ARC transfer are reported separately. No AGI or architecture winner is inferred.
"""
import argparse
from collections import defaultdict
import importlib.util
import json
import math
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
import torch
_spec=importlib.util.spec_from_file_location('langtrain099_common',Path(__file__).with_name('common.py'))
C=importlib.util.module_from_spec(_spec);sys.modules[_spec.name]=C;_spec.loader.exec_module(C)

def run_child(command,deadline):
    remaining=deadline-time.perf_counter()
    if remaining<=0:raise TimeoutError('registered evaluation deadline reached')
    subprocess.run(command,check=True,timeout=remaining)

def arrays(path):
    with np.load(path,allow_pickle=False) as z:
        return {n:z[n].copy() for n in z.files if n!='metadata'},json.loads(str(z['metadata'].item()))

def compare(left,right):
    a,ma=arrays(left);b,mb=arrays(right)
    if set(a)!=set(b):raise ValueError('fresh-process replay schema differs')
    for n in a:
        if np.issubdtype(a[n].dtype,np.floating) and (not np.isfinite(a[n]).all() or not np.isfinite(b[n]).all()):raise ValueError('nonfinite inference array')
        if not np.array_equal(a[n],b[n]):raise ValueError('fresh-process actual prediction replay differs: '+n)
    for name in ('checkpoint_sha256','source_commit'):
        if ma.get(name)!=mb.get(name):raise ValueError('fresh prediction source binding differs')

def language(args):
    C.setup(9901);A=C.load('actor');model,checkpoint=A.load_checkpoint(args.checkpoint,args.sha256,args.source_commit)
    records=C.read(args.data/'LANGUAGE_CASES.json');ids=[];nll=[];correct=[];bigram_nll=[]
    counts=np.load(args.data/'FIT_BIGRAM_COUNTS.npy',allow_pickle=False).astype(np.float64)
    if counts.shape!=(256,256) or np.any(counts<0):raise ValueError('authenticated FIT-only bigram baseline required')
    probability=(counts+.1)/(counts.sum(1,keepdims=True)+25.6)
    source_info={r['file']:r for r in C.read(args.data/'EVAL_MANIFEST.json')['language_sources']}
    raw={}
    for name,info in source_info.items():
        p=args.data/name
        if C.sha(p)!=info['sha256']:raise ValueError('held-out language bytes differ')
        raw[name]=p.read_bytes()
    with torch.inference_mode():
        for start in range(0,len(records),4):
            group=records[start:start+4];byte_rows=[list(raw[r['file']][r['offset']:r['offset']+257]) for r in group]
            if any(len(row)!=257 for row in byte_rows):raise ValueError('complete fixed source windows required')
            data=torch.tensor(byte_rows,dtype=torch.long);tokens,target=data[:,:-1],data[:,1:]
            logits,_=model(tokens,owners=tuple(r['id'] for r in group));lp=logits.log_softmax(-1)
            if not bool(torch.isfinite(lp).all()):raise ValueError('nonfinite actual language predictions')
            values=-lp.gather(-1,target[...,None]).squeeze(-1).mean(1)
            ids.extend(r['id'] for r in group);nll.extend(values.tolist());correct.extend((logits.argmax(-1)==target).sum(1).tolist())
            bigram_nll.extend((-np.log(probability[tokens.numpy(),target.numpy()])).mean(1).tolist())
    metadata={'checkpoint':checkpoint,'checkpoint_sha256':args.sha256,'source_commit':args.source_commit,'cases':records,'optimizer_steps':0}
    with args.output.open('xb') as f:np.savez_compressed(f,ids=np.array(ids),nll=np.array(nll),correct=np.array(correct),bigram_nll=np.array(bigram_nll),metadata=np.array(json.dumps(metadata,sort_keys=True)))

def score_episode(payload,meta,actor,judge,label):
    D=C.load('data').load_episodic();rows=[];lp=payload['log_probs'];generated=payload['generated_bytes'];available=payload['available'];lengths=payload['lengths']
    if lp.shape!=(5,72,256) or generated.shape!=(5,72,40) or available.shape!=(5,72) or not np.isfinite(lp).all():raise ValueError('complete original actor outputs required')
    if np.max(np.abs(np.logaddexp.reduce(lp.astype(float),axis=-1)))>5e-5:raise ValueError('unnormalized actual probabilities')
    if meta['case_ids']!=[r['id'] for r in actor] or meta['optimizer_steps']!=0:raise ValueError('actual case order differs')
    for i,a in enumerate(actor):
        gold=judge[a['id']];target=bytes(gold['target_bytes'])
        if lengths[i]!=a['generation_limit']:raise ValueError('gold-independent generation limit changed')
        for ci,condition in enumerate((*C.CONDITIONS,'binding_shuffle')):
            if condition=='binding_shuffle' and a['domain']!='arc':continue
            expected=gold['donor_available'] if condition=='donor' else gold['binding_shuffle_available'] if condition=='binding_shuffle' else True
            if bool(available[ci,i])!=expected:raise ValueError('registered unavailable control changed')
            row={'checkpoint_stage':label,'seed':meta['checkpoint']['seed'],'id':a['id'],'domain':a['domain'],'group':gold['group'],
                 'cluster':gold['cluster_id'],'condition':condition,'status':'MEASURED' if expected else 'INCONCLUSIVE'}
            if expected:
                raw=bytes(generated[ci,i,:lengths[i]])
                if raw[0]!=int(lp[ci,i].argmax()):raise ValueError('greedy first byte disagrees with distribution')
                if a['domain']=='lexical':valid=True;exact=raw==target
                else:
                    try:pred=D.decode_arc_output(raw);valid=True
                    except ValueError:pred=None;valid=False
                    exact=bool(valid and pred==D.decode_arc_output(target))
                probabilities=np.exp(lp[ci,i].astype(float));confidence=float(probabilities.max());top=int(probabilities.argmax())
                row.update(nll=-float(lp[ci,i,target[0]]),exact=int(exact),format_valid=int(valid),first_correct=int(top==target[0]),confidence=confidence,
                           brier=float(np.square(probabilities).sum()-2*probabilities[target[0]]+1),
                           correct_prefix=[int(raw[:j]==target[:j]) for j in range(1,min(4,len(target))+1)])
            else:row.update(nll=None,exact=None,format_valid=None)
            rows.append(row)
    return rows

def clustered(values,clusters):
    names=sorted(set(clusters));means=np.array([np.mean([v for v,c in zip(values,clusters) if c==n]) for n in names])
    rng=np.random.default_rng(99001);samples=means[rng.integers(0,len(names),(5000,len(names)))].mean(1)
    return {'mean':float(np.mean(values)),'cluster_equal_mean':float(means.mean()),'ci95':[float(v) for v in np.quantile(samples,[.025,.975])],'clusters':len(names)}

def summarize(rows,language_rows):
    result={'scope':'Three trained exact10M pilots; public exposed DEV, no AGI/architecture superiority qualification','seeds':list(C.SEEDS),'episode_summary':[],'language_summary':[],'effects':[]}
    for label in ('initial','trained'):
        for domain,group in [('lexical','positive'),('lexical','negative'),('lexical','uniform'),('arc','unknown_rule')]:
            for condition in (*C.CONDITIONS,'binding_shuffle'):
                selected=[r for r in rows if (r['checkpoint_stage'],r['domain'],r['group'],r['condition'])==(label,domain,group,condition)]
                if not selected:continue
                measured=[r for r in selected if r['status']=='MEASURED'];total=len(selected)
                result['episode_summary'].append({'stage':label,'domain':domain,'group':group,'condition':condition,'outcomes':total,
                    'unavailable':total-len(measured),'nll':float(np.mean([r['nll'] for r in measured])) if measured else None,
                    'exact_rate':float(np.mean([r['exact'] for r in measured])) if measured else None,
                    'format_rate':float(np.mean([r['format_valid'] for r in measured])) if measured else None,
                    'per_seed_exact':[{'seed':s,'correct':sum(r['exact'] for r in measured if r['seed']==s),'measured_cases':sum(r['seed']==s for r in measured)} for s in C.SEEDS]})
        for modality in sorted({r['modality'] for r in language_rows}):
            selected=[r for r in language_rows if r['stage']==label and r['modality']==modality]
            result['language_summary'].append({'stage':label,'modality':modality,'nll':float(np.mean([r['nll'] for r in selected])),
                'bits_per_byte':float(np.mean([r['nll'] for r in selected]))/math.log(2),'nextbyte_accuracy':float(np.mean([r['correct']/256 for r in selected])),
                'FIT_bigram_reference_nll':float(np.mean([r['bigram_nll'] for r in selected]))})
    language_lookup={(r['stage'],r['seed'],r['id']):r for r in language_rows}
    for modality in sorted({r['modality'] for r in language_rows}):
        cases={r['id']:r for r in language_rows if r['modality']==modality};values=[];clusters=[]
        for case,row in sorted(cases.items()):
            values.append(float(np.mean([language_lookup[('initial',s,case)]['nll']-language_lookup[('trained',s,case)]['nll'] for s in C.SEEDS])));clusters.append(row['owner'])
        result['effects'].append({'name':'heldout_language_training_nll_gain','modality':modality,'supported_effect':clustered(values,clusters)})
    # Average seeds within each case before resampling original owners/tasks.
    lookup={(r['checkpoint_stage'],r['seed'],r['id'],r['condition']):r for r in rows}
    for domain,group in [('lexical','positive'),('arc','unknown_rule')]:
        ids=sorted({r['id'] for r in rows if r['domain']==domain and r['group']==group})
        for name,left,right,metric in [('training_nll_gain',('initial','carry'),('trained','carry'),'nll'),
                ('reset_minus_carry',('trained','reset'),('trained','carry'),'nll'),
                ('donor_minus_carry',('trained','donor'),('trained','carry'),'nll'),
                ('carry_minus_reset_exact',('trained','carry'),('trained','reset'),'exact'),
                ('carry_minus_donor_exact',('trained','carry'),('trained','donor'),'exact')]:
            values=[];clusters=[];missing=0
            for case in ids:
                pairs=[(lookup[(left[0],s,case,left[1])],lookup[(right[0],s,case,right[1])]) for s in C.SEEDS]
                if any(a['status']!='MEASURED' or b['status']!='MEASURED' for a,b in pairs):missing+=1;continue
                values.append(float(np.mean([a[metric]-b[metric] for a,b in pairs])));clusters.append(pairs[0][0]['cluster'])
            entry={'name':name,'domain':domain,'metric':metric,'cases':len(ids),'unavailable_cases':missing,'status':'INCONCLUSIVE' if missing else 'MEASURED'}
            if values:entry['supported_effect']=clustered(values,clusters)
            result['effects'].append(entry)
    # ARC binding-control attribution is required separately from donor support.
    ids=sorted({r['id'] for r in rows if r['domain']=='arc'})
    values=[];clusters=[];missing=0
    for case in ids:
        pairs=[(lookup[('trained',s,case,'carry')],lookup[('trained',s,case,'binding_shuffle')]) for s in C.SEEDS]
        if any(a['status']!='MEASURED' or b['status']!='MEASURED' for a,b in pairs):missing+=1;continue
        values.append(float(np.mean([a['exact']-b['exact'] for a,b in pairs])));clusters.append(pairs[0][0]['cluster'])
    entry={'name':'carry_minus_binding_shuffle_exact','domain':'arc','metric':'exact','cases':len(ids),'unavailable_cases':missing,'status':'INCONCLUSIVE' if missing else 'MEASURED'}
    if values:entry['supported_effect']=clustered(values,clusters)
    result['effects'].append(entry)
    result['capability_gates']=[]
    for domain,group,required in [('lexical','positive',('carry_minus_reset_exact','carry_minus_donor_exact')),
                                  ('arc','unknown_rule',('carry_minus_reset_exact','carry_minus_binding_shuffle_exact'))]:
        carry=[r for r in rows if (r['checkpoint_stage'],r['domain'],r['group'],r['condition'])==('trained',domain,group,'carry')]
        nonzero=all(sum(r['exact'] for r in carry if r['seed']==s)>0 for s in C.SEEDS)
        effects=[next(e for e in result['effects'] if e.get('domain')==domain and e['name']==name) for name in required]
        unavailable=any(e['status']=='INCONCLUSIVE' for e in effects)
        positive=all(e.get('supported_effect',{}).get('ci95',[-1])[0]>0 for e in effects)
        result['capability_gates'].append({'domain':domain,'group':group,'status':'INCONCLUSIVE' if unavailable else 'PASS' if nonzero and positive else 'FAIL',
            'nonzero_carried_exact_in_every_seed':nonzero,'required_exact_attribution_effects':list(required),'positive_intervals':positive})
    return result

def evaluate(args):
    reg=C.read(C.HERE/'REGISTER.json');source=C.source_hashes(args.source_commit)
    if args.output.exists():raise FileExistsError('retain original evaluation')
    receipts=list(args.fits.rglob('FIT_RECEIPT.json'))
    if len(receipts)!=3:raise ValueError('all3 original fits required before any DEV access')
    matrix={}
    for p in receipts:
        r=C.read(p);seed=r['seed']
        if (seed not in C.SEEDS or seed in matrix or r['status']!='SEALED_BEFORE_DEV' or r['steps']!=reg['training']['steps']
                or r['source_commit']!=args.source_commit or r['source_hashes']!=source or r['trainable_parameters']!=10000000 or r['deployed_parameters']!=10000000):raise ValueError('incomplete exact10M matrix')
        if C.sha(p.parent/'MODEL.pt')!=r['checkpoint_sha256'] or C.sha(p.parent/'INITIAL_MODEL.pt')!=r['initial_checkpoint_sha256']:raise ValueError('original trained/initial archives differ')
        payload=torch.load(p.parent/'MODEL.pt',map_location='cpu',weights_only=True)
        actual=C.make(seed);actual.load_state_dict(payload['model'],strict=True)
        if C.state_digest(actual)!=r['model_sha256'] or r['model_sha256']==r['initial_model_sha256']:raise ValueError('actual changed trained weights required')
        slots=payload['optimizer']['state'];ids=[i for g in payload['optimizer']['param_groups'] for i in g['params']]
        if len(ids)!=len(list(actual.parameters())) or len(slots)!=len(ids):raise ValueError('complete original AdamW slots required')
        for identifier,parameter in zip(ids,actual.parameters()):
            slot=slots[identifier]
            if float(slot['step'])!=reg['training']['steps']:raise ValueError('actual optimizer slot does not have all updates')
            for name in ('exp_avg','exp_avg_sq'):
                if slot[name].shape!=parameter.shape or not bool(torch.isfinite(slot[name]).all()):raise ValueError('actual optimizer moment archive differs')
        if not isinstance(payload['rng']['torch'],torch.Tensor):raise ValueError('original RNG archive required')
        del actual,payload
        matrix[seed]=(p.parent,r)
    args.output.mkdir(parents=True);deadline=time.perf_counter()+1800
    C.save(args.output/'ALL_THREE_CHECKPOINTS_SEALED.json',{'source_commit':args.source_commit,'counts':10000000,'before_DEV':True,'receipts':[r for _,r in matrix.values()]})
    spec=importlib.util.spec_from_file_location('langtrain099_gate',C.ROOT/'scripts/validate_research_parity.py');gate=importlib.util.module_from_spec(spec);spec.loader.exec_module(gate)
    manifest=C.read(C.HERE/'FAIR_COMPARISON.json')
    for row in manifest['models']:row['measured']=matrix[row['seed']][1]
    C.save(args.output/'FINAL_RESOURCE_MANIFEST.json',manifest);C.save(args.output/'FINAL_RESOURCE_GATE.json',gate.check_manifest(manifest,'final'))
    eval_manifest=C.read(args.data/'EVAL_MANIFEST.json')
    if C.sha(args.data/'EVAL_MANIFEST.json')!=reg['data']['EVAL_MANIFEST.json']:raise ValueError('frozen evaluation manifest differs')
    for name,digest in eval_manifest['files'].items():
        if C.sha(args.data/name)!=digest:raise ValueError('original evaluation bytes differ: '+name)
    actor=C.read(args.data/'DEV_ACTOR.json');inputs=args.output/'actor-inputs';inputs.mkdir()
    C.save(inputs/'acquisition.json',[{k:r[k] for k in ('id','domain','acquisition_bytes')} for r in actor])
    C.save(inputs/'queries.json',[{k:r[k] for k in ('id','domain','query_bytes','generation_limit')} for r in actor])
    predictions=[];language_predictions=[];replays=[]
    for seed,(folder,receipt) in sorted(matrix.items()):
        for label,name in [('initial','INITIAL_MODEL.pt'),('trained','MODEL.pt')]:
            path=folder/name;digest=C.sha(path);out=args.output/f'{label}-{seed}';out.mkdir();state=out/'STATE.pt'
            common=[sys.executable,str(C.HERE/'actor.py'),'--checkpoint',str(path),'--sha256',digest,'--source-commit',args.source_commit,'--state',str(state)]
            run_child(common+['--mode','acquire','--acquisition',str(inputs/'acquisition.json'),'--controls',str(args.data/'DEV_CONTROL_ACTOR.json'),'--output',str(out/'ACQUISITION.json')],deadline)
            state_sha=C.sha(state)
            for suffix in ('original','replay'):
                run_child(common+['--mode','query','--queries',str(inputs/'queries.json'),'--donors',str(args.data/'DEV_DONORS.json'),'--output',str(out/(suffix+'.npz'))],deadline)
                run_child([sys.executable,str(Path(__file__)),'--mode','language','--checkpoint',str(path),'--sha256',digest,'--source-commit',args.source_commit,'--data',str(args.data),'--output',str(out/('language-'+suffix+'.npz'))],deadline)
            compare(out/'original.npz',out/'replay.npz');compare(out/'language-original.npz',out/'language-replay.npz')
            if C.sha(state)!=state_sha or C.sha(path)!=digest:raise ValueError('query mutated original state/weights')
            predictions.append((out/'original.npz',label));language_predictions.append((out/'language-original.npz',label))
            replays.append({'seed':seed,'stage':label,'status':'PASS','checkpoint_sha256':digest,'state_sha256':state_sha})
            print({'stage':'PREDICTIONS_REPLAYED','seed':seed,'checkpoint_stage':label},flush=True)
    raw=args.output/'RAW_HISTORY.npz';run_child([sys.executable,str(C.ROOT/'research/mathematical-engineering/langacquire097/raw_control.py'),'--actor',str(args.data/'DEV_ACTOR.json'),'--output',str(raw)],deadline)
    C.save(args.output/'ALL_PREDICTIONS_SEALED.json',{'before_target_scores':True,'replays':replays,'raw_sha256':C.sha(raw)})
    judge=C.read(args.data/'DEV_JUDGE.json');rows=[];language_rows=[]
    for path,label in predictions:
        payload,metadata=arrays(path);rows.extend(score_episode(payload,metadata,actor,judge,label))
    for path,label in language_predictions:
        payload,meta=arrays(path)
        for i,r in enumerate(meta['cases']):language_rows.append({**r,'seed':meta['checkpoint']['seed'],'stage':label,'nll':float(payload['nll'][i]),'correct':int(payload['correct'][i]),'bigram_nll':float(payload['bigram_nll'][i])})
    report=summarize(rows,language_rows);D=C.load('data').load_episodic();payload,meta=arrays(raw);lookup={str(x):i for i,x in enumerate(payload['ids'])};arc_lookup={str(x):i for i,x in enumerate(payload['arc_ids'])}
    raw_rows=[]
    for r in actor:
        g=judge[r['id']]
        if r['domain']=='lexical':
            i=lookup[r['id']];raw_rows.append({'id':r['id'],'domain':'lexical','group':g['group'],'exact':int(bytes(payload['generated'][i])==bytes(g['target_bytes'])),'nll':-float(payload['log_probs'][i,g['target_bytes'][0]])})
        else:
            i=arc_lookup[r['id']];valid=True
            try:prediction=D.decode_arc_output(payload['arc_generated'][i])
            except ValueError:prediction=None;valid=False
            raw_rows.append({'id':r['id'],'domain':'arc','exact':int(valid and prediction==D.decode_arc_output(bytes(g['target_bytes']))),'format_valid':int(valid)})
    report['references']=raw_rows;report['actual_trained_models']=3;report['parameters_per_model']=10000000;report['optimizer_steps_per_model']=reg['training']['steps'];report['replays']=replays
    C.save(args.output/'SCORED_EPISODES.json',rows);C.save(args.output/'SCORED_LANGUAGE.json',language_rows);C.save(args.output/'RESULTS.json',report)
    C.save(args.output/'EXECUTION.json',{'status':'COMPLETED_INTERNAL_RESEARCH','source_commit':args.source_commit,'models_trained':3,'parameters':10000000,'optimizer_steps_during_evaluation':0,'GPU_minutes':0,'replays':replays,'whole_model_IEEE_AD':'OPEN','architecture_superiority':False,'general_intelligence_established':False})
    print({'status':'COMPLETED_INTERNAL_RESEARCH','models_trained':3,'parameters':10000000},flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=('evaluate','language'),default='evaluate');p.add_argument('--fits',type=Path);p.add_argument('--data',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--source-commit',required=True);p.add_argument('--checkpoint',type=Path);p.add_argument('--sha256');a=p.parse_args();language(a) if a.mode=='language' else evaluate(a)
