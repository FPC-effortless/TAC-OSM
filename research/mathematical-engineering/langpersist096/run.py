"""Execute frozen lexical and general-capability development batteries on CPU.

Prediction workers are separate processes and receive no scorer targets.
Original outputs, failed attempts, controls and replay outputs are retained.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as handle:
        for part in iter(lambda:handle.read(1024*1024),b''): h.update(part)
    return h.hexdigest()


def write_json(path,value):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8') as handle:
        json.dump(value,handle,indent=2,sort_keys=True,allow_nan=False);handle.write('\n')


def verify_code_snapshot(commit):
    if re.fullmatch('[0-9a-f]{40}',commit or '') is None:
        raise ValueError('actual prospective source commit required')
    paths=list(HERE.glob('*.py'))+[HERE/p for p in ('PREREGISTRATION.md','FAIR_COMPARISON.json','FROZEN_INPUTS.json')]
    pins={}
    for path in paths:
        relative=path.relative_to(ROOT).as_posix()
        frozen=subprocess.run(['git','show',commit+':'+relative],cwd=ROOT,check=True,capture_output=True).stdout
        if frozen!=path.read_bytes(): raise ValueError('source changed since preregistration: '+relative)
        pins[relative]=digest(path)
    return pins


def load_predictions(path):
    with np.load(path,allow_pickle=False) as archive:
        payload={k:archive[k].copy() for k in archive.files if k!='metadata'}
        metadata=json.loads(str(archive['metadata'].item()))
    return payload,metadata


def verify_replay(left,right,keys):
    a,_=load_predictions(left);b,_=load_predictions(right)
    if any(k not in a or k not in b or not np.array_equal(a[k],b[k]) for k in keys):
        raise ValueError('fresh process prediction replay failed')


def verify_actor_metadata(meta,row,acquisition_sha,query_sha):
    if (meta.get('arm')!=row['arm'] or meta.get('checkpoint_sha256')!=row['checkpoint_sha256']
            or meta.get('optimizer_steps')!=0 or meta.get('acquisition_sha256')!=acquisition_sha
            or meta.get('query_sha256')!=query_sha):
        raise ValueError('actor prediction provenance or boundary differs')


def poison_fixed_sources(sources,records,acquisition,queries):
    """Mutate only future bytes in private copies; never reselect the cohort."""
    changed={key:bytearray(value) for key,value in sources.items()}
    for record in records:
        raw=changed[record['file']]
        start,end=record['target_offset'],record['target_end']
        raw[start:end]=bytes(value^1 for value in raw[start:end])
    a=np.stack([np.frombuffer(changed[r['file']][r['acquisition_start']:r['acquisition_end']],dtype=np.uint8)
                for r in records])
    q=np.stack([np.frombuffer(changed[r['file']][r['query_start']:r['query_end']],dtype=np.uint8)
                for r in records])
    if not np.array_equal(a,acquisition) or not np.array_equal(q,queries):
        raise ValueError('future-only source poisoning changed model observations')
    return q,{'status':'PASS_FIXED_EPISODE_OBSERVATION_NONINTERFERENCE',
              'original_source_hashes':{k:hashlib.sha256(v).hexdigest() for k,v in sources.items()},
              'poisoned_source_hashes':{k:hashlib.sha256(v).hexdigest() for k,v in changed.items()},
              'original_files_unchanged':True,'eligibility_recomputed':False}


def prepare(cache,frozen):
    import data as lexical_data
    import general_data
    import checkpoints
    start=time.perf_counter()
    lex=lexical_data.prepare(cache/'lexical')
    if lex.get('status')!='COHORT_READY': raise ValueError('INSUFFICIENT_COHORT')
    for name,expected in frozen['lexical'].items():
        if digest(cache/'lexical'/name)!=expected: raise ValueError('frozen lexical input differs: '+name)
    general=general_data.prepare_general_data(cache/'general')
    if (general['actor_sha256']!=frozen['general']['actor_sha256'] or
            general['judge_sha256']!=frozen['general']['judge_sha256'] or
            general['task_ids']!=frozen['general']['task_ids']):
        raise ValueError('frozen general-task selection differs')
    models=checkpoints.prepare_checkpoints(cache/'checkpoints')
    actual=[{k:r[k] for k in ('arm','seed','checkpoint_sha256')} for r in models['models']]
    if actual!=frozen['checkpoints']: raise ValueError('frozen checkpoint matrix differs')
    return lex,general,models,time.perf_counter()-start


def actor_command(script,mode,row,state,output,**inputs):
    cmd=[sys.executable,str(HERE/script),'--mode',mode,'--arm',row['arm'],
         '--checkpoint',row['path'],'--sha256',row['checkpoint_sha256'],
         '--state',str(state),'--output',str(output)]
    for key,value in inputs.items(): cmd += ['--'+key,str(value)]
    return cmd


def general_results(predictions,judge_records,references,costs):
    from general_judge import judge_case
    started=time.perf_counter();cases=[];groups=defaultdict(list)
    for (arm,seed),payload in predictions.items():
        generated=payload['generated_bytes'];lengths=payload['lengths']
        if generated.shape!=(2,24,128) or generated.dtype!=np.uint8 or lengths.shape!=(24,):
            raise ValueError('general prediction geometry invalid')
        for ci,condition in enumerate(('carry','reset')):
            for i,task_id in enumerate(references['task_ids']):
                record=judge_records[task_id]
                result=judge_case(record,generated[ci,i,:int(lengths[i])])
                item={'arm':arm,'seed':seed,'condition':condition,'task_id':task_id,
                      'domain':record['domain'],'result':result}
                cases.append(item);groups[(arm,seed,condition,record['domain'])].append(result)
    summary=[]
    for key,rows in sorted(groups.items()):
        supported=[r for r in rows if r.get('correct') is not None]
        correct=sum(r.get('correct') is True for r in supported)
        summary.append(dict(arm=key[0],seed=key[1],condition=key[2],domain=key[3],
                            correct=correct,selected=len(rows),judgeable=len(supported),
                            unsupported=len(rows)-len(supported),
                            accuracy_supported_subset=correct/len(supported) if supported else None,
                            status='MEASURED_INTERNAL' if len(supported)==len(rows) else 'INCONCLUSIVE_FULL_DOMAIN'))
    return {'schema':'langpersist096.general-results.v1','evidence_level':'INTERNAL_PUBLIC_DEVELOPMENT_PILOT',
            'general_intelligence_established':False,'single_agi_score':None,
            'task_count':24,'models':9,'conditions':['carry','reset'],
            'summary':summary,'cases':cases,'reference_validation':references,
            'verification_seconds':time.perf_counter()-started,'resources':costs,
            'limitations':['Eight development tasks per domain; no independent blind benchmark',
                           'Frozen byte-prediction models are not instruction-tuned',
                           'External verification is not an autonomously learned verifier',
                           'Unknown transformations test in-context transfer, not sustained autonomous improvement']}


def general_markdown(report):
    lines=['# LANGPERSIST096 general capability pilot','',
           'Same frozen models; public development tasks, not an AGI certificate.','',
           '| Model | Domain | Carry correct/judgeable | Reset correct/judgeable | Judgeable per seed |',
           '| --- | --- | --- | --- | --- |']
    for arm in ('tac','gru','ssm'):
        for domain in ('arc','gsm8k','humaneval'):
            rows=[r for r in report['summary'] if r['arm']==arm and r['domain']==domain]
            carry=sum(r['correct'] for r in rows if r['condition']=='carry')
            reset=sum(r['correct'] for r in rows if r['condition']=='reset')
            carry_n=sum(r['judgeable'] for r in rows if r['condition']=='carry')
            reset_n=sum(r['judgeable'] for r in rows if r['condition']=='reset')
            denom=sorted({r['judgeable'] for r in rows})
            lines.append(f'| {arm} | {domain} | {carry}/{carry_n} | {reset}/{reset_n} | {denom} |')
    lines += ['', 'Unsupported reference tasks remain inconclusive and are not replaced.',
              'Counts sum three seeds on the same eight cases; they are not24 independent tasks.','']
    return '\n'.join(lines)


def run(args):
    started=time.perf_counter();deadline=started+40*60
    if args.output.exists(): raise FileExistsError(args.output)
    code=verify_code_snapshot(args.preregistration_commit)
    frozen=json.loads((HERE/'FROZEN_INPUTS.json').read_text())
    args.output.mkdir(parents=True)
    write_json(args.output/'PREREGISTRATION_SEAL.json',{'commit':args.preregistration_commit,'source_hashes':code,
               'input_hashes':frozen,'before_first_model_predictions':True})
    lex,general,model_manifest,preparation_seconds=prepare(args.cache,frozen)
    write_json(args.output/'SOURCE_RECEIPTS.json',{'lexical':lex,'general':general,'models':model_manifest})
    log=[];battery_seconds={'lexical':0.,'general':0.}
    def execute(cmd,label):
        remaining=deadline-time.perf_counter()
        battery='lexical' if '-lex-' in label or label=='raw-history' else 'general'
        remaining=min(remaining,20*60-battery_seconds[battery])
        if remaining<=0: raise TimeoutError('registered cumulative battery/research budget exhausted')
        tic=time.perf_counter()
        with (args.output/(label+'.log')).open('x') as stream:
            result=subprocess.run(cmd,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,
                                  timeout=remaining,env={**os.environ,'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'})
        elapsed=time.perf_counter()-tic;battery_seconds[battery]+=elapsed
        log.append({'label':label,'battery':battery,'seconds':elapsed,'exit_code':result.returncode})
        write_json(args.output/(label+'.execution.json'),log[-1])
        if result.returncode: raise RuntimeError('actor failed: '+label)

    # Prepare source-free query inputs at a distinct location from raw acquisition.
    query_dir=args.output/'query-inputs';query_dir.mkdir()
    for name in ('queries.npy','donor_indices.npy'):
        (query_dir/name).write_bytes((args.cache/'lexical'/name).read_bytes())
    records=json.loads((args.cache/'lexical'/'records.json').read_text())
    sources={r['file']:(args.cache/'lexical'/'sources'/r['file']).read_bytes()
             for r in lex['source_rows'] if r['partition']=='TRANSFER_DEV'}
    poison_query,poison_audit=poison_fixed_sources(sources,records,
        np.load(args.cache/'lexical'/'acquisition.npy',allow_pickle=False),
        np.load(query_dir/'queries.npy',allow_pickle=False))
    with (query_dir/'future-poison-queries.npy').open('xb') as handle:
        np.save(handle,poison_query,allow_pickle=False)
    write_json(args.output/'SOURCE_FUTURE_POISON.json',poison_audit)
    actor=json.loads(Path(general['actor_path']).read_text())
    judge_records=json.loads(Path(general['judge_path']).read_text())
    acq_records=[{k:r[k] for k in ('task_id','domain','acquisition_bytes')} for r in actor]
    q_records=[{k:r[k] for k in ('task_id','domain','query_bytes','generation_limit')} for r in actor]
    write_json(args.output/'general-acquisition.json',acq_records)
    write_json(query_dir/'general-queries.json',q_records)
    from general_judge import validate_reference
    references={'task_ids':[r['task_id'] for r in actor],
                'checks':{r['task_id']:validate_reference(judge_records[r['task_id']]) for r in actor}}
    write_json(args.output/'REFERENCE_VALIDATION.json',references)
    for task_id,result in references['checks'].items():
        expected='UNSUPPORTED' if task_id=='HumanEval/160' else 'PASS'
        if result['status']!=expected:
            raise ValueError('public reference or confinement changed before prediction: '+task_id)

    lexical_predictions={};general_predictions={};lexical_costs={};general_costs={};replays=[]
    for row in model_manifest['models']:
        arm,seed=row['arm'],row['seed'];tag=f'{arm}-{seed}'
        folder=args.output/tag;folder.mkdir()
        state=folder/'lexical-state.pt'
        execute(actor_command('worker.py','acquire',row,state,folder/'lexical-acquisition.json',
                              acquisition=args.cache/'lexical'/'acquisition.npy'),tag+'-lex-acquire')
        saved_hash=digest(state)
        for suffix in ('original','replay'):
            execute(actor_command('worker.py','query',row,state,folder/('lexical-'+suffix+'.npz'),
                                  queries=query_dir/('queries.npy' if suffix=='original' else 'future-poison-queries.npy'),
                                  donors=query_dir/'donor_indices.npy'),tag+'-lex-'+suffix)
        verify_replay(folder/'lexical-original.npz',folder/'lexical-replay.npz',('log_probs','greedy'))
        if digest(state)!=saved_hash: raise ValueError('query mutated sealed acquisition state')
        payload,meta=load_predictions(folder/'lexical-original.npz')
        verify_actor_metadata(meta,row,frozen['lexical']['acquisition.npy'],frozen['lexical']['queries.npy'])
        if meta.get('conditions')!=['carry','reset','donor','corrupt']:
            raise ValueError('lexical condition ordering differs')
        lexical_predictions[(arm,seed)]=payload;lexical_costs[(arm,seed)]=meta['cost']
        lexical_costs[(arm,seed)]['per_condition_costs']=meta.get('per_condition_costs')
        state=folder/'general-state.pt'
        execute(actor_command('general_worker.py','acquire',row,state,folder/'general-acquisition.json',
                              acquisition=args.output/'general-acquisition.json'),tag+'-general-acquire')
        saved_hash=digest(state)
        for suffix in ('original','replay'):
            execute(actor_command('general_worker.py','query',row,state,folder/('general-'+suffix+'.npz'),
                                  queries=query_dir/'general-queries.json'),tag+'-general-'+suffix)
        verify_replay(folder/'general-original.npz',folder/'general-replay.npz',('generated_bytes','lengths'))
        if digest(state)!=saved_hash: raise ValueError('general query mutated acquisition state')
        payload,meta=load_predictions(folder/'general-original.npz')
        verify_actor_metadata(meta,row,digest(args.output/'general-acquisition.json'),digest(query_dir/'general-queries.json'))
        if meta.get('task_ids')!=references['task_ids'] or meta.get('conditions')!=['carry','reset']:
            raise ValueError('general task/condition ordering differs')
        general_predictions[(arm,seed)]=payload;general_costs[tag]=meta
        replays.append({'arm':arm,'seed':seed,'lexical':'PASS','general':'PASS','state_immutable':True})
        print(json.dumps({'stage':'model_completed','arm':arm,'seed':seed,'replay':'PASS'}),flush=True)

    execute([sys.executable,str(HERE/'raw_history.py'),'--acquisition',str(args.cache/'lexical'/'acquisition.npy'),
             '--queries',str(query_dir/'queries.npy'),'--output',str(args.output/'raw-history.npz')],'raw-history')
    payload,meta=load_predictions(args.output/'raw-history.npz')
    lexical_predictions[('raw_history',0)]=payload;lexical_costs[('raw_history',0)]=meta['costs']
    targets=np.load(args.cache/'lexical'/'targets.npy',allow_pickle=False)
    from analysis import analyze,write_reports
    lexical=analyze(lexical_predictions,targets,records,lexical_costs)
    write_reports(lexical,args.output/'LEXICAL_RESULTS.json',args.output/'LEXICAL_RESULTS.md')
    if lexical['status']=='FAIL': raise ValueError('lexical analysis failed: '+str(lexical.get('analysis_error')))
    broad=general_results(general_predictions,judge_records,references,general_costs)
    if time.perf_counter()>deadline:
        raise TimeoutError('registered total research budget exhausted during analysis/judging')
    write_json(args.output/'GENERAL_RESULTS.json',broad)
    with (args.output/'GENERAL_RESULTS.md').open('x') as handle: handle.write(general_markdown(broad))
    # Gold is owned only by this scorer; poisoning never changes actor inputs.
    write_json(args.output/'INFORMATION_BOUNDARY.json',{
        'query_inputs':['queries.npy','donor_indices.npy','general-queries.json','saved_state','checkpoint'],
        'forbidden_to_query':['acquisition.npy','general-acquisition.json','targets.npy','records.json','GENERAL_JUDGE.json'],
        'future_poison_original_sha256':hashlib.sha256(targets.tobytes()).hexdigest(),
        'future_poison_alternative_sha256':hashlib.sha256(np.bitwise_xor(targets,1).tobytes()).hexdigest(),
        'fresh_process_replays':replays,'scope':'source-reviewed worker inputs; not arbitrary-code OS noninterference proof'})
    for name,expected in frozen['lexical'].items():
        if digest(args.cache/'lexical'/name)!=expected: raise ValueError('input changed during scoring')
    for row in model_manifest['models']:
        if digest(row['path'])!=row['checkpoint_sha256']: raise ValueError('checkpoint changed during scoring')
    if time.perf_counter()>deadline:
        raise TimeoutError('registered total research budget exhausted during final integrity checks')
    write_json(args.output/'EXECUTION.json',{'status':'COMPLETED_INTERNAL_RESEARCH',
        'source_commit':args.preregistration_commit,'preparation_seconds':preparation_seconds,
        'elapsed_seconds':time.perf_counter()-started,'battery_subprocess_seconds':battery_seconds,
        'stages':log,'optimizer_steps':0,'gpu_minutes':0,
        'protected_repository_TEST_opened':False,'general_intelligence_established':False})


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--preregistration-commit',required=True)
    args=parser.parse_args()
    try: run(args)
    except Exception as exc:
        if args.output.exists():
            path=args.output/'FAILED_ATTEMPT.json'
            if not path.exists(): write_json(path,{'status':'FAILED_OR_INCONCLUSIVE_RETAINED','error':type(exc).__name__,'reason':str(exc)})
        raise


if __name__=='__main__': main()
