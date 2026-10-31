"""Finite forty-fit LANGCONTROL098 orchestration and access-separated scoring.

Ten FIT objective legs each seal four original checkpoints. The assembler
verifies the complete forty-model matrix and produces five seed seals.
A downstream evaluator must
verify all forty before DEV prediction. Source-authentication/poison producers
may inspect answers; prediction actors never do. Target-score computation starts
after every prediction/replay is sealed. No checkpoint selection occurs.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np

_PATH=Path(__file__).with_name('resources.py')
_SPEC=importlib.util.spec_from_file_location('langcontrol098_resources',_PATH)
R=importlib.util.module_from_spec(_SPEC);sys.modules[_SPEC.name]=R;_SPEC.loader.exec_module(R)


def execute(command,label,output,deadline,log):
    deadline.check();started=time.perf_counter();exit_code=None
    try:
        with (output/(label+'.log')).open('x') as stream:
            completed=subprocess.run(command,cwd=R.ROOT,stdout=stream,stderr=subprocess.STDOUT,
                timeout=deadline.remaining(),env={**os.environ,'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'})
        exit_code=completed.returncode
    finally:
        receipt={'label':label,'elapsed_seconds':time.perf_counter()-started,'exit_code':exit_code}
        log.append(receipt);R.write_json(output/(label+'.execution.json'),receipt)
    if exit_code!=0:raise RuntimeError('registered subprocess failed: '+label)
    deadline.check()


def fit(args):
    register=R.load_register(args.config)
    if register.get('fit_enabled') is not True:raise ValueError('measured no-fit profiles must be sealed before enabling fits')
    if args.seed not in R.SEEDS or args.objective not in R.OBJECTIVES:
        raise ValueError('one registered seed and objective per fit job')
    if args.output.exists():raise FileExistsError(args.output)
    source=R.verify_source(args.source_commit,args.config)
    # The fit artifact contains only these two files, never DEV actors/answers.
    trainer=R.load_local('train');trainer.validate_data(args.data,register)
    args.output.mkdir(parents=True)
    deadline=R.Deadline(register['training'].get('max_leg_seconds',register['training'].get('max_seed_seconds')));log=[];sealed=[]
    R.write_json(args.output/'FIT_PLAN_SEAL.json',{'source_commit':args.source_commit,'source_hashes':source,
        'seed':args.seed,'objective':args.objective,'register_sha256':R.sha256(args.config),
        'matrix':[[f,args.objective] for f in R.FAMILIES],
        'DEV_scoring_permitted':False,'before_first_optimizer_step':True})
    for family in R.FAMILIES:
        for objective in (args.objective,):
            name=family+'-'+objective;folder=args.output/name
            command=[sys.executable,str(R.HERE/'train.py'),'--family',family,'--objective',objective,
                '--seed',str(args.seed),'--data',str(args.data),'--config',str(args.config),'--output',str(folder),
                '--initials',str(args.initials),
                '--source-commit',args.source_commit]
            execute(command,name,args.output,deadline,log)
            receipt=R.read_json(folder/'FIT_RECEIPT.json')
            if (receipt['family'],receipt['objective'],receipt['seed'],receipt['status'])!=(family,objective,args.seed,'SEALED_BEFORE_DEV'):
                raise ValueError('completed fit receipt identity differs')
            if R.sha256(folder/'MODEL.pt')!=receipt['checkpoint_sha256']:raise ValueError('original checkpoint changed before seed seal')
            sealed.append({'family':family,'objective':objective,'seed':args.seed,'checkpoint':name+'/MODEL.pt',
                'checkpoint_sha256':receipt['checkpoint_sha256'],'receipt':name+'/FIT_RECEIPT.json',
                'receipt_sha256':R.sha256(folder/'FIT_RECEIPT.json')})
            print(json.dumps({'stage':'fit_sealed','family':family,'objective':objective,'seed':args.seed}),flush=True)
    deadline.check()
    R.write_json(args.output/'FIT_LEG_SEAL.json',{'schema':'langcontrol098.objective-leg-fit-seal.v1','seed':args.seed,
        'objective':args.objective,
        'source_commit':args.source_commit,'register_sha256':R.sha256(args.config),'models':sealed,
        'DEV_predictions_started':False,'elapsed_seconds':time.perf_counter()-deadline.started,'execution':log,
        'optimizer_steps_all_models':4*register['training']['steps'],'gpu_minutes':0})


def verify_model_checkpoint(folder,receipt,register,source_commit,source_hashes):
    """Read exact tensor/optimizer bytes; never execute a DEV/model forward."""
    import torch
    M=R.load_local('models')
    checkpoint=folder/'MODEL.pt';initial=folder/'INITIAL_MODEL.pt'
    if checkpoint.is_symlink() or initial.is_symlink():raise ValueError('original checkpoints must be regular artifact files')
    if R.sha256(checkpoint)!=receipt['checkpoint_sha256'] or R.sha256(initial)!=receipt['initial_checkpoint_sha256']:
        raise ValueError('original trained or initial checkpoint bytes changed')
    saved=torch.load(checkpoint,map_location='cpu',weights_only=True)
    beginning=torch.load(initial,map_location='cpu',weights_only=True)
    expected={'schema','metadata','model','optimizer','rng'}
    if set(saved)!=expected or set(beginning)!=expected or saved['schema']!='langcontrol098.checkpoint.v1' or beginning['schema']!='langcontrol098.checkpoint.v1':
        raise ValueError('complete original checkpoint/optimizer/RNG archive required')
    metadata=saved['metadata']
    for key in ('study_id','family','objective','objective_control','seed','optimizer_steps','source_commit','source_hashes',
                'register_sha256','data_manifest_sha256','fit_sha256','schedule_sha256',
                'initial_model_sha256','final_model_sha256','profile_initial_state_sha256','counts','status'):
        if metadata.get(key)!=receipt.get(key):raise ValueError('checkpoint and receipt differ: '+key)
    if (metadata['source_commit']!=source_commit or metadata['source_hashes']!=source_hashes
            or metadata['data_manifest_sha256']!=register['data']['manifest_sha256']
            or metadata['fit_sha256']!=register['data']['FIT.json_sha256']
            or metadata['optimizer_steps']!=register['training']['steps']):
        raise ValueError('original checkpoint source/recipe differs')
    family,objective,seed=receipt['family'],receipt['objective'],receipt['seed']
    if metadata.get('objective_control')!=R.load_local('objective').objective_mapping(objective):
        raise ValueError('checkpoint coefficient-control mapping differs')
    optimizer=saved['optimizer'];rng=saved['rng']
    if (not isinstance(optimizer,dict) or not optimizer.get('state') or not optimizer.get('param_groups')
            or not isinstance(rng,dict) or set(rng)!={'python','numpy','torch'}
            or not isinstance(rng['torch'],torch.Tensor) or rng['torch'].dtype!=torch.uint8 or rng['torch'].numel()==0):
        raise ValueError('completed original optimizer update state and RNG archives are required')
    for state in optimizer['state'].values():
        step=state.get('step')
        if (not isinstance(step,torch.Tensor) or step.numel()!=1 or not bool(torch.isfinite(step).all())
                or float(step)!=register['training']['steps']):
            raise ValueError('original optimizer update counts differ from the fixed complete fit')
    initial_metadata=beginning['metadata']
    if (initial_metadata.get('family'),initial_metadata.get('seed'),initial_metadata.get('status'))!=(family,seed,'UNTRAINED_FIT_REPLAY_ONLY'):
        raise ValueError('original pre-optimizer checkpoint identity differs')
    model=M.make_model(family,seed).eval()
    if len(optimizer['state'])!=len(list(model.parameters())):
        raise ValueError('original optimizer does not cover every active parameter tensor')
    model.load_state_dict(beginning['model'],strict=True)
    initial_digest=R.model_digest(model)
    family_contract=R.read_json(R.ROOT/register['manifest_paths'][family])
    profile=R.read_json(R.ROOT/family_contract['operator_profile_path'])
    frozen=next(row for row in profile['profiles'] if row['seed']==seed and row['objective']==objective)
    if (frozen.get('objective_control')!=metadata['objective_control']
            or initial_digest!=frozen['weights_before'] or initial_digest!=receipt['initial_model_sha256']
            or initial_digest!=receipt['actual_initial_model_sha256']
            or receipt.get('actual_profile_initial_state_sha256')!=frozen['initial_state_sha256']
            or metadata.get('profile_initial_state_sha256')!=frozen['initial_state_sha256']):
        raise ValueError('actual initial checkpoint differs from the source-bound profile')
    model.load_state_dict(saved['model'],strict=True)
    counts=R.assert_counts(model,M.PARAMETERS[family])
    if counts!=metadata['counts'] or R.model_digest(model)!=receipt['final_model_sha256']:
        raise ValueError('actual loaded model count/content differs from original receipt')
    if receipt['final_model_sha256']==initial_digest:raise ValueError('completed fit has unchanged original weights')


def model_artifacts(root,register,config,source_commit,source_hashes):
    """Validate the complete forty-artifact matrix before assembly or DEV access."""
    paths=list(Path(root).rglob('FIT_RECEIPT.json'))
    if len(paths)!=40:raise ValueError('all forty original model fit artifacts are required for assembly')
    expected={(family,objective,seed) for family in R.FAMILIES for objective in R.OBJECTIVES for seed in R.SEEDS}
    matrix={};register_sha=R.sha256(config)
    for path in paths:
        if path.is_symlink():raise ValueError('fit receipt must be a regular artifact file')
        receipt=R.read_json(path);key=receipt.get('family'),receipt.get('objective'),receipt.get('seed')
        if key not in expected or key in matrix:raise ValueError('duplicate or unsupported original model artifact')
        if (receipt.get('study_id')!='LANGCONTROL098' or receipt.get('status')!='SEALED_BEFORE_DEV'
                or receipt.get('objective_control')!=R.load_local('objective').objective_mapping(key[1])
                or receipt.get('source_commit')!=source_commit or receipt.get('source_hashes')!=source_hashes
                or receipt.get('register_sha256')!=register_sha or receipt.get('optimizer_steps')!=register['training']['steps']
                or receipt.get('actual_steps')!=register['training']['steps']
                or receipt.get('actual_examples')!=register['training']['steps']*register['training']['batch_size']
                or receipt.get('actual_dense_compute_tokens')!=register['training']['steps']*register['training']['batch_size']*register['training']['context']):
            raise ValueError('incomplete or source/config-mismatched model artifact')
        if receipt.get('replay',{}).get('cold_reload_bitwise_equal') is not True:
            raise ValueError('original fitted checkpoint lacks its completed FIT cold replay')
        if R.sha256(path.parent/'MODEL.pt')!=receipt.get('checkpoint_sha256'):
            raise ValueError('original model artifact checksum changed')
        matrix[key]={'folder':path.parent,'receipt':receipt,'receipt_path':path}
    if set(matrix)!=expected:raise ValueError('entire forty-model artifact matrix is required')
    for seed in R.SEEDS:
        rows=[matrix[(f,o,seed)]['receipt'] for f in R.FAMILIES for o in R.OBJECTIVES]
        for key in ('schedule_sha256','actual_tokens','actual_dense_compute_tokens','actual_examples','actual_steps'):
            if len({r[key] for r in rows})!=1:raise ValueError('original training exposure differs across models: '+key)
        for family in R.FAMILIES:
            paired=[matrix[(family,o,seed)]['receipt']['actual_initial_model_sha256'] for o in R.OBJECTIVES]
            if paired[0]!=paired[1]:raise ValueError('objective pair did not start from identical actual weights')
    legs=list(Path(root).rglob('FIT_LEG_SEAL.json'))
    if len(legs)!=10:raise ValueError('all ten successful objective-leg seals are required for assembly')
    seen=set()
    for path in legs:
        seal=R.read_json(path);identity=seal.get('seed'),seal.get('objective')
        if (identity not in {(s,o) for s in R.SEEDS for o in R.OBJECTIVES} or identity in seen
                or seal.get('schema')!='langcontrol098.objective-leg-fit-seal.v1'
                or seal.get('source_commit')!=source_commit or seal.get('register_sha256')!=register_sha
                or seal.get('DEV_predictions_started') is not False or len(seal.get('models',[]))!=4):
            raise ValueError('complete original objective-leg seal required')
        seen.add(identity);leg_families=set()
        for row in seal['models']:
            key=row.get('family'),row.get('objective'),row.get('seed')
            if key not in matrix or key[1:]!=(identity[1],identity[0]) or key[0] in leg_families:
                raise ValueError('objective-leg model matrix differs')
            leg_families.add(key[0]);item=matrix[key]
            if row.get('checkpoint_sha256')!=item['receipt']['checkpoint_sha256'] or row.get('receipt_sha256')!=R.sha256(item['receipt_path']):
                raise ValueError('objective-leg seal differs from original checkpoint/receipt')
    for key,item in sorted(matrix.items()):
        verify_model_checkpoint(item['folder'],item['receipt'],register,source_commit,source_hashes)
    return matrix


def assemble(args):
    """Join original artifacts only: no dataset, actor, scorer or optimizer access."""
    if args.output.exists():raise FileExistsError(args.output)
    register=R.load_register(args.config);source=R.verify_source(args.source_commit,args.config)
    matrix=model_artifacts(args.fits,register,args.config,args.source_commit,source)
    args.output.mkdir(parents=True)
    for seed in R.SEEDS:
        folder=args.output/f'seed-{seed}';folder.mkdir();sealed=[]
        for family in R.FAMILIES:
            for objective in R.OBJECTIVES:
                item=matrix[(family,objective,seed)];name=family+'-'+objective
                # Preserve every original diagnostic, optimizer/RNG checkpoint and log.
                shutil.copytree(item['folder'],folder/name)
                receipt=item['receipt']
                sealed.append({'family':family,'objective':objective,'seed':seed,'checkpoint':name+'/MODEL.pt',
                    'checkpoint_sha256':receipt['checkpoint_sha256'],'receipt':name+'/FIT_RECEIPT.json',
                    'receipt_sha256':R.sha256(folder/name/'FIT_RECEIPT.json')})
        R.write_json(folder/'FITS_SEAL.json',{'schema':'langcontrol098.seed-fit-seal.v1','seed':seed,
            'source_commit':args.source_commit,'register_sha256':R.sha256(args.config),'models':sealed,
            'DEV_predictions_started':False,'assembled_from_separately_bounded_objective_legs':True,
            'optimizer_steps_all_models':8*register['training']['steps'],'gpu_minutes':0})
    all_fits(args.output,register,args.config,args.source_commit)
    R.write_json(args.output/'ASSEMBLY.json',{'schema':'langcontrol098.artifact-assembly.v1','models_verified':40,
        'objective_legs_verified':10,'seed_seals':5,'source_commit':args.source_commit,
        'register_sha256':R.sha256(args.config),'datasets_opened':False,'DEV_predictions_started':False,
        'model_forward_calls':0,'optimizer_steps':0,'gpu_minutes':0})


def all_fits(root,register,config,source_commit):
    """Verify exact original bytes and the entire matrix without opening DEV."""
    paths=list(Path(root).rglob('FITS_SEAL.json'))
    if len(paths)!=5:raise ValueError('all five successful seed seals are required before any DEV predictions')
    matrix={};seeds=set();register_sha=R.sha256(config)
    for path in paths:
        seal=R.read_json(path);seed=seal.get('seed')
        if (seal.get('schema')!='langcontrol098.seed-fit-seal.v1' or seed not in R.SEEDS or seed in seeds
                or seal.get('source_commit')!=source_commit or seal.get('register_sha256')!=register_sha
                or seal.get('DEV_predictions_started') is not False or len(seal.get('models',[]))!=8):
            raise ValueError('complete original source-bound seed seal required')
        seeds.add(seed)
        for item in seal['models']:
            key=item.get('family'),item.get('objective'),item.get('seed')
            if key[0] not in R.FAMILIES or key[1] not in R.OBJECTIVES or key[2]!=seed or key in matrix:
                raise ValueError('duplicated or unsupported fit matrix row')
            for name in ('checkpoint','receipt'):
                relative=Path(item[name])
                if relative.is_absolute() or '..' in relative.parts:raise ValueError('fit seal path escapes artifact')
                local=path.parent/relative
                if local.is_symlink() or R.sha256(local)!=item[name+'_sha256']:raise ValueError('sealed original fit artifact changed')
            receipt=R.read_json(path.parent/item['receipt'])
            expected_control=R.load_local('objective').objective_mapping(key[1])
            if (receipt.get('study_id')!='LANGCONTROL098'
                    or (receipt.get('family'),receipt.get('objective'),receipt.get('seed'))!=key
                    or R.json_digest(receipt.get('objective_control'))!=R.json_digest(expected_control)
                    or receipt.get('register_sha256')!=register_sha
                    or receipt.get('status')!='SEALED_BEFORE_DEV'
                    or receipt.get('actual_steps')!=register['training']['steps']
                    or receipt.get('source_commit')!=source_commit
                    or receipt.get('checkpoint_sha256')!=item['checkpoint_sha256']):
                raise ValueError('incomplete or different fit is not eligible for DEV scoring')
            matrix[key]={'checkpoint':path.parent/item['checkpoint'],'sha256':item['checkpoint_sha256'],'receipt':receipt}
    expected={(f,o,s) for f in R.FAMILIES for o in R.OBJECTIVES for s in R.SEEDS}
    if set(matrix)!=expected:raise ValueError('entire forty-model matrix must seal before DEV')
    for seed in R.SEEDS:
        rows=[matrix[(f,o,seed)]['receipt'] for f in R.FAMILIES for o in R.OBJECTIVES]
        for key in ('schedule_sha256','actual_tokens','actual_dense_compute_tokens','actual_examples','actual_steps'):
            if len({r[key] for r in rows})!=1:raise ValueError('frozen source exposure differs across models: '+key)
    return matrix


def final_gates(matrix,register,output,deadline,log):
    gates={}
    for family in R.FAMILIES:
        manifest=R.read_json(R.ROOT/register['manifest_paths'][family])
        for row in manifest['models']:
            row['measured']=matrix[(family,row['name'],row['seed'])]['receipt']
        path=output/('FINAL_PARITY_'+family.upper()+'.json');R.write_json(path,manifest)
        command=[sys.executable,str(R.ROOT/'scripts/validate_research_parity.py'),'--manifest',str(path),'--phase','final']
        execute(command,'final-parity-'+family,output,deadline,log)
        gates[family]={'status':'PASS','manifest_sha256':R.sha256(path),'scope':'within-family complete operator parity only'}
    return gates


def load_predictions(path):
    with np.load(path,allow_pickle=False) as archive:
        values={k:archive[k].copy() for k in archive.files if k!='metadata'}
        metadata=json.loads(str(archive['metadata'].item()))
    return values,metadata


def replay(left,right):
    a,_=load_predictions(left);b,_=load_predictions(right)
    keys=('log_probs','generated_bytes','lengths','available')
    if any(k not in a or k not in b or not np.array_equal(a[k],b[k]) for k in keys):
        raise ValueError('fresh-process deterministic query replay differs')


def prediction_rows(payload,metadata,actor,judge):
    conditions=(*R.CONDITIONS,'binding_shuffle');n=len(actor)
    if (metadata.get('schema')!='langcontrol098.predictions.v1' or metadata.get('conditions')!=list(conditions)
            or metadata.get('case_ids')!=[r['id'] for r in actor] or metadata.get('optimizer_steps')!=0):
        raise ValueError('prediction ordering or inference provenance differs')
    lp=payload['log_probs'];generated=payload['generated_bytes'];available=payload['available'];lengths=payload['lengths']
    if (lp.shape!=(5,n,256) or generated.shape!=(5,n,40) or available.shape!=(5,n) or available.dtype!=bool
            or lengths.shape!=(n,) or not np.isfinite(lp).all() or generated.dtype!=np.uint8):
        raise ValueError('prediction geometry/dtype/finiteness differs')
    if np.max(np.abs(np.logaddexp.reduce(lp.astype(float),axis=-1)))>5e-5:raise ValueError('first-byte probabilities are not normalized')
    case_rows=[];binding=[];D=R.load_local('data');checkpoint=metadata['checkpoint']
    for i,observed in enumerate(actor):
        gold=judge[observed['id']]
        if int(lengths[i])!=observed['generation_limit'] or gold['domain']!=observed['domain']:
            raise ValueError('fixed generation contract differs')
        target=R.bytes_list(gold['target_bytes'],'scorer-owned authentic target')
        for ci,condition in enumerate(conditions):
            if condition=='binding_shuffle' and observed['domain']!='arc':
                if available[ci,i]:raise ValueError('ARC-only mechanism control leaked into lexical cohort')
                continue
            expected=(gold['donor_available'] if condition=='donor' else gold['binding_shuffle_available'] if condition=='binding_shuffle' else True)
            if bool(available[ci,i])!=expected:raise ValueError('predeclared control availability changed')
            row={'family':checkpoint['family'],'objective':checkpoint['objective'],'seed':checkpoint['seed'],
                 'domain':gold['domain'],'case_id':observed['id'],'cluster_id':gold['cluster_id'],
                 'group':gold['group'],'condition':condition}
            if not expected:
                row.update(status='INCONCLUSIVE',nll=None,exact_match=None,format_valid=None)
            else:
                values=generated[ci,i,:int(lengths[i])]
                if int(values[0])!=int(lp[ci,i].argmax()):raise ValueError('greedy first byte differs from first distribution')
                if gold['domain']=='lexical':
                    valid=True;correct=bool(np.array_equal(values,np.asarray(target,dtype=np.uint8)))
                else:
                    try:prediction=D.decode_arc_output(values);valid=True
                    except ValueError:prediction=None;valid=False
                    expected_grid=D.decode_arc_output(bytes(target));correct=bool(valid and prediction==expected_grid)
                row.update(status='MEASURED',nll=-float(lp[ci,i,target[0]]),exact_match=int(correct),format_valid=int(valid))
            (binding if condition=='binding_shuffle' else case_rows).append(row)
    return case_rows,binding


def raw_rows(path,actor,judge):
    payload,metadata=load_predictions(path);lookup={r['id']:r for r in actor}
    if metadata.get('schema')!='langcontrol098.raw-control.v1' or metadata.get('optimizer_steps')!=0:
        raise ValueError('raw observed-history-only reference required')
    rows=[];arc=[];D=R.load_local('data')
    ids=[str(x) for x in payload['ids']]
    if len(ids)!=64 or len(set(ids))!=64:raise ValueError('complete lexical raw control cohort required')
    for i,case_id in enumerate(ids):
        gold=judge[case_id]
        if gold['domain']!='lexical' or case_id not in lookup:raise ValueError('raw control case binding differs')
        target=gold['target_bytes'];lp=payload['log_probs'][i];generated=payload['generated'][i]
        rows.append({'case_id':case_id,'nll':-float(lp[target[0]]),'exact_match':int(np.array_equal(generated,np.asarray(target,dtype=np.uint8)))})
    arc_ids=[str(x) for x in payload['arc_ids']]
    if len(arc_ids)!=8 or len(set(arc_ids))!=8:raise ValueError('complete ARC copy reference cohort required')
    for i,case_id in enumerate(arc_ids):
        gold=judge[case_id]
        try:prediction=D.decode_arc_output(payload['arc_generated'][i]);valid=True
        except ValueError:prediction=None;valid=False
        correct=bool(valid and prediction==D.decode_arc_output(bytes(gold['target_bytes'])))
        arc.append({'case_id':case_id,'cluster_id':gold['cluster_id'],'format_valid':int(valid),'exact_match':int(correct),
            'scope':'untrained copy-last-observed-demonstration output; acquisition only, no query or gold'})
    return rows,arc,metadata


def evaluate(args):
    if args.output.exists():raise FileExistsError(args.output)
    register=R.load_register(args.config);source=R.verify_source(args.source_commit,args.config)
    matrix=all_fits(args.fits,register,args.config,args.source_commit)
    args.output.mkdir(parents=True);deadline=R.Deadline(40*60);log=[]
    R.write_json(args.output/'ALL_CHECKPOINTS_SEALED.json',{'models':[
        {'family':f,'objective':o,'seed':s,'checkpoint_sha256':matrix[(f,o,s)]['sha256']} for f,o,s in sorted(matrix)],
        'source_commit':args.source_commit,'source_hashes':source,'before_first_DEV_prediction':True,
        'before_evaluator_opens_DEV_inputs':True,'models_sealed':40})
    gates=final_gates(matrix,register,args.output,deadline,log)
    D=R.load_local('data');data_manifest=D.prepare(args.data)
    if R.sha256(args.data/'MANIFEST.json')!=register['data']['manifest_sha256']:raise ValueError('frozen evaluation source manifest differs')
    for name,expected in register['data']['artifact_sha256'].items():
        if R.sha256(args.data/name)!=expected:raise ValueError('frozen data bytes differ: '+name)
    actor=R.read_json(args.data/'DEV_ACTOR.json')
    alternate,poison_audit=D.future_poison_audit(args.data)
    if alternate!=actor:raise ValueError('future-only source poisoning changed actual actor observations')
    R.write_json(args.output/'SOURCE_FUTURE_POISON.json',poison_audit)
    acq=[{k:r[k] for k in ('id','domain','acquisition_bytes')} for r in actor]
    queries=[{k:r[k] for k in ('id','domain','query_bytes','generation_limit')} for r in actor]
    poison_queries=[{k:r[k] for k in ('id','domain','query_bytes','generation_limit')} for r in alternate]
    inputs=args.output/'actor-inputs';inputs.mkdir()
    R.write_json(inputs/'acquisition.json',acq);R.write_json(inputs/'queries.json',queries);R.write_json(inputs/'future-poison-queries.json',poison_queries)
    replay_receipts=[];artifacts=[]
    for family,objective,seed in sorted(matrix):
        sealed=matrix[(family,objective,seed)];name=f'{family}-{objective}-{seed}';folder=args.output/name;folder.mkdir()
        state=folder/'STATE.pt';common=[sys.executable,str(R.HERE/'evaluate.py'),'--checkpoint',str(sealed['checkpoint']),
            '--sha256',sealed['sha256'],'--config',str(args.config),'--source-commit',args.source_commit,'--state',str(state)]
        execute(common+['--mode','acquire','--acquisition',str(inputs/'acquisition.json'),'--controls',str(args.data/'DEV_CONTROL_ACTOR.json'),
            '--output',str(folder/'ACQUISITION.json')],name+'-acquire',args.output,deadline,log)
        state_hash=R.sha256(state)
        for suffix,query_file in [('original','queries.json'),('replay','future-poison-queries.json')]:
            execute(common+['--mode','query','--queries',str(inputs/query_file),'--donors',str(args.data/'DEV_DONORS.json'),
                '--output',str(folder/(suffix+'.npz'))],name+'-'+suffix,args.output,deadline,log)
        replay(folder/'original.npz',folder/'replay.npz')
        if R.sha256(state)!=state_hash:raise ValueError('query mutated sealed state file')
        payload,metadata=load_predictions(folder/'original.npz')
        if (metadata['checkpoint_sha256']!=sealed['sha256'] or metadata['checkpoint']['family']!=family
                or metadata['checkpoint']['objective']!=objective or metadata['checkpoint']['seed']!=seed
                or metadata['query_sha256']!=R.sha256(inputs/'queries.json')
                or metadata['acquisition_sha256']!=R.sha256(inputs/'acquisition.json')):raise ValueError('actual prediction source binding differs')
        artifacts.append((folder/'original.npz',metadata))
        replay_receipts.append({'family':family,'objective':objective,'seed':seed,'status':'PASS','state_immutable':True,
            'prediction_sha256':R.sha256(folder/'original.npz'),'replay_sha256':R.sha256(folder/'replay.npz')})
        print(json.dumps({'stage':'prediction_replayed','family':family,'objective':objective,'seed':seed}),flush=True)
    raw=args.output/'RAW_HISTORY.npz'
    execute([sys.executable,str(R.HERE/'raw_control.py'),'--actor',str(args.data/'DEV_ACTOR.json'),'--output',str(raw)],'raw-reference',args.output,deadline,log)
    R.write_json(args.output/'ALL_PREDICTIONS_SEALED.json',{'models':40,'replays':replay_receipts,
        'before_target_score_computation':True,'raw_sha256':R.sha256(raw),
        'scope':'Source/poison producers can inspect targets; query actor processes receive no judge input.'})
    # Target scoring begins here; the authentication/poison producer read gold earlier.
    judge=R.read_json(args.data/'DEV_JUDGE.json');rows=[];binding=[]
    for path,metadata in artifacts:
        payload,_=load_predictions(path);main,aux=prediction_rows(payload,metadata,actor,judge)
        rows.extend(main);binding.extend(aux)
    reference,arc_reference,raw_metadata=raw_rows(raw,actor,judge)
    integrity={k:'PASS' for k in ('source_provenance','split_isolation','prediction_boundary','fresh_process_replay',
        'parameter_budget','training_exposure','within_family_compute','gradient_to_acquisition')}
    # Each PASS above is tied to enforced actual source/fit/replay checks, not a fixture.
    if set(data_manifest['fit_owners'])&set(data_manifest['dev_owners']):raise ValueError('original source owners overlap')
    if any(r['receipt']['acquisition_gradient'][d]['status']!='PASS' for r in matrix.values() for d in ('lexical','arc')):
        raise ValueError('actual query-to-acquisition gradient check failed')
    A=R.load_local('analysis');report=A.analyze(rows,judge,integrity,raw_rows=reference,binding_rows=binding,
                                             arc_reference_rows=arc_reference)
    if report['status']=='FAIL':raise ValueError('frozen analysis rejected actual outputs: '+str(report.get('analysis_error')))
    R.write_json(args.output/'SCORED_ROWS.json',rows);R.write_json(args.output/'BINDING_ROWS.json',binding)
    R.write_json(args.output/'ARC_COPY_REFERENCE.json',{'rows':arc_reference,'scope':'untrained practical reference, not neural parity',
        'correct':sum(r['exact_match'] for r in arc_reference),'format_valid':sum(r['format_valid'] for r in arc_reference),'cases':len(arc_reference)})
    A.write_reports(report,args.output/'RESULTS.json',args.output/'RESULTS.md')
    deadline.check()
    for key,item in matrix.items():
        if R.sha256(item['checkpoint'])!=item['sha256']:raise ValueError('original model changed during evaluation')
    for name,expected in register['data']['artifact_sha256'].items():
        if R.sha256(args.data/name)!=expected:raise ValueError('original data changed during evaluation')
    R.write_json(args.output/'EXECUTION.json',{'status':'COMPLETED_INTERNAL_RESEARCH','source_commit':args.source_commit,
        'models_fitted':40,'fresh_process_replays':replay_receipts,'family_final_parity':gates,
        'elapsed_evaluation_seconds':time.perf_counter()-deadline.started,'execution':log,'raw_metadata':raw_metadata,
        'optimizer_steps_during_evaluation':0,'gpu_minutes':0,'protected_repository_TEST_opened':False,
        'general_intelligence_established':False,'cross_architecture_superiority_eligible':False})


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--mode',choices=('fit','assemble','evaluate'),required=True)
    p.add_argument('--seed',type=int);p.add_argument('--objective',choices=R.OBJECTIVES)
    p.add_argument('--fits',type=Path);p.add_argument('--data',type=Path)
    p.add_argument('--initials',type=Path)
    p.add_argument('--config',type=Path,default=R.HERE/'REGISTER.json');p.add_argument('--source-commit',required=True)
    p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    try:
        if args.mode=='fit':
            if args.data is None or args.initials is None:raise ValueError('FIT-only data and original zero-update initialization artifacts required')
            fit(args)
        elif args.mode=='assemble':
            if args.fits is None or args.seed is not None or args.objective is not None or args.data is not None or args.initials is not None:
                raise ValueError('assembly accepts only original fit artifacts, not data or a selected arm')
            assemble(args)
        else:
            if args.fits is None or args.seed is not None or args.objective is not None or args.data is None or args.initials is not None:
                raise ValueError('evaluation requires all five complete seed artifacts and frozen DEV data')
            evaluate(args)
    except BaseException as error:
        if args.output.exists() and not (args.output/'FAILED_ATTEMPT.json').exists():
            R.write_json(args.output/'FAILED_ATTEMPT.json',{'status':'FAILED_OR_INCONCLUSIVE_RETAINED','reason':type(error).__name__+': '+str(error)})
        raise

if __name__=='__main__':main()
