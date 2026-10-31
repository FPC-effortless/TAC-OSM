#!/usr/bin/env python3
"""Independent, read-only LANGCONTROL098 tensor/receipt audit.

This script lives outside the frozen source tree. It never calls a model
forward, optimizer, trainer or predictor. It does not recompute capability
scores. It loads original tensors, independently recomputes hashes/counts,
and checks existing original/fresh-query-replay provenance.
"""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import re
import subprocess
import sys

import numpy as np
import torch

FAMILIES=('tac','gru','ssm','transformer')
SEEDS=(1701,1702,1703,1704,1705)
OBJECTIVES=('full','query')
COUNTS={'tac':100521,'gru':100661,'ssm':100404,'transformer':100484}
EXPECTED={(f,o,s) for f in FAMILIES for o in OBJECTIVES for s in SEEDS}
ZERO_WORK_PROFILER={'profiler::_record_function_enter_new','profiler::_record_function_exit'}
EXPECTED_SOURCE='50bf7f2818f38c31f7c4d8280c1352c06f33bd4f'
STUDY=Path('research/mathematical-engineering/langcontrol098')
REGISTER_CONTROL={'domain_weights':{'lexical':.5,'arc':.5},'prefix_lambda_by_objective':{'full':1.,'query':0.},
    'reduction':'fixed_domain_case_segment_means','normalize_by_total_coefficient_mass':False}
SHARED=('study_id','objective_control','source_commit','source_hashes','register_sha256','data_manifest_sha256','fit_sha256',
        'schedule_sha256','family','objective','seed','optimizer_steps','counts',
        'initial_model_sha256','final_model_sha256','profile_initial_state_sha256','status')


class AuditError(ValueError):pass


def require(value,reason):
    if not value:raise AuditError(reason)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def canonical(value):return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
def digest(value):return hashlib.sha256(canonical(value)).hexdigest()
def read(path):
    path=Path(path)
    require(path.is_file() and not path.is_symlink(), 'missing/nonregular original JSON: '+str(path))
    return json.loads(path.read_text())


def expected_control(objective):
    require(objective in OBJECTIVES, 'unknown registered objective')
    return {'objective_label':objective,'prefix_lambda':1. if objective=='full' else 0.,
            'future_component_coefficient':1.,'domain_weights':{'lexical':.5,'arc':.5},
            'reduction':'fixed_domain_case_segment_means','normalize_by_total_coefficient_mass':False}


def frozen_file(repo,relative,commit=EXPECTED_SOURCE):
    require(commit==EXPECTED_SOURCE, 'audit source differs from frozen original fit source')
    relative=Path(relative)
    require(not relative.is_absolute() and '..' not in relative.parts, 'source path escapes frozen repository')
    path=repo/relative
    require(path.is_file() and not path.is_symlink() and path.resolve().is_relative_to(repo), 'missing/escaping frozen source')
    original=subprocess.check_output(['git','show',commit+':'+relative.as_posix()],cwd=repo)
    require(path.read_bytes()==original, 'disk bytes differ from frozen original source: '+relative.as_posix())
    return hashlib.sha256(original).hexdigest()


def audit_loss_history(receipt,objective,steps=256):
    """Recompute the registered scalar decomposition, not gradients or inference."""
    require(receipt['objective_control']==expected_control(objective), 'fit coefficient control changed')
    history=receipt['history']
    require(len(history)==steps and [row['step'] for row in history]==list(range(1,steps+1)), 'original update history incomplete')
    lam=1. if objective=='full' else 0.
    maximum=0.;clipped=0
    for row in history:
        require(row['prefix_lambda']==lam, 'training history prefix coefficient differs')
        domains=row['per_domain_losses']
        require(set(domains)=={'lexical','arc'} and all(set(parts)=={'future','prefix'} for parts in domains.values()), 'loss component domain/segment fields differ')
        values=[row['loss'],row['future_loss'],row['prefix_loss'],row['gradient_norm_before_clip']]
        values.extend(value for parts in domains.values() for value in parts.values())
        require(all(isinstance(value,(int,float)) and not isinstance(value,bool) and math.isfinite(value) and value>=0 for value in values), 'invalid original scalar loss/gradient diagnostic')
        expected_future=.5*domains['lexical']['future']+.5*domains['arc']['future']
        expected_prefix=.5*domains['lexical']['prefix']+.5*domains['arc']['prefix']
        for actual,expected in ((row['future_loss'],expected_future),(row['prefix_loss'],expected_prefix),
                                (row['loss'],row['future_loss']+lam*row['prefix_loss'])):
            delta=abs(actual-expected);maximum=max(maximum,delta)
            require(math.isclose(actual,expected,rel_tol=2e-6,abs_tol=2e-6), 'actual scalar objective is not fixed domain/case-segment coefficient sum')
        expected_clipped=row['gradient_norm_before_clip']>1.
        require(type(row['gradient_clipped'])is bool and row['gradient_clipped']==expected_clipped, 'gradient clip history differs')
        clipped+=int(expected_clipped)
    require(receipt['gradient_clipped_steps']==clipped and receipt['gradient_clip_frequency']==clipped/steps, 'gradient clipping aggregate differs')
    return {'status':'PASS_SAVED_SCALAR_COEFFICIENT_DECOMPOSITION','updates':steps,
            'max_float32_component_sum_difference':maximum,'gradient_clipped_steps':clipped,
            'scope':'Scalar saved losses only; not a gradient-contribution or update-share claim.'}



def tensor_digest(state):
    h=hashlib.sha256()
    for name,value in sorted(state.items()):
        require(isinstance(value,torch.Tensor),'non-tensor model state')
        h.update(name.encode());h.update(str(tuple(value.shape)).encode())
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def saved_state_digest(states):
    h=hashlib.sha256()
    for state in states:
        h.update(json.dumps([state['arm'],state['owners'],state['clock']]).encode())
        for name,value in sorted(state['tensors'].items()):
            h.update(name.encode());h.update(str(tuple(value.shape)).encode())
            h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def load_payload(path):
    require(not path.is_symlink(),'symlink checkpoint is not an original archive')
    payload=torch.load(path,map_location='cpu',weights_only=True)
    require(isinstance(payload,dict) and set(payload)=={'schema','metadata','model','optimizer','rng'},'incomplete checkpoint/optimizer/RNG archive')
    require(payload['schema']=='langcontrol098.checkpoint.v1','unexpected checkpoint schema')
    return payload


def check_ledger(rows,stage):
    require(isinstance(rows,list) and bool(rows),'missing complete operator ledger')
    seen=set();calls=0;markers={}
    for row in rows:
        require(set(row)=={'stage','signature','calls'} and row['stage']==stage,'operator ledger stage/schema differs')
        require(type(row['calls']) is int and row['calls']>0,'operator call count is not positive')
        signature=json.loads(row['signature'])
        require(isinstance(signature,list) and len(signature)==3 and isinstance(signature[0],str),'invalid actual operator signature')
        if signature[0] in ZERO_WORK_PROFILER:
            require(stage=='optimizer' and signature[1:]==[[],[]] and row['signature']==canonical(signature).decode(),'unexpected nonempty/noncanonical profiler marker')
            markers[signature[0]]=row['calls']
        else:
            require(signature[0].startswith('aten::'),'unknown non-ATen arithmetic operator')
        require(row['signature'] not in seen,'duplicate operator signature')
        seen.add(row['signature']);calls+=row['calls']
        for side in signature[1:]:
            require(isinstance(side,list),'operator tensor signature missing')
            for dtype,shape in side:
                require(isinstance(dtype,str) and isinstance(shape,list) and all(type(n)is int and n>=0 for n in shape),'invalid operator dtype/shape')
    def order(row):
        name,inputs,outputs=json.loads(row['signature'])
        normalized=lambda side:tuple((dtype,tuple(shape)) for dtype,shape in side)
        return row['stage'],name,normalized(inputs),normalized(outputs)
    require(rows==sorted(rows,key=order),'operator rows not canonical by typed tensor signature')
    require(not markers or (set(markers)==ZERO_WORK_PROFILER and len(set(markers.values()))==1),'incomplete/unbalanced zero-tensor profiler marker pair')
    require(any(json.loads(row['signature'])[0].startswith('aten::') for row in rows),'profiler-only ledger has no arithmetic work')
    return digest(rows),calls


def import_models(repo):
    path=repo/'research/mathematical-engineering/langcontrol098/models.py'
    name='langcontrol098_independent_models'
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec)
    sys.modules[name]=module;spec.loader.exec_module(module)
    return module


def load_contracts(repo,register_path,profiles_override=None):
    frozen_file(repo, register_path.relative_to(repo))
    register=read(register_path)
    require(register['schema']=='langcontrol098.register.v1' and register['study_id']=='LANGCONTROL098' and register['objective_control']==REGISTER_CONTROL,'registered coefficient-control contract differs')
    require(register['seeds']==list(SEEDS) and register['families']==list(FAMILIES),'frozen matrix differs')
    require(register['objectives']==list(OBJECTIVES) and register['training']['steps']==256,'frozen objective/update recipe differs')
    require(register['training']['batch_size']==4 and register['training']['context']==272,'frozen dense geometry differs')
    profiles={};contracts={}
    for family in FAMILIES:
        frozen_file(repo,register['manifest_paths'][family])
        contract=read(repo/register['manifest_paths'][family]);contracts[family]=contract
        require(contract['initial_state_schema']=='langcontrol098.initial-state.v1' and contract['optimizer_marker_contract']=='balanced-empty-optimizer-profiler-v1','prospective source/marker contract differs')
        path=(profiles_override/f'PROFILE_{family}.json') if profiles_override else repo/contract['operator_profile_path']
        require(sha(path)==contract['operator_profile_sha256'],'original operator profile bytes differ')
        profile=read(path);require(profile['family']==family and profile['optimizer_steps']==0,'original no-fit profile identity differs')
        require(digest(profile['source_hashes'])==profile['implementation_sha256']==contract['implementation_sha256'],'profile implementation digest differs')
        for relative,expected in profile['source_hashes'].items():
            p=repo/relative
            require(not Path(relative).is_absolute() and not p.is_symlink() and p.resolve().is_relative_to(repo),'profile source escapes frozen repository')
            require(sha(p)==expected,'actual profiled source changed: '+relative)
            require(frozen_file(repo,relative)==expected,'source profile does not bind original Git object')
        require(digest(profile['training_recipe'])==profile['recipe_sha256']==contract['training_recipe_sha256'],'training recipe digest differs')
        require(profile['training_recipe']==register['training'],'actual registered training recipe differs')
        require(len(profile['profiles'])==10,'incomplete original seed/objective profile matrix')
        for row in profile['profiles']:
            require(row['objective_control']==expected_control(row['objective']),'profile coefficient-control formula differs')
            key=(family,row['objective'],row['seed']);require(key in EXPECTED and key not in profiles,'profile matrix duplicated/unknown')
            ledger_sha,_=check_ledger(row['operators'],'forward_backward')
            require(ledger_sha==row['operator_sha256'],'original operator fingerprint differs')
            require(row['weights_before']==row['weights_after'],'no-fit profiling altered model weights')
            profiles[key]=row
    require(set(profiles)==EXPECTED,'incomplete actual profile matrix')
    return register,contracts,profiles


def audit_initials(models,profiles,initials):
    reports=[]
    for family in FAMILIES:
        for seed in SEEDS:
            full,query=(profiles[(family,o,seed)] for o in OBJECTIVES)
            require(full['initial_state_sha256']==query['initial_state_sha256'] and full['weights_before']==query['weights_before'],'paired source initial weights differ')
            name=full['initial_state_file'];require(name==f'INIT_{family}_{seed}.pt','original initialization name differs')
            path=initials/name;require(sha(path)==full['initial_state_sha256'],'original initialization archive bytes differ')
            payload=torch.load(path,map_location='cpu',weights_only=True)
            require(set(payload)=={'schema','family','seed','model','model_sha256','optimizer_steps'},'initial archive schema differs')
            require((payload['schema'],payload['family'],payload['seed'],payload['optimizer_steps'])==('langcontrol098.initial-state.v1',family,seed,0),'initial archive is not original zero-update model')
            require(tensor_digest(payload['model'])==payload['model_sha256']==full['weights_before'],'actual zero-update tensor digest differs')
            model=models.make_model(family,seed);model.load_state_dict(payload['model'],strict=True)
            count=sum(p.numel() for p in model.parameters() if p.requires_grad);deployed=sum(p.numel() for p in model.parameters())
            require(count==deployed==COUNTS[family] and 95000<=count<=105000,'actual original active/deployed count differs')
            reports.append({'family':family,'seed':seed,'state_archive_sha256':sha(path),'model_sha256':payload['model_sha256'],'trainable_parameters':count,'deployed_parameters':deployed,'optimizer_steps':0})
    return reports


def check_rng(rng):
    require(isinstance(rng,dict) and set(rng)=={'python','numpy','torch'},'all original RNG states required')
    require(isinstance(rng['python'],tuple) and len(rng['python'])==3 and isinstance(rng['python'][1],tuple) and len(rng['python'][1])>=624,'Python RNG state incomplete')
    require(isinstance(rng['numpy'],tuple) and len(rng['numpy'])==5 and rng['numpy'][0]=='MT19937' and len(rng['numpy'][1])==624,'NumPy RNG state incomplete')
    require(isinstance(rng['torch'],torch.Tensor) and rng['torch'].dtype==torch.uint8 and rng['torch'].ndim==1 and rng['torch'].numel()>0,'Torch RNG state incomplete')


def check_optimizer(model,optimizer,training):
    require(isinstance(optimizer,dict) and set(optimizer)=={'state','param_groups'},'original optimizer state archive incomplete')
    params=list(model.named_parameters());groups=optimizer['param_groups']
    require(isinstance(groups,list) and len(groups)==1,'one source AdamW parameter group required')
    group=groups[0];identities=group['params']
    require(len(identities)==len(set(identities))==len(params) and set(identities)==set(optimizer['state']),'original optimizer does not cover all active parameter tensors')
    for field,wanted in [('lr',training['learning_rate']),('weight_decay',training['weight_decay']),('eps',training['eps']),('foreach',False),('fused',False),('amsgrad',False),('maximize',False),('capturable',False),('differentiable',False)]:
        require(group.get(field)==wanted,'actual AdamW contract differs: '+field)
    require(tuple(group['betas'])==tuple(training['betas']),'actual optimizer momentum differs')
    actual_steps=[]
    for ident,(name,param) in zip(identities,params):
        state=optimizer['state'][ident]
        require(set(state)=={'step','exp_avg','exp_avg_sq'},'AdamW slot fields differ: '+name)
        step=state['step'];require(isinstance(step,torch.Tensor) and step.numel()==1 and bool(torch.isfinite(step)) and float(step)==training['steps'],'actual AdamW slot update count differs: '+name)
        for field in ('exp_avg','exp_avg_sq'):
            value=state[field]
            require(isinstance(value,torch.Tensor) and value.shape==param.shape and value.dtype==param.dtype and bool(torch.isfinite(value).all()),'original AdamW moment tensor invalid: '+name)
        require(bool((state['exp_avg_sq']>=0).all()),'AdamW squared moment is negative: '+name)
        actual_steps.append(int(step))
    return {'parameter_tensors':len(params),'slot_count':len(optimizer['state']),'all_slot_steps':sorted(set(actual_steps)),'optimizer_contract':'AdamW exact registered dense/foreach/fused settings'}


def audit_one(path,models,register,contracts,profiles,repo,register_sha,source_commit,fit_rows):
    receipt=read(path);key=(receipt['family'],receipt['objective'],receipt['seed']);require(key in EXPECTED,'unregistered original fit identity')
    family,objective,seed=key;profile=profiles[key];folder=path.parent
    require(receipt['study_id']=='LANGCONTROL098' and receipt['status']=='SEALED_BEFORE_DEV','fit not complete before DEV')
    require(receipt['objective_control']==expected_control(objective)==profile['objective_control'],'fit/profile coefficient-control mapping differs')
    require(receipt['source_commit']==source_commit and receipt['register_sha256']==register_sha,'original source/registration differs')
    for relative,expected in receipt['source_hashes'].items():
        p=repo/relative;require(not Path(relative).is_absolute() and p.resolve().is_relative_to(repo) and not p.is_symlink(),'source path escapes frozen repository')
        require(sha(p)==expected,'actual original fit source differs: '+relative)
    require(receipt['data_manifest_sha256']==register['data']['manifest_sha256'] and receipt['fit_sha256']==register['data']['FIT.json_sha256'],'actual source data differs')
    require(receipt['optimizer_steps']==receipt['actual_steps']==256 and receipt['actual_examples']==1024 and receipt['actual_dense_compute_tokens']==278528,'actual original fit exposure differs')
    contract=next(r for r in contracts[family]['models'] if r['name']==objective and r['seed']==seed)
    require(receipt['actual_sample_order_sha256']==receipt['schedule_sha256']==profile['schedule_sha256']==contract['training_sample_order_sha256'],'actual sample-order hash differs')
    prefit=read(folder/'PREFIT.json');require(digest(prefit['schedule_ids'])==receipt['schedule_sha256'],'actual preserved FIT schedule differs')
    require(prefit['source_hashes']==receipt['source_hashes'] and prefit['before_first_optimizer_step'] is True,'pre-optimizer source receipt differs')
    require(len(prefit['schedule_ids'])==256 and all(len(batch)==4 for batch in prefit['schedule_ids']),'original batch journal incomplete')
    ordered=list(fit_rows.values());groups={d:[i for i,r in enumerate(ordered) if r['domain']==d] for d in ('lexical','arc')}
    rng=np.random.default_rng(seed);independent=[]
    for _ in range(256):
        indices=[]
        for domain in ('lexical','arc'):indices.extend(int(i) for i in rng.choice(groups[domain],size=2,replace=False))
        independent.append([ordered[i]['id'] for i in indices])
    require(independent==prefit['schedule_ids'] and digest(independent)==receipt['schedule_sha256'], 'independent frozen-seed sampler reproduction differs')
    observed=dense=supervised=0;domains=Counter()
    for batch in prefit['schedule_ids']:
        require([fit_rows[case]['domain'] for case in batch]==['lexical','lexical','arc','arc'],'original domain sampling differs')
        for case in batch:
            row=fit_rows[case];tokens=sum(len(row[k]) for k in ('acquisition_bytes','query_bytes','target_bytes'))-1
            require(tokens<=272,'episode exceeds original dense mask geometry');observed+=tokens;dense+=272
            domains[row['domain']]+=tokens;supervised+=tokens if objective=='full' else len(row['target_bytes'])
    require(observed==receipt['actual_tokens']==receipt['observed_input_bytes']==profile['planned_tokens_seen']==contract['budget']['tokens_seen'],'actual observed byte count differs')
    require(dense==278528 and receipt['end_padding_positions']==dense-observed and receipt['supervised_targets']==supervised,'original supervision/padding accounting differs')
    require(dict(domains)==receipt['domain_observed_input_bytes'],'actual domain exposures differ')
    model_path=folder/'MODEL.pt';initial_path=folder/'INITIAL_MODEL.pt'
    require(sha(model_path)==receipt['checkpoint_sha256'] and sha(initial_path)==receipt['initial_checkpoint_sha256'],'original checkpoint bytes differ')
    saved=load_payload(model_path);initial=load_payload(initial_path)
    for field in SHARED:require(saved['metadata'].get(field)==receipt.get(field),'original checkpoint/receipt metadata differs: '+field)
    require((initial['metadata']['family'],initial['metadata']['seed'],initial['metadata']['status'])==(family,seed,'UNTRAINED_FIT_REPLAY_ONLY'),'original initial checkpoint differs')
    require(initial['optimizer']['state']=={},'initial checkpoint already has optimizer updates')
    check_rng(saved['rng']);check_rng(initial['rng'])
    initial_hash=tensor_digest(initial['model']);final_hash=tensor_digest(saved['model'])
    require(initial_hash==profile['weights_before']==receipt['initial_model_sha256']==receipt['actual_initial_model_sha256'],'actual original paired initialization differs')
    require(receipt['profile_initial_state_sha256']==receipt['actual_profile_initial_state_sha256']==profile['initial_state_sha256'],'actual zero-update source archive differs')
    require(final_hash==receipt['final_model_sha256'] and final_hash!=initial_hash,'model weights did not genuinely change')
    model=models.make_model(family,seed);model.load_state_dict(saved['model'],strict=True)
    active=sum(p.numel() for p in model.parameters() if p.requires_grad);deployed=sum(p.numel() for p in model.parameters())
    require(active==deployed==COUNTS[family] and receipt['trainable_parameters']==active and receipt['deployed_parameters']==deployed,'actual instantiated parameter counts differ')
    require(receipt['counts']==model.counts(),'actual source model count/buffer receipt differs')
    changed=[];unchanged=[]
    for name,param in model.named_parameters():
        (changed if not torch.equal(initial['model'][name],saved['model'][name]) else unchanged).append(name)
        require(bool(torch.isfinite(param).all()),'nonfinite fitted weights: '+name)
    require(bool(changed),'no learned parameter tensor changed')
    for name,_ in model.named_buffers():require(torch.equal(initial['model'][name],saved['model'][name]),'deterministic source buffer changed: '+name)
    optimizer=check_optimizer(model,saved['optimizer'],register['training'])
    history=receipt['history']
    scalar_audit=audit_loss_history(receipt,objective)
    require(len(history)==256 and [r['step'] for r in history]==list(range(1,257)), 'original update history incomplete')
    require(all(math.isfinite(r['loss']) and math.isfinite(r['gradient_norm_before_clip']) and r['gradient_norm_before_clip']>=0 for r in history), 'original update loss/gradient diagnostics invalid')
    fb=receipt['forward_backward_operator_step_sha256'];opt=receipt['optimizer_operator_step_sha256']
    require(len(fb)==256 and set(fb)=={profile['operator_sha256']},'not every actual forward/backward matches original operator profile')
    for kind in ('first','warm'):
        actual_sha,_=check_ledger(receipt['forward_backward_operators_'+kind],'forward_backward')
        require(actual_sha==profile['operator_sha256'],'saved original forward/backward example differs')
    first_sha,first_calls=check_ledger(receipt['optimizer_operators_first'],'optimizer')
    warm_sha,warm_calls=check_ledger(receipt['optimizer_operators_warm'],'optimizer')
    require(len(opt)==256 and opt[0]==first_sha and set(opt[1:])=={warm_sha},'original first/warm optimizer operator coverage differs')
    require(receipt['replay']['cold_reload_bitwise_equal'] is True and receipt['initial_replay']['cold_reload_bitwise_equal'] is True,'original FIT cold replay missing')
    gradients=read(folder/'FIRST_UPDATE_GRADIENTS.json')
    require(gradients['all_connected_and_finite'] is True and set(gradients['parameters'])=={n for n,p in model.named_parameters()},'actual original active gradient coverage incomplete')
    for domain in ('lexical','arc'):
        g=receipt['acquisition_gradient'][domain]
        require(g['status']=='PASS' and any(v['connected'] and v['finite'] and v['l2']>0 for v in g['state_components'].values()),'actual query gradient does not reach acquisition state: '+domain)
    require(receipt['gpu_minutes']==0 and receipt['protected_TEST_opened'] is False and receipt['capability_result'] is None,'fit leaked into forbidden evaluation/promotion scope')
    require(receipt['hardware']['cuda_used'] is False and receipt['hardware']['torch_threads']==1,'actual runtime differs from single-thread CPU contract')
    return key,{'family':family,'objective':objective,'seed':seed,'status':'PASS_TENSOR_AND_RECEIPT_AUDIT',
        'checkpoint_sha256':sha(model_path),'initial_checkpoint_sha256':sha(initial_path),'initial_model_sha256':initial_hash,
        'final_model_sha256':final_hash,'actual_parameters':active,'actual_steps':256,'actual_observed_inputs':observed,
        'actual_supervised_targets':supervised,'actual_dense_compute_positions':dense,'changed_parameter_tensors':len(changed),
        'unchanged_parameter_names':unchanged,'optimizer':optimizer,'forward_backward_steps_verified':len(fb),
        'optimizer_steps_verified':len(opt),'optimizer_first_calls':first_calls,'optimizer_warm_calls':warm_calls,
        'objective_control':expected_control(objective),'saved_scalar_loss_audit':scalar_audit,
        'hardware':receipt['hardware'],'elapsed_fit_seconds':receipt['elapsed_seconds'],
        'fit_loss_first':history[0]['loss'],'fit_loss_last':history[-1]['loss'],
        'fit_loss_scope':'Training diagnostic only; fixed future coefficients, lambda1/0 prefix contribution. Coefficients do not equal gradient/AdamW update shares.'},receipt


def load_npz(path):
    with np.load(path,allow_pickle=False) as archive:
        values={k:archive[k].copy() for k in archive.files if k!='metadata'}
        metadata=json.loads(str(archive['metadata'].item()))
    return values,metadata


def audit_evaluation(directory,matrix,register,register_sha,source_commit,data):
    original_seal=read(directory/'ALL_CHECKPOINTS_SEALED.json');prediction_seal=read(directory/'ALL_PREDICTIONS_SEALED.json')
    require(original_seal['models_sealed']==40 and original_seal['before_first_DEV_prediction'] is True,'all40 checkpoint-before-prediction barrier missing')
    require(original_seal['source_commit']==source_commit and prediction_seal['models']==40 and prediction_seal['before_target_score_computation'] is True,'prediction barrier/source differs')
    seal_models={(r['family'],r['objective'],r['seed']):r for r in original_seal['models']}
    replays={(r['family'],r['objective'],r['seed']):r for r in prediction_seal['replays']}
    require(set(seal_models)==set(replays)==EXPECTED,'complete original/replay matrix missing')
    query=directory/'actor-inputs/queries.json';acq=directory/'actor-inputs/acquisition.json';poison=directory/'actor-inputs/future-poison-queries.json'
    require(query.read_bytes()==poison.read_bytes(),'future-only source poison changed prediction observations')
    query_rows=read(query);acq_rows=read(acq)
    expected_artifacts=register['data']['artifact_sha256']
    for name in ('DEV_ACTOR.json','DEV_DONORS.json','DEV_CONTROL_ACTOR.json'):
        require(sha(data/name)==expected_artifacts[name], 'frozen actor/control artifact differs: '+name)
    original_actor=read(data/'DEV_ACTOR.json');donors=read(data/'DEV_DONORS.json');controls=read(data/'DEV_CONTROL_ACTOR.json')
    require(acq_rows==[{k:r[k] for k in ('id','domain','acquisition_bytes')} for r in original_actor], 'actual acquired observations differ from frozen source')
    require(query_rows==[{k:r[k] for k in ('id','domain','query_bytes','generation_limit')} for r in original_actor], 'actual query observations differ from frozen source')
    require(len(query_rows)==len(acq_rows)==72,'complete actual actor cohort missing')
    for q,a in zip(query_rows,acq_rows):
        require(set(q)=={'id','domain','query_bytes','generation_limit'} and set(a)=={'id','domain','acquisition_bytes'},'query/acquisition actor schema includes missing or forbidden fields')
        require(q['id']==a['id'] and q['domain']==a['domain'] and re.fullmatch('097-[a-f0-9]{64}',q['id']),'actual source identities leaked into actor IDs')
        require(q['generation_limit']=={'lexical':4,'arc':40}[q['domain']],'actor generation length depends on gold')
    reports=[]
    for key in sorted(EXPECTED):
        family,objective,seed=key;receipt=matrix[key];folder=directory/f'{family}-{objective}-{seed}'
        original=folder/'original.npz';fresh=folder/'replay.npz';state=folder/'STATE.pt'
        values,metadata=load_npz(original);again,replay_metadata=load_npz(fresh)
        for field in ('log_probs','generated_bytes','lengths','available'):
            require(np.array_equal(values[field],again[field]),'fresh-process numeric/byte replay differs: '+str(key)+' '+field)
        require(values['log_probs'].shape==(5,72,256) and values['generated_bytes'].shape==(5,72,40) and values['available'].shape==(5,72) and values['available'].dtype==bool,'actual output geometry differs')
        require(values['available'][[0,1,3]].all(), 'retained/reset/corrupt coverage missing')
        require(values['available'][2].tolist()==[r['available'] for r in donors] and values['available'][4].tolist()==[r['available'] for r in controls], 'predeclared donor/binding availability differs')
        require(np.isfinite(values['log_probs']).all(),'actual output probabilities nonfinite')
        require(np.max(np.abs(np.logaddexp.reduce(values['log_probs'].astype(float),axis=-1)))<5e-5,'actual first-byte distribution unnormalized')
        for meta,path in ((metadata,original),(replay_metadata,fresh)):
            require(meta['checkpoint_sha256']==receipt['checkpoint_sha256']==seal_models[key]['checkpoint_sha256'],'query used a different original checkpoint')
            require(meta['optimizer_steps']==0 and meta['checkpoint']['optimizer_steps']==256 and meta['checkpoint']['source_commit']==source_commit,'actor inference/fit step provenance differs')
            for field in SHARED:require(meta['checkpoint'].get(field)==receipt.get(field),'actor checkpoint metadata differs: '+field)
            require(meta['query_sha256']==sha(query) and meta['acquisition_sha256']==sha(acq) and meta['state_file_sha256']==sha(state),'actor observation/state binding differs')
            require(meta['donor_sha256']==expected_artifacts['DEV_DONORS.json'], 'actor fixed-donor provenance differs')
            require(meta['case_ids']==[r['id'] for r in query_rows],'actual query identity order differs')
        require(replays[key]['status']=='PASS' and replays[key]['state_immutable'] is True and replays[key]['prediction_sha256']==sha(original) and replays[key]['replay_sha256']==sha(fresh),'saved fresh-process replay seal differs')
        saved=torch.load(state,map_location='cpu',weights_only=True);info=json.loads(saved['metadata'])
        require(info['checkpoint']==metadata['checkpoint'] and info['acquisition_sha256']==sha(acq),'acquired states bound to different inputs/model')
        require(info['control_sha256']==expected_artifacts['DEV_CONTROL_ACTOR.json'], 'past-only binding-acquisition provenance differs')
        require(saved_state_digest(saved['states'])==info['state_digest'],'actual original acquired-state digest differs')
        auxiliary=[s for s in saved['auxiliary_states'] if s is not None]
        require(saved_state_digest(auxiliary)==info['aux_state_digest'],'actual binding-control state digest differs')
        require(len(saved['states'])==len(saved['auxiliary_states'])==72,'actual state cohort incomplete')
        for i,(native,a) in enumerate(zip(saved['states'],acq_rows)):
            require(native['arm']==family and native['owners']==(f'case-{i}',) and native['clock']==len(a['acquisition_bytes']),'actual state owner/legitimate clock differs')
            for tensor in native['tensors'].values():require(tensor.dtype==torch.float32 and tensor.shape[0]==1 and bool(torch.isfinite(tensor).all()),'invalid native acquired tensor')
        reports.append({'family':family,'objective':objective,'seed':seed,'status':'PASS_ORIGINAL_AND_FRESH_QUERY_PROVENANCE',
            'prediction_sha256':sha(original),'fresh_query_sha256':sha(fresh),'state_file_sha256':sha(state),'all_saved_arrays_bitwise_identical':True})
    execution=read(directory/'EXECUTION.json')
    require(execution['status']=='COMPLETED_INTERNAL_RESEARCH' and execution['models_fitted']==40 and execution['optimizer_steps_during_evaluation']==0,'complete no-update evaluation execution missing')
    require(execution['gpu_minutes']==0 and execution['protected_repository_TEST_opened'] is False,'evaluation exceeded authorized research scope')
    return reports


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo',type=Path,default=Path(__file__).resolve().parents[5])
    p.add_argument('--register',type=Path);p.add_argument('--profiles',type=Path)
    p.add_argument('--initials',type=Path)
    p.add_argument('--data',type=Path,required=True)
    p.add_argument('--fits-dir',type=Path);p.add_argument('--evaluation-dir',type=Path)
    p.add_argument('--source-commit',default='50bf7f2818f38c31f7c4d8280c1352c06f33bd4f')
    p.add_argument('--report',type=Path,required=True);args=p.parse_args()
    require(not args.report.exists(),'audit output is create-once')
    repo=args.repo.resolve();register_path=(args.register or repo/STUDY/'REGISTER.json').resolve()
    args.initials=args.initials or repo/STUDY/'profiles'
    report={'schema':'langcontrol098.independent-checkpoint-audit.v1','scope':'Internal independent tensor/receipt/code audit, not external qualification or capability scoring',
        'source_commit':args.source_commit,'auditor_source_sha256':sha(Path(__file__)),'model_forward_calls':0,'optimizer_steps':0,
        'optimizer_parser_scope':{'prospectively_registered_contract':'balanced-empty-optimizer-profiler-v1',
            'permitted_empty_tensor_serialization_markers':sorted(ZERO_WORK_PROFILER),'raw_ledgers_and_hashes_retained':True,
            'all_aten_signatures_shapes_calls_preserved':True},
        'fits':[],'errors':[],'evaluation':None}
    try:
        require(args.source_commit==EXPECTED_SOURCE,'exact original fit source commit required')
        register,contracts,profiles=load_contracts(repo,register_path,args.profiles)
        require(sha(args.data/'FIT.json')==register['data']['FIT.json_sha256'],'independent actual FIT bytes differ')
        require(sha(args.data/'MANIFEST.json')==register['data']['manifest_sha256'],'independent actual source manifest differs')
        fit_rows={r['id']:r for r in read(args.data/'FIT.json')};require(len(fit_rows)==152,'complete frozen FIT cohort required')
        models=import_models(repo);report['initial_snapshots']=audit_initials(models,profiles,args.initials)
        report['initial_snapshot_status']='PASS_20_ORIGINAL_ZERO_UPDATE_TENSOR_ARCHIVES'
        if args.fits_dir is None:
            report['status']='READY_INITIAL_SNAPSHOTS_VERIFIED_NO_FIT_ARTIFACTS_REQUESTED'
        else:
            paths=list(args.fits_dir.rglob('FIT_RECEIPT.json'));report['fit_receipts_found']=len(paths)
            require(len(paths)==40,'all forty original fit receipt/checkpoint pairs required; no partial matrix promoted')
            matrix={}
            for path in sorted(paths):
                key=None
                try:
                    key,result,receipt=audit_one(path,models,register,contracts,profiles,repo,sha(register_path),args.source_commit,fit_rows)
                    require(key not in matrix,'duplicate original trained model identity')
                    matrix[key]=receipt;report['fits'].append(result)
                except Exception as error:report['errors'].append({'path':str(path),'key':key,'error':type(error).__name__+': '+str(error)})
            require(not report['errors'] and set(matrix)==EXPECTED,'one or more original checkpoint audits failed or matrix incomplete')
            legs=list(args.fits_dir.rglob('FIT_LEG_SEAL.json'));require(len(legs)==10,'all ten original objective-leg seals required')
            seen_legs=set()
            for leg_path in legs:
                leg=read(leg_path);identity=leg['seed'],leg['objective']
                require(identity in {(s,o) for s in SEEDS for o in OBJECTIVES} and identity not in seen_legs,'duplicate/unknown original objective leg')
                seen_legs.add(identity)
                require(leg['schema']=='langcontrol098.objective-leg-fit-seal.v1' and leg['source_commit']==args.source_commit and leg['register_sha256']==sha(register_path),'original leg source/registration differs')
                require(leg['DEV_predictions_started'] is False and leg['optimizer_steps_all_models']==1024 and leg['gpu_minutes']==0 and len(leg['models'])==4,'original objective leg incomplete or exceeded research scope')
                found=set()
                for item in leg['models']:
                    key=item['family'],item['objective'],item['seed'];require(key in matrix and key[1:]==(identity[1],identity[0]) and key[0] not in found,'leg seal model identity differs')
                    found.add(key[0]);require(item['checkpoint_sha256']==matrix[key]['checkpoint_sha256'],'leg checkpoint differs')
                    receipt_path=leg_path.parent/item['receipt'];checkpoint_path=leg_path.parent/item['checkpoint']
                    require(sha(receipt_path)==item['receipt_sha256'] and sha(checkpoint_path)==item['checkpoint_sha256'],'actual original leg artifact bytes differ')
                plan=read(leg_path.parent/'FIT_PLAN_SEAL.json')
                require(plan['seed']==identity[0] and plan['objective']==identity[1] and plan['before_first_optimizer_step'] is True and plan['DEV_scoring_permitted'] is False,'original per-leg pre-optimizer/DEV boundary plan missing')
            report['objective_legs_verified']=10
            for family in FAMILIES:
                for seed in SEEDS:
                    a,b=(matrix[(family,o,seed)] for o in OBJECTIVES)
                    require(a['initial_model_sha256']==b['initial_model_sha256'],'full/query actual initial weights differ')
                    require(a['forward_backward_operator_step_sha256']==b['forward_backward_operator_step_sha256'],'paired actual forward/backward work differs')
                    require(a['optimizer_operators_first']==b['optimizer_operators_first'] and a['optimizer_operators_warm']==b['optimizer_operators_warm'],'paired actual original optimizer work differs')
            require(max(COUNTS.values())/min(COUNTS.values())-1<=.01,'parameter gap exceeds matched 1% bound')
            report['status']='PASS_ALL_40_ORIGINAL_CHECKPOINTS'
            if args.evaluation_dir:
                report['evaluation']=audit_evaluation(args.evaluation_dir,matrix,register,sha(register_path),args.source_commit,args.data)
                report['status']='PASS_ALL_40_CHECKPOINTS_AND_FRESH_QUERY_REPLAYS'
    except Exception as error:
        report['status']='FAIL_OR_INCOMPLETE_NOT_PROMOTED';report['errors'].append({'error':type(error).__name__+': '+str(error)})
    args.report.parent.mkdir(parents=True,exist_ok=True)
    with args.report.open('x') as stream:json.dump(report,stream,indent=2,sort_keys=True,allow_nan=False);stream.write('\n')
    print(json.dumps({'status':report['status'],'fits_verified':len(report['fits']),'errors':report['errors'],'report':str(args.report)}))
    return 0 if not report['errors'] else 2

if __name__=='__main__':raise SystemExit(main())
