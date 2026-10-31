"""Fit one preregistered LANGCONTROL098 arm; no DEV answers are read here."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import re
import signal
import subprocess
import sys
import time
PROCESS_START=time.perf_counter()

import numpy as np
import torch

# Use a unique name: generic helpers and historical source aliases are unsafe.
import importlib.util
_PATH=Path(__file__).with_name('resources.py')
_SPEC=importlib.util.spec_from_file_location('langcontrol098_resources',_PATH)
R=importlib.util.module_from_spec(_SPEC);sys.modules[_SPEC.name]=R;_SPEC.loader.exec_module(R)
M=R.load_local('models');O=R.load_local('objective')
sys.path.insert(0,str(R.ROOT/'src'))
from tac_osm.research_objective_budget import OperatorLedger


def fit_rows(path):
    rows=R.read_json(path)
    if not isinstance(rows,list) or len(rows)!=152:raise ValueError('the complete frozen FIT cohort is required')
    seen=set();counts={'lexical':0,'arc':0}
    for row in rows:
        if not isinstance(row,dict) or row.get('partition')!='FIT' or row.get('domain') not in counts:
            raise ValueError('training accepts only registered FIT episodes')
        if not isinstance(row.get('id'),str) or not row['id'] or row['id'] in seen:raise ValueError('duplicate/missing FIT identity')
        seen.add(row['id']);counts[row['domain']]+=1
        for key in ('acquisition_bytes','query_bytes','target_bytes'):R.bytes_list(row.get(key),key)
        if sum(len(row[k]) for k in ('acquisition_bytes','query_bytes','target_bytes'))>272:
            raise ValueError('real episode exceeds registered observed-byte bound')
    if counts!={'lexical':128,'arc':24}:raise ValueError('FIT domain cohort differs')
    return rows


def schedule(rows,seed,steps,batch_size=4):
    """Architecture/objective-independent input order, with two cases per domain."""
    if seed not in R.SEEDS or type(steps) is not int or steps<1 or batch_size!=4:
        raise ValueError('registered seed/steps/batch required')
    groups={domain:[i for i,r in enumerate(rows) if r['domain']==domain] for domain in ('lexical','arc')}
    if any(not indices for indices in groups.values()):raise ValueError('both learning domains required')
    rng=np.random.default_rng(seed)
    planned=[]
    for _ in range(steps):
        batch=[]
        for domain in ('lexical','arc'):
            batch.extend(int(i) for i in rng.choice(groups[domain],size=2,replace=False))
        planned.append(batch)
    return planned


def collate(rows,indices,context=272):
    """Only end padding; no padded byte can precede a scored authentic target."""
    if type(context) is not int or context!=272 or len(indices)!=4:raise ValueError('registered dense geometry required')
    inputs=torch.zeros((4,context),dtype=torch.long);targets=torch.zeros_like(inputs)
    valid=torch.zeros_like(inputs,dtype=torch.bool);query=torch.zeros_like(valid)
    observed={'lexical':0,'arc':0};supervised={'full':0,'query':0};domains=[]
    for bi,index in enumerate(indices):
        row=rows[index]
        if row.get('domain') not in O.DOMAINS:raise ValueError('registered loss-only domain required')
        domains.append(row['domain'])
        raw=row['acquisition_bytes']+row['query_bytes']+row['target_bytes']
        length=len(raw)-1;start=len(row['acquisition_bytes'])+len(row['query_bytes'])-1
        if not 0<=start<length<=context:raise ValueError('future target/padding alignment')
        inputs[bi,:length]=torch.tensor(raw[:-1]);targets[bi,:length]=torch.tensor(raw[1:])
        valid[bi,:length]=True;query[bi,start:length]=True
        observed[row['domain']]+=length;supervised['full']+=length;supervised['query']+=length-start
    return inputs,targets,query,valid,{'observed_by_domain':observed,'supervised_targets':supervised,
        'loss_domains':domains,'loss_domains_are_model_inputs':False,
        'actual_observed_inputs':sum(observed.values()),'padded_compute_positions':4*context,
        'end_padding_positions':int((~valid).sum())}


def gradient_receipt(model):
    rows={}
    for name,p in model.named_parameters():
        g=p.grad
        rows[name]={'connected':g is not None,'finite':bool(torch.isfinite(g).all()) if g is not None else None,
                    'l2':float(torch.linalg.vector_norm(g.detach())) if g is not None else None}
    if any(not r['connected'] or not r['finite'] for r in rows.values()):
        raise ValueError('missing or nonfinite active parameter gradient')
    return {'parameters':rows,'all_connected_and_finite':True,
            'all_nonzero':all(r['l2']>0 for r in rows.values()),
            'zero_gradient_names':[n for n,r in rows.items() if r['l2']==0]}


def acquisition_gradient(model,row):
    """Actual FIT-only graph check; no optimizer update or extra labels."""
    inputs,targets,mask=O.episode_tensors(row['acquisition_bytes'],row['query_bytes'],row['target_bytes'])
    model.zero_grad(set_to_none=True)
    result=O.sequence_loss(model,inputs,targets,mask,objective='query',domains=(row['domain'],),
                           allow_single_domain=True,owners=('gradient-diagnostic',),
                           acquisition_end=len(row['acquisition_bytes']),retain_acquisition_grad=True)
    result.loss.backward();receipt=O.acquisition_gradient_receipt(result)
    if receipt['status']!='PASS':raise ValueError('query loss does not reach an acquisition-state write')
    receipt['scope']='Actual frozen FIT episode gradient connectivity; no optimizer step and no capability claim'
    model.zero_grad(set_to_none=True)
    return receipt


def validate_data(data,register):
    manifest=R.read_json(data/'MANIFEST.json')
    expected=register.get('data',{})
    if R.sha256(data/'MANIFEST.json')!=expected.get('manifest_sha256'):
        raise ValueError('frozen data manifest differs')
    fit_hash=R.sha256(data/'FIT.json')
    if fit_hash!=expected.get('FIT.json_sha256') or fit_hash!=manifest['artifact_sha256']['FIT.json']:
        raise ValueError('FIT bytes differ from pre-fit seal')
    if manifest.get('counts')!={'FIT':{'lexical':128,'arc':24},'DEV':{'lexical':64,'arc':8}}:
        raise ValueError('frozen complete cohort required')
    return manifest,fit_rows(data/'FIT.json')


def parity_preflight(manifest):
    result=subprocess.run([sys.executable,str(R.ROOT/'scripts/validate_research_parity.py'),
        '--manifest',str(manifest),'--phase','preflight'],cwd=R.ROOT,capture_output=True,text=True,check=True)
    return json.loads(result.stdout)


def checkpoint(path,model,optimizer,metadata):
    payload={'schema':'langcontrol098.checkpoint.v1','metadata':metadata,
             'model':R.cpu_tree(model.state_dict()),'optimizer':R.cpu_tree(optimizer.state_dict()),
             'rng':R.capture_rng()}
    with Path(path).open('xb') as handle:torch.save(payload,handle)
    return R.sha256(path)


def load_initial_state(model,profile,directory,family,seed):
    """Load the exact zero-update profile snapshot, never an extra pretrained model."""
    name=profile.get('initial_state_file')
    if (not isinstance(name,str) or len(name)>128 or Path(name).name!=name
            or re.fullmatch(r'[A-Za-z0-9_.-]+\.pt',name) is None):
        raise ValueError('bounded original initialization basename required')
    directory=Path(directory);path=directory/name
    if directory.is_symlink() or path.is_symlink() or not path.resolve().is_relative_to(directory.resolve()):
        raise ValueError('original initialization path escapes its pinned artifact')
    expected=profile.get('initial_state_sha256')
    if not isinstance(expected,str) or re.fullmatch('[0-9a-f]{64}',expected) is None or R.sha256(path)!=expected:
        raise ValueError('original zero-update initialization bytes differ')
    saved=torch.load(path,map_location='cpu',weights_only=True)
    if (not isinstance(saved,dict) or set(saved)!={'schema','family','seed','model','model_sha256','optimizer_steps'}
            or saved['schema']!='langcontrol098.initial-state.v1' or saved['family']!=family
            or type(saved['seed']) is not int or saved['seed']!=seed
            or type(saved['optimizer_steps']) is not int or saved['optimizer_steps']!=0
            or saved['model_sha256']!=profile['weights_before']):
        raise ValueError('source-bound zero-update initialization identity differs')
    model.load_state_dict(saved['model'],strict=True)
    if R.model_digest(model)!=profile['weights_before']:
        raise ValueError('actual loaded initialization differs from the frozen profile weights')
    return expected


def cold_reload(model,path,register,fit_sample):
    payload=torch.load(path,map_location='cpu',weights_only=True)
    metadata=payload['metadata']
    cold=M.make_model(metadata['family'],metadata['seed']).eval();cold.load_state_dict(payload['model'],strict=True)
    R.assert_counts(cold,M.PARAMETERS[metadata['family']])
    model.eval();tokens=torch.tensor([fit_sample['acquisition_bytes']+fit_sample['query_bytes']],dtype=torch.long)
    owners=('cold-fit-replay',)
    with torch.inference_mode():
        a,_=model(tokens,owners=owners);b,_=cold(tokens,owners=owners)
        split=len(fit_sample['acquisition_bytes'])
        early,state=cold(tokens[:,:split],owners=owners)
        late,_=cold(tokens[:,split:],state=state,owners=owners)
        streamed=torch.cat((early,late),dim=1)
    exact=bool(torch.equal(a,b));delta=float((a-streamed).abs().max())
    if not exact or not torch.allclose(a,streamed,atol=3e-5,rtol=1e-4):
        raise ValueError('sealed model cold/streaming FIT replay failed')
    return {'cold_reload_bitwise_equal':exact,'streaming_max_abs_difference':delta,
            'streaming_atol':3e-5,'streaming_rtol':1e-4,'before_any_DEV_prediction':True}


def train(args):
    if os.environ.get('GITHUB_ACTIONS')!='true':
        raise ValueError('registered model fitting is routed to GitHub Actions CPU')
    if args.output.exists():raise FileExistsError('never overwrite an attempted fit')
    register=R.load_register(args.config);training=register['training']
    if register.get('fit_enabled') is not True:
        raise ValueError('fitting disabled until measured source-bound profiles are sealed')
    if args.family not in R.FAMILIES or args.objective not in R.OBJECTIVES or args.seed not in R.SEEDS:
        raise ValueError('unregistered model/seed/objective')
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed)
    source=R.verify_source(args.source_commit,args.config)
    manifest,rows=validate_data(args.data,register)
    family_manifest=R.ROOT/register['manifest_paths'][args.family]
    preflight=parity_preflight(family_manifest)
    model=M.make_model(args.family,args.seed).train();counts=R.assert_counts(model,M.PARAMETERS[args.family])
    family_contract=R.read_json(family_manifest)
    declared=family_contract['models']
    matches=[r for r in declared if r.get('name')==args.objective and r.get('seed')==args.seed]
    if len(matches)!=1 or matches[0]['trainable_parameters']!=counts['trainable'] or matches[0]['deployed_parameters']!=counts['deployed']:
        raise ValueError('live model absent from family-specific pre-fit parity matrix')
    operator_profile=R.read_json(R.ROOT/family_contract['operator_profile_path'])
    frozen_profile=next(p for p in operator_profile['profiles'] if p['seed']==args.seed and p['objective']==args.objective)
    if frozen_profile.get('objective_control')!=O.objective_mapping(args.objective):
        raise ValueError('frozen profile uses a different objective coefficient contract')
    profile_initial_sha=load_initial_state(model,frozen_profile,args.initials,args.family,args.seed)
    args.output.mkdir(parents=True)
    planned=schedule(rows,args.seed,training['steps'],training['batch_size'])
    planned_ids=[[rows[i]['id'] for i in batch] for batch in planned]
    sample_hash=R.json_digest(planned_ids);initial=R.model_digest(model)
    if initial!=frozen_profile['weights_before']:
        raise ValueError('live initial model differs from the frozen actual profile weights')
    if matches[0].get('training_sample_order_sha256')!=sample_hash:
        raise ValueError('actual sample schedule differs from frozen family registration')
    R.write_json(args.output/'PREFIT.json',{'source_commit':args.source_commit,'source_hashes':source,
        'objective_control':O.objective_mapping(args.objective),
        'register_sha256':R.sha256(args.config),'family_manifest_sha256':R.sha256(family_manifest),
        'data_manifest_sha256':R.sha256(args.data/'MANIFEST.json'),'fit_sha256':R.sha256(args.data/'FIT.json'),
        'counts':counts,'schedule_sha256':sample_hash,'schedule_ids':planned_ids,'initial_model_sha256':initial,
        'profile_initial_state_sha256':profile_initial_sha,
        'hardware':R.runtime_hardware(),'preflight':preflight,'before_first_optimizer_step':True})
    optimizer=torch.optim.AdamW(model.parameters(),lr=training['learning_rate'],weight_decay=training['weight_decay'],
                                betas=tuple(training['betas']),eps=training['eps'],foreach=False,fused=False)
    deadline=R.Deadline(training['max_arm_seconds_by_family'][args.family]);actual_steps=0;history=[];seen={d:0 for d in ('lexical','arc')}
    observed=padding=supervised=0;optimizer_seconds=0.;gradient=None;acquisition=None
    forward_steps=[];optimizer_steps=[];operator_examples={};clipped_steps=0
    try:
        initial_meta={'family':args.family,'seed':args.seed,'status':'UNTRAINED_FIT_REPLAY_ONLY'}
        initial_path=args.output/'INITIAL_MODEL.pt'
        initial_checkpoint_sha=checkpoint(initial_path,model,optimizer,initial_meta)
        initial_replay=cold_reload(model,initial_path,register,rows[planned[0][0]])
        acquisition={domain:acquisition_gradient(model,next(r for r in rows if r['domain']==domain))
                     for domain in ('lexical','arc')}
        R.write_json(args.output/'ACQUISITION_GRADIENT.json',acquisition)
        model.train()
        for step,indices in enumerate(planned):
            deadline.check();optimizer.zero_grad(set_to_none=True)
            inputs,targets,mask,valid,exposure=collate(rows,indices,training['context'])
            with OperatorLedger('forward_backward') as forward_ledger:
                result=O.sequence_loss(model,inputs,targets,mask,objective=args.objective,valid_mask=valid,
                                       domains=tuple(exposure['loss_domains']),
                                       owners=tuple(f'train-{i}' for i in range(4)))
                result.loss.backward()
            forward_steps.append(forward_ledger.sha256())
            if forward_steps[-1]!=frozen_profile['operator_sha256']:
                raise ValueError('actual forward/backward operators differ from the frozen no-fit profile')
            current=gradient_receipt(model)
            if gradient is None:gradient=current;R.write_json(args.output/'FIRST_UPDATE_GRADIENTS.json',gradient)
            tic=time.perf_counter()
            with OperatorLedger('optimizer') as optimizer_ledger:
                norm=torch.nn.utils.clip_grad_norm_(model.parameters(),training['grad_clip'],foreach=False)
                if not bool(torch.isfinite(norm)):raise ValueError('nonfinite gradient norm')
                optimizer.step()
            optimizer_seconds+=time.perf_counter()-tic
            optimizer_steps.append(optimizer_ledger.sha256())
            if step<2:
                operator_examples['forward_backward_operators_'+('first' if step==0 else 'warm')]=forward_ledger.rows()
                operator_examples['optimizer_operators_'+('first' if step==0 else 'warm')]=optimizer_ledger.rows()
            if step>1 and optimizer_steps[-1]!=optimizer_steps[1]:
                raise ValueError('actual warm optimizer operator trace changed')
            actual_steps=step+1;observed+=exposure['actual_observed_inputs'];padding+=exposure['end_padding_positions']
            supervised+=result.supervised_targets
            for domain,n in exposure['observed_by_domain'].items():seen[domain]+=n
            was_clipped=float(norm)>training['grad_clip'];clipped_steps+=int(was_clipped)
            loss=float(result.loss.detach())
            history.append({'step':actual_steps,'loss':loss,
                'future_loss':float(result.future_loss.detach()),'prefix_loss':float(result.prefix_loss.detach()),
                'prefix_lambda':result.prefix_lambda,'gradient_norm_before_clip':float(norm),
                'gradient_clipped':was_clipped,
                'per_domain_losses':{domain:{name:float(value.detach()) for name,value in parts.items()}
                                     for domain,parts in result.per_domain_losses.items()}})
            deadline.check()
            if actual_steps%32==0:print(json.dumps({'family':args.family,'objective':args.objective,'seed':args.seed,'optimizer_step':actual_steps,'loss':loss}),flush=True)
        final_counts=R.assert_counts(model,M.PARAMETERS[args.family]);final=R.model_digest(model)
        if final==initial:raise ValueError('optimizer executed but original weights did not change')
        metadata={'study_id':'LANGCONTROL098','family':args.family,'objective':args.objective,'seed':args.seed,
            'objective_control':O.objective_mapping(args.objective),
            'optimizer_steps':actual_steps,'counts':final_counts,'source_commit':args.source_commit,'source_hashes':source,
            'register_sha256':R.sha256(args.config),'data_manifest_sha256':R.sha256(args.data/'MANIFEST.json'),
            'fit_sha256':R.sha256(args.data/'FIT.json'),'schedule_sha256':sample_hash,'observed_input_bytes':observed,
            'supervised_targets':supervised,'padded_compute_positions':training['steps']*4*training['context'],
            'end_padding_positions':padding,'initial_model_sha256':initial,'final_model_sha256':final,
            'profile_initial_state_sha256':profile_initial_sha,
            'status':'SEALED_BEFORE_DEV','selection':'fixed_final_step_no_DEV_selection'}
        path=args.output/'MODEL.pt';digest=checkpoint(path,model,optimizer,metadata)
        replay=cold_reload(model,path,register,rows[planned[0][0]])
        receipt={**metadata,'checkpoint_sha256':digest,'checkpoint_path':str(path),'trainable_parameters':final_counts['trainable'],
            'deployed_parameters':final_counts['deployed'],'model_bytes':sum(t.numel()*t.element_size() for t in model.state_dict().values()),
            'actual_steps':actual_steps,'actual_examples':actual_steps*4,'actual_tokens':observed,
            'domain_observed_input_bytes':seen,'elapsed_seconds':time.perf_counter()-deadline.started,
            'optimizer_seconds':optimizer_seconds,'peak_rss_bytes':R.peak_rss_bytes(),'history':history,'replay':replay,
            'gradient_clipped_steps':clipped_steps,'gradient_clip_frequency':clipped_steps/actual_steps,
            'component_loss_scope':'Fixed domain/case segment means before lambda; coefficients are not gradient or AdamW update shares.',
            'actual_sample_order_sha256':sample_hash,'actual_dense_compute_tokens':training['steps']*4*training['context'],
            'actual_initial_model_sha256':initial,
            'actual_profile_initial_state_sha256':profile_initial_sha,
            'forward_backward_operator_step_sha256':forward_steps,'optimizer_operator_step_sha256':optimizer_steps,
            **operator_examples,'setup_seconds':deadline.started-PROCESS_START,
            'elapsed_process_seconds':time.perf_counter()-PROCESS_START,
            'hardware':R.runtime_hardware(),
            'initial_checkpoint_sha256':initial_checkpoint_sha,'initial_replay':initial_replay,
            'acquisition_gradient':acquisition,
            'gpu_minutes':0,'protected_TEST_opened':False,'capability_result':None,
            'compute_profile':{'operator_profile_sha256':family_contract['operator_profile_sha256'],
                               'forward_backward_operator_sha256':frozen_profile['operator_sha256'],
                               'absolute_training_flops':'INCONCLUSIVE_NOT_CLAIMED'},
            'compute_scope':'Dense forward/backward positions include masked end padding; actual byte and target counts are separate.'}
        R.write_json(args.output/'FIT_RECEIPT.json',receipt)
        return receipt
    except BaseException as error:
        metadata={'study_id':'LANGCONTROL098','family':args.family,'objective':args.objective,'seed':args.seed,
                  'objective_control':O.objective_mapping(args.objective),
                  'optimizer_steps':actual_steps,'status':'FAILED_PARTIAL_NOT_FOR_EVALUATION','source_commit':args.source_commit,
                  'source_hashes':source,'schedule_sha256':sample_hash,'reason':type(error).__name__+': '+str(error)}
        try:
            digest=checkpoint(args.output/'PARTIAL_MODEL.pt',model,optimizer,metadata)
            R.write_json(args.output/'FAILED_FIT.json',{**metadata,'checkpoint_sha256':digest,'history':history,
                'elapsed_seconds':time.perf_counter()-deadline.started,'observed_input_bytes':observed,'supervised_targets':supervised,
                'gradient_clipped_steps':clipped_steps,
                'gradient_clip_frequency':clipped_steps/actual_steps if actual_steps else None,
                'forward_backward_operator_step_sha256':forward_steps,'optimizer_operator_step_sha256':optimizer_steps,
                **operator_examples})
        except BaseException as archive_error:
            R.write_json(args.output/'FAILED_ARCHIVE.json',{**metadata,'archive_error':str(archive_error)})
        raise


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--family',choices=R.FAMILIES,required=True);p.add_argument('--objective',choices=R.OBJECTIVES,required=True)
    p.add_argument('--seed',type=int,choices=R.SEEDS,required=True);p.add_argument('--data',type=Path,required=True)
    p.add_argument('--initials',type=Path,required=True)
    p.add_argument('--config',type=Path,default=R.HERE/'REGISTER.json');p.add_argument('--output',type=Path,required=True)
    p.add_argument('--source-commit',required=True)
    train(p.parse_args())

if __name__=='__main__':main()
