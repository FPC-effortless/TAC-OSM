"""Source-authenticated, zero-update LANGMATH101 schedule/profile preflight.

No optimizer, DEV loader, target scorer or model fitting is invoked. The same
4096-item FIT schedule is frozen for each seed before profiling. Conventional
BPE fitting uses only the actual valid observations in its first512 items;
its preprocessing exposures/time are reported separately from neural training.
"""
from __future__ import annotations
import argparse
import importlib.util
import json
import math
from pathlib import Path
import sys
import time

HERE=Path(__file__).resolve().parent
_sp=importlib.util.spec_from_file_location('langmath101_common',HERE/'common.py')
C=importlib.util.module_from_spec(_sp);sys.modules[_sp.name]=C;_sp.loader.exec_module(C)


def relative(path):
    p=Path(path).resolve()
    if not p.is_relative_to(C.ROOT.resolve()):
        raise ValueError('Profiles/manifests must live under the actual repository root')
    return p.relative_to(C.ROOT.resolve()).as_posix()


def observed_counts(bundle,plan):
    """Exact source lengths without repeatedly constructing dense Torch batches."""
    result=[]
    for item in plan:
        if item['mode']=='language':
            if len(item['indices'])!=4:raise ValueError('Original B4 source schedule required')
            result.append(4*128)
        elif item['mode']=='episodic':
            rows=[bundle.fit_episodes[i] for i in item['indices']]
            result.append(sum(len(e['acquisition_bytes'])+len(e['query_bytes'])+len(e['target_bytes'])-1 for e in rows))
        else:raise ValueError('Unknown genuine FIT source schedule mode')
    return result


def exposure(prefix_counts,steps):
    if not 1<=steps<=len(prefix_counts):raise ValueError('Frozen schedule prefix outside4000-item source cohort')
    return {'steps':steps,'examples_seen':4*steps,'tokens_seen':sum(prefix_counts[:steps]),
            'dense_compute_tokens':4*272*steps}


def fit_tokenizer(bundle,plan,directory):
    """No future-only target byte or PAD is admitted to tokenizer fitting."""
    chunks=[]
    for item in plan[:512]:
        batch=C.padded_batch(bundle,item)
        for row,valid in zip(batch['tokens'],batch['valid']):
            raw=bytes(row[valid].tolist())
            if not raw:raise ValueError('Actual FIT observation chunk is empty')
            chunks.append(raw)
    if len(chunks)!=2048:raise ValueError('Exactly512 B4 preprocessing source items required')
    started=time.perf_counter();cpu=time.process_time()
    tokenizer=C.adaptive().FrozenBPE.fit(chunks,partition='FIT')
    receipt=tokenizer.receipt()
    C.save(directory/'TOKENIZER.json',receipt)
    C.save(directory/'TOKENIZER_FIT_COST.json',{
        'partition':'FIT','schedule_items':512,'examples':len(chunks),
        'observed_source_bytes':sum(map(len,chunks)),
        'source_chunks_sha256':C.digest(receipt['source_sha256']),
        'source_schedule_prefix_sha256':C.digest(plan[:512]),
        'merge_passes':32,'optimizer_steps':0,'elapsed_seconds':time.perf_counter()-started,
        'cpu_seconds':time.process_time()-cpu,'DEV_access':False,'protected_TEST_access':False,
        'convention':'Classical FIT-only preprocessing counted separately from neural training; each chunk is an actual observed valid byte prefix without PAD or the last unseen next-byte target.'})
    return receipt


def source_receipt(bundle):
    rows=bundle.fit_sources
    return {'name':'Authenticated original LANGTRAIN099 licensed code/documentation FIT cohort',
            'sha256':bundle.data_identity['source_manifest_sha256'],'observed_real_outcomes':True,
            'publisher_url':'https://github.com/'+rows[0]['publisher'],
            'publishers':sorted({r['publisher'] for r in rows}),
            'license':', '.join(sorted({r['license'] for r in rows})),
            'license_evidence_url':'https://github.com/python/cpython/blob/60403a5409ff2c3f3b07dd2ca91a7a3e096839c7/LICENSE',
            'source_receipt':'Original099 FIT_MANIFEST.json SHA256 '+bundle.data_identity['FIT_manifest_sha256']+
                '; data.load_fit authenticates every original publisher file, license, byte length/hash and lexical target/observation source join.',
            'actual_source_files':len(rows),'actual_source_bytes':sum(len(b) for b in bundle.fit_corpus),
            'fit_data_identity':bundle.data_identity}


def shared_contracts(D):
    return {
      'train_split_sha256':D.FIT_MANIFEST_SHA,
      'dev_split_sha256':D.EVAL_MANIFEST_SHA,
      'feature_contract_sha256':C.digest({'source':'actual original FIT observed byte IDs0..255',
          'input_geometry':[4,272],'source_language_context':128,
          'availability':'causal only; no targets, future state, oracle or judge features',
          'mechanism_scope':'same observed source stream; stateless current/window and patch candidates deliberately differ in retained information; no full information-equivalence/superiority claim'}),
      'target_contract_sha256':C.digest({'target':'actual source next byte','vocab':256,
          'loss':'equal case mean future CE plus0.5 mean valid prefix CE; language valid positions only',
          'teacher_forcing':'earlier answer bytes only; final next-byte target never enters source x',
          'padding':'causal trailing PAD never scored; explicitly counted dense geometry'}),
      'optimizer_schedule_sha256':C.digest(C.RECIPE),
      'tuning_policy_sha256':C.digest({'trials':0,'source':'fixed recipe','DEV_selection':False}),
      'selection_policy_sha256':C.digest({'parameter_lane':'final512th update',
          'counted_lane':'first actual cumulative counted-dense target crossing',
          'checkpoint_selection':'no best checkpoint, no loss-based stopping or reordering'}),
      'evaluation_contract_sha256':C.sha(HERE/'PROTOCOL.json')}


def build_manifest(profile,profile_path,specs,plans,counts,source,contracts):
    G=C.local('owner_two_lane')
    maximum=max(r['counted_dense_units'] for r in profile['models'])
    minimum=min(r['counted_dense_units'] for r in profile['models'])
    # Shared target is set from original source profiles before any fitting.
    # A0/other dense arms count all actual forward+back mm/addmm/bmm; adaptive
    # route-dependent runtime consumption still controls the actual stop point.
    target=512*maximum;bound=int(target*.005)
    if maximum>bound:raise ValueError('Profile charge exceeds the frozen0.5% first-crossing headroom')
    rows=[];estimates=[]
    for pr in profile['models']:
        arm,seed=pr['architecture'],pr['seed'];charge=pr['counted_dense_units']
        estimate=math.ceil(target/charge)
        if not 200<=estimate<=4096:
            raise ValueError(f'Counted source-work estimate exceeds frozen200..4096 bounds: {arm} seed{seed}: {estimate}')
        expected_wall=pr['profile_wall_seconds']*max(512,estimate)*1.25+30
        if expected_wall>C.RECIPE['max_arm_seconds']:
            raise ValueError(f'Actual source profile predicts over-budget training for {arm} seed{seed}: {expected_wall:.2f}s; retain receipt and redesign before fitting')
        estimates.append({'architecture':arm,'seed':seed,'counted_step_estimate':estimate,
                          'profile_counted_units_per_step':charge,'conservative_wall_seconds':expected_wall})
        for lane in G.LANES:
            steps=512 if lane==G.LANES[0] else estimate
            b=exposure(counts[seed],steps)
            b.update(batch_size=4,history_events=272,tuning_trials=0,
                     common_schedule_sha256=C.digest(plans[seed]))
            if lane==G.LANES[0]:
                b.update(planned_counted_units=512*charge,max_counted_units=512*bound)
            else:
                b.update(planned_counted_units=target,max_counted_units=target+bound,
                  dynamic_stop_rule={'kind':'first_step_reaching_counted_target',
                    'target_counted_units':target,'max_single_step_counted_units':bound,
                    'min_steps':200,'max_steps':4096})
            row={'lane':lane,'architecture':arm,'seed':seed,
                 'trainable_parameters':pr['trainable_parameters'],'deployed_parameters':pr['deployed_parameters'],
                 'training_sample_order_sha256':C.digest(plans[seed][:steps]),'budget':b}
            row.update(contracts);row['dataset_sha256']=source['sha256'];rows.append(row)
    manifest={'schema_version':'tac-osm.parity.v1','study_id':G.STUDY,
      'study_type':'owner_two_lane_program','owner_contract':G.CONTRACT,
      'owner_instruction_quote':G.QUOTE,'execution_level':'internal_research',
      'registered_before_first_fit':True,'no_architecture_superiority_claim':True,
      'protected_TEST_access':False,'independently_qualified':False,'checkpoint_promotion':False,
      'real_world_source':source,'arithmetic_convention':'actual_ATen_mm_addmm_bmm_units',
      'full_model_FLOPs_status':'INCONCLUSIVE_NOT_CLAIMED',
      'comparison_scope':'Owner-requested split-lane internal pilot. Parameter lane matches capacity and source exposure but not actual computation; counted lane matches declared partial dense arithmetic, not complete FLOPs, source exposure, equivalent available history or architectural superiority.',
      'source_schedules':specs,'training_recipe_sha256':profile['training_recipe_sha256'],
      'operator_profile_path':relative(profile_path),'operator_profile_sha256':C.sha(profile_path),
      'arms':rows,'counted_target_policy':{'common_target':target,'maximum_profile_step':maximum,
        'minimum_profile_step':minimum,'max_single_step_bound':bound,
        'definition':'512 times maximum sealed zero-update actual source forward/back dense count across every architecture and seed; freeze before fitting.'}}
    protocol=json.loads((HERE/'PROTOCOL.json').read_text())
    for field in ('research_scope','split_policy','causal_protocol','checkpoint_policy'):
        if field not in protocol:raise ValueError('Frozen PROTOCOL is missing outer-gate field '+field)
        manifest[field]=protocol[field]
    return manifest,estimates


def prepare(a):
    import torch
    if not (HERE/'PROTOCOL.json').is_file():raise FileNotFoundError('Freeze PROTOCOL.json before actual prefit profiles')
    output=a.output.resolve();relative(output)
    if output.exists():raise FileExistsError('Keep original source/profile evidence; choose a new output directory')
    if not a.profile_only and a.manifest.exists():raise FileExistsError('Never overwrite a frozen prefit manifest')
    output.mkdir(parents=True)
    D=C.local('data');M=C.local('meter');G=C.local('owner_two_lane')
    bundle=D.load_fit(a.fit)
    before_source=C.source_hashes()
    seeds=[a.only_seed] if a.only_seed else list(C.SEEDS)
    names=[a.only_arm] if a.only_arm else C.arm_names()
    if len(set(names))!=len(names) or 'factorized098' in names:raise ValueError('Canonical unique architecture list required')
    if not a.profile_only and (seeds!=list(C.SEEDS) or names!=C.arm_names()):
        raise ValueError('Full preflight must contain all registered architectures and seeds')
    profile={'schema':'langmath101.owner-two-lane-profile.v1','optimizer_steps':0,
      'source_sha256':bundle.data_identity['source_manifest_sha256'],
      'training_recipe':C.RECIPE,'training_recipe_sha256':C.digest(C.RECIPE),
      'source_hashes':before_source,'implementation_sha256':C.digest(before_source),
      'tokenizer_receipts_sha256':{},'models':[],'scope':'Original authenticated FIT forward/backward and tokenizer preprocessing only; zero optimizer steps, no DEV load/scoring'}
    plans={};specs={};counts={};source=source_receipt(bundle)
    for seed in seeds:
        directory=output/str(seed);directory.mkdir()
        plan=D.schedule(bundle,seed,4096);C.save(directory/'SCHEDULE.json',plan)
        plans[seed]=plan;specs[str(seed)]={'path':relative(directory/'SCHEDULE.json'),'sha256':C.sha(directory/'SCHEDULE.json')}
        counts[seed]=observed_counts(bundle,plan)
        tokenizer=fit_tokenizer(bundle,plan,directory)
        profile['tokenizer_receipts_sha256'][str(seed)]=C.sha(directory/'TOKENIZER.json')
        v=C.padded_batch(bundle,plan[3])
        assert tuple(v['tokens'].shape)==(4,272) and plan[3]['mode']=='episodic'
        for arm in names:
            C.setup(seed);model=C.make_model(arm,tokenizer)
            binding=M.verify_common_bindings(model,C.state_digest)
            initial=C.state_digest(model);name=f'INIT_{arm}_{seed}.pt'
            checkpoint_sha=C.checkpoint(directory/name,model,None,
              {'arm':arm,'seed':seed,'optimizer_steps':0,'stage':'INITIAL_BEFORE_DEV',
               'model_sha256':initial,'tokenizer_receipt':tokenizer,'source_hashes':before_source,
               'implementation_sha256':profile['implementation_sha256'],'recipe':C.RECIPE})
            started=time.perf_counter();cpu=time.process_time()
            pr=M.profile_without_update(model,lambda m:C.loss(m,v),architecture=arm,seed=seed,input_shape=[4,272])
            pr.update(profile_wall_seconds=time.perf_counter()-started,
                      profile_cpu_seconds=time.process_time()-cpu,
                      source_schedule_index=3,source_schedule_item=plan[3],
                      source_input_sha256=C.digest(v['tokens'].tolist()),
                      source_targets_sha256=C.digest(v['targets'].tolist()),
                      observed_source_tokens=v['counts']['observed'],
                      dense_compute_tokens=4*272,
                      initial_checkpoint_path=relative(directory/name),initial_checkpoint_sha256=checkpoint_sha,
                      source_digest_binding=binding,hardware=C.hardware())
            assert pr['weights_before']==initial==pr['weights_after']
            if pr['nonzero_gradient_tensor_count']!=pr['parameter_tensor_count']:
                raise ValueError('At least one actual parameter tensor has zero real-source profile gradient: '+arm)
            C.save(directory/f'PROFILE_{arm}.json',pr);profile['models'].append(pr)
            del model
            print(json.dumps({'stage':'PROFILE_NO_UPDATE','arm':arm,'seed':seed,
                              'counted_units':pr['counted_dense_units'],'seconds':pr['profile_wall_seconds']}),flush=True)
    if C.source_hashes()!=before_source:raise ValueError('Executable source changed during prefit preparation')
    pp=output/'PROFILE_INDEX.json';C.save(pp,profile)
    C.save(output/'SOURCE_RECEIPT.json',source)
    if a.profile_only:
        C.save(output/'PROFILE_ONLY.json',{'status':'PARTIAL_NO_FIT_PROFILE_ONLY','preflight_eligible':False,
          'architectures':names,'seeds':seeds,'optimizer_steps':0})
        print(json.dumps({'status':'PARTIAL_NO_FIT_PROFILE_ONLY','profile':str(pp)}));return
    manifest,estimates=build_manifest(profile,pp,specs,plans,counts,source,shared_contracts(D))
    C.save(output/'RESOURCE_ESTIMATES.json',estimates)
    result=G.validate_program(manifest,'preflight',root=C.ROOT)
    C.save(a.manifest,manifest);C.save(output/'PREFLIGHT.json',result)
    print(json.dumps({'status':result['status'],'manifest':str(a.manifest),'model_profiles':len(profile['models']),
      'fit_rows':len(manifest['arms']),'counted_target':manifest['counted_target_policy']['common_target']}))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--fit',type=Path,required=True)
    p.add_argument('--output',type=Path,default=HERE/'profiles')
    p.add_argument('--manifest',type=Path,default=HERE/'FAIR_COMPARISON.json')
    p.add_argument('--profile-only',action='store_true',help='Explicit partial technical profile; never a valid program preflight')
    p.add_argument('--only-arm',choices=C.arm_names());p.add_argument('--only-seed',type=int,choices=C.SEEDS)
    prepare(p.parse_args())
