#!/usr/bin/env python3
"""Original-source reload/observability audit, no fits or performance scoring."""
from __future__ import annotations
import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path
import sys
import tempfile

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'src')]
from tac_osm.history_source_controls import read_fit_members,make_control,declared_control_reader
from tac_osm.history_source_probe import HistorySourceWindow,adapter
from tac_osm.research_operator_budget import canonical,digest
from scripts.original_history_trace import verify_implementation,load_original_modules,sha_file,tensor_sha,verify_causal_features
from scripts.validate_research_parity import check_manifest

DIR=ROOT/'research/mathematical-engineering/history_source017'


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n')


def reload_source(source,original,path,expected,parent):
    captured={};native=source.source_frames
    def capture(*args,**kwargs):
        data=native(*args,**kwargs)
        captured.update({k:tensor_sha(data[k]) for k in ('center','spread')})
        return data
    with declared_control_reader(source,path,expected,original_sha=parent):
        source.source_frames=capture
        try:data=original.source_observations(path)
        finally:source.source_frames=native
    return data,captured


def window(d,rows,source):
    import torch
    owners=torch.tensor([s for s,_ in rows],dtype=torch.long)
    anchors=torch.tensor([t for _,t in rows],dtype=torch.long)
    hours=anchors[:,None]+torch.arange(-23,65)[None,:]
    clocks=anchors[:,None]+torch.arange(0,64,8)[None,:]
    return HistorySourceWindow(*(d[k][owners[:,None],hours] for k in ('values','valid','age','last')),
        owners,hours,source.calendar(clocks.flatten().tolist()).reshape(len(rows),8,6),
        source.calendar((anchors+70).tolist())[:,None,:])


def probe(model,s):
    import torch
    from torch.nn import functional as F
    from tac_osm.latent_fusion_004_factorial import head_weights
    a=adapter();a.validate(model,s)
    projection_names=('embed','key_map','val_map','retention_map','write_map','write_router')
    captured={};handles=[]
    for name in (*projection_names,'historic_clock_query','belief_query','read_router','historic_event_head'):
        def hook(module,args,output,name=name):captured.setdefault(name,[]).append((args[0].detach().clone(),output.detach().clone()))
        handles.append(getattr(model,name).register_forward_hook(hook))
    state=a.start(model,s);timeline=[];keys=[];writers=[];update_error=0.
    try:
        for i in range(88):
            captured.clear();old=state.associative.detach().clone()
            state=a.advance(model,state,s,i)
            if any(len(captured.get(n,[]))!=1 for n in projection_names):raise ValueError('ambiguous original writer calls')
            x={n:captured[n][0][1] for n in projection_names}
            key=F.normalize(x['key_map'].reshape(len(s.owners),2,8),dim=-1,eps=1e-8)
            route,eligible=head_weights(x['write_router'],model.log_head_temperature,head_count=2,soft=True)
            eligible=eligible&s.valid[:,i].any(-1)[:,None]
            alpha=torch.where(eligible,torch.sigmoid(x['retention_map']).expand(-1,2),torch.ones_like(route))
            beta=torch.sigmoid(x['write_map'])*route*eligible
            retained=alpha[:,:,None,None]*old.reshape(-1,16,2,8).permute(0,2,1,3)
            directional=beta[:,:,None,None]*(torch.tanh(x['val_map'])[:,None,:,None]-retained@key.unsqueeze(-1))*key[:,:,None,:]
            rebuilt=(retained+directional).permute(0,2,1,3).contiguous().reshape(-1,16,16)
            error=float((rebuilt-state.associative).abs().max());update_error=max(update_error,error)
            if error>2e-6:raise ValueError('actual source writer-to-state reconstruction failed')
            keys.append(key);writers.append(directional.flatten(1).ne(0).any(-1))
            timeline.append(dict(hours=s.hours[:,i].tolist(),
                input={k:tensor_sha(getattr(s,k)[:,i]) for k in ('values','valid','age','last')},
                modules={k:tensor_sha(v) for k,v in x.items()},
                states={k:tensor_sha(getattr(state,k)) for k in state._fields}))
        captured.clear()
        outputs=a.forecast(model,state,s)
        # Native historical calls precede the forecast calls in each hook list.
        q=F.normalize((torch.tanh(captured['historic_clock_query'][0][1])+
            captured['belief_query'][0][1][:,None]).reshape(len(s.owners),8,2,8),dim=-1,eps=1e-8)
        route,_=head_weights(captured['read_router'][0][1],model.log_head_temperature,head_count=2,soft=True)
        matrix=state.associative
        if model.__class__.__module__.endswith('latent_fusion_008_nonassoc_control'):
            matrix=state.fast[:,:16].unsqueeze(-1)*state.slow[:,:16].unsqueeze(1)
        read=torch.einsum('bhvk,bnhk->bnhv',matrix.reshape(-1,16,2,8).permute(0,2,1,3),q)
        read=(read*route[:,None,:,None]).sum(2)
        actual_head_input=captured['historic_event_head'][0][0]
        read_error=float((read-actual_head_input).abs().max())
        if read_error>2e-6:raise ValueError('native query-to-read correspondence failed')
        key_history=torch.stack(keys,1);actual_writes=torch.stack(writers,1)
        geometry=[]
        for b,owner in enumerate(s.owners.tolist()):
            queries=[]
            for j,offset in enumerate(range(0,64,8)):
                idx=23+offset
                cosine=(key_history[b]*q[b,j][None,:,:]).sum(-1).mean(-1)
                eligible=actual_writes[b]
                present=bool(eligible[idx])
                queries.append(dict(target_hour=int(s.hours[b,idx]),actual_target_write=present,
                    target_cosine=float(cosine[idx]) if present else None,
                    strictly_higher_cosine_legal_writes=int(((cosine>cosine[idx]+1e-6)&eligible).sum()) if present else None,
                    tied_legal_writes=int(((cosine-cosine[idx]).abs()<=1e-6).logical_and(eligible).sum()) if present else None,
                    legal_actual_writes=int(eligible.sum()),status='MEASURED_GEOMETRY_ONLY' if present else 'INCONCLUSIVE_NO_ACTUAL_TARGET_WRITE'))
            pair_cos=torch.einsum('nhd,mhd->nmh',q[b],q[b]).mean(-1)
            offdiag=~torch.eye(8,dtype=torch.bool)
            geometry.append(dict(owner=owner,anchor_hour=int(s.hours[b,23]),query_hour=int(s.hours[b,-1]),queries=queries,
                mean_different_clock_query_key_cosine=float(pair_cos[offdiag].mean())))
        result=dict(input_sha256=digest({k:tensor_sha(getattr(s,k)) for k in s.__dataclass_fields__}),
            timeline=timeline,outputs={k:dict(sha256=tensor_sha(v),values=v.tolist()) for k,v in outputs.items()},
            final_state={k:tensor_sha(getattr(state,k)) for k in state._fields},
            update_reconstruction_max_abs_error=update_error,read_reconstruction_max_abs_error=read_error,geometry=geometry)
        return result
    finally:
        for h in handles:h.remove()


def first_divergence(base,changed):
    if base['input_sha256']!=changed['input_sha256']:return 'actual_encoder_input_window'
    for i,(x,y) in enumerate(zip(base['timeline'],changed['timeline'])):
        if x!=y:
            for key in ('input','modules','states'):
                if x[key]!=y[key]:return f'step/{i}/{key}'
    if base['outputs']!=changed['outputs']:return 'historical_or_forecast_output'
    if base['final_state']!=changed['final_state']:return 'final_state_including_owner_flags'
    return None


def execute(args):
    import torch
    spec=json.loads((DIR/'FAIR_COMPARISON.json').read_text());check_manifest(spec,'preflight')
    root=verify_implementation(args.implementation_root,spec['implementation_commit'])
    if sha_file(args.source)!=spec['real_world_source']['sha256']:raise ValueError('original publisher archive differs')
    plan_path=ROOT/spec['frozen_plan_path']
    if sha_file(plan_path)!=spec['frozen_plan_sha256']:raise ValueError('frozen source identities changed')
    plan=json.loads(plan_path.read_text());rows=plan['primary']+plan['secondary']
    if digest(rows)!=spec['split_policy']['cohort_sha256']:raise ValueError('frozen source cohort differs')
    source,original=load_original_modules(root)
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    members=read_fit_members(args.source,spec['real_world_source']['sha256'])
    parent=source.SOURCE_SHA
    baseline,base_intermediate=reload_source(source,original,args.source,parent,parent)
    verify_causal_features(baseline,source,max(t+64 for _,t in rows))
    models={};before={}
    for r in spec['models']:
        path=ROOT/r['path']
        if sha_file(path)!=r['sha256']:raise ValueError('frozen original checkpoint differs')
        checkpoint=torch.load(path,map_location='cpu',weights_only=False)
        if checkpoint['source_sha256']!=parent or checkpoint['seed']!=r['seed'] or checkpoint['architecture']!=r['arm'] or checkpoint['steps']!=64:
            raise ValueError('original checkpoint/source identity differs')
        model=original.create_model(r['seed'],r['arm']);model.load_state_dict(checkpoint['model'],strict=True);model.eval()
        name=f"{r['seed']}:{r['arm']}";models[name]=model
        before[name]={k:tensor_sha(v) for k,v in model.state_dict().items()}
    report=dict(study_id=spec['study_id'],evidence_level='internal_source_mechanism_audit_not_capability_evidence',
        source_sha256=parent,cohort_sha256=digest(rows),windows=len(rows),source_controls=[],groups=[],positive_control={},
        probe_adapter=adapter().name,local_AD='INCONCLUSIVE_NOT_IMPLEMENTED',whole_model_PNDS_IEEE_AD='OPEN',
        input_semantics='13 physical channels/missingness/elapsed/last; writer encoder receives NO absolute calendar or owner ID; owner remains implicit in separate stream state',
        source_view_hash_policy='Actual derived hashes are declared explicitly in a scoped isolated original reader; parent publisher hash never forged or replaced in provenance',
        synthetic_controls_are_not_authentic_outcomes=True,training_steps=0,protected_TEST=False)
    args.output.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='history-source017-') as temp,torch.no_grad():
        temp=Path(temp);path=temp/'FIT-only.zip';receipt=make_control(members,path)
        try:source.source_frames(path,phase='fit')
        except RuntimeError as e:
            if 'original source SHA mismatch' not in str(e):raise
            report['default_original_hash_guard']='PASS_REJECTS_UNDECLARED_DERIVATIVE'
        else:raise ValueError('original checksum guard unexpectedly accepted derived source')
        identity,identity_intermediate=reload_source(source,original,path,receipt['sha256'],parent)
        if any(tensor_sha(identity[k])!=tensor_sha(baseline[k]) for k in ('raw','values','valid','age','last','center','spread')):
            raise ValueError('FIT-only identity reload changed actual original tensor values')
        report['FIT_only_identity_reload']='PASS_EXACT'
        report['source_controls'].append(receipt)
        for cutoff in spec['cutoff_hours']:
            cases=[r for r in rows if r[1]+64==cutoff]
            if not cases:raise ValueError('empty preregistered source cutoff')
            base={name:probe(model,window(baseline,cases,source)) for name,model in models.items()}
            repeated={name:probe(model,window(baseline,cases,source)) for name,model in models.items()}
            group=dict(cutoff_hour=cutoff,source_cases=cases,
                baseline_determinism={n:first_divergence(base[n],repeated[n]) for n in models},
                baseline_geometry={n:base[n]['geometry'] for n in models},
                original_forward_trace_sha256={n:digest(base[n]) for n in models},controls=[])
            for mode in ('values','missingness'):
                path=temp/f'{cutoff}-{mode}.zip'
                receipt=make_control(members,path,cutoff=source.START+dt.timedelta(hours=cutoff),mode=mode)
                control,temporary=reload_source(source,original,path,receipt['sha256'],parent)
                future_changed=tensor_sha(control['raw'][:,cutoff+1:])!=tensor_sha(baseline['raw'][:,cutoff+1:])
                if not future_changed:raise ValueError('source poison did not reach actual reloaded raw FIT table')
                prefix={k:tensor_sha(control[k][:,:cutoff+1])==tensor_sha(baseline[k][:,:cutoff+1]) for k in ('raw','values','valid','age','last')}
                normalizer={k:tensor_sha(control[k])==tensor_sha(baseline[k]) for k in ('center','spread')}
                item=dict(mode=mode,archive=receipt,raw_future_changed=True,consumed_prefix_invariant=prefix,
                    consumed_normalizer_invariant=normalizer,
                    unused_intermediate_source_frames_normalizer_invariant=temporary==base_intermediate,
                    unused_intermediate_note='Only raw from source_frames is consumed by original.source_observations; temporary future-fitted transforms are discarded',models={})
                for name,model in models.items():
                    got=probe(model,window(control,cases,source));divergence=first_divergence(base[name],got)
                    item['models'][name]=dict(first_divergence=divergence,trace_sha256=digest(got),
                        status='PASS_EXACT_INVARIANCE' if divergence is None else 'FAIL_EXACT_INVARIANCE_CAUSE_UNQUALIFIED')
                group['controls'].append(item)
                path.unlink()
            report['groups'].append(group)
            write(args.output/'RESULTS.json',report)
            print('SOURCE_RELOAD',cutoff,'windows',len(cases),'completed both future controls',flush=True)
        # Source-visible edit proves the perturbation reaches the real encoder boundary.
        site,anchor=rows[0];when=source.START+dt.timedelta(hours=anchor+64)
        path=temp/'positive.zip';receipt=make_control(members,path,mode='positive',positive=(source.SITES[site],when))
        control,_=reload_source(source,original,path,receipt['sha256'],parent)
        report['positive_control']=dict(archive=receipt,owner=site,query_hour=anchor+64,models={})
        for name,model in models.items():
            b=probe(model,window(baseline,[[site,anchor]],source));p=probe(model,window(control,[[site,anchor]],source))
            changed=b['timeline'][-1]['modules']['embed']!=p['timeline'][-1]['modules']['embed']
            report['positive_control']['models'][name]=dict(first_divergence=first_divergence(b,p),actual_encoder_activation_changed=changed)
        report['checkpoint_and_parameter_immutability']=all(before[n]=={k:tensor_sha(v) for k,v in m.state_dict().items()} and all(p.grad is None for p in m.parameters()) for n,m in models.items()) and all(sha_file(ROOT/r['path'])==r['sha256'] for r in spec['models'])
    report['all_exercised_end_to_end_invariance_checks_pass']=report['checkpoint_and_parameter_immutability'] and all(
        all(x is None for x in g['baseline_determinism'].values()) and all(
            all(c['consumed_prefix_invariant'].values()) and all(c['consumed_normalizer_invariant'].values()) and all(x['first_divergence'] is None for x in c['models'].values()) for c in g['controls']) for g in report['groups']) and all(x['actual_encoder_activation_changed'] for x in report['positive_control']['models'].values())
    write(args.output/'RESULTS.json',report)
    if not report['all_exercised_end_to_end_invariance_checks_pass']:raise SystemExit(2)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--implementation-root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    execute(p.parse_args())
if __name__=='__main__':main()
