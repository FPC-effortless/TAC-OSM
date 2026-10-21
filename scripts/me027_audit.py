"""Recorded-only ME027 audit, state diagnostics and post-fit FLOP subtotals."""
from __future__ import annotations
import argparse
import json
import time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from scripts.me027_state_research import C,make_model
from scripts.me023_predictive_research import sha,load_source,atomic_checkpoint
from scripts.me025_real_transformer_parity import fit_only_popularity
from scripts.me014b_exact_action_memory import batch
from scripts.me022_native_readout import predictions,scores,save_json


def selected(row,label):return next(q for q in row['checkpoints'] if label in q['labels'])


def restore(root,row,mean,scale,pop):
    ck=torch.load(root/row['checkpoint'],weights_only=True)
    net=torch.jit.script(make_model(ck['arm'],mean,scale,pop));net.load_state_dict(ck['state_dict'],strict=True)
    return ck,net.eval()


@torch.no_grad()
def state_probe(net,train,dev):
    n=C['feature_mean_fit_windows']
    features=torch.cat([net.latent_features(train['items'][i:i+256],train['metadata'][i:i+256]) for i in range(0,n,256)])
    mean=features.mean(0,keepdim=True);out=[]
    for i in range(0,len(dev['y']),384):
        sl=slice(i,i+384);meta=dev['metadata'][sl]
        common=net.observables(dev['items'][sl],meta,dev['action'][sl],dev['duration'][sl],128)
        prior=(meta[:,-32:,1].sum(-1,keepdim=True)+1)/34
        out.append(net.predict_features(mean.expand(len(meta),-1),common,prior).sigmoid().flatten())
    return mean,torch.cat(out).numpy()


def profile_subtotal(net,train):
    n=C['profiler_fit_windows'];net.train();net.zero_grad(set_to_none=True)
    truth=torch.cat((train['metadata'][:n,C['prefix_depths'],1],train['y'][:n]),-1)
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU],record_shapes=True,with_flops=True) as prof:
        logits=net.prefix_logits(*batch(train,slice(0,n)))
        loss=(F.binary_cross_entropy_with_logits(logits,truth,reduction='none')*torch.tensor(C['prefix_weights']).reshape(1,4)).sum(-1).mean()
        loss.backward()
    ops=[{'operator':e.key,'calls':e.count,'reported_flops':e.flops or 0} for e in prof.key_averages()]
    net.zero_grad(set_to_none=True);net.eval()
    return {'fit_windows':n,'optimizer_steps':0,'forward_and_backward':True,'profiler_reported_flops_lower_bound':sum(x['reported_flops'] for x in ops),
        'all_operations_counted':False,'full_training_FLOPs_measured':False,'operations':ops}


def audit(root,parent,profile=False,verify_only=False):
    root=Path(root);r=json.loads((root/'results.json').read_text());assert r['contract']==C and len(r['models'])==12
    assert {(m['arm'],m['seed']) for m in r['models']}=={(a,s) for a in C['arms'] for s in C['seeds']}
    required={f'step{n}' for n in C['exposure_steps']}|{f'cpu{b}' for b in C['cpu_budgets_seconds']}
    # All checkpoint/prediction/export digests before cached response access.
    for m in r['models']:
        labels=[x for q in m['checkpoints'] for x in q['labels']];assert set(labels)==required and len(labels)==len(required)
        for q in m['checkpoints']:
            assert sha(root/q['checkpoint'])==q['checkpoint_sha256'];assert sha(root/q['prediction'])==q['prediction_sha256']
            if q['standalone']:assert sha(root/q['standalone'])==q['standalone_sha256']
    data=load_source(parent);train,dev=data['train'],data['dev'];torch.set_num_threads(1)
    assert sha(Path(parent)/'data-snapshot.pt')==C['parent_snapshot_sha256']
    pop=fit_only_popularity(train);mean=r['duration_mean'];scale=r['duration_scale']
    assert mean==float(train['duration'].mean()) and scale==float(train['duration'].std(unbiased=False))
    errors=[];export_errors=[];budget_records=[];states=[];profiles={};primary={}
    previous=json.loads((root/'replay.json').read_text()) if verify_only else None
    for m in r['models']:
        assert m['parameters']==C['parameters'][m['arm']]
        assert abs(m['parameters']-C['parameter_target'])/C['parameter_target']<=C['parameter_relative_tolerance']
        gen=torch.Generator().manual_seed(m['seed']+300);current=0
        for q in sorted(m['checkpoints'],key=lambda x:x['step']):
            ck,net=restore(root,q,mean,scale,pop)
            assert ck['contract']==C and ck['arm']==m['arm'] and ck['seed']==m['seed'] and ck['step']==q['step']
            assert ck['duration_mean']==mean and ck['duration_scale']==scale
            assert sum(p.numel() for p in net.parameters())==m['parameters']
            torch.testing.assert_close(net.item_popularity,pop,atol=0,rtol=0)
            assert all(torch.isfinite(v).all() for v in ck['state_dict'].values())
            assert len(ck['optimizer']['state'])==len(list(net.parameters()))
            for v in ck['optimizer']['state'].values():
                assert float(v['step'])==q['step'] and torch.isfinite(v['exp_avg']).all() and torch.isfinite(v['exp_avg_sq']).all()
            for _ in range(current,q['step']):torch.randint(len(train['y']),(C['batch'],),generator=gen)
            current=q['step'];assert torch.equal(gen.get_state(),ck['sampler_rng'])
            assert q['sampled_windows']==q['step']*128 and q['historical_event_presentations']==q['step']*128*128
            assert q['supervised_responses']==q['step']*128*4
            assert q['charged_cpu_seconds']<=C['maximum_charged_cpu_seconds_per_fit']
            assert abs(q['charged_cpu_seconds']-q['optimizer_cpu_seconds']-q['setup_cpu_seconds']-q['shared_preprocessing_cpu_seconds_charged'])<1e-9
            for label in q['labels']:
                if label.startswith('step'):assert q['step']==int(label[4:])
                else:assert q['charged_cpu_seconds']>=int(label[3:])
            p=predictions(net,dev);old=np.load(root/q['prediction']);error=float(np.max(np.abs(p-old)))
            assert error<=C['replay_probability_atol'];errors.append(error)
            for k,v in scores(p,dev['y'].numpy()).items():assert abs(q['dev'][k]-v)<=C['replay_metric_atol']
            if q['standalone']:
                exported=torch.jit.load(str(root/q['standalone'])).eval();e=float(np.max(np.abs(predictions(exported,dev)-old)))
                assert e<=C['replay_probability_atol'];export_errors.append(e);primary[m['arm'],m['seed']]=p
                if m['arm']!='mlp':
                    fm,mp=state_probe(net,train,dev);metrics=scores(mp,dev['y'].numpy())
                    filename=f"{m['arm']}-seed{m['seed']}-feature-mean"
                    if not verify_only:
                        atomic_checkpoint(root/f'{filename}.pt',{'features':fm})
                        np.save(root/f'{filename}-predictions.npy',mp)
                    else:
                        torch.testing.assert_close(fm,torch.load(root/f'{filename}.pt',weights_only=True)['features'],atol=0,rtol=0)
                        assert np.max(np.abs(mp-np.load(root/f'{filename}-predictions.npy')))<=C['replay_probability_atol']
                    states.append({'arm':m['arm'],'seed':m['seed'],'replaced_dev':metrics,'Brier_delta':metrics['Brier']-q['dev']['Brier'],
                        'state_head_weight_norm':float(net.head.weight[:,:64].detach().norm())})
            if verify_only:
                resource=next(v for v in previous['budget_resources'] if v['arm']==m['arm'] and v['seed']==m['seed'] and v['step']==q['step'])
                assert resource['labels']==q['labels'] and resource['charged_cpu_seconds']==q['charged_cpu_seconds']
                assert resource['inference_wall_median_seconds_batch128']==float(np.median([v['wall_seconds'] for v in resource['inference_samples']]))
                budget_records.append(resource)
            else:
                timing=[]
                with torch.no_grad():
                    for _ in range(3):net(*batch(dev,slice(0,128)))
                    for _ in range(9):
                        t,c=time.perf_counter(),time.process_time();net(*batch(dev,slice(0,128)))
                        timing.append({'wall_seconds':time.perf_counter()-t,'cpu_seconds':time.process_time()-c})
                budget_records.append({'arm':m['arm'],'seed':m['seed'],'labels':q['labels'],'step':q['step'],
                    'charged_cpu_seconds':q['charged_cpu_seconds'],'parameters':m['parameters'],'nonembedding_parameters':m['parameters']-128000,
                    'nontrainable_buffer_bytes':sum(x.numel()*x.element_size() for x in net.buffers()),
                    'checkpoint_bytes_including_optimizer_RNG':(root/q['checkpoint']).stat().st_size,
                    'inference_wall_median_seconds_batch128':float(np.median([x['wall_seconds'] for x in timing])),
                    'inference_samples':timing,'isolated_inference_memory_measured':False})
            if profile and m['seed']==C['seeds'][0] and 'step300' in q['labels']:profiles[m['arm']]=profile_subtotal(net,train)
    report={'status':'PASS','budget_checkpoints':len(errors),'primary_standalone_exports':len(export_errors),
        'predictions_per_budget_path':len(errors)*len(dev['y']),'primary_export_predictions':len(export_errors)*len(dev['y']),
        'maximum_checkpoint_prediction_error':max(errors),'maximum_export_prediction_error':max(export_errors),
        'optimizer_states_and_sampler_reconstructed':True,'new_fits':0,'new_test_outcomes':0,
        'first_crossing_CPU_selection':'frozen driver condition; no independently recorded per-step clock trace',
        'state_replacement_diagnostics':states,'budget_resources':budget_records}
    save_json(root/('verification.json' if verify_only else 'replay.json'),report)
    if profile:save_json(root/'profiler-subtotals.json',{'interpretation':'matrix FLOP lower bounds; unsupported/opaque operations retained; NOT full compute parity','models':profiles})
    y=dev['y'].numpy().flatten().astype(np.float64)
    losses=np.mean([(primary['tac',s].astype(np.float64)-y)**2-(primary['mlp',s].astype(np.float64)-y)**2 for s in C['seeds']],axis=0)
    users=np.array(dev['users']);unique=np.unique(users);sums=np.array([losses[users==u].sum() for u in unique]);counts=np.array([(users==u).sum() for u in unique])
    rng=np.random.default_rng(C['bootstrap_seed']);draws=rng.integers(len(unique),size=(C['bootstrap_user_clusters'],len(unique)))
    values=sums[draws].sum(1)/counts[draws].sum(1)
    save_json(root/'primary-user-bootstrap.json',{'mean_loss_difference':float(losses.mean()),'interval_95_percent':np.quantile(values,[.025,.975]).tolist(),
        'unique_users':len(unique),'draws':len(values),'seed':C['bootstrap_seed'],'conditional_on_three_training_seeds':True,'independent_validation':False})
    print(json.dumps({k:v for k,v in report.items() if k not in ['budget_resources','state_replacement_diagnostics']}));return report


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',default='results/me027');p.add_argument('--parent',default='results/me022');p.add_argument('--profile',action='store_true');p.add_argument('--verify-only',action='store_true')
    a=p.parse_args();audit(a.root,a.parent,a.profile,a.verify_only)
