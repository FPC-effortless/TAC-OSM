"""Recorded-only replay and chronological real-FIT Jacobian probes; never fits."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from scripts.me022_native_readout import predictions, scores, save_json
from scripts.me024_coupled_research import C, sha, load_source, make_model, feature_diagnostic
from tac_osm.coupled_native import CoupledNative, joint_jacobian


@torch.no_grad()
def probes(net, train):
    net = net.double().eval()
    rows = []
    for i in range(C['probe_fit_windows']):
        state = torch.zeros(1, 64, dtype=torch.float64)
        product = torch.eye(64, dtype=torch.float64)
        for t in range(128):
            u = torch.cat((net.emb(train['items'][i:i+1,t]),train['metadata'][i:i+1,t].double()),-1)
            j,a,d,b,c = joint_jacobian(net,u[0],state[0])
            product = j@product
            state = net.advance_features(u,state)
            if t+1 in C['probe_jacobian_depths']:
                singular = torch.linalg.svdvals(product)
                rows.append({'fit_window':i,'depth':t+1,'product_singular_values':singular.tolist(),
                    'product_spectral_norm':float(singular[0]),'product_smallest_singular':float(singular[-1]),
                    'local_feedback_spectral_norm':float(torch.linalg.matrix_norm(d,ord=2)),
                    'local_joint_spectral_norm':float(torch.linalg.matrix_norm(j,ord=2)),
                    'state_max_abs':float(state.abs().max())})
    return rows


def audit(root, parent, previous):
    root=Path(root); result=json.loads((root/'results.json').read_text())
    assert result['contract']==C and len(result['models'])==12
    assert {(r['arm'],r['seed']) for r in result['models']}=={(a,s) for a in C['arms'] for s in C['seeds']}
    # Validate every model before any data or labels are loaded.
    for row in result['models']:
        assert sha(root/row['checkpoint'])==row['checkpoint_sha256']
        assert sha(root/row['standalone'])==row['standalone_sha256']
    data=load_source(parent,previous); train,dev=data['train'],data['dev']
    torch.set_num_threads(1); sampler={}; errors=[]; stream_errors=[]; jac=[]
    for row in result['models']:
        ck=torch.load(root/row['checkpoint'],weights_only=True)
        assert ck['contract']==C and ck['arm']==row['arm'] and ck['seed']==row['seed']
        assert ck['step']==row['steps']==C['updates']
        assert ck['train_cpu_seconds']==row['train_cpu_seconds']
        assert all(torch.isfinite(v).all() for v in ck['state_dict'].values())
        assert len(ck['optimizer']['state'])==len(list(make_model(row['arm'],ck['duration_mean'],ck['duration_scale']).parameters()))
        for state in ck['optimizer']['state'].values():
            assert float(state['step'])==C['updates']
            assert torch.isfinite(state['exp_avg']).all() and torch.isfinite(state['exp_avg_sq']).all()
        assert row['parameters']==C['parameters']
        assert row['train_cpu_seconds']<=C['cpu_seconds_ceiling_per_fit']
        assert row['sampled_windows']==C['updates']*C['batch']
        assert row['historical_event_presentations']==row['sampled_windows']*128
        assert row['supervised_response_presentations']==row['sampled_windows']*4
        assert [t['step'] for t in row['trace']]==C['trace_steps']
        gen=torch.Generator().manual_seed(row['seed']+300)
        for _ in range(C['updates']): torch.randint(len(train['y']),(C['batch'],),generator=gen)
        assert torch.equal(gen.get_state(),ck['sampler_rng'])
        if row['seed'] in sampler: assert torch.equal(sampler[row['seed']],ck['sampler_rng'])
        sampler[row['seed']]=ck['sampler_rng']
        net=make_model(row['arm'],ck['duration_mean'],ck['duration_scale'])
        net.load_state_dict(ck['state_dict'],strict=True)
        script=torch.jit.script(net).eval(); exported=torch.jit.load(str(root/row['standalone'])).eval()
        old=np.load(root/f"{row['arm']}-seed{row['seed']}-predictions.npz")
        p=predictions(script,dev)
        replaced,mean,mp=feature_diagnostic(script,train,dev)
        off=torch.jit.script(CoupledNative(0.,'rms' in row['arm'],ck['duration_mean'],ck['duration_scale']))
        off.load_state_dict(ck['state_dict'],strict=True); op=predictions(off,dev)
        paths=[(p,old['probability']),(predictions(exported,dev),p),
               (mp,old['mean_feature_probability']),(op,old['feedback_disabled_probability'])]
        err=[float(np.max(np.abs(a-b))) for a,b in paths]; assert max(err)<=1e-7; errors.append(err)
        for recorded,actual in [(row['dev'],scores(p,dev['y'].numpy())),
                (row['feature_mean_dev'],replaced),(row['feedback_disabled_dev'],scores(op,dev['y'].numpy()))]:
            for k,v in actual.items(): assert abs(recorded[k]-v)<=1e-9
        mean_old=torch.load(root/f"{row['arm']}-seed{row['seed']}-feature-mean.pt",weights_only=True)['features']
        torch.testing.assert_close(mean,mean_old,atol=0,rtol=0)
        with torch.no_grad():
            state=torch.zeros(32,64)
            for t in range(128): state=exported.stream_step(dev['items'][:32,t],dev['metadata'][:32,t],state)
            ws=script.encode(dev['items'][:32],dev['metadata'][:32])
            prior=(dev['metadata'][:32,-32:,1].sum(-1,keepdim=True)+1)/34
            logit=exported.readout(state,dev['action'][:32],dev['duration'][:32],prior)
            direct=script(dev['items'][:32],dev['metadata'][:32],dev['action'][:32],dev['duration'][:32])
            se=[float((state-ws).abs().max()),float((logit-direct).abs().max())]
            assert max(se)<=1e-6; stream_errors.append(se)
        jac.append({'arm':row['arm'],'seed':row['seed'],'probes':probes(net,train)})
    save_json(root/'jacobian-probes.json',{'dtype':'float64','dataset':'first 8 authenticated FIT windows',
        'product_order':'J_t @ ... @ J_1, evaluated along actual model trajectory',
        'interpretation':'descriptive local derivatives; no causal or generalization conclusion','models':jac})
    report={'status':'PASS','checkpoints':12,'standalone_exports':12,'predictions_per_path':12*len(dev['y']),
        'path_order':['checkpoint','standalone','FIT_feature_mean','feedback_disabled'],
        'maximum_prediction_errors':np.max(errors,axis=0).tolist(),
        'streaming_real_windows_per_model':32,'maximum_streaming_state_and_logit_errors':np.max(stream_errors,axis=0).tolist(),
        'exact_sampler_rng_reconstruction':True,'exact_feature_means':True,
        'jacobian_trajectories':12*C['probe_fit_windows'],'new_fits':0,'new_test_outcomes':0}
    save_json(root/'replay.json',report); print(json.dumps(report)); return report


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--root',default='results/me024')
    p.add_argument('--parent',default='results/me022'); p.add_argument('--previous',default='results/me023')
    a=p.parse_args(); audit(a.root,a.parent,a.previous)
