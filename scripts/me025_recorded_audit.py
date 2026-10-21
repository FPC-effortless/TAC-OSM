"""Replay the completed original ME025 archive. Never trains or downloads data."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from scripts.me025_real_transformer_parity import model_for, fit_only_popularity, prob_scores, PARAM_COUNTS, SEEDS, ARMS, STEPS
from scripts.me022_native_readout import scores,save_json
from scripts.me023_predictive_research import load_source,sha

CONTRACT=Path(__file__).resolve().parents[1]/'research/mathematical-engineering/me026/contract.json'
C=json.loads(CONTRACT.read_text())


def audit(root,parent,out,archive):
    root=Path(root); out=Path(out); out.mkdir(parents=True,exist_ok=True)
    assert sha(Path(archive))==C['original_archive_sha256']
    r=json.loads((root/'manifest.json').read_text())
    assert len(r['model_results'])==9 and r['all_original_models_completed']
    assert {(m['arm'],m['seed']) for m in r['model_results']}=={(a,s) for a in ARMS for s in SEEDS}
    for row in r['model_results']:
        assert sha(root/row['checkpoint'])==row['checkpoint_sha256']
        assert sha(root/row['prediction_file'])==row['prediction_sha256']
    data=load_source(parent); train,dev=data['train'],data['dev']; torch.set_num_threads(1)
    mean=float(train['duration'].mean()); scale=float(train['duration'].std(unbiased=False))
    assert r['duration_fit_mean']==mean and r['duration_fit_scale']==scale
    rate=fit_only_popularity(train); digest=hashlib.sha256(rate.numpy().tobytes()).hexdigest()
    rows=[]
    for row in r['model_results']:
        ck=torch.load(root/row['checkpoint'],weights_only=True)
        assert ck['step']==row['updates']==300 and ck['params']==row['parameters']==PARAM_COUNTS[row['arm']]
        assert ck['arm']==row['arm'] and ck['seed']==row['seed']
        assert ck['duration_fit_mean']==mean and ck['duration_fit_scale']==scale
        assert ck['fit_only_popularity_sha256']==digest
        assert ck['source_sha256']==r['official_publisher_zip_sha256']==data['source']['zip_sha256']
        assert ck['charged_optimizer_CPU_seconds']==row['charged_optimizer_CPU_seconds']<=600
        assert row['sampled_windows']==38400 and row['genuine_event_history_presentations']==4915200
        assert row['supervised_true_next_responses']==153600 and [t['step'] for t in row['trace']]==list(STEPS)
        gen=torch.Generator().manual_seed(row['seed']+300)
        for _ in range(300): torch.randint(len(train['y']),(128,),generator=gen)
        assert torch.equal(gen.get_state(),ck['sampler_rng'])
        net=model_for(row['arm'],mean,scale,rate); net.load_state_dict(ck['state_dict'],strict=True)
        assert sum(p.numel() for p in net.parameters())==PARAM_COUNTS[row['arm']]
        assert all(torch.isfinite(p).all() for p in net.parameters())
        if row['arm']!='tac_rms': torch.testing.assert_close(net.item_popularity,rate,atol=0,rtol=0)
        assert len(ck['optimizer']['state'])==len(list(net.parameters()))
        for state in ck['optimizer']['state'].values():
            assert float(state['step'])==300
            assert torch.isfinite(state['exp_avg']).all() and torch.isfinite(state['exp_avg_sq']).all()
        p=prob_scores(net,dev); old=np.load(root/row['prediction_file'])
        error=float(np.max(np.abs(p-old))); assert error<=C['replay_probability_atol']
        metrics=scores(p,dev['y'].numpy()); metric_errors={k:abs(metrics[k]-row['dev'][k]) for k in metrics}
        assert max(metric_errors.values())<=C['metric_atol'],metric_errors
        rows.append({'arm':row['arm'],'seed':row['seed'],'maximum_probability_error':error,
                     'metric_errors':metric_errors,'replayed_scores':metrics})
    primary=r['registered_primary']; groups={(q['arm'],q['seed']):q for q in r['model_results']}
    delta=[groups['tac_kernel',s]['dev']['Brier']-groups['tac_rms',s]['dev']['Brier'] for s in SEEDS]
    assert delta==primary['per_seed_deltas']
    passed=all(x<0 for x in delta) and np.mean(delta)<=-.001 and primary['ECE10_kernel_minus_native_mean']<=.005
    assert bool(passed)==primary['passed_development_only_engineering_gate']==False
    report={'status':'PASS','original_neural_checkpoints':9,'predictions_per_path':9*len(dev['y']),
        'maximum_probability_error':max(x['maximum_probability_error'] for x in rows),'models':rows,
        'original_primary_gate_passed':False,'primary':primary,'mean_scores':r['descriptive_means'],
        'MLP_reference_weights_and_probabilities_available':False,'MLP_reference_independently_replayed':False,
        'parameter_parity':'PASS','data_exposure_parity':'PASS','tuning_selection_parity':'PASS',
        'full_compute_parity':'UNESTABLISHED','architecture_superiority_claim_allowed':False,
        'training_CPU_seconds_by_arm':{a:sum(m['charged_optimizer_CPU_seconds'] for m in r['model_results'] if m['arm']==a) for a in ARMS},
        'new_fits':0,'new_test_outcomes':0,'original_archive_sha256':C['original_archive_sha256']}
    save_json(out/'me025-replay.json',report); print(json.dumps({k:v for k,v in report.items() if k!='models'}))
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--root',default='/workspace/me025-workflow-original/results/me025')
    p.add_argument('--parent',default='results/me022');p.add_argument('--out',default='results/me026')
    p.add_argument('--archive',default='/workspace/me025-original-workflow.zip')
    a=p.parse_args();audit(a.root,a.parent,a.out,a.archive)
