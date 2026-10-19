"""Frozen room/horizon qualification with paired seed/two-day-block inference."""
import argparse
import csv
import io
import json
from pathlib import Path
import numpy as np
from data_real_maintenance_003 import write,ROOMS,HORIZONS


def bootstrap(difference,seed=973003,draws=2000):
    # Preserve paired model/control cases; repeat entire adjacent two-day blocks.
    x=np.asarray(difference,dtype=np.float64)
    assert x.ndim==4 and x.shape[0]==20 and x.shape[2:]==(2,10)
    rng=np.random.default_rng(seed);n,d=x.shape[:2];samples=[]
    for _ in range(draws):
        seeds=rng.integers(n,size=n);starts=rng.integers(d,size=(d+1)//2)
        days=((starts[:,None]+np.arange(2)[None])%d).reshape(-1)[:d]
        samples.append(x[seeds][:,days].mean((0,1)))
    return np.array(samples)


def compact(record):
    return {key:value for key,value in record.items() if key not in (
        'daily_absolute_error','daily_persistence_error','day_ids','absolute_error',
        'window_day','origin_seconds')}


def run(directory,output):
    root=Path(directory);summary=json.loads((root/'summary.json').read_text())
    assert summary['status'] in ('COMPLETE','VALIDATION_REJECTED','DATA_QUALITY_REJECTED')
    selection=json.loads((root/'selection.json').read_text()) if (root/'selection.json').exists() else None
    pilots=[json.loads(p.read_text()) for p in sorted(root.glob('pilot-*/*/report.json'))]
    result=dict(status=summary['status'],run=summary,selection=selection,
        pilots=[{key:r[key] for key in ('arm','seed','learning_rate','status','updates')}|
                {'validation':compact(r['validation'])} for r in pilots],
        commercial_qualified=False,
        scope='Measured point forecasts for10room channels/2horizons, one building. Not159analog channels, calibrated beliefs, actions, actual repair costs or independent commercial validation.')
    if summary['status']!='COMPLETE':
        assert not summary['test_opened'] and not (root/'TEST-EXPOSURE.json').exists()
        assert not (root/'_datasets/test.pt').exists() and not list(root.glob('group-*/*/test.json'))
        result.update(decision='NO_PROMOTION_TEST_SENSOR_VALUES_UNOPENED',
            fresh_validation=[dict(arm=r['arm'],seed=r['seed'],validation=compact(r['validation']))
                for r in (json.loads(p.read_text()) for p in sorted(root.glob('group-*/*/report.json')))])
        write(output,result);return
    assert summary['test_opened'] and (root/'TEST-EXPOSURE.json').exists()
    candidate=selection['candidate'];assert candidate is not None
    specs=json.loads((root/'specs.json').read_text());arms={};csvrows=[];full={}
    for arm in specs:
        reports=[]
        for path in sorted(root.glob(f'group-*/{arm}/test.json')):
            test=json.loads(path.read_text());fit=json.loads((path.parent/'report.json').read_text())
            reports.append(test)
            for hi,horizon in enumerate(HORIZONS):
                for channel,name in enumerate(ROOMS):
                    csvrows.append(dict(arm=arm,seed=fit['seed'],horizon_minutes=horizon*15,
                        channel=name,raw_MAE=test['raw_MAE'][hi][channel],MASE=test['MASE'][hi][channel],
                        relative_MAE=test['relative_MAE'][hi][channel],
                        all_cells_no_worse=test['all_cells_no_worse']))
        assert len(reports)==20 and len({tuple(r['day_ids']) for r in reports})==1
        full[arm]=reports
        arms[arm]=dict(individual_seeds_no_point_regression=sum(r['all_cells_no_worse'] for r in reports),
            per_seed={str(p.parent.parent.name):compact(r) for p,r in zip(sorted(root.glob(f'group-*/{arm}/test.json')),reports,strict=True)},
            mean_raw_MAE=np.mean([r['raw_MAE'] for r in reports],axis=0).tolist(),
            mean_MASE=float(np.mean([r['mean_MASE'] for r in reports])))
    reference_name=selection['conventional_reference']
    reference=json.loads((root/'references'/reference_name/'test.json').read_text())
    reference_reports={p.parent.name:compact(json.loads(p.read_text())) for p in sorted(root.glob('references/*/test.json'))}
    rows=full[candidate]
    assert all(row['day_ids']==reference['day_ids'] for row in rows)
    errors=np.array([r['daily_absolute_error'] for r in rows]);naive=np.array(rows[0]['daily_persistence_error'])
    assert all(r['daily_persistence_error']==rows[0]['daily_persistence_error'] for r in rows)
    differences=errors-naive[None]
    samples=bootstrap(differences)
    upper=np.quantile(samples,1-.05/20,axis=0)
    ratios=[r['mean_relative_MAE'] for r in rows]
    average_relative=float(np.mean(ratios)) if all(v is not None for v in ratios) else None
    # TRAIN-only scaling, never substitute a heldout denominator into learning.
    import torch
    train_scale=torch.load(root/'_datasets/normalization.pt',weights_only=True)['naive_scale'].numpy()
    control=np.array(reference['daily_absolute_error'])
    reference_bootstrap=bootstrap((errors-control[None])/train_scale[None,None])
    ref_means=reference_bootstrap.mean((1,2))
    ref_difference=float(((errors-control[None])/train_scale[None,None]).mean())
    gates=dict(individual18of20_no_point_regression=arms[candidate]['individual_seeds_no_point_regression']>=18,
        mean_relative_improvement5percent=average_relative is not None and average_relative<=.95,
        simultaneous20_cell_upper_no_regression=bool((upper<=0).all()),
        mean_MASE_vs_preselected_reference=float(np.quantile(ref_means,.95))<=0,
        replay='PENDING_SEPARATE_AUDIT',latency='PENDING_SEPARATE_MEASUREMENT')
    result.update(arms=arms,references=reference_reports,candidate=candidate,
        candidate_mean_relative_MAE=average_relative,
        candidate_minus_persistence_raw_MAE=differences.mean((0,1)).tolist(),
        simultaneous_one_sided95_upper_difference=upper.tolist(),
        candidate_minus_reference_mean_MASE=dict(reference=reference_name,difference=ref_difference,
            ordinary_one_sided95_upper=float(np.quantile(ref_means,.95))),
        bootstrap=dict(draws=2000,seed=973003,day_blocks=2,seeds=20,test_days=errors.shape[1]),
        gates=gates,decision='PENDING_REPLAY' if all(v is True for v in gates.values() if isinstance(v,bool)) else 'NO_PROMOTION_REGISTERED_FORECAST_GATES_FAILED')
    write(output,result)
    stream=io.StringIO();writer=csv.DictWriter(stream,fieldnames=list(csvrows[0]),lineterminator='\n')
    writer.writeheader();writer.writerows(csvrows);Path(output).with_suffix('.csv').write_text(stream.getvalue())
    print(json.dumps(dict(candidate=candidate,gates=gates)),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--directory',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();run(args.directory,args.output)
