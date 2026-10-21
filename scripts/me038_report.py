"""Report model construction, both resource lanes and prospective power outcomes."""
import json,shutil
from collections import defaultdict
import numpy as np
import scripts.me038_research as b


def run():
    frozen=b.verify();result=json.loads((b.OUT/'results.json').read_text());g=defaultdict(list)
    for r in result['models']:g[(r['lane'],r['dataset'],r['horizon'],r['arm'])].append(r)
    cells={k:float(np.mean([r['test']['MSE'] for r in v])) for k,v in g.items()}
    powers={k:float(np.mean([r['test']['power_MSE'] for r in v])) for k,v in g.items() if k[1]=='tetouan'}
    legacy={a:float(np.mean([cells[('exposure',d,h,a)] for d in b.old.DATASETS for h in b.HORIZONS])) for a in b.ARMS}
    house={a:float(np.mean([cells[('exposure','tetouan',h,a)] for h in b.HORIZONS])) for a in b.ARMS}
    power={a:float(np.mean([powers[('exposure','tetouan',h,a)] for h in b.HORIZONS])) for a in b.ARMS}
    cpower={a:float(np.mean([powers[('compute','tetouan',h,a)] for h in (96,720)])) for a in b.ARMS}
    def gain(values,other):return 1-values['innovation_tac']/values[other]
    complete=all(r['status']=='PASS' for r in frozen['fits'])
    summary={'legacy_exposure_MSE':legacy,'prospective_Tetouan_all_channel_MSE':house,
        'prospective_Tetouan_power_MSE':power,'compute_lane_Tetouan_power_MSE':cpower,
        'legacy_relative_gain_vs_AR':gain(legacy,'ar_adapt'),'legacy_relative_gain_vs_context':gain(legacy,'context_cross'),
        'prospective_power_gain_vs_AR':gain(power,'ar_adapt'),'prospective_power_gain_vs_context':gain(power,'context_cross'),
        'compute_power_gain_vs_AR':gain(cpower,'ar_adapt'),'compute_power_gain_vs_context':gain(cpower,'context_cross'),
        'failed_fits':[r['key'] for r in frozen['fits'] if r['status']!='PASS'],
        'architectural_superiority_claim':False,'global_novelty_claim':False}
    summary['legacy_engineering_gate']=bool(complete and gain(legacy,'ar_adapt')>=.02 and gain(legacy,'context_cross')>=.01)
    for lane,values in [('exposure',power),('compute',cpower)]:
        long=powers[(lane,'tetouan',720,'innovation_tac')]/powers[(lane,'tetouan',720,'ar_adapt')]-1
        summary[lane+'_power_H720_regression_vs_AR']=long
        summary[lane+'_prospective_power_gate']=bool(complete and gain(values,'ar_adapt')>=.02 and gain(values,'context_cross')>=.01 and long<=.02)
    previous=json.loads((b.ROOT/'research/mathematical-engineering/me037/summary.json').read_text())['legacy_development_MSE']['tac_horizon']
    summary['ME037_native_legacy_MSE']=previous;summary['historical_unmatched_relative_gain']=1-legacy['innovation_tac']/previous
    parents=json.loads((b.OUT/'parents.json').read_text())['parents']
    summary['actual_new_parent_training_CPU_seconds']=sum(r['training_CPU_seconds'] for r in parents if r['dataset']=='tetouan')
    summary['actual_new_joint_training_CPU_seconds']=sum(r['joint_CPU_seconds'] for r in frozen['fits'])
    summary['actual_selection_serialization_CPU_seconds']=sum(r['selection_serialization_CPU_seconds'] for r in frozen['fits'] if r['status']=='PASS')
    summary['worker_startup_CPU_seconds']=sum(r['worker_startup_CPU_seconds'] for r in frozen['fits'] if r['status']=='PASS')
    summary['final_neural_scalar_forecasts']=sum(r['test']['scalar_targets'] for r in result['models'])
    manifest=[];fit={r['key']:r for r in frozen['fits']};metadata=json.loads((b.OUT/'data.json').read_text());meta={r['name']:r for r in metadata}
    for r in result['models']:
        f=fit[r['key']]
        manifest.append({'key':r['key'],'source':meta[r['dataset']]['source'],'split_edges':meta[r['dataset']]['split_edges'],
            'input_history':96,'channels':meta[r['dataset']]['channels'],'horizon':r['horizon'],'lane':r['lane'],'arm':r['arm'],'seed':r['seed'],
            'trainable_parameters':r['parameters'],'parent':f['parent']['selected'],'parent_updates_allocated_consumed':1000,
            'joint_updates_allocated':500 if r['lane']=='exposure' else 10000,'joint_updates_consumed':r['updates'],
            'joint_training_windows':f['joint_windows_seen'],'pipeline_windows':f['pipeline_windows_seen'],
            'joint_CPU_budget':f['joint_CPU_budget'],'joint_CPU_consumed':f['joint_CPU_seconds'],
            'deadline_overshoot':f['compute_deadline_overshoot'],'selection_serialization_CPU':f['selection_serialization_CPU_seconds'],
            'attributed_pipeline_CPU_with_selection_startup':f['pipeline_measured_CPU_including_selection_and_startup'],
            'same_CPU_model':frozen['CPU_model'],'threads':1,'max_concurrent_fit_workers':2,
            'selection':'Lowest validation MSE of three positive endpoints; no step0 selection.',
            'parity':{'parameters':'PASS' if r['arm']!='ar_adapt' else 'FAIL_UNMATCHED_REFERENCE',
                'data':'PASS','exposure':'PASS' if r['lane']=='exposure' else 'FAIL_DIFFERENT_UPDATES',
                'compute':'FAIL_DIFFERENT_SPEND' if r['lane']=='exposure' else 'FAIL_END_TO_END_UNMATCHED_TRAINING_DEADLINE_ONLY',
                'selection':'PASS','evaluation':'PASS'},'architecture_superiority_claim':False,'test':r['test']})
    b.dump(b.REPORT/'summary.json',summary);b.dump(b.REPORT/'comparison-manifest.json',manifest)
    for name in ('results.json','audit.json','metric-audit.json','runtime.json','resources.json'):
        if (b.OUT/name).exists():shutil.copyfile(b.OUT/name,b.REPORT/name)
    labels={'innovation_tac':'Innovation TAC','context_cross':'Matched context','gru_cross':'Matched GRU','ar_adapt':'AR adaptation'}
    lines=['# ME038 — a new native state-transition variant','',
        'Native fast/slow dynamics now receive associative recall and perform prediction-error matrix writes. '
        'Historical cross-channel context is shared equally across neural arms. This is a new implemented TAC variant; '
        'associative/delta-rule primitives are established, so no first-in-literature novelty claim is made.',
        '', '## Exposure-matched complete models','',
        'Two seeds, 500 joint updates after identical validation-selected 1,000-update AR parents. '
        'Five legacy sources are exposed development evidence. Tetouan is a separately reserved temporal test '
        'within one city, not zero-shot or multi-city generalization. Lower MSE is better.',
        '', '| Model | Legacy mean MSE | Tetouan all 8 channels | Tetouan 3 power channels |',
        '|---|---:|---:|---:|']
    for a in b.ARMS:lines.append(f'| {labels[a]} | {legacy[a]:.6f} | {house[a]:.6f} | {power[a]:.6f} |')
    lines+=['',f"Development gain vs AR: {gain(legacy,'ar_adapt')*100:.2f}%; vs context: {gain(legacy,'context_cross')*100:.2f}%. "
        f"Registered development target **{'PASS' if summary['legacy_engineering_gate'] else 'FAIL'}**.",
        '',f"Prospective power gain vs AR: {gain(power,'ar_adapt')*100:.2f}%; vs context: {gain(power,'context_cross')*100:.2f}%. "
        f"Registered primary power target **{'PASS' if summary['exposure_prospective_power_gate'] else 'FAIL'}**. "
        f"H720 power regression vs AR: {summary['exposure_power_H720_regression_vs_AR']*100:+.2f}% (positive worse).",
        '',f"ME037 historical native macro MSE was {previous:.6f}; new candidate relative change: "
        f"{abs(summary['historical_unmatched_relative_gain'])*100:.2f}% {'lower' if summary['historical_unmatched_relative_gain']>=0 else 'higher'} MSE. This historical comparison is unmatched "
        'for parameters, state capacity, cross-channel information paths and actual CPU; it is not architecture superiority.',
        '', '## Separate training-CPU-budget lane','',
        'Five process-CPU-second optimization deadline, finish the crossing update, at most 10,000 updates. '
        'Sample exposure therefore differs. Validation/serialization and worker startup are measured outside this '
        'training deadline and included in reported component cost sums. Shared data loading and final trace/report writing are not exhaustively metered; these are not complete end-to-end totals. This is not equal end-to-end compute/data parity.',
        '', '| Model | Tetouan power MSE (H96/720) | Mean updates, all compute cells | Mean training CPU seconds |',
        '|---|---:|---:|---:|']
    for a in b.ARMS:
        rr=[r for r in frozen['fits'] if r['lane']=='compute' and r['arm']==a and r['status']=='PASS']
        lines.append(f"| {labels[a]} | {cpower[a]:.6f} | {np.mean([r['updates'] for r in rr]):.1f} | {np.mean([r['joint_CPU_seconds'] for r in rr]):.4f} |")
    lines+=['',f"Compute-lane power gain vs AR: {gain(cpower,'ar_adapt')*100:.2f}%; vs context: {gain(cpower,'context_cross')*100:.2f}%. "
        f"Separate power target **{'PASS' if summary['compute_prospective_power_gate'] else 'FAIL'}**.",
        '', '## All dataset/horizon cells','',
        '| Lane | Dataset | H | Native innovation | Context | GRU | AR |','|---|---|---:|---:|---:|---:|---:|']
    for lane in ('exposure','compute'):
        for d in b.DATASETS:
            for h in b.HORIZONS:
                if (lane,d,h,b.ARMS[0]) not in cells:continue
                lines.append('| '+lane+' | '+d+' | '+str(h)+' | '+' | '.join(f'{cells[(lane,d,h,a)]:.6f}' for a in b.ARMS)+' |')
    lines+=['', '## Power-only outcomes','',
        '| Lane | H | Native | Context | GRU | AR |','|---|---:|---:|---:|---:|---:|']
    for lane in ('exposure','compute'):
        for h in b.HORIZONS:
            if (lane,'tetouan',h,b.ARMS[0]) in powers:
                lines.append('| '+lane+' | '+str(h)+' | '+' | '.join(f"{powers[(lane,'tetouan',h,a)]:.6f}" for a in b.ARMS)+' |')
    lines+=['', '## Power MAE and channel trade-offs','',
        'The following zone MAEs are rescaled using each train-only channel std. Units remain the original publisher numeric units, which UCI does not specify; no Wh/kW conversion or measured savings is inferred.',
        '', '| Lane | H | Model | Power standardized MAE | Zone1 MAE | Zone2 MAE | Zone3 MAE |',
        '|---|---:|---|---:|---:|---:|---:|']
    for lane in ('exposure','compute'):
        for h in b.HORIZONS:
            for arm in b.ARMS:
                rows=g.get((lane,'tetouan',h,arm),[])
                if not rows:continue
                stdmae=np.mean([r['test']['power_MAE'] for r in rows])
                physical=np.mean([r['test']['channel_MAE'][5:] for r in rows],axis=0)*np.asarray(meta['tetouan']['scale'][5:])
                lines.append(f'| {lane} | {h} | {labels[arm]} | {stdmae:.6f} | '+ ' | '.join(f'{x:.3f}' for x in physical)+' |')
    lines+=['','## Evidence boundaries','',
        'All attempted fits, checkpoint selection, data hashes, parameter/exposure/cost contracts and per-channel '
        'scores are retained. No fit replacement or post-test modification. Parameter gaps are under 1% among '
        'neural arms; native recurrent state is 128 float32 values/channel, twice ME037. AR and persistence are '
        'unmatched practical references. Resource/selection costs and supported operator FLOP subtotals are in '
        'resources.json and comparison-manifest.json; unsupported work means no full FLOP/energy parity.',
        '', 'Neither lane passes all architecture-parity gates: exposure lane spends different CPU; CPU lane '
        'has different sample presentations and total end-to-end spend. **No architectural superiority claim.** '
        'Two fitted seeds, one new city and previously exposed legacy benchmarks limit generalization. '
        'Small differences are descriptive, not statistical superiority results.',
        '', 'Original-source units are preserved, with no invented power units, measured savings, actions or costs. '
        'The model forecasts all eight measurements; the primary power metric is the three actual zone channels. '
        'No future weather, calendar variables, synthetic fit evidence, reopened household/KuaiRec data, '
        'action/controller/VoI capability, commercial qualification or constant-time streaming claim. '
        'Full 96-step replay remains necessary. Prior Weather anomalies and ME035B precision failure remain intact.']
    if (b.OUT/'audit.json').exists():
        a=json.loads((b.OUT/'audit.json').read_text());lines+=['',f"Replay: {a['original_checkpoints_replayed']:,} parent/adaptation endpoints, {a['bit_exact_validation_forecast_values']:,} "
            f"saved validation values bit exact; {a['bit_exact_initial_AR_forecast_values']:,} initial AR values exact. "
            'These arrays are not the complete final test matrix.']
    if (b.OUT/'metric-audit.json').exists():
        a=json.loads((b.OUT/'metric-audit.json').read_text());lines+=['',f"Independent final-metric reduction audit: {a['models_and_references']} Tetouan models/references, "
            f"{a['scalar_test_forecasts_recomputed']:,} test forecast values recomputed; {a['status']}. "
            'Uses the preregistered float64 accumulation bound; does not imply bit-exact storage/replay of every test array or a legacy full-matrix rerun.']
    (b.REPORT/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    (b.REPORT/'README.md').write_text('# ME038\n\n[Results](RESULTS.md) · [State equations](MATHEMATICS.md) · [Protocol](PROTOCOL.md) · [Final-metric audit contract](AUDIT-ADDENDUM.md) · [Original models](ARTIFACT.md)\n')
    print(json.dumps(summary,indent=2),flush=True)

if __name__=='__main__':run()
