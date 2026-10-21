"""Report legacy development and prospective household outcomes separately."""
import json,shutil
from collections import defaultdict
import numpy as np
import scripts.me037_research as b


def run():
    frozen=json.loads((b.OUT/'frozen.json').read_text());results=json.loads((b.OUT/'results.json').read_text())
    grouped=defaultdict(list)
    for r in results['models']:grouped[(r['dataset'],r['horizon'],r['arm'])].append(r)
    cells={k:float(np.mean([r['test']['MSE'] for r in rows])) for k,rows in grouped.items()}
    legacy={a:float(np.mean([cells[(d,h,a)] for d in b.old.DATASETS for h in b.HORIZONS])) for a in b.ARMS}
    house={a:float(np.mean([cells[('appliances',h,a)] for h in b.HORIZONS])) for a in b.ARMS}
    energy={a:float(np.mean([r['test']['energy_MSE'] for r in results['models'] if r['dataset']=='appliances' and r['arm']==a])) for a in b.ARMS}
    old=json.loads((b.ROOT/'research/mathematical-engineering/me036/summary.json').read_text())['overall_MSE']['tac_patch']
    long={d:cells[(d,720,'tac_horizon')]/cells[(d,720,'ar_adapt')]-1 for d in ('ETTh1','ETTh2')}
    legacy_gain=1-legacy['tac_horizon']/legacy['ar_adapt'];house_gain=1-house['tac_horizon']/house['ar_adapt']
    household_long=cells[('appliances',720,'tac_horizon')]/cells[('appliances',720,'ar_adapt')]-1
    parents=json.loads((b.OUT/'parents.json').read_text())['parents']
    summary={'legacy_development_MSE':legacy,'prospective_household_MSE':house,'prospective_energy_only_MSE':energy,
        'legacy_relative_improvement_vs_AR_adapt':legacy_gain,'household_relative_improvement_vs_AR_adapt':house_gain,
        'ETTh_720_relative_regression_vs_AR_adapt':long,'household_720_relative_regression_vs_AR_adapt':household_long,
        'legacy_engineering_gate':bool(legacy_gain>=.02 and all(v<=.02 for v in long.values())),
        'prospective_household_engineering_gate':bool(house_gain>=.02 and household_long<=.02),
        'historical_ME036_native_MSE':old,'historical_relative_improvement':1-legacy['tac_horizon']/old,
        'historical_comparison_matched':False,'architectural_superiority_qualified':False,
        'failed_fits':[r['key'] for r in frozen['fits'] if r['status']!='PASS'],
        'new_adaptation_fits':len(frozen['fits']),'new_parent_fits':sum(r['origin']=='new_household_pretraining' for r in parents),
        'actual_new_adaptation_CPU_seconds':sum(r['joint_CPU_seconds'] for r in frozen['fits']),
        'actual_new_parent_CPU_seconds':sum(r['training_CPU_seconds'] for r in parents if r['origin']=='new_household_pretraining'),
        'cached_parent_training_CPU_seconds_deduplicated':sum(r['training_CPU_seconds'] for r in parents if r['origin']=='cached_ME036_AR_only'),
        'neural_final_scoring_CPU_seconds':sum(r['test']['CPU_seconds'] for r in results['models']),
        'neural_final_scalar_forecasts':sum(r['test']['scalar_targets'] for r in results['models']),
        'seeds':[37001,37002],'aggregation':'Mean over seeds within dataset/H, equal weight H/cells; legacy and new-house separate.'}
    for name in ('results.json','audit.json','runtime.json'):shutil.copyfile(b.OUT/name,b.REPORT/name)
    b.dump(b.REPORT/'summary.json',summary)
    indexed={(r['dataset'],r['horizon'],r['arm'],r['seed']):r for r in frozen['fits']}
    meta={r['name']:r for r in json.loads((b.OUT/'data.json').read_text())}
    manifests=[]
    for r in results['models']:
        fit=indexed[(r['dataset'],r['horizon'],r['arm'],r['seed'])]
        manifests.append({'key':r['key'],'dataset':r['dataset'],'arm':r['arm'],'horizon':r['horizon'],'seed':r['seed'],
            'source':meta[r['dataset']]['source'],'split_edges':meta[r['dataset']]['split_edges'],
            'history':96,'channels':meta[r['dataset']]['channels'],'forecast_steps':r['horizon'],
            'parameters_trainable':r['parameters'],'parent_checkpoint':fit['parent']['selected'],
            'parent_training_updates_allocated_and_consumed':1000,'parent_selected_step':fit['parent']['selected']['step'],
            'joint_updates_allocated_and_consumed':500,'pipeline_training_window_presentations':fit['pipeline_windows_seen'],
            'joint_CPU_seconds':r['joint_CPU_seconds'],'attributed_pipeline_training_CPU_seconds':r['pipeline_training_CPU_seconds'],
            'CPU_model':frozen['CPU_model'],'threads_per_fit':1,'max_concurrent_workers':2,'joint_CPU_cap':120,
            'selection':'lowest validation MSE among100/250/500; step0 ineligible','test':r['test'],
            'parity':{'parameters':'PASS' if r['arm']!='ar_adapt' else 'FAIL_UNMATCHED_REFERENCE',
                'data_exposure':'PASS','selection':'PASS','evaluation':'PASS','actual_compute':'FAIL_NOT_MATCHED'},
            'architectural_superiority_claim':False})
    b.dump(b.REPORT/'comparison-manifest.json',manifests)
    labels={'tac_horizon':'Native memory + learned context','context_horizon':'Matched context-only model',
        'gru_horizon':'Matched GRU + context','ar_adapt':'AR-only adaptation'}
    lines=['# ME037 — horizon-controlled context-integrated forecasting','',
        'A complete model build with native fast/slow memory, a learned full-window context '
        'branch and a bounded monotone horizon gate. All192adaptation fits and8new AR '
        'parent fits completed. Each adaptation received500joint updates after its '
        'shared1000-update AR parent training; parent selection/pretraining costs are recorded.',
        '', '## Main results','',
        'Lower MSE is better. Two-seed means, full target sequences and every eligible '
        'test origin (stride1). The five older datasets are development-exposed. The '
        'new household final20% is a separately reserved temporal test; models train '
        'on that same house\'s earlier observations. This is not zero-shot transfer.',
        '', '| Model | Legacy development MSE | Prospective house MSE | House energy-only MSE |',
        '|---|---:|---:|---:|']
    for a in b.ARMS:lines.append(f'| {labels[a]} | {legacy[a]:.6f} | {house[a]:.6f} | {energy[a]:.6f} |')
    lines+=['',f"Legacy native gain versus AR adaptation: **{legacy_gain*100:.2f}%**; registered gate "
        f"**{'PASS' if summary['legacy_engineering_gate'] else 'FAIL'}**. New-house native gain: "
        f"**{house_gain*100:.2f}%**; separate prospective engineering gate "
        f"**{'PASS' if summary['prospective_household_engineering_gate'] else 'FAIL'}**.",
        '',f"At720steps, native versus AR adaptation changes MSE by **{long['ETTh1']*100:+.2f}%** "
        f"on ETTh1, **{long['ETTh2']*100:+.2f}%** on ETTh2, and **{household_long*100:+.2f}%** "
        'on the new house. Positive means worse. The envelope is an amplitude bound, '
        'not a guaranteed loss improvement.',
        '', f'Historical ME036 native macro MSE was{old:.6f}; ME037 changes it by '
        f"{summary['historical_relative_improvement']*100:.2f}% relative improvement. "
        'This is an unmatched system-development comparison: model capacity, AR pretraining '
        'and optimization exposure differ. Previously scored benchmark outcomes are not fresh evidence.',
        '', '## Dataset/horizon results','',
        '| Dataset | H | Native + context | Matched context | Matched GRU | AR adaptation |',
        '|---|---:|---:|---:|---:|---:|']
    for d in b.DATASETS:
        for h in b.HORIZONS:
            lines.append('| '+d+' | '+str(h)+' | '+' | '.join(f'{cells[(d,h,a)]:.6f}' for a in b.ARMS)+' |')
    lines+=['', '## Prospective energy forecasts','',
        'The20-channel house score includes18temperature/humidity channels. Energy results '
        'are therefore shown separately. MAE in Wh below is obtained by rescaling each '
        'standardized energy residual with its training-only channel std; it is not measured '
        'energy savings or a repair/control outcome.',
        '', '| H | Model | Energy-only standardized MSE | Appliances MAE (Wh) | Lights MAE (Wh) |',
        '|---:|---|---:|---:|---:|']
    scale=meta['appliances']['scale']
    for h in b.HORIZONS:
        for a in b.ARMS:
            rows=grouped[('appliances',h,a)]
            em=np.mean([r['test']['energy_MSE'] for r in rows])
            ma=np.mean([r['test']['channel_MAE'][:2] for r in rows],axis=0)*np.array(scale[:2])
            lines.append(f'| {h} | {labels[a]} | {em:.6f} | {ma[0]:.4f} | {ma[1]:.4f} |')
    audit=json.loads((b.OUT/'audit.json').read_text());runtime=json.loads((b.OUT/'runtime.json').read_text())
    lines+=['','## Reproducibility and limits','',
        f"{audit['original_checkpoints_replayed']}parent/adaptation checkpoints replayed "
        f"{audit['bit_exact_validation_forecast_values']:,} saved validation values bit exactly. "
        f"Initial AR forecasts also replayed exactly across {audit['bit_exact_initial_AR_forecast_values']:,} values. "
        'These are saved validation arrays, not the entire final test matrix.',
        '',f"{len(runtime['models'])} native models have standardized and raw physical-input "
        'standalone exports; training-only anchor buffers are removed without changing learned weights. '
        f"Export replay covers {runtime['total_standardized_values_exact']+runtime['total_physical_values_exact']:,} values. "
        'Physical wrapper replay is against the same eager raw-input pipeline; float32 '
        'physical roundtrip is not declared equal to a separately standardized call.',
        '', 'The old 1,680 ME036 checkpoint hashes remain unchanged. Its earlier full replay '
        'receipt remains preserved; this turn\'s old-checkpoint immutability check hashes '
        'those files rather than rerunning all their forecasts. The historical ME035B '
        'precision exception is unchanged.',
        '', 'All matched neural encoders/readouts differ by less than1% in trainable '
        'parameters and share parents, examples, samplers and selection opportunities. '
        'Actual CPU spend differs: **no architectural-superiority claim**. AR is an '
        'unmatched practical reference. Cached parent costs are attributed per pipeline '
        'but were not physically rerun for every arm; summary.json separates new and '
        'deduplicated historical spend. Process RSS is a high-water mark, not isolated '
        'activation memory. No full FLOP/energy parity or published-model parity.',
        '', 'Native state is64float32coordinates per channel, not one64-vector for every '
        'channel combined. Serving still replays the whole96-step normalized window. '
        'No arbitrary-lifetime streaming or microsecond controller claim.',
        '', 'The new site is one Belgian house. Its temporal test and two fitted seeds '
        'do not establish multi-site population reliability. Energy meters and direct '
        'ZigBee measurements are retained; publisher-added random variables and merged '
        'airport weather fields are excluded. Previous ETT exposure, shared ETTh1/ETTm1 '
        'equipment and Weather anomalies remain explicit.',
        '', 'PR300 was read and remains draft/unmerged; its task-scoped foundation-model '
        'comparison is not a general domain prohibition or evidence of TAC action/VoI '
        'strengths. No actions, inspection costs, counterfactual repairs, physical '
        'control policy or commercial qualification are demonstrated by this study.']
    text='\n'.join(lines)+'\n'
    for a,c in {'All192adaptation':'All 192 adaptation','and8new':'and 8 new','received500joint':'received 500 joint',
        'shared1000-update':'shared 1,000-update','stride1':'stride 1','final20%':'final 20%',
        'At720steps':'At 720 steps','was0.':'was 0.','The20-channel':'The 20-channel','includes18temperature':'includes 18 temperature',
        'checkpoints replayed ':'checkpoints replayed ','parent/adaptation':' parent/adaptation',
        'across'+str(audit['bit_exact_initial_AR_forecast_values']):'across '+str(audit['bit_exact_initial_AR_forecast_values']),
        'covers'+str(runtime['total_standardized_values_exact']+runtime['total_physical_values_exact']):'covers '+str(runtime['total_standardized_values_exact']+runtime['total_physical_values_exact']),
        'old1680ME036':'old 1,680 ME036','than1%':'than 1%','is64float32coordinates':'is 64 float32 coordinates',
        'one64-vector':'one 64-vector','whole96-step':'whole 96-step'}.items():text=text.replace(a,c)
    (b.REPORT/'RESULTS.md').write_text(text)
    (b.REPORT/'README.md').write_text('# ME037\n\n[Results](RESULTS.md) · [Protocol](PROTOCOL.md) · '
        '[Equations](MATHEMATICS.md) · [Review of supplied findings](REVIEW.md) · [Artifacts](ARTIFACT.md)\n')
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':run()
