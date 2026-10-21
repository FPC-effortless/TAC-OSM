#!/usr/bin/env python3
"""Render recorded complete-model results without retraining or selection."""
import argparse,json
from pathlib import Path
import numpy as np

def render(a,b,destination):
    a=Path(a);b=Path(b);dest=Path(destination);dest.mkdir(parents=True,exist_ok=True)
    reports=[json.loads((p/'results.json').read_text()) for p in (a,b)]
    assert all(r['completed'] for r in reports)
    lines=['# ME043 / ME043B: complete compute-adaptive architecture results','',
        'Original authentic-data training; historical development only. No newly sealed test, merge or promotion.',
        '', 'A: 96 complete downstream fits. B: 64 separately registered follow-up fits after A failure. Eight original train-only AR parents are reused by B, not refitted.',
        '', 'A compresses context and trains both exits densely. B retains the full original contextual predictor and trains native memory/router on every eighth update. Both perform real hard conditional native dispatch at inference. These are new trained architectures, not equivalent execution changes.', '']
    for root,r in zip((a,b),reports):
        lines += [f"## {r['study']}", '']
        for domain,data in r['domains'].items():
            target='three-zone power' if domain=='tetouan' else 'oil temperature'
            lines += [f'### {domain}: {target} standardized historical DEV MSE','',
                '| Complete model | 250 updates | 5 optimizer CPU seconds | Native fraction, updates / CPU |',
                '|---|---:|---:|---:|']
            for arm in data['evaluation']['exposure']:
                ex=data['evaluation']['exposure'][arm];cp=data['evaluation']['cpu'][arm]
                frac='n/a' if ex['native_fraction'] is None else f"{ex['native_fraction']:.3f} / {cp['native_fraction']:.3f}"
                lines.append(f"| {arm} | {ex['primary_MSE']:.6f} | {cp['primary_MSE']:.6f} | {frac} |")
            lines += ['', 'Locked per-candidate gates:','', '| Model | >=1% accuracy gain | Horizon safety | Equal-CPU accuracy | All gates |', '|---|---|---|---|---|']
            for arm,g in data['evaluation']['gates'].items():
                lines.append('| '+arm+' | '+' | '.join('PASS' if g[k] else 'FAIL' for k in
                    ('one_percent_accuracy','no_large_horizon_regression','equal_CPU_accuracy','all'))+' |')
            lines.append('')
        replay=json.loads((root/'replay.json').read_text());audit=json.loads((root/'metric-audit.json').read_text())
        lines += [f"Replay: **{replay['models']} original model checkpoints, {replay['forecast_scalars_bit_exact']:,} forecast values bit exact**, including authentic truth and routing-array verification. Independent saved-array float64 metric recomputation: {audit['original_models']} models, maximum float64 reduction discrepancy {audit['max_metric_absolute_error']} (checked against an operation-count-derived float64 bound). No new fits or test rows in either audit.", '']
    la=json.loads((a/'latency.json').read_text())['rows'];lb=json.loads((b/'latency.json').read_text())['rows']
    lines += ['## Measured batch1 complete forecast CPU latency','',
        'Each datum includes96-observation normalization, full-history AR, context encoder, router/dispatch and native/readout work where actually selected. Five warmups then5x20calls, single Torch CPU thread. Table geometric-means per-checkpoint medians across both horizons/seeds. Not microsecond streaming, training speed or persistent state reuse.', '',
        '| Study / domain / lane | Model | Full forecast CPU ms |','|---|---|---:|']
    for study,rows in [('A',la),('B',lb)]:
        for domain in ('tetouan','etth1'):
            for lane in ('exposure','cpu'):
                arms=sorted({v['arm'] for v in rows})
                for arm in arms:
                    subset=[v['median_CPU_seconds'] for v in rows if v['arm']==arm and v['domain']==domain and v['lane']==lane]
                    assert len(subset)==4
                    val=float(np.exp(np.mean(np.log(subset))))*1000
                    lines.append(f'| {study} / {domain} / {lane} | {arm} | {val:.3f} |')
    secondary={}
    for tag,report,rows,context,native in [('A',reports[0],la,'context_cross','innovation_tac'),
                                          ('B',reports[1],lb,'wide_context','wide_native')]:
        secondary[tag]={}
        for domain,data in report['domains'].items():
            secondary[tag][domain]={}
            for lane in ('exposure','cpu'):
                scores=data['evaluation'][lane]
                mean_latency=lambda arm:float(np.exp(np.mean(np.log([r['median_CPU_seconds'] for r in rows
                    if r['arm']==arm and r['domain']==domain and r['lane']==lane]))))
                secondary[tag][domain][lane]={}
                for arm in data['evaluation']['gates']:
                    speed=mean_latency(native)/mean_latency(arm)
                    accuracy=scores[arm]['primary_MSE']<=scores[context]['primary_MSE']
                    secondary[tag][domain][lane][arm]={'speedup_vs_native':speed,
                        'MSE_no_worse_than_context':accuracy,'speed_at_least_1_25':speed>=1.25,
                        'both_secondary_targets':bool(accuracy and speed>=1.25)}
    (dest/'secondary-latency-gates.json').write_text(json.dumps(secondary,indent=2,sort_keys=True)+'\n')
    lines += ['', 'The registered secondary speed/accuracy checks are retained in [secondary-latency-gates.json](secondary-latency-gates.json). Faster cheap-only routes are not counted as accuracy wins.', '']
    lines += ['', '## Comparison contract and remaining limits','',
        '- All active parameter counts within each registered arm matrix differ by <1%; counts and every original model are retained. B has more capacity than A, so B-versus-A improvements are unmatched engineering observations.',
        '- Exposure lanes match250 batch4 observed training windows and source-index stream per seed. Actual CPU differs. CPU lanes match5 optimizer-CPU-second budgets with final-step overshoot; actual examples differ. B additionally varies native expert-update exposure by design. Full three-way parity FAIL: **NO ARCHITECTURAL SUPERIORITY CLAIM**.',
        '- All evaluated endpoints fixed before execution; three A and two B cost candidates are registered. B is explicitly post-A development, with repeated historical-DEV access, not an independent confirmation. No validation-selected final-test opening.',
        '- Per-fit process CPU and process high-water RSS are recorded. Per-fit elapsed wall time was omitted by the original runner despite the A protocol; it is not fabricated. High-water RSS is not isolated model activation memory. No FLOP/energy parity established.',
        '- A route collapsing to cheap-only is preserved. Forecast gains from the complete system cannot be attributed to native memory without later mechanism studies. Routing probability is not a physical safety/decision policy.',
        '- The first independent metric audit demanded bitwise float64 reduction equality and failed; those logs remain. Original strided forecast arrays and reloaded contiguous arrays can reduce in different orders. The corrected metric check uses a float64 operation-count gamma bound, derived from epsilon and target count rather than the observed discrepancy. Original float32 forecasts and router arrays still replay bit exactly; no forecast tolerance changed.',
        '- Old ME035B strict ridge-equivalence tolerance failure remains unchanged. Current local exact checkpoint replay does not guarantee identical predictions on every CPU/library combination; separate CI replay reports its actual result.',
        '- Default runtime and earlier model choices are unchanged. No action benefit, counterfactual qualification, adaptive System2, arbitrary-lifetime streaming or commercial readiness is established.', '']
    (dest/'RESULTS.md').write_text('\n'.join(lines))
    for name,root in [('A',a),('B',b)]:
        for file in ('results.json','replay.json','metric-audit.json','latency.json'):
            (dest/f'{name}-{file}').write_bytes((root/file).read_bytes())

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--a',required=True);p.add_argument('--b',required=True)
    p.add_argument('--destination',required=True);x=p.parse_args();render(x.a,x.b,x.destination)
