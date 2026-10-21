#!/usr/bin/env python3
"""Publish the locked complete-model outcomes; no fitting or model selection."""
import argparse,json
from pathlib import Path
import numpy as np

def render(source,destination):
    root=Path(source);dest=Path(destination);dest.mkdir(parents=True,exist_ok=True)
    report=json.loads((root/'results.json').read_text());audit=json.loads((root/'audit.json').read_text())
    assert report['completed'] and audit['status']=='PASS'
    lines=['# ME044: separate multiscale native innovation decoder','',
        'Complete architecture-first training on actual measurements. Historical development only; no newly sealed test, merge or promotion.','',
        'The model retains the original context predictor and gives native memory a separate bounded forecast decoder. Six observed-token fast/slow/associative updates predict16 spline/seasonal coefficients. Both cadence priors and forecast basis are shared with equally sized context and GRU competitors. Native expert learning occurs every fourth update; inference always executes the expert.','',
        '**80 complete downstream fits +8 new AR parent fits**. Three capacity-matched systems and two unmatched practical references. Fixed1000-update and separate10optimizer-CPU-second training lanes; two seeds, horizons96/720, two measured domains. No post-score changes or early stopping.','']
    for domain,data in report['domains'].items():
        lines += [f'## {domain}: registered primary MSE','',
            ('Three measured power zones.' if domain=='tetouan' else 'Measured oil temperature.')+' Train-only standardized units, lower better. Do not compare raw scores across domains.','',
            '| Complete system | 1000 updates | 10 optimizer CPU seconds | Mean CPU at1000 updates |',
            '|---|---:|---:|---:|']
        for arm in data['evaluation']['exposure']:
            ex=data['evaluation']['exposure'][arm];cp=data['evaluation']['cpu'][arm]
            lines.append(f"| {arm} | {ex['primary_MSE']:.6f} | {cp['primary_MSE']:.6f} | {ex['CPU']:.3f}s |")
        lines += ['', 'Original fixed-update checkpoints and per-horizon/per-seed scores are retained in results.json. Matched systems have active parameter gaps<0.04%; original_context and previous_scheduled are deliberately unmatched practical references.','',
            '| Registered native gate | Result |','|---|---|']
        for name,value in data['evaluation']['native_gates'].items():lines.append(f"| {name} | {'PASS' if value else 'FAIL'} |")
        lines.append('')
    lines += ['## Original model verification','',
        f"- {audit['models']} original optimizers/checkpoints checked; {audit['forecast_scalars_bit_exact']:,} saved full-forecast values replay bit exactly.",
        f"- {audit['cheap_trajectory_checks_exact']} actual fixed-update state comparisons confirm every shared cheap-path coefficient remains bit-identical to original_context. Cheap forecasts, actual truths and trust/router arrays also replay exactly.",
        '- Independent float64 MSE/MAE recomputation uses canonical C-layout reductions and the preregistered epsilon/operation-count bound. No forecast replay tolerance relaxed. Original ME043 strict reduction-equality failure and ME035B ridge-equivalence exception remain unchanged.',
        '- Per-parameter optimizer clocks verify native/context/GRU expert updates every fourth step and previous scheduled expert/router every eighth step. Actual process/wall time, samples, parent cost and peak process RSS are recorded.','',
        '## Actual full-window inference cost','',
        'Batch1, original first DEV window,5warmups then5x20calls; geometric mean of per-checkpoint median CPU across horizons/seeds. Includes normalization, full-history AR, context, native/GRU computation, basis decode and all actual outputs. Process CPU is not a streaming update claim.','',
        '| Domain / training lane | System | Full forecast CPU ms |','|---|---|---:|']
    timing=json.loads((root/'latency.json').read_text())['rows']
    for domain in report['domains']:
        for lane in ('exposure','cpu'):
            for arm in report['domains'][domain]['evaluation'][lane]:
                r=[v['median_CPU_seconds'] for v in timing if v['domain']==domain and v['lane']==lane and v['arm']==arm]
                assert len(r)==4
                mean=float(np.exp(np.mean(np.log(r))))*1000
                lines.append(f'| {domain} / {lane} | {arm} | {mean:.3f} |')
    inference=json.loads((root/'inference.json').read_text())
    lines += ['', '## Optional compiled native execution','',
        'A separately registered post-fit execution optimization compiles only the native branch. Original learned weights and all other components remain unchanged; no new fitting or predictive accuracy result.', '',
        f"Recorded result: {inference['status']}. Target >=1.25x geometric-mean full-forecast speedup; exact-output guard compulsory. Full details in [inference.json](inference.json).", '']
    if 'geometric_mean_full_forecast_speedup' in inference:
        lines += [f"All{inference['models']} native checkpoints replay{inference['forecast_scalars_exact']:,} full forecast values exactly after compilation. Paired full-window CPU speedup **{inference['geometric_mean_full_forecast_speedup']:.3f}x**, target **{'PASS' if inference['speed_target_1_25_pass'] else 'FAIL'}**. Compilation time is separate, default backend unchanged.", '']
    lines += ['', '## Comparison manifest and interpretation','',
        '- Parameter parity PASS only within the three registered pyramid systems. All modes/priors/readout information match there. Practical references are unmatched; no dummy weights.',
        '- Data/exposure PASS within fixed1000-update matched systems: identical observed source rows, target windows, sampler sequence and expert schedule. Actual CPU differs. Compute-lane budget PASS with final-step overshoot if the recorded cap was reached; sample/expert exposures differ. Full three-way parity FAIL: **NO ARCHITECTURAL SUPERIORITY CLAIM**.',
        '- Same fixed endpoint and parent selection rules, one model configuration per arm, both seeds and horizons preserved. No result-dependent target switching. Whole-study design follows repeated historical DEV exposure, not independent new-site evidence.',
        '- Cheap-trajectory retention is a software/optimization guarantee, not proof that native state or its additional residual improves future prediction. Bounded corrections do not establish safety or calibration.',
        '- Modular stop-gradient training preserves cheap-path weights; the full expert objective does not backpropagate through that cheap encoder. Inference always runs the expert, unlike ME043 hard routing. No persistent streaming, adaptive System2, causal action, counterfactual, commercial or foundation-model claim.',
        '- All original weights, optimizers/RNG, losses, source hashes, genuine predicted/observed arrays, timing and negative gates are retained in repository-owned lossless indexed archives. Archive restoration and CI establish their actual delivery/replay status separately.','']
    (dest/'RESULTS.md').write_text('\n'.join(lines))
    for name in ('results.json','audit.json','latency.json','inference.json'):(dest/name).write_bytes((root/name).read_bytes())

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('source');p.add_argument('destination');a=p.parse_args();render(a.source,a.destination)
