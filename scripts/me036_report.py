"""Build ME036's factual research ledger from completed execution records."""
import json, shutil
from collections import defaultdict
from pathlib import Path
import numpy as np
import scripts.me036_research as b


def run():
    out=b.ROOT/'results/me036b';r=json.loads((out/'results.json').read_text())
    frozen=json.loads((out/'frozen.json').read_text());report=b.REPORT
    rows=r['models']+r.get('ridge',[])
    groups=defaultdict(list)
    for row in rows:groups[(row['dataset'],row['horizon'],row['arm'])].append(row['test']['MSE'])
    cell={k:float(np.mean(v)) for k,v in groups.items()}
    arms=(*b.ARMS,'ridge');overall={a:float(np.mean([v for (d,h,arm),v in cell.items() if arm==a])) for a in arms}
    best=min(v for a,v in overall.items() if a!='tac_patch')
    improvement=1-overall['tac_patch']/overall['ar'];gap=overall['tac_patch']/best-1
    for filename in ('frozen.json','results.json','replay.json','runtime.json'):
        shutil.copyfile(out/filename,report/filename)
    shutil.copyfile(b.OUT/'frozen.json',report/'initial-frozen.json')
    shutil.copyfile(b.OUT/'data.json',report/'data.json')
    summary={'overall_MSE':overall,'AR_relative_improvement':improvement,'best_baseline_relative_gap':gap,
        'success_gate':bool(improvement>=.02 and gap<=.05),'neural_models':len(frozen['fits']),
        'failed_models':[r['key'] for r in frozen['fits'] if r['status']!='PASS'],
        'ridge_models':len(frozen['ridge']),
        'total_neural_training_CPU_seconds':sum(r.get('charged_training_CPU_seconds',0) for r in frozen['fits']),
        'total_ridge_training_CPU_seconds':sum(r['charged_training_CPU_seconds'] for r in frozen['ridge']),
        'total_validation_CPU_seconds':sum(e['validation']['CPU_seconds'] for r in frozen['fits'] for e in r['endpoints']),
        'total_neural_test_CPU_seconds':sum(r['test']['CPU_seconds'] for r in rows if r['arm']!='ridge'),
        'total_ridge_test_CPU_seconds':sum(r['test']['CPU_seconds'] for r in rows if r['arm']=='ridge'),
        'total_scored_neural_scalar_predictions':sum(r['test']['scalar_targets'] for r in rows if r['arm']!='ridge'),
        'macro_aggregation':'equal weight dataset/horizon cells; mean fitted-seed MSE within cell',
        'test_stride':1,'architecture_superiority_qualified':False,'paper_parity_qualified':False}
    b.dump(report/'summary.json',summary)
    manifest=[]
    for row in rows:
        rid=row['arm']=='ridge';matched=row['arm'] in ('tac_patch','context','gru')
        manifest.append({**{k:row[k] for k in ('dataset','horizon','arm','seed','parameters','charged_training_CPU_seconds')},
            'source_sha256':json.loads((report/'sources.json').read_text())[row['dataset']+'.csv']['sha256'],
            'history':96,'full_output_steps':row['horizon'],'updates':None if rid else 1000,
            'training_windows':None if rid else 4000,'selection':'none' if rid else 'validation MSE over6endpoints',
            'parity':{'parameters':'PASS' if matched else 'FAIL_UNMATCHED_REFERENCE',
                'data_exposure':'FAIL_RIDGE_DIFFERENT_EXPOSURE' if rid else 'PASS',
                'actual_compute':'FAIL_NOT_MATCHED','selection':'UNMATCHED_RIDGE' if rid else 'PASS',
                'evaluation':'PASS'},'architectural_superiority_claim':False})
    b.dump(report/'comparison-manifest.json',manifest)
    lines=['# ME036 — full-horizon patched-native forecasting results','',
        'Architecture-first real-data development on five forecasting benchmarks. '
        'The original100-update stage was extended prospectively to1000updates before '
        'any final test score was opened. All arms received the same continuation opportunity.',
        '',f"{len(frozen['fits'])} neural trajectories (two seeds), {len(frozen['ridge'])} fitted ridge references. "
        f"Failed trajectories: {len(summary['failed_models'])}. No model promoted or merged.",
        '', '## Full-test results','',
        'Train-only standardized MSE; all output steps/channels; every eligible test anchor '
        '(stride1). Separate models for horizons96/192/336/720. Macro mean gives each of '
        'the20dataset/horizon cells equal weight. Two-seed means are descriptive; no '
        'independent retraining interval or superiority test is claimed.','',
        '| Model | Macro MSE | Native parameter parity |','|---|---:|---|']
    labels={'tac_patch':'Patched native TAC','ar':'AR only','context':'AR + context correction',
        'gru':'AR + GRU correction','dlinear':'Compact shared DLinear-style','patchtst_small':'Compact PatchTST-style',
        'itransformer_small':'Compact iTransformer-style','ridge':'Shared-channel ridge'}
    for a in arms:lines.append(f"| {labels[a]} | {overall[a]:.6f} | {'within1%' if a in ('tac_patch','context','gru') else 'unmatched'} |")
    lines+=['',f"TAC improvement over AR: **{improvement*100:.2f}%**. Gap to the best macro baseline: "
        f"**{gap*100:.2f}%**. Registered practical gate: **{'PASS' if summary['success_gate'] else 'FAIL'}**.",
        '', '## Dataset/horizon detail','', '| Dataset | H | TAC | AR | Context | GRU | DLinear | PatchTST | iTransformer | Ridge |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for d in b.DATASETS:
        for h in b.HORIZONS:lines.append('| '+d+' | '+str(h)+' | '+' | '.join(f'{cell[(d,h,a)]:.6f}' for a in arms)+' |')
    lines+=['','## Engineering and comparison limits','',
        'TAC uses12ordered8-step patches, learned fast32/slow32 native dynamics, residual '
        'slow updates, trainable multiscale rates, a SwiGLU/32-basis correction and joint '
        'channel-shared linear forecast path. All models start from scratch; ridge is '
        'a separately fitted reference, not a TAC initializer.',
        '', 'TAC/context/GRU parameter gaps are below1%; all other model sizes differ. '
        'Examples, samplers,1000-update ceilings and selection privileges match for '
        'the neural models. Actual CPU spend differs: the strict compute-parity gate '
        '**fails**, so there is **no architectural-superiority claim**.',
        '', f"Actual neural fitting CPU: {summary['total_neural_training_CPU_seconds']:.2f}s; "
        f"validation: {summary['total_validation_CPU_seconds']:.2f}s; "
        f"neural final scoring: {summary['total_neural_test_CPU_seconds']:.2f}s. "
        f"Ridge fitting: {summary['total_ridge_training_CPU_seconds']:.2f}s. "
        'Initialization/optimizer/update costs are charged; checkpoint I/O and export '
        'are separate. RSS in fit records is process high-water RSS, not isolated '
        'per-model activation memory. CPU timing is process time, single-thread.',
        '', 'Standard split/horizon/stride definitions are now used, but compact '
        'independent baseline implementations, lookback96,1000-update training and '
        'stride24 validation differ from published recipes. These scores do not '
        'reproduce the published DLinear/PatchTST/iTransformer tables. Weather and '
        'Electricity are authenticated THUML benchmark copies, not raw instrument files. '
        'Earlier ETT period exposure remains declared; no pristine-population claim. '
        'Weather contains81training-only-9999codes, one duplicate timestamp and one100-minute '
        'gap; original rows remain unchanged under the registered benchmark convention. '
        'Its test targets contain no-9999codes, but training normalization includes them. '
        'Weather results are sensitive to this source-quality limitation.',
        '', 'Serving still re-encodes the entire normalized96-step window. Twelve patch '
        'updates do not prove an eightfold runtime speedup or constant-time streaming. '
        'Runtime JSON records actual B1 latency and weight/input/output/state bytes. '
        'Standardized exports and physical-input exports are both supplied; physical exports '
        'include training mean/scale buffers and inversion. data.json records those statistics. The old ME035B1e-6 equivalence failure remains recorded '
        'and unchanged; new linear numerical tests use a prospective gamma_n bound.',
        '', 'See summary.json, results.json (per seed), comparison-manifest.json, '
        'frozen.json, replay.json and runtime.json for machine-readable evidence. '
        'Final metrics stream without retaining billions of forecast values; checkpoint '
        'audits replay the saved validation probes, not the entire test metric array.']
    (report/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    (report/'README.md').write_text('# ME036\n\n[Results](RESULTS.md) · [Registration](PROTOCOL.md) · '
        '[Prospective continuation](CONTINUATION.md) · [Equations](MATHEMATICS.md) · [Artifacts](ARTIFACT.md)\n')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':run()
