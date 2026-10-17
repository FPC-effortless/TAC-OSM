"""Report frozen DEV056/057 results without modifying acceptance rules."""
import argparse
import csv
import json
from pathlib import Path
import shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from run_learned_stack_dev055 import ROOT,write


def learning_plot(directory,p,reports):
    curves=[]
    for seed,arms in reports.items():
        for arm,record in arms.items():
            for point in record['training']['curves']:curves.append({'seed':seed,'arm':arm,**point})
    with (directory/'learning-curves.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=['seed','arm','update','train','validation'])
        writer.writeheader();writer.writerows(curves)
    fig,axes=plt.subplots(1,2,figsize=(11,4),sharey=True)
    for arm in p['arms']:
        rows=[[r for r in curves if r['seed']==str(s) and r['arm']==arm] for s in p['model_seeds']]
        xs=[r['update'] for r in rows[0]]
        for axis,metric in zip(axes,('train','validation')):
            values=np.array([[r[metric] for r in row] for row in rows])
            line,=axis.plot(xs,values.mean(0),label=arm)
            axis.fill_between(xs,values.min(0),values.max(0),color=line.get_color(),alpha=.12)
            axis.set(title=metric.capitalize(),xlabel='Optimizer updates');axis.grid(alpha=.2)
    axes[0].set_ylabel('Post-action fault BCE (nats)');axes[1].legend(fontsize=8)
    fig.suptitle('DEV-056: three-seed mean/range; fixed 1,024 updates, no convergence claim')
    fig.tight_layout();fig.savefig(directory/'learning-curves.png',dpi=160);plt.close(fig)


def report56(source):
    source=Path(source);out=ROOT/'research/dev056'
    p=json.loads((out/'protocol.json').read_text());summary=json.loads((source/'summary.json').read_text())
    reports=json.loads((source/'reports.json').read_text());controls=json.loads((source/'controls.json').read_text())
    shutil.copyfile(source/'summary.json',out/'summary.json');shutil.copyfile(source/'manifest.json',out/'raw-evidence-manifest.json')
    learning_plot(out,p,reports)
    selected=summary['validation_selected_control'];cells=[]
    for variant in ('mtsk','masked_mtsk'):
        for seed in p['model_seeds']:
            for family in ('slow','fast'):
                a=reports[str(seed)][variant]['evaluation'][family]
                for ref in ('gru_matched',selected):
                    b=controls[ref][family] if ref in controls else reports[str(seed)][ref]['evaluation'][family]
                    cells.append({'variant':variant,'seed':seed,'family':family,'reference':ref,
                                  **{m:a[m]-b[m] for m in ('cost','fault_exposure','final_faults')},
                                  'fault_guard':a['fault_exposure']<=b['fault_exposure'] and a['final_faults']<=b['final_faults']})
    write(out/'reliability-cells.json',cells)
    masks=[];diagnostics={}
    for seed in p['model_seeds']:
        report=reports[str(seed)]['masked_mtsk'];evaluations=report['evaluation'].values()
        possible=sum(e['potential_memory_updates'] for e in evaluations)
        actual=sum(e['memory_backend_sample_updates'] for e in evaluations)
        masks.append({'seed':seed,'possible_updates':possible,'backend_sample_updates':actual,'skipped':possible-actual,
                      'gate_weight_change_l1':report['training']['write_gate_weight_change_l1']})
    write(out/'mask-diagnostics.json',masks)
    for arm in p['arms']:
        rows=[reports[str(s)][arm]['training'] for s in p['model_seeds']]
        diagnostics[arm]={'trainable_parameters':rows[0]['trainable_parameters'],
                          'stored_parameter_bytes':rows[0]['stored_parameter_bytes'],
                          'retained_state_bytes':rows[0]['retained_state_bytes'],
                          'mean_training_seconds':float(np.mean([r['seconds'] for r in rows])),
                          'mean_final_validation_bce':float(np.mean([r['curves'][-1]['validation'] for r in rows]))}
    write(out/'fit-diagnostics.json',diagnostics)
    retired=summary['decision']=='RETIRE_MEMORY_NOVELTY_PURSUIT'
    title='DEV-056: deadline reached; retire MTSK-specific superiority pursuit' if retired else 'DEV-056: development candidate requires fresh replication'
    lines=[f'# {title}','',
           f'The complete twelve-fit matrix finished. The registered decision is **{summary["decision"]}**. '
           'This decision is limited to this research program; it does not assert that MTSK can never help any task, '
           'and it does not retire TAC-OSM, learned world models or selective computation.','',
           '## Cost and reliability','',
           'All costs are per two-entity world over 24 actions, including sixteen paid randomized acquisition actions '
           'and eight productive actions. Fault exposure accumulates unresolved faults over ticks; final faults '
           'are unresolved machines at the final tick. These quantities are different. The 384 independent worlds '
           'are shared across initialization seeds; seed-world rows are not independent samples.','',
           '| Arm | Total cost/world | Fault exposure/world | Final faults/world |',
           '| --- | ---: | ---: | ---: |']
    for arm,m in sorted(summary['metrics'].items(),key=lambda x:x[1]['cost']):
        lines.append(f'| {arm} | {m["cost"]:.4f} | {m["fault_exposure"]:.4f} | {m["final_faults"]:.4f} |')
    lines+=['',f'The adaptive comparator **{selected}** was selected on a separate 128-world validation stream before '
            'held-out outcomes were generated. Both conventional controls remain reported. The fitted HMM also '
            'uses privileged training regime labels; the neural GRU/MTSK comparisons have matched supervision.','',
            '| Variant minus comparator | Cost difference | Simultaneous interval | Fault guard |',
            '| --- | ---: | --- | --- |']
    for variant,result in summary['variants'].items():
        for ref,delta in result['comparisons'].items():
            lo,hi=delta['interval']
            lines.append(f'| {variant} − {ref} | {delta["mean"]:+.4f} | [{lo:+.4f}, {hi:+.4f}] | {delta["fault_guard"]} |')
    lines+=['','Intervals use the preregistered four-comparison Bonferroni familywise95% procedure, with paired '
            'world-cluster and global initialization-seed resampling. [Every seed/hazard fault-guard cell]'
            '(../research/dev056/reliability-cells.json) is retained. No bad cell or variant was dropped.','',
            '## Public-information headroom','',
            'The oracle receives only public sensor/inspection receipts and the declared hazard. It filters the '
            'joint eight-state distribution using known probability laws, rather than reading sampled physical '
            'faults or the hidden regime. Its update matches explicit eight-state Bayes enumeration in a disjoint '
            'test. It remains a one-step known-model policy, not a long-horizon Bayes optimum.','',
            '| Reference minus public oracle | Saving/entity | Paired95% interval | Reliability feasible | Adequate |',
            '| --- | ---: | --- | --- | --- |']
    for ref,d in summary['headroom'].items():
        lo,hi=d['interval'];lines.append(f'| {ref} | {d["mean"]:.4f} | [{lo:.4f}, {hi:.4f}] | {d["reliability_feasible"]} | {d["adequate"]} |')
    lines+=['', 'Adequate headroom requires a saving of at least0.1/entity, a positive lower interval, and no '
            'increased aggregate fault exposure or final faults for either hazard versus both comparators. '
            f'The complete headroom gate is **{summary["adequate_public_headroom"]}**. Insufficient headroom '
            'would limit the benchmark but cannot postpone the registered retirement deadline.','',
            '## A separately identified mask','',
            'This study implements a genuine write mask. When closed, both memory and recurrent responsibilities '
            'are held exactly, including decay; inference skips the backend for closed samples. A forced-open '
            'mask matches unmasked recurrence with shared weights. The gate has a hard forward choice and biased '
            'straight-through training derivative. This is distinct from CDL relevance or CASM operator routing.','',
            '| Initialization | Potential writes | Backend updates | Skipped | Gate weight change L1 |',
            '| --- | ---: | ---: | ---: | ---: |']
    for r in masks:lines.append(f'| {r["seed"]} | {r["possible_updates"]} | {r["backend_sample_updates"]} | {r["skipped"]} | {r["gate_weight_change_l1"]:.4f} |')
    if all(m['skipped']==0 for m in masks):
        lines+=['','**The trained gates skipped zero backend updates in this evaluation.** The mask mechanism '
                'is implemented and checked, but this recipe did not learn selective writing. No masked-memory '
                'sparsity or efficiency result follows.']
    lines+=['','## Exposure and retained state','',
            '![All-arm learning curves](../research/dev056/learning-curves.png)','',
            'All fits use final checkpoints after1,024 updates, with the same public histories, factual outcome '
            'supervision and structural decoder. Curves include each fit; shading is the three-seed range, '
            'not a confidence interval. No plateau/global-convergence claim.','',
            '| Arm | Used trainable parameters | Retained state bytes | Final validation BCE |',
            '| --- | ---: | ---: | ---: |']
    for arm,d in diagnostics.items():lines.append(f'| {arm} | {d["trainable_parameters"]} | {d["retained_state_bytes"]} | {d["mean_final_validation_bce"]:.5f} |')
    lines+=['','Unused inherited paths are frozen. Matching counts actual used trainable parameters, within2% '
            'for both MTSK variants versus the GRU, without padding. Parameter/state accounting and training '
            'timings are diagnostics; no speed or storage superiority is asserted.','',
            '## Provenance and reproduction','',
            f'Registration `3f39f8d`; main training source `{summary["source"]}`. Main elapsed '
            f'{summary["elapsed_seconds"]:.2f} seconds. The protocol transparently records that implementation '
            'drafts/disjoint unit tests preceded registration after an initial shell failure; no main numerical '
            'outcomes preceded registration. This is preregistered development evidence, not confirmation.','',
            'The [audit](../research/dev056/audit.json) restores every checkpoint and independently reruns '
            'decisions in another process, then replays complete trajectories through pinned native Veritas with '
            '24-action budgets. Same author, shared generator and recorded effect tables limit independence. '
            'Unrepaired final outcomes are preserved. Native replay is not real-world verification.','',
            'Download [all checkpoints and raw evidence](../research/dev056/evidence.tar.gz), check '
            '[SHA256/provenance](../research/dev056/evidence-index.json), then extract into `results/`.','',
            '```bash','mkdir -p results','tar -xzf research/dev056/evidence.tar.gz -C results',
            '.venv/bin/python scripts/audit_public_memory_dev056.py --directory results/dev056-main --output results/dev056-replay-audit.json --veritas-root /path/to/pinned/veritas',
            '.venv/bin/python scripts/demo_public_memory_dev056.py --checkpoint results/dev056-main/7561-gru_matched.pt --events research/dev056/example-public-events.jsonl',
            '# Fresh registered reproduction requires a clean committed worktree:',
            '.venv/bin/python scripts/run_public_memory_dev056.py --output results/dev056-reproduction','```','',
            'Retirement ends MTSK-specific superiority tuning in this lane. Reuse of the implemented backend as '
            'a non-superior research component is still allowed. The next architectural question is acquisition '
            'and useful selective computation; [DEV-057](DEV-057-RESULTS.md) establishes reference costs, not '
            'learned superiority.']
    (ROOT/'docs/DEV-056-RESULTS.md').write_text('\n'.join(lines)+'\n')


def report57(source):
    source=Path(source);out=ROOT/'research/dev057'
    summary=json.loads((source/'summary.json').read_text());cells=json.loads((source/'cells.json').read_text())
    shutil.copyfile(source/'summary.json',out/'summary.json');shutil.copyfile(source/'audit.json',out/'audit.json')
    shutil.copyfile(source/'manifest.json',out/'raw-evidence-manifest.json')
    rows=[]
    for c in cells:
        rows.append({k:c[k] for k in ('seed','history','queries','arm','success','coverage','missing_facts','stale_facts',
                                    'payload_bytes','resident_container_bytes','amortized_declared_operations','build_seconds','query_seconds')})
    with (out/'cells.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    histories=sorted({c['history'] for c in cells});arms=list(dict.fromkeys(c['arm'] for c in cells))
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for arm in arms:
        rates=[np.mean([c['success'] for c in cells if c['history']==h and c['queries']==64 and c['arm']==arm]) for h in histories]
        work=[np.mean([c['amortized_declared_operations'] for c in cells if c['history']==h and c['queries']==64 and c['arm']==arm]) for h in histories]
        axes[0].plot(histories,rates,marker='o',label=arm);axes[1].plot(histories,work,marker='o',label=arm)
    for axis in axes:axis.set_xscale('log',base=4);axis.set_xlabel('Public history events');axis.grid(alpha=.2)
    axes[0].set_ylabel('Exact current-fact circuit success');axes[0].set_ylim(-.03,1.03)
    axes[1].set_ylabel('Declared API/record operations per query');axes[1].set_yscale('log');axes[1].legend(fontsize=7)
    fig.suptitle('DEV-057 controls: Q=64, three-seed mean; build and verification charged')
    fig.tight_layout();fig.savefig(out/'reference-curves.png',dpi=160);plt.close(fig)
    lines=['# DEV-057: acquisition references reproduced; learned C5 remains open','',
           '**All180 registered cells reproduce.** Full scan and hand-coded indexed acquisition preserve exact '
           'answers across every cell. Fixed-capacity scalar/hash stores do not preserve arbitrary history. '
           'These are baseline and instrument results; there is no learned challenger and no new TAC-OSM '
           'selective-computation superiority claim.','',
           'Each query requests the latest facts for eight distinct public entity keys, then executes seven '
           'public XOR nodes. Four independently shuffled write epochs create overwritten records. Missing '
           'inputs force abstention, rather than counting a lucky XOR guess as success. Keys and the circuit '
           'are supplied; there is no learned relevance relation or symbolic program discovery.','',
           '## Capability and work','',
           '![Reference capability/work curves](../research/dev057/reference-curves.png)','',
           'At64 queries per built history, exact success averaged over three seeds is:','',
           '| Arm | H64 | H256 | H1024 | H4096 |','| --- | ---: | ---: | ---: | ---: |']
    for arm in arms:
        rates=[np.mean([c['success'] for c in cells if c['history']==h and c['queries']==64 and c['arm']==arm]) for h in histories]
        lines.append('| '+arm+' | '+' | '.join(f'{x:.4f}' for x in rates)+' |')
    lines+=['','The index reads all H incoming writes during construction and performs eight logical record '
            'reads per query. Full scan reads H records per query. These record counts are distinct from address '
            'operations, maintenance writes, circuit operations and verification. All are reported, including '
            'build cost amortized over Q=1,8,64. Query prefixes share data and are not independent replications.','',
            'Scalar/direct-hash/two-way-hash state is exactly64 records ×3int32 =768bytes/6144bits per arm, '
            'including key/tick/bit and validity. There is no retained external history or lookup table for '
            'these three arms. Equal-bits means equal among these controls; it is **not** a matched-state '
            'comparison against MTSK or a learned model. Hash collisions are detected as missing keys.','',
            '## Storage and build trade-off','',
            'At H4096 and Q64, the first registered seed gives:','',
            '| Arm | Payload bytes | Resident container estimate | Declared operations/query |',
            '| --- | ---: | ---: | ---: |']
    for arm in arms:
        c=next(c for c in cells if c['seed']==7571 and c['history']==4096 and c['queries']==64 and c['arm']==arm)
        lines.append(f'| {arm} | {c["payload_bytes"]} | {c["resident_container_bytes"]} | {c["amortized_declared_operations"]:.2f} |')
    lines+=['','The unbounded index is not capacity-matched to the finite controls. Its payload holds latest '
            'records, but Python mapping overhead makes its resident container estimate larger than the full '
            'history array in this cell. Resident figures are recursive container estimates, not RSS or full '
            'allocator accounting. Work units are declared API/record operations, not CPU instructions; Python '
            'dictionary internals are explicitly unmeasured. Wall times are descriptive and do not prove an '
            'asymptotic or production-speed advantage. Verification reads synthetic gold only in the evaluator '
            'and is counted separately; it is not deployable ground-truth verification.','',
            '## Audit and next falsifier','',
            f'Training-free source `{summary["source"]}`; registration `3f39f8d`. '
            'The separate-process [audit](../research/dev057/audit.json) regenerated every exact answer, capacity '
            'and work count, excluding timing. Adversarial tests cover collisions, stale writes, absent keys '
            'and hidden extra state. [All cells](../research/dev057/cells.csv) and '
            '[complete evidence](../research/dev057/evidence.tar.gz) are preserved; see '
            '[archive SHA256](../research/dev057/evidence-index.json).','',
            '```bash','mkdir -p results','tar -xzf research/dev057/evidence.tar.gz -C results',
            '.venv/bin/python scripts/run_acquisition_controls_dev057.py --output results/dev057-main --audit','```','',
            'A future learned-acquisition experiment must add nontrivial relevance, match total state/index '
            'storage, charge every write/build/address/verification operation and evaluate actual learned '
            'CASM execution under equal answer quality. Public-key lookup controls do not satisfy those '
            'conditions. C5 remains open.']
    (ROOT/'docs/DEV-057-RESULTS.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--dev056');parser.add_argument('--dev057')
    args=parser.parse_args()
    if args.dev056:report56(args.dev056)
    if args.dev057:report57(args.dev057)
