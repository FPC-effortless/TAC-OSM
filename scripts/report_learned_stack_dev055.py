"""Descriptive DEV-055 report; never changes the frozen acceptance rule."""
import argparse
import csv
import json
from pathlib import Path
import shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from run_learned_stack_dev055 import ROOT, PROTOCOL, write, bootstrap


def run(directory):
    directory=Path(directory)
    out=ROOT/'research/dev055'
    p=json.loads(PROTOCOL.read_text())
    summary=json.loads((directory/'summary.json').read_text())
    reports=json.loads((directory/'reports.json').read_text())
    controls=json.loads((directory/'controls.json').read_text())
    shutil.copyfile(directory/'summary.json',out/'summary.json')
    shutil.copyfile(directory/'manifest.json',out/'raw-evidence-manifest.json')
    curves=[]
    for seed,arms in reports.items():
        for arm,report in arms.items():
            for point in report['training']['curves']:
                curves.append({'seed':seed,'arm':arm,**point})
    with (out/'learning-curves.csv').open('w') as stream:
        writer=csv.DictWriter(stream,fieldnames=['seed','arm','update','train','validation'])
        writer.writeheader(); writer.writerows(curves)
    fig, axes=plt.subplots(1,2,figsize=(12,4),sharey=True)
    for arm in p['arms']:
        data=[[r for r in curves if r['arm']==arm and r['seed']==str(seed)] for seed in p['model_seeds']]
        xs=[r['update'] for r in data[0]]
        for axis,metric in zip(axes,('train','validation')):
            ys=np.array([[r[metric] for r in seed] for seed in data])
            line,=axis.plot(xs,ys.mean(0),label=arm)
            axis.fill_between(xs,ys.min(0),ys.max(0),color=line.get_color(),alpha=.12)
            axis.set(title=metric.capitalize(),xlabel='Optimizer updates')
            axis.grid(alpha=.2)
    axes[0].set_ylabel('Post-action fault BCE (nats)')
    axes[1].legend(fontsize=8)
    fig.suptitle('DEV-055: three-seed mean and range; fixed 512 updates, no convergence claim')
    fig.tight_layout(); fig.savefig(out/'learning-curves.png',dpi=160); plt.close(fig)
    def values(arm,metric='cost'):
        if arm in controls:
            return np.repeat(np.array([[r[metric] for f in ('coupling','drift') for r in controls[arm][f]['rows']]]),3,axis=0)
        return np.array([[r[metric] for f in ('coupling','drift') for r in reports[str(s)][arm]['evaluation'][f]['rows']] for s in p['model_seeds']])
    best=min(p['controls'][:-1],key=lambda a:values(a).mean())
    # Descriptive comparison requested in the review. This post-run selection
    # is NOT substituted for the preregistered fitted-HMM acceptance endpoint.
    descriptive={'selected_best_control':best,'selection':'Lowest observed pooled non-oracle control cost; retrospective descriptive only',
                 'mtsk_minus_control':bootstrap(values('mtsk'),values(best),np.random.default_rng(855155)),
                 'oracle_headroom_per_entity':bootstrap(values(best)/2,values('oracle')/2,np.random.default_rng(855156))}
    write(out/'best-control-descriptive.json',descriptive)
    stats={}
    for arm in p['arms']:
        fits=[reports[str(seed)][arm]['training'] for seed in p['model_seeds']]
        stats[arm]={'trainable_parameters':fits[0]['trainable_parameters'],
                    'stored_parameter_bytes':fits[0]['stored_parameter_bytes'],
                    'retained_state_bytes':fits[0]['retained_state_bytes'],
                    'mean_training_seconds':float(np.mean([f['seconds'] for f in fits])),
                    'final_train_bce':float(np.mean([f['curves'][-1]['train'] for f in fits])),
                    'final_validation_bce':float(np.mean([f['curves'][-1]['validation'] for f in fits])),
                    'gradients':{str(seed):reports[str(seed)][arm]['training']['first_step_gradient_l1'] for seed in p['model_seeds']}}
    write(out/'fit-diagnostics.json',stats)
    lines=['# DEV-055: learned dynamics, CDL-style routing and neural CASM execute; superiority fails','',
           'The complete public-history learned model trains, restores and drives a paid physical simulation loop. '
           'All 15 fits completed 512 updates. This establishes an executable development path; '
           '**it does not establish MTSK, relevance-routing or CASM superiority.** The registered reliability guard fails.','',
           '## Results','',
           'Costs are per two-machine, twenty-action world, including twelve paid randomized acquisition actions '
           'and residual-fault penalties at every step. Fault exposure is the sum of unresolved faults across ticks, '
           'not final residual faults or independent failures. There are 384 independent world clusters across two '
           'families, reused across three model initializations.','',
           '| Arm | Total stream cost/world | Fault exposure/world |','| --- | ---: | ---: |']
    for arm,metrics in sorted(summary['costs'].items(),key=lambda x:x[1]['total_stream_cost']):
        lines.append(f'| {arm} | {metrics["total_stream_cost"]:.4f} | {metrics["fault_exposure"]:.4f} |')
    lines += ['','| MTSK minus comparator | Paired cost difference | Interval |', '| --- | ---: | --- |']
    for arm,delta in summary['contrasts'].items():
        lo,hi=delta['interval']; lines.append(f'| {arm} | {delta["mean"]:+.4f} | [{lo:+.4f}, {hi:+.4f}] |')
    d=descriptive['mtsk_minus_control']; lo,hi=d['interval']
    lines += ['',f'The strongest observed conventional control is **{best}**, cost {values(best).mean():.4f}. '
              f'MTSK costs {d["mean"]:+.4f} more (descriptive paired 95% interval [{lo:+.4f}, {hi:+.4f}]). '
              'This post-run comparator selection is disclosed; it does not replace the registered HMM endpoint.', '',
              'The registered HMM oracle headroom is {:.4f}/entity, interval [{:.4f}, {:.4f}], with no increase in '
              'fault exposure for either family. This clears the registered floor. The oracle sees latent physical '
              'state and is a one-step reference, not an attainable public-information Bayes optimum.'.format(
                  summary['oracle_headroom_per_entity']['mean'],*summary['oracle_headroom_per_entity']['interval']), '',
              'Carried memory beats the retrained reset control on cost, with the adjusted interval excluding zero. '
              'That is a bounded memory-use result. Uniform routing and bypassing CASM have lower point costs than '
              'the full model; neither comparison establishes a benefit from the extra learned mechanism. '
              'No architecture or efficiency win follows.','',
              '## What is learned','',
              '- An event encoder and recurrent MTSK state, trained from public noisy sensors and paid inspection receipts.',
              '- The existing `LearnedCDL` scorer over expert views and learned structural-workspace edges/messages.',
              '- The existing sparse `ConditionalExecutor` operator bank, selector and halt head.',
              '- Action-conditioned predictions of post-action fault marginals. Simulator-only fault labels supervise training; runtime cannot accept those labels.', '',
              'The physics generator, two entity addresses, five-action grammar, costs and one-step argmin controller '
              'remain programmed. Public events are ten typed scalar features. This is CDL-style task-trained routing, '
              'not original CDL teacher distillation; neural conditional operators, not general symbolic CASM-S '
              'program discovery. The learned fault predictor is not a safety verifier. The model does not learn '
              'a long-horizon contingent planner or the value of inspections. Fixed-expert exhaustive scoring does '
              'not test indexed acquisition or C5.','',
              '## Training evidence','',
              '![Per-arm learning curves](../research/dev055/learning-curves.png)','',
              'Every arm receives the same 512-update exposure and uses its final checkpoint. Shading is the '
              'minimum-to-maximum range of three seeds. Full per-fit curves are in '
              '[learning-curves.csv](../research/dev055/learning-curves.csv). No plateau or global-convergence claim is made.', '',
              '| Arm | Used trainable parameters | Retained state bytes | Final validation BCE |',
              '| --- | ---: | ---: | ---: |']
    for arm,s in stats.items(): lines.append(f'| {arm} | {s["trainable_parameters"]} | {s["retained_state_bytes"]} | {s["final_validation_bce"]:.5f} |')
    lines += ['', 'Parameter matching counts used trainable parameters, without padding; inherited unused MTSK '
              'paths are frozen and their stored bytes are separately reported. Reset and bypass arms are mechanism '
              'ablations, not parameter-matched architecture controls. Timings include development-environment '
              'contention and are not production benchmarks. Frozen raw per-world `work` diagnostics used '
              'batch-total integer division, rounding down uneven early halting. Original bytes remain '
              'preserved. The audit reports exact trace-counted calls and diagnostic mismatches; the runner '
              'is corrected for future runs. Predictions, actions, costs, faults and registered acceptance '
              'outcomes are unaffected.','',
              '## Verification and reproduction','',
              'Training source: `{}`. Protocol registration: `5067a82`. Main elapsed time: {:.2f} seconds. '
              'The separate [audit record](../research/dev055/audit.json) is authoritative for restored checkpoint, '
              'physical replay and native Veritas counts. Native effect tables encode the recorded newly authored '
              'synthetic trajectories. Veritas checks execution, receipts, accounting and final state; shared physics '
              'limits independence. Failed final-world outcomes are retained. No real-world outcome verification follows.'.format(summary['source'],summary['elapsed_seconds']), '',
              '```bash', 'uv venv .venv',
              'uv pip install --python .venv/bin/python torch==2.14.1+cpu --index-url https://download.pytorch.org/whl/cpu',
              'uv pip install --python .venv/bin/python numpy==2.5.4 pytest==9.1.1 matplotlib==3.11.2',
              '# Optional native replay: install the pinned Veritas checkout as described below.',
              '.venv/bin/python scripts/run_learned_stack_dev055.py --output results/dev055-reproduction',
              '.venv/bin/python scripts/audit_learned_stack_dev055.py --directory results/dev055-reproduction --output results/dev055-reproduction-audit.json --veritas-root /path/to/veritas',
              '.venv/bin/python scripts/demo_learned_stack_dev055.py --checkpoint results/dev055-reproduction/7551-mtsk.pt --events research/dev055/example-public-events.jsonl', '```', '',
              'Use a clean committed worktree for the registered full run. Check out Veritas '
              '`b21d829b62d3cab2811e746267ca305e110ed004` and install it with '
              '`uv pip install --python .venv/bin/python -e /path/to/veritas`. The input example and all tests '
              'use disjoint development namespaces. The raw evidence manifest covers all fifteen checkpoints, '
              'complete reports, controls and fitted filters.','',
              '## Next bounded decisions','',
              'DEV-056 is the registered memory-novelty deadline: matched GRU superiority with reliability and '
              'adequate oracle headroom must be established, or the memory-novelty claim is retired for this program. '
              'Inadequate headroom cannot be used to claim success or postpone the deadline indefinitely. '
              'DEV-057 must return to the distinct C5 questions: indexed acquisition, equal-bits scalar state and '
              'hash baselines, with one-time index construction, reads, writes, addressing and verification charged. '
              'Those experiments have not been run here. All findings remain development evidence.']
    (ROOT/'docs/DEV-055-RESULTS.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('--directory',required=True)
    run(parser.parse_args().directory)
