"""Posthoc descriptions for the NEXT hypothesis; no fitting or promotion rule."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from run_learned_stack_dev055 import write


def run(directory, output):
    root = Path(directory)
    assert json.loads((root/'summary.json').read_text())['status'] == 'COMPLETE'
    records = torch.load(root/'test-fixed.pt', weights_only=True)
    arms = json.loads((root/'specs.json').read_text())
    descriptions, training = {}, {}
    for arm in arms:
        values, transitions = [], []
        for path in sorted(root.glob(f'group-*/{arm}/report.json')):
            report = json.loads(path.read_text())
            test = json.loads((path.parent/'test.json').read_text())
            values.append(test['fixed']['per_query_KL'])
            warm = next(c for c in report['curves'] if c['update'] == 1024)
            transitions.append(dict(seed=report['seed'], warmup_validation64_KL=warm['validation']['kl'],
                                    final_validation64_KL=report['curves'][-1]['validation']['kl']))
        assert len(values) == 20
        query_means = np.mean(values, axis=0)
        descriptions[arm] = []
        for index, record in enumerate(records):
            types = record['events'][:, :6].sum(0).tolist()
            descriptions[arm].append(dict(index=index, mean_KL=float(query_means[index]),
                depth=record['depth'], sensor_error=float(record['prices'][0]),
                graph=int(record['posterior'].reshape(4, 3).sum(0).argmax()),
                event_counts=types, prior_namespace=record['world_seed'], world_index=record['world_index']))
        # Preserve every query, ranked; do not only show a favorable subset.
        descriptions[arm].sort(key=lambda row: (-row['mean_KL'], row['index']))
        training[arm] = dict(per_seed=transitions,
            median_warmup_validation64_KL=float(np.median([r['warmup_validation64_KL'] for r in transitions])),
            median_final_validation64_KL=float(np.median([r['final_validation64_KL'] for r in transitions])))
    paired_warmup = []
    for seed in range(96501, 96521):
        reports = [json.loads((root/f'group-{seed}'/arm/'report.json').read_text()) for arm in arms if arm.startswith('tac_')]
        values = [next(c['validation']['kl'] for c in r['curves'] if c['update'] == 1024) for r in reports]
        assert max(values)-min(values) == 0, 'Factorial arms differed before the intervention phase'
        paired_warmup.append(dict(seed=seed, validation64_KL=values[0]))
    write(output, dict(status='POSTHOC_DESCRIPTION_ONLY', all_query_contexts=descriptions,
        warmup_to_joint=training, factorial_warmup_parity=paired_warmup,
        limits='No refitting, new gates, causal conclusion or checkpoint selection. Global gradient clipping can couple encoder step magnitude to actor losses despite a detached semantic interface; mixed inputs also remove encoder decision gradients on exact-target rows. Factorial contrasts describe the registered recipes.'))


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--directory', required=True); p.add_argument('--output', required=True)
    a = p.parse_args(); run(a.directory, a.output)
