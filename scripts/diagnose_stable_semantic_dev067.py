"""Posthoc descriptions of every test query and stopping context; no new gate."""
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
    contexts, stopping = {}, {}
    for arm in arms:
        values, missed = [], []
        noncompletion = certain_noncompletion = known_nonstop = 0
        for path in sorted(root.glob(f'group-*/{arm}/report.json')):
            report = json.loads(path.read_text())
            test = json.loads((path.parent/'test.json').read_text())
            values.append(test['fixed']['per_query_KL'])
            for cell in test['cells'][:4]:
                for row in cell['rows']:
                    certain = [i for i, q in enumerate(row['public_posterior']) if sum(q[:3]) >= 1-1e-12]
                    if not row['completion']:
                        noncompletion += 1; certain_noncompletion += bool(certain)
                    for tick in certain:
                        if row['actions'][tick]['kind'] == 'stop':
                            continue
                        known_nonstop += 1
                        q, p = np.array(row['public_posterior'][tick]), np.array(row['posterior'][tick])
                        missed.append(dict(seed=report['seed'], world=row['index'], history=cell['history'],
                            step=tick, predicted_healthy=float(p[:3].sum()),
                            mass_on_publicly_impossible_states=float(p[q == 0].sum()),
                            stop_probability=row['stop_probability'][tick],
                            chosen_kind=row['actions'][tick]['kind'], final_completion=row['completion']))
        assert len(values) == 20
        query_means = np.mean(values, axis=0)
        contexts[arm] = []
        for index, record in enumerate(records):
            contexts[arm].append(dict(index=index, mean_KL=float(query_means[index]),
                depth=record['depth'], sensor_error=float(record['prices'][0]),
                graph=int(record['posterior'].reshape(4, 3).sum(0).argmax()),
                event_counts=record['events'][:, :6].sum(0).tolist(),
                prior_namespace=record['world_seed'], world_index=record['world_index']))
        contexts[arm].sort(key=lambda row: (-row['mean_KL'], row['index']))
        stopping[arm] = dict(noncompletion_rollouts=noncompletion,
            noncompletion_with_public_certain_healthy_decision=certain_noncompletion,
            publicly_certain_healthy_nonstop_decisions=known_nonstop, all_missed_contexts=missed)
    references = json.loads((Path(output).parent/'public-reference.json').read_text())
    reference_harm = {}
    for name in ('public', 'privileged'):
        actions = [a for cell in references[name] for row in cell['rows'] for a in row['actions']]
        reference_harm[name] = dict(rollouts=sum(c['worlds'] for c in references[name]),
            new_fault_bits_during_actions=sum(sum(not b and v for b, v in zip(a['before'], a['after'], strict=True)) for a in actions),
            transient_fault_exposure=sum(sum(a['after']) for a in actions))
    write(output, dict(status='POSTHOC_DESCRIPTION_ONLY', all_query_contexts=contexts,
        stopping_contexts=stopping, reference_transient_harm=reference_harm,
        limits='No refitting, new threshold, checkpoint selection or causal conclusion. These synthetic queries and repeated starts are not independent field trials.'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--directory', required=True)
    parser.add_argument('--output', required=True); args = parser.parse_args()
    run(args.directory, args.output)
