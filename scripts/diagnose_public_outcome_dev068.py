"""Interim VALIDATION interventions. Not sealed tests or promotion evidence."""
import argparse
import json
from pathlib import Path
import torch
from run_public_outcome_dev068 import make, rollout
from run_learned_stack_dev055 import digest, write


def run(directory, output):
    torch.set_num_threads(1); torch.set_deterministic_debug_mode('error')
    root = Path(directory); gate = torch.load(root/'_datasets/gate.pt', weights_only=True)
    result = dict(status='RUNNING', scope='Interim fixed first14 training seeds on already-exposed validation worlds ONLY; diagnostic, no selection or promotion',
        diagnostic_source_sha256=digest(__file__), frozen_fit_source='a6d8cf30ddf69fe3c14980c54e27ba47059a8854',
        checkpoints=0, intervention_worlds=0, per_fit={}, commercial_qualified=False)
    assert not (root/'TEST-OPENED.json').exists()
    for seed in range(96801, 96815):
        for arm in ('tac_outcome', 'gru_outcome', 'no_casm_outcome', 'tac_teacher_belief'):
            folder = root/f'group-{seed}'; report = json.loads((folder/f'{arm}-report.json').read_text())
            assert digest(folder/f'{arm}.pt') == report['checkpoint_sha256']
            model = make(report['spec'], seed, gate).double().eval()
            model.load_state_dict(torch.load(folder/f'{arm}.pt', weights_only=True))
            exact = [rollout(model, c['seed'], c['history'], len(c['rows']), c['error'], c['inspection'], intervention='exact')
                     for c in report['validation']]
            def concise(cells):
                rows = [r for c in cells for r in c['rows']]
                return dict(completions=sum(r['completion'] for r in rows), worlds=len(rows),
                    fault_bits=sum(r['faults'] for r in rows),
                    false_stops=sum(r['stopped'] and r['faults'] > 0 for r in rows),
                    healthy_exhaustion=sum(not r['stopped'] and r['faults'] == 0 for r in rows),
                    physical_cost=sum(r['cost'] for r in rows)/len(rows))
            result['per_fit'][f'{arm}/{seed}'] = dict(predicted=concise(report['validation']), exact_belief=concise(exact))
            result['checkpoints'] += 1; result['intervention_worlds'] += sum(len(c['rows']) for c in exact)
            write(output, result)
    result['per_arm'] = {}
    for arm in ('tac_outcome', 'gru_outcome', 'no_casm_outcome', 'tac_teacher_belief'):
        fits = [r for name, r in result['per_fit'].items() if name.startswith(arm+'/')]
        result['per_arm'][arm] = {mode: dict(
            worlds=sum(r[mode]['worlds'] for r in fits), completions=sum(r[mode]['completions'] for r in fits),
            fault_bits=sum(r[mode]['fault_bits'] for r in fits),
            false_stops=sum(r[mode]['false_stops'] for r in fits),
            healthy_exhaustion=sum(r[mode]['healthy_exhaustion'] for r in fits))
            for mode in ('predicted', 'exact_belief')}
    result['status'] = 'COMPLETE'; write(output, result)
    print(json.dumps({k:v for k,v in result.items() if k != 'per_fit'}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--directory', required=True); parser.add_argument('--output', required=True)
    args = parser.parse_args(); run(args.directory, args.output)
