"""Known-law world-interface intervention on exposed VALIDATION worlds only.

Not deployed; not training; not an optimal oracle. Learned Q stays unchanged,
so replacing its training-distribution features is an interface/OOD diagnostic.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from run_public_outcome_dev068 import make, rollout
from data_belief_dev062 import BeliefWorld
from data_public_outcome_dev068 import graph_from_public
from tac_osm.development.belief_agent import cosine, choose
from tac_osm.development.uncertain_world import STATES, COMMANDS
from run_learned_stack_dev055 import write, digest


@torch.no_grad()
def intervene(model, seed, history, count):
    worlds = [BeliefWorld(seed, i, history) for i in range(count)]
    beliefs = [w.reference() for w in worlds]
    graphs = [graph_from_public(w.events, w.identities) for w in worlds]
    keys = torch.tensor(np.stack([w.identities for w in worlds]), dtype=torch.float64)
    for depth in range(10):
        if all(w.stopped for w in worlds): break
        payload = [w.candidates() for w in worlds]
        candidates = keys.new_tensor(np.stack([p[0] for p in payload]))
        prices = keys.new_tensor([[w.error, w.inspection] for w in worlds])
        remaining = keys.new_full((count, 1), (10-depth)/10)
        public = torch.cat((candidates[:, :, :3], cosine(candidates[:, :, 3:], keys),
                            prices[:, None].expand(-1, 5, -1), remaining[:, None].expand(-1, 5, -1)), -1)
        future = keys.new_zeros(count, 5, 21); cost = keys.new_zeros(count, 5, 1)
        for i, (belief, world) in enumerate(zip(beliefs, worlds, strict=True)):
            for choice, action in enumerate(payload[i][1]):
                kind, _ = COMMANDS[int(action)]
                after = belief.p @ belief.transitions[action]
                terminal = action == 0 or depth == 9
                for physical, mass in enumerate(after):
                    index = 12+3*int(STATES[physical].sum())+graphs[i] if terminal else 3*physical+graphs[i]
                    future[i, choice, index] += float(mass)
                immediate = .01 if kind == 'stop' else world.inspection+.02 if kind == 'inspect' else .63
                cost[i, choice, 0] = (immediate+(20*float(after@STATES.sum(-1)) if terminal else 0))/40
        features = torch.cat((future, cost, public), -1)
        q = model.policy(features).squeeze(-1)
        selected = choose(-q, torch.ones(count, 5, dtype=torch.bool))
        for i, world in enumerate(worlds):
            if world.stopped: continue
            action = int(payload[i][1][int(selected[i])]); receipt = world.act(action)
            beliefs[i].update(action, receipt)
    rows = [w.finish() for w in worlds]
    return dict(worlds=count, completions=sum(r['completion'] for r in rows),
        fault_bits=sum(r['faults'] for r in rows), false_stops=sum(r['stopped'] and r['faults'] > 0 for r in rows),
        healthy_exhaustion=sum(not r['stopped'] and r['faults'] == 0 for r in rows),
        physical_cost=sum(r['cost'] for r in rows)/count)


def run(directory, output):
    torch.set_num_threads(1); torch.set_deterministic_debug_mode('error')
    root = Path(directory); gate = torch.load(root/'_datasets/gate.pt', weights_only=True)
    assert not (root/'TEST-OPENED.json').exists()
    baseline = json.loads(Path('research/dev068-public-outcome/interim-validation-interventions.json').read_text())
    result = dict(status='RUNNING', per_fit={}, checkpoints=0, worlds=0,
        scope='Fixed first14 seeds; already-exposed validation worlds. Known-law expected future/cost interface, unchanged learned Q; NOT training/deployment/oracle-optimality evidence.',
        source_sha256=digest(__file__), prior_diagnostic_sha256=digest('research/dev068-public-outcome/interim-validation-interventions.json'),
        commercial_qualified=False)
    for seed in range(96801, 96815):
        for arm in ('tac_outcome', 'gru_outcome', 'no_casm_outcome', 'tac_teacher_belief'):
            folder = root/f'group-{seed}'; r = json.loads((folder/f'{arm}-report.json').read_text())
            model = make(r['spec'], seed, gate).double().eval()
            model.load_state_dict(torch.load(folder/f'{arm}.pt', weights_only=True))
            cells = [intervene(model, c['seed'], c['history'], len(c['rows'])) for c in r['validation']]
            row = {key: sum(c[key] for c in cells) for key in ('worlds', 'completions', 'fault_bits', 'false_stops', 'healthy_exhaustion')}
            result['per_fit'][f'{arm}/{seed}'] = dict(known_world=row, **baseline['per_fit'][f'{arm}/{seed}'])
            result['checkpoints'] += 1; result['worlds'] += row['worlds']; write(output, result)
    result['per_arm'] = {}
    for arm in ('tac_outcome', 'gru_outcome', 'no_casm_outcome', 'tac_teacher_belief'):
        fits = [r for name, r in result['per_fit'].items() if name.startswith(arm+'/')]
        result['per_arm'][arm] = {mode: {key: sum(r[mode][key] for r in fits)
            for key in ('worlds', 'completions', 'fault_bits', 'false_stops', 'healthy_exhaustion')}
            for mode in ('predicted', 'exact_belief', 'known_world')}
    result['status'] = 'COMPLETE'; write(output, result)
    print(json.dumps({k:v for k,v in result.items() if k != 'per_fit'}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--directory', required=True); parser.add_argument('--output', required=True)
    args = parser.parse_args(); run(args.directory, args.output)
