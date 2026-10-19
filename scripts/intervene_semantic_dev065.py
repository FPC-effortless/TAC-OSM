"""Closed-loop semantic substitutions on fresh worlds; never trains weights."""
import argparse
import gzip
import json
from pathlib import Path
import time
import numpy as np
import torch
from data_semantic_dev065 import BeliefWorld
from run_semantic_dev065 import make, PROTOCOL
from run_learned_stack_dev055 import write, digest
from analyze_belief_dev063 import metrics
from summarize_memory_dev061 import interval


def restore(folder):
    report = json.loads((folder/'report.json').read_text())
    assert digest(folder/'agent.pt') == report['checkpoint_sha256']
    weights = torch.load(folder/'agent.pt', weights_only=True)
    gate = {k.removeprefix('gate.'): v.float() for k, v in weights.items() if k.startswith('gate.')}
    model = make(report['arm'], report['spec'], report['seed'], gate).double().eval()
    model.load_state_dict(weights)
    return report, model


@torch.no_grad()
def rollout(model, seed, history, count, error, mode):
    worlds = [BeliefWorld(seed, i, history, error, .1) for i in range(count)]
    beliefs = [w.reference() for w in worlds]
    keys = torch.tensor(np.stack([w.identities for w in worlds]), dtype=torch.float64)
    state = model.initial(keys)
    for i, world in enumerate(worlds):
        events = keys.new_tensor(np.stack(world.events))
        keep = events[model.gate(events[:, :6]).squeeze(-1) >= 0][-64:]
        state['events'][i, :len(keep)] = keep
        state['present'][i, :len(keep)] = True
    for step in range(10):
        if all(w.stopped for w in worlds):
            break
        payload = [w.candidates() for w in worlds]
        candidates = keys.new_tensor(np.stack([p[0] for p in payload]))
        prices = keys.new_tensor([[w.error, w.inspection] for w in worlds])
        # Exact substitution bypasses encoding which would be discarded anyway.
        # This changes evaluation work only, not the actor or supplied information.
        original = None if mode == 'exact' else model.decide(state, candidates, prices)
        q = keys.new_zeros(len(keys), 12) if original is None else original['posterior']
        if mode == 'exact':
            q = torch.zeros_like(q)
            for i, b in enumerate(beliefs):
                # The reference parsed these parents from public relation records.
                graph = 1 if b.parents[1] == 0 else 2 if b.parents[0] == 1 else 0
                q[i].reshape(4, 3)[:, graph] = keys.new_tensor(b.p)
        elif mode == 'blurred':
            q = .5*q + .5/12
        elif mode != 'predicted':
            raise ValueError(mode)
        scores = model.act_from_belief(q, keys, candidates, prices)
        choices = model.choice(scores)
        if mode == 'predicted':
            assert torch.equal(choices, model.choice(original))
        pending = []
        for i, w in enumerate(worlds):
            if w.stopped:
                pending.append([])
                continue
            before = len(w.events)
            action = int(payload[i][1][int(choices[i])])
            beliefs[i].update(action, w.act(action))
            pending.append(w.events[before:])
        for tick in range(0 if mode == 'exact' else max(map(len, pending), default=0)):
            present = torch.tensor([tick < len(e) for e in pending])
            event = keys.new_tensor(np.stack([v[tick] if tick < len(v) else np.zeros(46) for v in pending]))
            state = model.observe(state, event, present)
    return dict(seed=seed, history=history, error=error, inspection=.1,
                worlds=count, horizon=10, mode=mode, rows=[w.finish() for w in worlds])


@torch.no_grad()
def run(directory, output, details, smoke=False):
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    protocol = json.loads(PROTOCOL.read_text()); root = Path(directory)
    summary = json.loads((root/'summary.json').read_text())
    assert summary['status'] == ('SMOKE_ONLY' if smoke else 'COMPLETE')
    start = time.process_time()
    result = dict(status='RUNNING', checkpoints=0, trajectories=0, per_seed={}, arms={},
                  scope='Each mode follows its own public history. Exact posterior is evaluation-only, never privileged physical state or deployed oracle.')
    count = 2 if smoke else protocol['intervention_worlds']
    seed = protocol['intervention_seed'] + 10000000*int(smoke)
    Path(details).parent.mkdir(parents=True, exist_ok=True)
    try:
        with gzip.open(details, 'wt', encoding='utf-8') as stream:
            for path in sorted(root.glob('group-*/*/report.json')):
                report, model = restore(path.parent)
                allcells = {mode: [] for mode in ('predicted', 'exact', 'blurred')}
                for history in protocol['intervention_histories']:
                    for error in protocol['intervention_errors']:
                        for mode in allcells:
                            if time.process_time()-start > protocol['intervention_cpu_cap_seconds']:
                                raise TimeoutError('Registered intervention CPU cap')
                            cell = rollout(model, seed, history, count, error, mode)
                            allcells[mode].append(cell)
                            stream.write(json.dumps(dict(arm=report['arm'], training_seed=report['seed'], cell=cell))+'\n')
                            result['trajectories'] += count
                result['per_seed'][report['arm']+'/'+str(report['seed'])] = {m: dict(
                    overall=metrics(cells), strata=[metrics([c]) for c in cells]) for m, cells in allcells.items()}
                result['checkpoints'] += 1
                write(output, result)
                print('SEMANTIC_INTERVENTION', report['arm'], report['seed'], flush=True)
        expected = 6 if smoke else 120
        assert result['checkpoints'] == expected
        for arm in json.loads((root/'specs.json').read_text()):
            rows = [v for k, v in result['per_seed'].items() if k.startswith(arm+'/')]
            result['arms'][arm] = {mode: dict(
                mean_cost=float(np.mean([v[mode]['overall']['cost'] for v in rows])),
                mean_completion=float(np.mean([v[mode]['overall']['completion'] for v in rows])),
                total_faults=sum(v[mode]['overall']['faults'] for v in rows))
                for mode in ('predicted', 'exact', 'blurred')}
            result['arms'][arm]['exact_minus_predicted_cost'] = interval([
                v['exact']['overall']['cost']-v['predicted']['overall']['cost'] for v in rows])
        result['status'] = 'SMOKE_ONLY' if smoke else 'COMPLETE'
    except Exception:
        import traceback
        result['status'] = 'FAILED'; result['failure'] = traceback.format_exc()
        raise
    finally:
        result['cpu_seconds'] = time.process_time()-start
        result['details_sha256'] = digest(details)
        write(output, result)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--directory', required=True)
    p.add_argument('--output', required=True); p.add_argument('--details', required=True); p.add_argument('--smoke', action='store_true')
    a = p.parse_args(); run(a.directory, a.output, a.details, a.smoke)
