"""Restore DEV058 checkpoints; audit autonomous, standalone and native replay.

Veritas tables are derived from recorded synthetic outcomes. This audits
execution/accounting, not independent physics or real-world verification.
"""
import argparse
import ast
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
import numpy as np
import torch
from run_fully_learned_dev058 import ROOT, PROTOCOL, LearnedAgent, AgentConfig, World, evaluate, functional
from run_learned_stack_dev055 import digest, write

PIN = 'b21d829b62d3cab2811e746267ca305e110ed004'


@torch.no_grad()
def prediction_diagnostics(model, cells):
    """Evaluator-only proper scores on autonomous visited states/actions."""
    completion, outcome, cost = [], [], []
    selection = [0] * model.config.operators
    calls = candidates = admissions = observations = 0
    for cell in cells:
        worlds = [World(cell['seed'], i, cell['entities'], cell['history']) for i in range(cell['worlds'])]
        state = model.initial(len(worlds))
        for t in range(cell['history']):
            state, trace = model.observe(state, torch.tensor(np.stack([w.events[t] for w in worlds])))
            admissions += trace['writes']; observations += len(worlds)
        active = list(range(len(worlds)))
        for tick in range(2 * cell['entities'] + 4):
            if not active:
                break
            public = [worlds[i].candidates() for i in active]
            values = [torch.from_numpy(np.stack([p[k] for p in public])) for k in range(4)]
            output = model((state[0][active], state[1][active]), *values)
            choices = output['logits'].argmax(-1)
            candidates += int(values[1].sum())
            calls += model.last_work['operator_calls']
            for selected in output['operator_selection']:
                for op in range(model.config.operators):
                    selection[op] += int(((selected == op) & values[1]).sum())
            remaining, receipts = [], []
            for j, i in enumerate(active):
                w = worlds[i]
                completion.append((float(output['completion'][j].sigmoid()) - float(not w.bits.any())) ** 2)
                action = int(choices[j])
                from tac_osm.development.learned_agent_world import outcomes
                expected_cost = sum(p * c for p, _, c in outcomes(w.bits, w.parents, public[j][4][action]))
                cost.append((float(output['cost'][j, action]) * 20 - expected_cost) ** 2)
                receipt = w.act(public[j][4][action])
                predicted = output['outcomes'][j, action, :w.entities].sigmoid().numpy()
                outcome.extend(((predicted - w.bits) ** 2).tolist())
                assert w.actions[-1] == cell['rows'][i]['actions'][tick]
                if not w.stopped:
                    remaining.append(i); receipts.append(receipt)
            memory, occupied = state
            for t in range(max(map(len, receipts), default=0)):
                present = torch.tensor([t < len(r) for r in receipts])
                events = torch.tensor(np.stack([r[t] if t < len(r) else np.zeros(22, np.float32) for r in receipts]))
                updated, trace = model.observe((memory[remaining], occupied[remaining]), events, present)
                memory = memory.index_copy(0, torch.tensor(remaining), updated[0])
                occupied = occupied.index_copy(0, torch.tensor(remaining), updated[1])
                admissions += trace['writes']; observations += int(present.sum())
            state = memory, occupied
            active = remaining
    return {'completion_brier': float(np.mean(completion)), 'action_outcome_brier': float(np.mean(outcome)),
            'action_cost_rmse': float(np.sqrt(np.mean(cost))), 'visited_decisions': len(completion),
            'operator_selector_proposals_including_halted_nodes': selection, 'operator_calls': calls,
            'possible_operator_calls': candidates * model.config.depth,
            'observations': observations, 'admitted_writes': admissions, 'skipped_writes': observations - admissions,
            'scope': 'Selected policy actions on held-out visited states; forecast heads are learned predictions, not an exact verifier.'}


def native_replay(row):
    from tac_osm.development.veritas_public_challenge import make_episode
    from investigation_world.operational.models import (
        ActionKind, HiddenActionEffect, PublicActionSpec, StateAssertion, EpisodeSubmission)
    from investigation_world.operational.runtime import OperationalRuntime
    ep = make_episode(row['seed'], row['index'])
    n = row['entities']
    initial = {f'e{e}.fault': int(row['actions'][0]['before'][e]) for e in range(n)}
    initial.update({'world.tick': 0, 'world.closed': 0})
    effects = []
    for tick, action in enumerate(row['actions']):
        effects.append(HiddenActionEffect(
            action_name=action['kind'], required_parameters={'tick': tick, 'entity': action['target']},
            required_state=[StateAssertion(object_id='world', field_name='tick', expected_value=tick),
                            *[StateAssertion(object_id=f'e{e}', field_name='fault', expected_value=action['before'][e]) for e in range(n)]],
            set_state={**{f'e{e}.fault': action['after'][e] for e in range(n)},
                       'world.tick': tick + 1, 'world.closed': int(action['kind'] == 'stop')},
            observable_result={'events': action['receipt']}))
    ep.task.available_actions = [PublicActionSpec(name=kind, kind=ActionKind.EXECUTE,
        system='plant', description='DEV058 public transport', parameter_names=['tick', 'entity'], cost=cost)
        for kind, cost in (('inspect', 10), ('repair', 60), ('stop', 0))]
    ep.task.constraints = ['Replay recorded public API commands and receipts.']
    ep.oracle.initial_state = initial
    ep.oracle.action_effects = effects
    ep.oracle.target_state = [StateAssertion(object_id=f'e{e}', field_name='fault', expected_value=0) for e in range(n)] + [StateAssertion(object_id='world', field_name='closed', expected_value=1)]
    ep.oracle.invariants = []
    ep.oracle.forbidden_actions = []
    ep.oracle.required_action_counts = {}
    ep.oracle.max_cost = 10000
    ep.oracle.max_tool_calls = len(row['actions'])
    runtime = OperationalRuntime(ep)
    for tick, action in enumerate(row['actions']):
        receipt = runtime.act(action['kind'], tick=tick, entity=action['target'])
        assert receipt['events'] == action['receipt'], 'native receipt mismatch'
    assert not any(e.blocked for e in runtime.events), 'native action blocked'
    total = runtime.budget.spent / 100 + row['observation_cost'] + row['write_cost'] + 20 * row['final_faults']
    assert abs(total - row['cost']) < 1e-7, 'native cost mismatch'
    verified = runtime.submit(EpisodeSubmission(conclusion='DEV058 synthetic replay'))
    assert bool(verified.state == 1) == row['success'], 'native completion mismatch'
    return len(row['actions'])


def export(directory, report, destination):
    destination.mkdir(parents=True, exist_ok=True)
    shutil.copy(ROOT / 'src/tac_osm/development/learned_agent.py', destination / 'agent_model.py')
    shutil.copy(ROOT / 'scripts/learned_agent_runtime_dev058.py', destination / 'runtime.py')
    shutil.copy(directory / 'agent.pt', destination / 'agent.pt')
    write(destination / 'config.json', report['config'])
    (destination / 'README.txt').write_text(
        'Requires Python and PyTorch 2.14.1+cpu. Run: python runtime.py < requests.jsonl\n'
        'Runnable recorded public trace: python runtime.py < demo-requests.jsonl\n'
        'Compare its JSON lines with demo-expected.jsonl. No world package is needed.\n'
        'One request per line: events (new public 22-vectors), candidates (13x11),\n'
        'valid (13 booleans), queries (6x8), query_valid (6 booleans), reset (optional).\n'
        'All values are public API features. Output index selects the supplied candidate.\n'
        'Event features: kind one-hot4 (initial status, relation, receipt, noise),\n'
        'primary opaque identity one-hot8, related identity one-hot8, status bit, tick/256.\n'
        'Candidate: kind one-hot3 (stop, inspect, repair), opaque identity one-hot8.\n'
        'Query: opaque identity one-hot8. Padding masks describe tensor padding only.\n'
        'No simulator, teacher, TAC-OSM installation, analytic planner or hidden state is used.\n'
        'Trained on the bounded DEV058 synthetic task; no industrial/general competence claim.\n')


@torch.no_grad()
def standalone(directory, model, seed):
    """Exercise real multi-step trajectories through a persistent external process."""
    requests, expected, rows = [], [], []
    for i in range(12):
        w = World(seed, i, (2, 4, 6)[i % 3], (32, 96, 128)[i % 3])
        state = model.initial(1)
        incoming = w.events.copy()
        for tick in range(2 * w.entities + 4):
            d, v, q, qv, commands = w.candidates()
            requests.append({'reset': tick == 0, 'events': [e.tolist() for e in incoming],
                             'candidates': d.tolist(), 'valid': v.tolist(),
                             'queries': q.tolist(), 'query_valid': qv.tolist()})
            for event in incoming:
                state, _ = model.observe(state, torch.tensor([event.tolist()]))
            result = model(state, torch.tensor(d[None]), torch.tensor(v[None]), torch.tensor(q[None]), torch.tensor(qv[None]))
            index = int(result['logits'].argmax(-1))
            expected.append((index, result['logits'][0].tolist()))
            incoming = w.act(commands[index])
            if w.stopped:
                break
        rows.append(w.finish(0)['success'])
    # Isolated cwd contains only the exported files; deny all research imports.
    guard = """import builtins,runpy,sys
sys.path.insert(0, '.')
import torch
original=builtins.__import__
def checked(name,*args,**kwargs):
    if name.split('.')[0] in ('tac_osm','investigation_world','run_fully_learned_dev058'):
        raise RuntimeError('research dependency crossed standalone boundary: '+name)
    return original(name,*args,**kwargs)
builtins.__import__=checked
runpy.run_path('runtime.py',run_name='__main__')
"""
    completed = subprocess.run([sys.executable, '-I', '-B', '-c', guard], cwd=directory,
        input=''.join(json.dumps(r) + '\n' for r in requests), text=True, capture_output=True, check=True)
    actual = [json.loads(line) for line in completed.stdout.splitlines()]
    assert len(actual) == len(expected)
    for response, (index, logits) in zip(actual, expected):
        assert response['index'] == index and response['logits'] == logits, 'standalone neural decision mismatch'
    (directory / 'demo-requests.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in requests))
    (directory / 'demo-expected.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in actual))
    return {'worlds': len(rows), 'decisions': len(actual), 'exact_logits': True,
            'research_imports_forbidden': True, 'successful_worlds': sum(rows)}


def run(directory, output, veritas_root=None, export_root=None):
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    started = time.monotonic()
    root = Path(directory)
    summary = json.loads((root / 'summary.json').read_text())
    assert not summary['smoke']
    assert digest(PROTOCOL) == summary['protocol_sha256']
    for name, value in json.loads((root / 'manifest.json').read_text()).items():
        assert digest(root / name) == value, name
    source = ROOT / 'src/tac_osm/development/learned_agent.py'
    tree = ast.parse(source.read_text())
    imports = [n.module if isinstance(n, ast.ImportFrom) else alias.name
               for n in ast.walk(tree) if isinstance(n, (ast.Import, ast.ImportFrom))
               for alias in (n.names if isinstance(n, ast.Import) else [None])]
    assert all(n.split('.')[0] in ('torch', 'dataclasses', 'math') for n in imports)
    result = {'status': 'RUNNING', 'restored_checkpoints': 0, 'restored_cells': 0,
              'worlds': 0, 'actions': 0, 'native_worlds': 0, 'native_actions': 0,
              'failed_completion_worlds': 0, 'residual_fault_worlds': 0, 'residual_faults': 0,
              'missing_stop_worlds': 0, 'exports': {}, 'runtime_imports': imports,
              'limitation': 'Same author/shared synthetic physics; native effects encode recorded outcomes. No independent replication or real-world verifier.'}
    if veritas_root:
        vr = Path(veritas_root).resolve()
        assert subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=vr, text=True).strip() == PIN
        sys.path.insert(0, str(vr / 'src'))
        result['veritas_commit'] = PIN
    for report in json.loads((root / 'development.json').read_text()):
        checkpoint_dir = root / f"{report['recipe']['name']}-{report['seed']}"
        checkpoint = checkpoint_dir / 'agent.pt'
        assert digest(checkpoint) == report['checkpoint_sha256']
        model = LearnedAgent(AgentConfig(**report['config'])).eval()
        model.load_state_dict(torch.load(checkpoint, weights_only=True))
        initial_model = LearnedAgent(model.config, report['seed'])
        result.setdefault('parameter_changes_l1', {})[str(report['seed'])] = {
            name: float((value - initial_model.state_dict()[name]).abs().sum())
            for name, value in model.state_dict().items()}
        cells = list(report['validation'])
        test = checkpoint_dir / 'test.json'
        if test.exists():
            testing = json.loads(test.read_text())
            assert functional(testing['cells']) == testing['functional_test']
            cells += testing['cells']
            cells += [c for values in testing['lesions'].values() for c in values]
            untrained = LearnedAgent(model.config, report['seed'] + 100_000).eval()
            for cell in testing['untrained']:
                assert evaluate(untrained, cell['seed'], cell['entities'], cell['history'], cell['worlds']) == cell
            result.setdefault('prediction_diagnostics', {})[str(report['seed'])] = prediction_diagnostics(model, testing['cells'])
        for cell in cells:
            replay = evaluate(model, cell['seed'], cell['entities'], cell['history'], cell['worlds'], cell['lesion'])
            assert replay == cell, f"restored autonomous mismatch {report['seed']}/{cell['lesion']}"
            result['restored_cells'] += 1
            for row in cell['rows']:
                w = World(row['seed'], row['index'], row['entities'], row['history'])
                for action in row['actions']:
                    w.act((action['kind'], action['target']))
                assert w.finish(row['writes']) == row, 'physical replay mismatch'
                result['worlds'] += 1
                result['actions'] += len(row['actions'])
                result['failed_completion_worlds'] += int(not row['success'])
                result['residual_fault_worlds'] += int(row['final_faults'] > 0)
                result['residual_faults'] += row['final_faults']
                result['missing_stop_worlds'] += int(not row['stopped'])
                if veritas_root:
                    result['native_actions'] += native_replay(row)
                    result['native_worlds'] += 1
        if report['seed'] in summary['selected_seeds'] and report['recipe'] == summary['selected_recipe']:
            destination = Path(export_root) / str(report['seed']) if export_root else Path(tempfile.mkdtemp(prefix='dev058-export-'))
            export(checkpoint_dir, report, destination)
            result['exports'][str(report['seed'])] = standalone(destination, model, 5589001)
            shutil.rmtree(destination / '__pycache__', ignore_errors=True)
            result['exports'][str(report['seed'])]['files_sha256'] = {
                file.name: digest(file) for file in sorted(destination.iterdir()) if file.is_file()}
        result['restored_checkpoints'] += 1
        print('audited', report['recipe']['name'], report['seed'], flush=True)
    result.update(status='PASS', seconds=time.monotonic() - started, training_source=summary['source'],
                  functional_target_met=summary['status']=='FUNCTIONAL_TARGET_MET_UNAUDITED' and bool(result['exports']))
    write(output, result)
    print(json.dumps({key:result[key] for key in ('status','functional_target_met','restored_checkpoints',
        'restored_cells','worlds','actions','native_actions','seconds')}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--veritas-root')
    parser.add_argument('--export-root')
    args = parser.parse_args()
    run(args.directory, args.output, args.veritas_root, args.export_root)
