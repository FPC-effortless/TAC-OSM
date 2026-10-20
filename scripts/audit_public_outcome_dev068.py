"""Recorded-only checkpoint, public-data, boundary and native verifier replay."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
import torch
from run_public_outcome_dev068 import (ROOT, PROTOCOL, make, rollout, public_metrics)
from data_public_outcome_dev068 import corpus, teacher_targets
from data_belief_dev062 import BeliefWorld
from run_learned_stack_dev055 import write, digest
from audit_identity_dev0615 import compare
from audit_fully_learned_dev058 import native_replay, PIN


def export(folder, report, destination):
    destination.mkdir(parents=True, exist_ok=False)
    sources = [
        ('belief_agent.py', 'agent_base.py', {}),
        ('belief_bottleneck.py', 'agent_model.py', {'.belief_agent':'agent_base'}),
        ('semantic_state.py', 'agent_semantic.py', {'.belief_agent':'agent_base', '.belief_bottleneck':'agent_model'}),
        ('public_outcome.py', 'agent_outcome.py', {'.belief_agent':'agent_base', '.semantic_state':'agent_semantic'})]
    for source, target, replacements in sources:
        text = (ROOT/'src/tac_osm/development'/source).read_text()
        for old, new in replacements.items(): text = text.replace('from '+old+' import', 'from '+new+' import')
        (destination/target).write_text(text)
    shutil.copy(ROOT/'scripts/runtime_public_outcome_dev068.py', destination/'runtime.py')
    shutil.copy(folder/f"{report['arm']}.pt", destination/'agent.pt')
    write(destination/'metadata.json', dict(arm=report['arm'], spec=report['spec'], seed=report['seed'],
        evidence_stage='DIAGNOSTIC_PILOT_ONLY_NOT_PROMOTED' if folder.name.startswith('pilot') else 'FRESH_REGISTERED_FIT',
        source_commit=report['source'], checkpoint_sha256=report['checkpoint_sha256'],
        state_labels='health-pair inspection outcomes x3 public graph types',
        future_labels='nonterminal paired inspection12; terminal verified fault-count xgraph9',
        inference_dtype='float64', teacher_control=report['spec']['teacher'], commercial_qualified=False))
    (destination/'LICENSE-NOTICE.txt').write_text('TAC-OSM project metadata declares Apache-2.0.\n')
    (destination/'README.txt').write_text(
        'Python + PyTorch only. python runtime.py < demo-requests.jsonl\n'
        'Inputs: reset, keys2x16, new events Nx46, shuffled choices5x19, prices2, remaining_fraction.\n'
        'State12 predicts paid inspection outcomes (healthpair4 x publicgraph3).\n'
        'Future21 predicts next paired inspection12 or terminal verificationfaultcount3 x graph3.\n'
        'Policy reads only learned future probabilities and cost plus public choices/prices/budget.\n'
        'No analytical posterior, action teacher, physical law, search or forced stop at inference.\n'
        'Explicit structure: typed schema, cosine identity binding, 2entities/5actions, state vocabulary, FIFO64 and supervised noise gate.\n'
        'Outcome-arm training uses extra separately charged inspection labels WITHHELD from student history.\n'
        'Teacher-control encoder alone uses explicit known-law public posterior targets.\n'
        'Reported probabilities are unqualified learned estimates; external verification is required.\n'
        'Pilot export is diagnostic; not commercially qualified or evidence of architectural superiority.\n')


@torch.no_grad()
def standalone(folder, report, model, destination):
    export(folder, report, destination)
    world = BeliefWorld(6681002, 0, 128)
    state = model.initial(torch.tensor(world.identities, dtype=torch.float64)[None])
    pending = world.events.copy(); requests, expected = [], []
    for depth in range(10):
        candidates, permutation = world.candidates()
        requests.append(dict(reset=depth == 0, keys=world.identities.tolist(),
            events=[e.tolist() for e in pending], candidates=candidates.tolist(),
            prices=[world.error, world.inspection], remaining_fraction=(10-depth)/10))
        for event in pending: state = model.observe(state, torch.tensor(event, dtype=torch.float64)[None])
        out = model.decide(state, torch.tensor(candidates, dtype=torch.float64)[None],
            torch.tensor([[world.error, world.inspection]], dtype=torch.float64), torch.tensor([[(10-depth)/10]], dtype=torch.float64))
        index = int(model.choice(out)[0])
        expected.append(dict(index=index, scores=out['logits'][0].tolist(),
            paired_measurement_probabilities=out['posterior'][0].tolist(),
            future_verification_probabilities=out['future'][0].softmax(-1).tolist(),
            normalized_physical_cost_predictions=out['cost'][0].tolist()))
        start = len(world.events); world.act(int(permutation[index])); pending = world.events[start:]
        if world.stopped: break
    text = ''.join(json.dumps(r)+'\n' for r in requests)
    (destination/'demo-requests.jsonl').write_text(text)
    (destination/'demo-expected.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in expected))
    guard = """import sys,runpy,builtins
sys.path.insert(0,'.')
import torch
original=builtins.__import__
def checked(name,globals=None,locals=None,fromlist=(),level=0):
    if level==0 and name.split('.')[0] in ('numpy','tac_osm','investigation_world'):raise ImportError('research dependency forbidden')
    return original(name,globals,locals,fromlist,level)
builtins.__import__=checked
runpy.run_path('runtime.py',run_name='__main__')
"""
    response = subprocess.run([sys.executable, '-I', '-B', '-c', guard], cwd=destination,
        input=text, text=True, capture_output=True)
    assert response.returncode == 0, response.stderr
    actual = [json.loads(line) for line in response.stdout.splitlines()]
    compare(actual, expected, dict(values=0, max_abs=0.))
    (destination/'demo-output.jsonl').write_text(response.stdout)
    return len(expected)


@torch.no_grad()
def run(directory, output, exports=None, veritas=None, reconstruct=False):
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    start = time.monotonic(); root = Path(directory)
    summary = json.loads((root/'summary.json').read_text())
    assert summary['status'] in ('VALIDATION_REJECTED', 'TEST_COMPLETE', 'SMOKE_ONLY')
    manifest = json.loads((root/'manifest.json').read_text())
    for name, sha in manifest.items(): assert digest(root/name) == sha, name
    if summary['status'] == 'VALIDATION_REJECTED':
        assert not summary['tests_opened'] and not (root/'TEST-OPENED.json').exists()
        assert not list(root.glob('group-*/*-test.json'))
    if veritas:
        assert subprocess.check_output(['git', '-C', veritas, 'rev-parse', 'HEAD'], text=True).strip() == PIN
        sys.path.insert(0, str(Path(veritas)/'src'))
    result = dict(status='RUNNING', checkpoints=0, trajectories=0, native_actions=0,
        portable_decisions=0, clock_cases=0, interface_cases=0, reconstructed_datasets=0,
        telemetry=dict(values=0, max_abs=0.), frozen_source=summary['source'],
        commercial_qualified=False, tests_opened=summary['tests_opened'])
    files = ['src/tac_osm/development/'+name for name in ('public_outcome.py', 'semantic_state.py',
        'belief_bottleneck.py', 'belief_agent.py', 'uncertain_world.py')]
    files += ['scripts/'+name for name in ('run_public_outcome_dev068.py', 'data_public_outcome_dev068.py',
        'data_belief_dev062.py', 'run_semantic_dev065.py')]
    files += ['research/dev068-public-outcome/'+name for name in ('PROTOCOL.md', 'protocol.json')]
    for name in files:
        assert subprocess.check_output(['git', 'show', summary['source']+':'+name], cwd=ROOT) == (ROOT/name).read_bytes(), name
    gate = torch.load(root/'_datasets/gate.pt', weights_only=True)
    validation = torch.load(root/'_datasets/validation.pt', mmap=True, weights_only=True)
    if reconstruct:
        protocol = json.loads(PROTOCOL.read_text())
        for name, seed, count in [('train', protocol['train_seed'], protocol['transitions']),
                                  ('validation', protocol['validation_seed'], protocol['validation_transitions'])]:
            spec = json.loads((root/'specs.json').read_text())['tac_outcome']
            fresh, episodes, coverage = corpus(seed, count, make(spec, 96800, gate).gate)
            stored = torch.load(root/'_datasets'/f'{name}.pt', mmap=True, weights_only=True)
            for key in stored: torch.testing.assert_close(fresh[key], stored[key], atol=0, rtol=0)
            assert episodes == json.loads((root/'_datasets'/f'{name}-traces.json').read_text())
            compare(coverage, json.loads((root/'_datasets'/f'{name}-coverage.json').read_text()), result['telemetry'])
            if name == 'train':
                torch.testing.assert_close(teacher_targets(episodes, len(fresh['keys'])),
                    torch.load(root/'_datasets/teacher-only.pt', weights_only=True), atol=0, rtol=0)
            result['reconstructed_datasets'] += 1
    reports = sorted(root.glob('*/*-report.json'))
    assert len(reports) == summary['fits']
    for path in reports:
        report = json.loads(path.read_text()); folder = path.parent
        assert report['source'] == summary['source']
        assert digest(folder/f"{report['arm']}.pt") == report['checkpoint_sha256']
        model = make(report['spec'], report['seed'], gate).double().eval()
        model.load_state_dict(torch.load(folder/f"{report['arm']}.pt", weights_only=True))
        compare(public_metrics(model, validation), report['validation_fixed'], result['telemetry'])
        cells = report['validation']
        test = folder/f"{report['arm']}-test.json"
        if test.exists():
            value = json.loads(test.read_text()); cells += value['primary']+value['secondary']
        for cell in cells:
            actual = rollout(model, cell['seed'], cell['history'], len(cell['rows']), cell['error'], cell['inspection'])
            compare(actual, cell, result['telemetry'])
            result['trajectories'] += len(cell['rows'])
            if veritas and folder.name in ('pilot-0', 'group-96801') and cell['inspection'] == .1:
                for index, row in enumerate(cell['rows']):
                    world = BeliefWorld(cell['seed'], index, cell['history'], cell['error'], cell['inspection'])
                    for action in row['actions']: world.act(action)
                    native = dict(world.finish(), entities=2, final_faults=row['faults'],
                        success=row['completion'], write_cost=0., observation_cost=row['costs']['observations'])
                    result['native_actions'] += native_replay(native)
        data = {k: validation[k][:1].double() if validation[k].is_floating_point() else validation[k][:1]
                for k in ('events', 'present', 'keys', 'candidates', 'prices', 'remaining')}
        base = model(**data)
        for scale, shift in ((1, 10000), (1000, 0)):
            changed = {**data, 'events': data['events'].clone()}
            changed['events'][:, :, 42] = data['events'][:, :, 42]*scale+shift
            altered = model(**changed)
            for key in ('posterior', 'future', 'cost', 'logits'):
                torch.testing.assert_close(base[key], altered[key], atol=0, rtol=0)
            result['clock_cases'] += 1
        boundary = model.world_from_state(base['posterior'], data['keys'], data['candidates'], data['prices'], data['remaining'])
        torch.testing.assert_close(model.policy(boundary['features']).squeeze(-1), base['q'], atol=0, rtol=0)
        result['interface_cases'] += 1
        export_group = 'group-96801' if (root/'group-96801/summary.json').exists() else 'pilot-0'
        if exports and folder.name == export_group:
            result['portable_decisions'] += standalone(folder, report, model, Path(exports)/report['arm'])
        result['checkpoints'] += 1; write(output, result)
        print('restored', str(path.relative_to(root)), flush=True)
    result.update(status='COMPLETE', elapsed_seconds=time.monotonic()-start,
        source_files_sha256={name: digest(ROOT/name) for name in files})
    write(output, result); print(json.dumps(result), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--directory', required=True)
    parser.add_argument('--output', required=True); parser.add_argument('--exports')
    parser.add_argument('--veritas'); parser.add_argument('--reconstruct-data', action='store_true')
    args = parser.parse_args()
    run(args.directory, args.output, args.exports, args.veritas, args.reconstruct_data)
