"""Recover the frozen DEV068 matrix after executor loss; no recipe changes."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import math
import multiprocessing
from pathlib import Path
import subprocess
import time
import torch
from run_public_outcome_dev068 import ROOT, PROTOCOL, group, make, rollout, eligible
from run_learned_stack_dev055 import write, digest


def run(directory):
    root = Path(directory); protocol = json.loads(PROTOCOL.read_text())
    frozen = 'a6d8cf30ddf69fe3c14980c54e27ba47059a8854'
    assert subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip() == frozen
    # Commit remains unchanged while group() captures source at entry. The
    # recovery wrapper is hash-frozen in the run before any restarted worker.
    for name in ('scripts/run_public_outcome_dev068.py', 'scripts/data_public_outcome_dev068.py',
                 'src/tac_osm/development/public_outcome.py', 'research/dev068-public-outcome/PROTOCOL.md',
                 'research/dev068-public-outcome/protocol.json'):
        assert subprocess.check_output(['git', 'show', frozen+':'+name], cwd=ROOT) == (ROOT/name).read_bytes()
    assert not (root/'TEST-OPENED.json').exists() and not (root/'summary.json').exists()
    specs = json.loads((root/'specs.json').read_text())
    selection = json.loads((root/'selection.json').read_text())
    primary = selection['primary']; prior_nll = selection['prior_NLL']
    pilots = json.loads((root/'pilots.json').read_text())
    fresh = json.loads((root/'fresh.json').read_text())
    assert len({g['seed'] for g in fresh}) == len(fresh)
    assert all(g['status'] == 'COMPLETE' for g in fresh+pilots)
    assert all(r['source'] == frozen for g in fresh+pilots for r in g['reports'])
    pending = [s for s in protocol['seeds'] if s not in {g['seed'] for g in fresh}]
    partial = []
    for seed in pending:
        folder = root/f'group-{seed}'
        if folder.exists():
            assert not (folder/'summary.json').exists()
            target = root/'interrupted-executor'/folder.name; target.parent.mkdir(exist_ok=True)
            assert not target.exists(); folder.rename(target)
            partial.append(dict(seed=seed, path=str(target.relative_to(root)),
                files_sha256={str(p.relative_to(target)): digest(p) for p in target.rglob('*') if p.is_file()},
                conservative_cpu_charge=protocol['group_cpu_cap_seconds']))
    # Exact original parent process accounting was lost with the executor.
    # Bound setup with twice its recorded wall interval plus ten seconds of
    # startup (two-CPU quota). Charge a FULL seed cap for each interrupted fit.
    setup_wall = (root/'data-manifest.json').stat().st_mtime-(root/'specs.json').stat().st_mtime
    setup_charge = math.ceil(2*(setup_wall+10))
    overhead = setup_charge+sum(v['conservative_cpu_charge'] for v in partial)
    recovery = dict(reason='executor restart; processes lost, no numerical-failure reselection', frozen=frozen,
        wrapper_sha256=digest(__file__), locked_selection_sha256=digest(root/'selection.json'),
        completed_seed_groups=len(fresh), pending_seeds=pending, interrupted=partial,
        exact_interrupted_cpu='UNKNOWN; full registered group cap charged per interrupted process',
        setup_cpu='UNKNOWN exact; conservative two-CPU wall bound plus startup allowance',
        setup_wall_seconds=setup_wall, setup_charge_cpu_seconds=setup_charge,
        schedule='Two workers; single final admission permitted if only one full group cap fits. No training changes.',
        global_cpu_cap=protocol['fit_cpu_cap_seconds'], test_opened=False)
    write(root/'resumption.json', recovery)
    with ProcessPoolExecutor(max_workers=2, mp_context=multiprocessing.get_context('spawn')) as executor:
        cursor = 0
        while cursor < len(pending):
            used = overhead+sum(g['cpu_seconds'] for g in fresh+pilots)
            slots = min(2, len(pending)-cursor, int((protocol['fit_cpu_cap_seconds']-used)//protocol['group_cpu_cap_seconds']))
            if slots < 1: break
            seeds = pending[cursor:cursor+slots]; cursor += slots
            jobs = [(seed, selection['lrs'], specs, str(root/f'group-{seed}'), str(root/'_datasets'), False) for seed in seeds]
            for future in as_completed([executor.submit(group, job) for job in jobs]):
                result = future.result(); fresh.append(result)
                write(root/'fresh.json', fresh)
                print('resumed', result['seed'], result['status'], round(result['cpu_seconds'], 1), flush=True)
    complete = len(fresh) == len(protocol['seeds']) and all(g['status'] == 'COMPLETE' for g in fresh)
    passing = [r for g in fresh for r in g['reports'] if r['arm'] == primary and eligible(r, prior_nll)]
    status = 'READY_FOR_TEST' if complete and len(passing) >= 18 else 'VALIDATION_REJECTED' if complete else 'INCOMPLETE'
    if status == 'READY_FOR_TEST':
        write(root/'TEST-OPENED.json', dict(source=frozen, selection=digest(root/'selection.json'),
            fresh=digest(root/'fresh.json'), resumption=digest(root/'resumption.json')))
        torch.set_num_threads(1); torch.set_deterministic_debug_mode('error')
        gate = torch.load(root/'_datasets/gate.pt', weights_only=True)
        for g in sorted(fresh, key=lambda v: v['seed']):
            for arm, spec in specs.items():
                model = make(spec, g['seed'], gate).double().eval()
                model.load_state_dict(torch.load(root/f"group-{g['seed']}"/f'{arm}.pt', weights_only=True))
                cells = [rollout(model, protocol['test_seed'], h, protocol['test_worlds']) for h in protocol['histories']]
                secondary = [rollout(model, protocol['secondary_seed'], h, 32, error=.2, inspection=.4) for h in (128, 1024)]
                write(root/f"group-{g['seed']}"/f'{arm}-test.json', dict(primary=cells, secondary=secondary))
            print('tested', g['seed'], flush=True)
        status = 'TEST_COMPLETE'
    summary = dict(status=status, source=frozen, preparation_cpu_seconds=None,
        preparation_cpu_upper_charge=setup_charge,
        interrupted_cpu_upper_charge=sum(v['conservative_cpu_charge'] for v in partial),
        fit_cpu_seconds=sum(g['cpu_seconds'] for g in fresh+pilots),
        charged_total_cpu_upper=overhead+sum(g['cpu_seconds'] for g in fresh+pilots),
        fits=sum(len(g['reports']) for g in fresh+pilots), complete_fresh_groups=len(fresh),
        tests_opened=(root/'TEST-OPENED.json').exists(), commercial_qualified=False,
        recovery_wrapper_sha256=digest(__file__), schedule_deviation='Single-group admission allowed after executor interruption; all fitting updates/seeds/gates unchanged')
    assert summary['charged_total_cpu_upper'] <= protocol['fit_cpu_cap_seconds']
    write(root/'summary.json', summary)
    write(root/'manifest.json', {str(p.relative_to(root)): digest(p) for p in sorted(root.rglob('*')) if p.is_file() and p.name != 'manifest.json'})
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--directory', required=True)
    run(parser.parse_args().directory)
