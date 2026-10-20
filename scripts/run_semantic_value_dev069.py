"""Freeze world/memory; compare neural and probability-linear Q readouts."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import copy
import hashlib
import json
import multiprocessing
import shutil
from pathlib import Path
import subprocess
import sys
import time
ROOT = Path(__file__).resolve().parents[1]; sys.path[:0] = [str(ROOT/'src'), str(ROOT/'scripts')]
import numpy as np
import torch
from torch.nn import functional as F
from tac_osm.development.semantic_value import from_parent
from run_public_outcome_dev068 import cached_states, cached_features, rollout
from run_learned_stack_dev055 import write, digest
PROTOCOL = ROOT/'research/dev069-semantic-value/protocol.json'
BACKENDS = ('tac_outcome', 'gru_outcome', 'no_casm_outcome')


def save_training_state(path, raw, ema, optimizers, generator, curves, step, source, lrs, cpu_seconds):
    value = dict(source=source, lrs=lrs, update=step, generator=generator.get_state(), curves=curves,
        cpu_seconds=cpu_seconds, raw={a:m.state_dict() for a,m in raw.items()},
        ema={a:m.state_dict() for a,m in ema.items()}, optimizers={a:o.state_dict() for a,o in optimizers.items()})
    path = Path(path); temporary = path.with_suffix(path.suffix+'.tmp')
    torch.save(value, temporary); temporary.replace(path)


def restore_training_state(path, raw, ema, optimizers, generator, source, lrs):
    checkpoint = torch.load(path, weights_only=True)
    assert checkpoint['source'] == source and checkpoint['lrs'] == lrs
    for arm in raw:
        raw[arm].load_state_dict(checkpoint['raw'][arm]); ema[arm].load_state_dict(checkpoint['ema'][arm])
        optimizers[arm].load_state_dict(checkpoint['optimizers'][arm])
    generator.set_state(checkpoint['generator'])
    return checkpoint


def body_hash(model):
    value = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        if not name.startswith('policy.'):
            value.update(name.encode()); value.update(tensor.cpu().contiguous().numpy().tobytes())
    return value.hexdigest()


def parent_model(parent, parent_seed, backend, policy_seed, readout, pilot=False):
    folder = Path(parent)/('pilot-0' if pilot else f'group-{parent_seed}')
    report = json.loads((folder/f'{backend}-report.json').read_text())
    assert report['seed'] == parent_seed and not report['spec']['teacher']
    assert digest(folder/f'{backend}.pt') == report['checkpoint_sha256']
    model = from_parent(torch.load(folder/f'{backend}.pt', weights_only=True),
        report['spec'], parent_seed, policy_seed, readout)
    return model, report


@torch.no_grad()
def ema_update(target, raw):
    for dst, src in zip(target.policy.parameters(), raw.policy.parameters(), strict=True):
        dst.mul_(.995).add_(src, alpha=.005)


def group(job):
    seed, parent_seed, lrs, folder, parent, pilot, smoke = job
    torch.set_num_threads(1); torch.set_deterministic_debug_mode('error')
    start = time.process_time(); folder = Path(folder); folder.mkdir(parents=True, exist_ok=True)
    protocol = json.loads(PROTOCOL.read_text()); updates = 4 if smoke else protocol['updates']
    source = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    if (folder/'summary.json').exists():
        previous = json.loads((folder/'summary.json').read_text())
        assert previous['status'] in ('COMPLETE', 'SMOKE_ONLY')
        assert all(r['source'] == source and r['lr'] == lrs[r['arm']] for r in previous['reports'])
        return previous
    assert not (folder/'failure.json').exists(), 'Do not resume numerical/cap failures as executor interruptions'
    data = torch.load(Path(parent)/'_datasets/train.pt', weights_only=True, mmap=True)
    models, raw, ema, opts, caches, parent_reports, bodies = {}, {}, {}, {}, {}, {}, {}
    for backend in BACKENDS:
        for readout in ('mlp', 'expected'):
            arm = backend.removesuffix('_outcome')+'_'+readout
            model, report = parent_model(parent, parent_seed, backend, seed, readout, pilot)
            raw[arm] = model
            ema[arm] = copy.deepcopy(model).eval().requires_grad_(False)
            opts[arm] = torch.optim.AdamW(model.policy.parameters(), lr=lrs[arm], weight_decay=1e-4)
            bodies[arm] = body_hash(model); parent_reports[arm] = report
            if backend not in caches:
                state = cached_states(model, data); caches[backend] = cached_features(model, data, state)
    generator = torch.Generator().manual_seed(seed); curves = {a: [] for a in raw}; first = 1
    resume = folder/'training-state.pt'
    if resume.exists():
        checkpoint = restore_training_state(resume, raw, ema, opts, generator, source, lrs)
        curves = checkpoint['curves']; first = checkpoint['update']+1
    cpu_before = checkpoint['cpu_seconds'] if first > 1 else 0.
    status = 'RUNNING'; step = first-1

    def cpu(): return cpu_before+time.process_time()-start

    def snapshot(step):
        save_training_state(resume, raw, ema, opts, generator, curves, step, source, lrs, cpu())

    try:
        for step in range(first, updates+1):
            ids = torch.randint(len(data['state']), (256,), generator=generator)
            si = data['state'][ids]; ni = data['next'][ids]; rows = torch.arange(len(ids))
            for arm, model in raw.items():
                backend = parent_reports[arm]['arm']; features = caches[backend]
                q = model.policy(features[si]).squeeze(-1)
                with torch.no_grad():
                    next_action = model.policy(features[ni]).squeeze(-1).argmin(-1)
                    continuation = ema[arm].policy(features[ni]).squeeze(-1)[rows, next_action]
                    target = data['train_cost'][ids].double()/40+(~data['done'][ids]).double()*continuation
                observed = q[rows, data['choice'][ids]]
                bellman = F.mse_loss(observed, target)
                conservative = (torch.logsumexp(-q, -1)+observed).mean()
                loss = bellman+.01*conservative; assert torch.isfinite(loss)
                opts[arm].zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(model.policy.parameters(), 1., error_if_nonfinite=True)
                opts[arm].step(); ema_update(ema[arm], model)
                if step in (1, updates) or step % 512 == 0:
                    curves[arm].append(dict(update=step, loss=float(loss.detach()), Bellman_MSE=float(bellman.detach()), CQL=float(conservative.detach())))
            if step == updates or step % 512 == 0:
                write(folder/'progress.json', dict(update=step, cpu_seconds=cpu(), latest={a:v[-1] for a,v in curves.items()}))
            if step == updates or step % 1024 == 0: snapshot(step)
            if cpu() > protocol['group_cpu_cap_seconds']: raise TimeoutError('Registered group CPU cap')
        status = 'COMPLETE'
    except Exception:
        import traceback
        status = 'CAPPED' if cpu() > protocol['group_cpu_cap_seconds'] else 'FAILED'
        write(folder/'failure.json', dict(error=traceback.format_exc(), last_update=step)); snapshot(step)
    reports = []
    for arm, model in ema.items():
        assert body_hash(model) == bodies[arm] == body_hash(raw[arm])
        torch.save(raw[arm].state_dict(), folder/f'{arm}-raw.pt')
        torch.save(model.state_dict(), folder/f'{arm}.pt')
        cells = []
        if status == 'COMPLETE':
            cells = [rollout(model, protocol['validation_seed']+10000000*int(smoke), h, 2 if smoke else 32) for h in (32, 128)]
        report = dict(arm=arm, seed=seed, parent_seed=parent_seed, parent_arm=parent_reports[arm]['arm'],
            parent_checkpoint_sha256=parent_reports[arm]['checkpoint_sha256'], parent_spec=parent_reports[arm]['spec'],
            source=source, readout=arm.rsplit('_', 1)[-1], body_sha256=bodies[arm], lr=lrs[arm],
            parameters=sum(p.numel() for p in model.parameters()), policy_parameters=sum(p.numel() for p in model.policy.parameters()),
            updates=step, curves=curves[arm], validation=cells, checkpoint_sha256=digest(folder/f'{arm}.pt'))
        write(folder/f'{arm}-report.json', report); reports.append(report)
    if cpu() > protocol['group_cpu_cap_seconds']: status = 'CAPPED'
    result = dict(seed=seed, parent_seed=parent_seed, status='SMOKE_ONLY' if smoke and status == 'COMPLETE' else status,
        cpu_seconds=cpu(), reports=reports)
    write(folder/'summary.json', result); return result


def eligible(report):
    return report['readout'] == 'expected' and len(report['validation']) == 2 and all(c['completion'] >= .95 and c['faults'] == 0 for c in report['validation'])


def run(directory, parent, smoke=False, resume_run=False):
    torch.set_num_threads(1); torch.set_deterministic_debug_mode('error'); start = time.process_time()
    root = Path(directory); root.mkdir(parents=True, exist_ok=resume_run); parent = Path(parent)
    assert not (root/'summary.json').exists(), 'Completed runs are replayed, not refitted'
    protocol = json.loads(PROTOCOL.read_text()); parent_summary = json.loads((parent/'summary.json').read_text())
    assert parent_summary['source'] == 'a6d8cf30ddf69fe3c14980c54e27ba47059a8854'
    assert parent_summary['complete_fresh_groups'] == 20 and parent_summary['fits'] == 88
    assert parent_summary['status'] == 'VALIDATION_REJECTED' and not parent_summary['tests_opened']
    assert not (parent/'TEST-OPENED.json').exists()
    if not smoke and subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=no'], cwd=ROOT, text=True).strip():
        raise RuntimeError('Commit registration and actual source before fitting')
    source = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    manifest = json.loads((parent/'manifest.json').read_text())
    for path in [parent/'_datasets/train.pt']+list(parent.glob('group-*/*_outcome.pt'))+list(parent.glob('pilot-0/*_outcome.pt')):
        assert digest(path) == manifest[str(path.relative_to(parent))]
    references = {str(p.relative_to(parent)): digest(p) for p in [parent/'manifest.json', parent/'summary.json', parent/'_datasets/train.pt']}
    for path in parent.glob('group-*/*_outcome-report.json'): references[str(path.relative_to(parent))] = digest(path)
    provenance = dict(source=parent_summary['source'], files_sha256=references,
        reuse='20 frozen parent encoders/worlds; ONLY newly initialized policy heads train', parent_status='VALIDATION_REJECTED')
    if (root/'parent-index.json').exists():
        assert resume_run and provenance == json.loads((root/'parent-index.json').read_text())
    else: write(root/'parent-index.json', provenance)
    recovery_charge = 0.
    if resume_run:
        records = json.loads((root/'resumptions.json').read_text()) if (root/'resumptions.json').exists() else []
        interrupted = [p.name for p in root.glob('*') if p.is_dir() and p.name.startswith(('pilot-', 'group-')) and not (p/'summary.json').exists()]
        preserved = {}
        for name in interrupted:
            folder = root/name; destination = root/'interrupted-attempts'/str(len(records))/name
            destination.mkdir(parents=True, exist_ok=False)
            preserved[name] = {}
            for path in folder.glob('*'):
                if path.is_file():
                    shutil.copy(path, destination/path.name); preserved[name][path.name] = digest(path)
        records.append(dict(interrupted=interrupted, preserved_files_sha256=preserved, source=source,
            conservative_lost_cpu_charge=len(interrupted)*protocol['group_cpu_cap_seconds'],
            reason='executor interruption only; restore atomic optimizer/RNG/raw/EMA when present, otherwise restart same seed before first snapshot'))
        write(root/'resumptions.json', records)
        recovery_charge = sum(r['conservative_lost_cpu_charge'] for r in records)
    arms = [backend.removesuffix('_outcome')+'_'+readout for backend in BACKENDS for readout in ('mlp', 'expected')]
    groups = []; offset = 10000000*int(smoke)
    preparation_cpu = time.process_time()-start
    with ProcessPoolExecutor(max_workers=2, mp_context=multiprocessing.get_context('spawn')) as executor:
        jobs = [(96900+offset, 96800, {a:lr for a in arms}, str(root/f'pilot-{i}'), str(parent), True, smoke)
                for i, lr in enumerate(protocol['lrs'][:1] if smoke else protocol['lrs'])]
        pilots = []
        for future in as_completed([executor.submit(group, job) for job in jobs]):
            value = future.result(); groups.append(value); pilots.append(value)
            write(root/'pilots.json', pilots); print('pilot', value['status'], round(value['cpu_seconds'], 1), flush=True)
        available = [r for g in pilots for r in g['reports']]
        candidate = [r for r in available if eligible(r)] if all(g['status'] in ('COMPLETE', 'SMOKE_ONLY') for g in pilots) else []
        if not candidate and not smoke: status = 'VALIDATION_REJECTED'
        else:
            def rank(r): return (-min(c['completion'] for c in r['validation']), sum(c['faults'] for c in r['validation']), sum(c['cost'] for c in r['validation']), r['lr'])
            lrs = {a:min([r for r in available if r['arm'] == a], key=rank)['lr'] for a in arms}
            winners = [r for r in candidate if r['lr'] == lrs[r['arm']]]
            primary = 'tac_expected' if smoke else min(winners, key=lambda r:(sum(c['cost'] for c in r['validation']), r['arm']))['arm']
            write(root/'selection.json', dict(primary=primary, lrs=lrs, locked_before_fresh_fits=True))
            fresh = []
            seeds = protocol['seeds'][:1] if smoke else protocol['seeds']
            for index in range(0, len(seeds), 2):
                pending = seeds[index:index+2]
                if preparation_cpu+recovery_charge+sum(g['cpu_seconds'] for g in groups)+len(pending)*protocol['group_cpu_cap_seconds'] > protocol['fit_cpu_cap_seconds']: break
                jobs = [(s+offset, 96801+s-96901, lrs, str(root/f'group-{s+offset}'), str(parent), False, smoke) for s in pending]
                for future in as_completed([executor.submit(group, job) for job in jobs]):
                    value = future.result(); groups.append(value); fresh.append(value)
                    write(root/'fresh.json', fresh); print('fresh', value['seed'], value['status'], round(value['cpu_seconds'], 1), flush=True)
            complete = len(fresh) == len(seeds) and all(g['status'] in ('COMPLETE', 'SMOKE_ONLY') for g in fresh)
            passcount = sum(r['arm'] == primary and eligible(r) for g in fresh for r in g['reports'])
            status = 'READY_FOR_TEST' if complete and passcount >= min(18,len(seeds)) else 'VALIDATION_REJECTED' if complete else 'INCOMPLETE'
            if status == 'READY_FOR_TEST' and not smoke:
                write(root/'TEST-OPENED.json', dict(source=source, selection=digest(root/'selection.json'), fresh=digest(root/'fresh.json')))
                for g in fresh:
                    for r in g['reports']:
                        model, _ = parent_model(parent, r['parent_seed'], r['parent_arm'], r['seed'], r['readout'])
                        model.load_state_dict(torch.load(root/f"group-{g['seed']}"/f"{r['arm']}.pt", weights_only=True)); model.eval()
                        cells = [rollout(model, protocol['test_seed'], h, 64) for h in (32,128,512,1024)]
                        secondary = [rollout(model, protocol['secondary_seed'], h, 32, error=.2, inspection=.4) for h in (128,1024)]
                        write(root/f"group-{g['seed']}"/f"{r['arm']}-test.json", dict(primary=cells, secondary=secondary))
                status = 'TEST_COMPLETE'
    if smoke: status = 'SMOKE_ONLY'
    summary = dict(status=status, source=source, fits=sum(len(g['reports']) for g in groups),
        preparation_and_parent_cpu_seconds=time.process_time()-start, fit_cpu_seconds=sum(g['cpu_seconds'] for g in groups),
        fitting_preparation_cpu_seconds=preparation_cpu,
        conservative_interrupted_cpu_charge=recovery_charge,
        tests_opened=(root/'TEST-OPENED.json').exists(), parent_index_sha256=digest(root/'parent-index.json'),
        whole_world_model_qualified=False, commercial_qualified=False)
    write(root/'summary.json', summary)
    write(root/'manifest.json', {str(p.relative_to(root)): digest(p) for p in root.rglob('*') if p.is_file() and p.name != 'manifest.json'})
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--directory', required=True); parser.add_argument('--parent', required=True)
    parser.add_argument('--smoke', action='store_true'); parser.add_argument('--resume', action='store_true')
    args = parser.parse_args(); run(args.directory, args.parent, args.smoke, args.resume)
