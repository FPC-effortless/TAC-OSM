"""Fresh staged-interface factorial; public teachers never enter deployment."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import copy
import hashlib
import json
import multiprocessing
from pathlib import Path
import resource
import subprocess
import sys
import time
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT/'src'))
import torch
from torch.nn import functional as F
from tac_osm.development.stable_semantic import StableSemanticAgent, configurations
from data_stable_semantic_dev067 import corpus, legacy_training, prepare, take
from run_semantic_dev065 import gate_fit, rollout, fixed_metrics
from analyze_belief_dev063 import metrics as cell_metrics
from run_learned_stack_dev055 import write, digest
PROTOCOL = ROOT/'research/dev067-stable-interface/protocol.json'
HEADS = ('actor.', 'future.', 'cost.', 'policy.')


def head_parameters(model):
    return [p for name, p in model.named_parameters() if name.startswith(HEADS)]


def encoder_parameters(model):
    return [p for name, p in model.named_parameters() if not name.startswith(HEADS) and p.requires_grad]


def make(arm, spec, seed, gate):
    model = StableSemanticAgent(spec['backend'], spec['width'], spec['aux'], seed,
        clock_mode=spec['clock_mode'], bypass=False, coordinate=spec['coordinate'])
    model.gate.load_state_dict(gate)
    for p in model.gate.parameters(): p.requires_grad_(False)
    return model


@torch.no_grad()
def update_ema(ema, raw, heads):
    source = dict(raw.named_parameters())
    for name, value in ema.named_parameters():
        if name.startswith(HEADS) == heads and source[name].requires_grad:
            value.mul_(.995).add_(source[name].to(value.dtype), alpha=.005)


def compose(encoder, actor, spec, seed, gate, raw=False):
    model = make('', spec, seed, gate).double().eval()
    state = {name: (actor if name.startswith(HEADS) else encoder).state_dict()[name].detach().clone()
             for name in model.state_dict()}
    if raw: return state
    model.load_state_dict(state)
    return model


def encoder_hash(model):
    value = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        if not name.startswith(HEADS) and name != 'temperature':
            value.update(name.encode()); value.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return value.hexdigest()


def actor_loss(model, x, posterior):
    output = model.act_from_belief(posterior, x['keys'], x['candidates'], x['prices'])
    policy = -(x['policy']*output['logits'].log_softmax(-1)).sum(-1).mean()
    future_target = x['future']/x['future'].sum(-1, keepdim=True)
    future = F.kl_div(output['future'].log_softmax(-1), future_target, reduction='batchmean')/5
    cost = F.smooth_l1_loss(output['cost'], x['cost']/40)
    loss = policy+.25*future+.1*cost
    return loss, {key: float(value.detach()) for key, value in dict(loss=loss, policy=policy, future=future, cost=cost).items()}


@torch.no_grad()
def cached_posteriors(model, data):
    return torch.cat([model.encode(x['events'], x['present'], x['keys'])[1]
        for x in (take(data, torch.arange(start, min(start+128, len(data['keys']))), torch.float64)
                  for start in range(0, len(data['keys']), 128))])


def group(job):
    seed, lrs, specs, folder, dataset, smoke = job
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    folder = Path(folder); folder.mkdir(parents=True, exist_ok=False)
    start = time.process_time(); protocol = json.loads(PROTOCOL.read_text())
    source = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    data = torch.load(Path(dataset)/'prepared-train.pt', weights_only=True, mmap=True)
    val = torch.load(Path(dataset)/'prepared-validation.pt', weights_only=True, mmap=True)
    rawval = torch.load(Path(dataset)/'validation.pt', weights_only=True)
    gate = gate_fit(); backends = sorted({s['backend'] for s in specs.values()})
    prototypes = {backend: next(s for s in specs.values() if s['backend'] == backend) for backend in backends}
    encoders = {b: make('', s, seed, gate) for b, s in prototypes.items()}
    for model in encoders.values():
        for name, p in model.named_parameters():
            if name.startswith(HEADS): p.requires_grad_(False)
    ema_encoders = {b: copy.deepcopy(m).double().eval().requires_grad_(False) for b, m in encoders.items()}
    actors = {a: make(a, s, seed, gate).double() for a, s in specs.items()}
    for model in actors.values():
        for name, p in model.named_parameters():
            if not name.startswith(HEADS): p.requires_grad_(False)
    ema_actors = {a: copy.deepcopy(m).eval().requires_grad_(False) for a, m in actors.items()}
    enc_opts = {b: torch.optim.AdamW(encoder_parameters(m), lr=lrs[b], weight_decay=1e-4) for b, m in encoders.items()}
    act_opts = {a: torch.optim.AdamW(head_parameters(m), lr=lrs[specs[a]['backend']], weight_decay=1e-4) for a, m in actors.items()}
    enc_steps = 8 if smoke else protocol['encoder_updates']; act_steps = 8 if smoke else protocol['actor_updates']
    warm = 2 if smoke else protocol['warmup']
    generator = torch.Generator().manual_seed(seed)
    enc_indices = torch.randint(len(data['keys']), (enc_steps, 64), generator=generator)
    act_indices = torch.randint(len(data['keys']), (act_steps, 64), generator=generator)
    masks = torch.rand(act_steps, 64, generator=generator) < .5
    encoder_curves = {b: [] for b in backends}; actor_curves = {a: [] for a in actors}
    actor_counts = {a: 0 for a in actors}; encoder_count = 0; status = 'RUNNING'
    validation_ids = torch.arange(min(128, len(val['keys'])))

    def check_budget():
        if time.process_time()-start > protocol['group_cpu_cap_seconds']:
            raise TimeoutError('Registered seed-group CPU cap')

    def train_actor(arm, step, q, x, version):
        model = actors[arm]
        belief = torch.where(masks[step-1, :, None], x['posterior'], q)
        assert not belief.requires_grad
        loss, values = actor_loss(model, x, belief)
        assert torch.isfinite(loss)
        act_opts[arm].zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(head_parameters(model), 1., error_if_nonfinite=True)
        act_opts[arm].step(); update_ema(ema_actors[arm], model, True)
        actor_counts[arm] += 1
        if step in (1, act_steps) or step % 256 == 0:
            with torch.no_grad():
                v = take(val, validation_ids, torch.float64)
                vp = ema_encoders[specs[arm]['backend']].encode(v['events'], v['present'], v['keys'])[1]
                _, predicted = actor_loss(ema_actors[arm], v, vp)
                _, exact = actor_loss(ema_actors[arm], v, v['posterior'])
            actor_curves[arm].append(dict(actor_update=step, encoder_version=version,
                train=values, validation_predicted=predicted, validation_exact=exact))
            write(folder/f'{arm}-progress.json', actor_curves[arm])

    try:
        for step in range(1, enc_steps+1):
            x = take(data, enc_indices[step-1], torch.float32)
            for backend, model in encoders.items():
                latent, _ = model.encode(x['events'], x['present'], x['keys'])
                logp = model.belief(latent).log_softmax(-1).double()
                target = x['posterior']/x['posterior'].sum(-1, keepdim=True)
                kl = F.kl_div(logp, target, reduction='batchmean'); loss = 4*kl
                assert torch.isfinite(loss)
                enc_opts[backend].zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(encoder_parameters(model), 1., error_if_nonfinite=True)
                enc_opts[backend].step(); update_ema(ema_encoders[backend], model, False)
                if step in (1, warm, enc_steps) or step % 256 == 0:
                    with torch.no_grad():
                        v = take(val, validation_ids, torch.float64)
                        _, vp = ema_encoders[backend].encode(v['events'], v['present'], v['keys'])
                        validation_kl = F.kl_div(vp.clamp_min(1e-12).log(), v['posterior'], reduction='batchmean')
                    encoder_curves[backend].append(dict(update=step, train_KL=float(kl.detach()), validation_KL=float(validation_kl)))
                    write(folder/f'{backend}-encoder-progress.json', encoder_curves[backend])
            encoder_count = step
            if step > warm:
                actor_step = step-warm; ax = take(data, act_indices[actor_step-1], torch.float64)
                with torch.no_grad(): q = ema_encoders['tac'].encode(ax['events'], ax['present'], ax['keys'])[1]
                for arm, spec in specs.items():
                    if spec['schedule'] == 'moving': train_actor(arm, actor_step, q, ax, step)
            check_budget()
        cached = {b: cached_posteriors(m, data) for b, m in ema_encoders.items()}
        moving_done = enc_steps-warm
        for step in range(1, act_steps+1):
            x = take(data, act_indices[step-1], torch.float64)
            for arm, spec in specs.items():
                if spec['schedule'] == 'frozen' or step > moving_done:
                    train_actor(arm, step, cached[spec['backend']][act_indices[step-1]], x, enc_steps)
            check_budget()
        assert all(n == act_steps for n in actor_counts.values())
        status = 'COMPLETE'
    except Exception:
        import traceback
        status = 'CAPPED' if time.process_time()-start > protocol['group_cpu_cap_seconds'] else 'FAILED'
        write(folder/'failure.json', dict(error=traceback.format_exc(), encoder_updates=encoder_count, actor_updates=actor_counts))
    reports = []
    for arm, spec in specs.items():
        dest = folder/arm; dest.mkdir(); backend = spec['backend']
        torch.save(compose(encoders[backend], actors[arm], spec, seed, gate, raw=True), dest/'raw.pt')
        model = compose(ema_encoders[backend], ema_actors[arm], spec, seed, gate)
        fm, cells = None, []
        if status == 'COMPLETE':
            fm = fixed_metrics(model, rawval, copy.deepcopy(model.gate).float())
            temperature = min(fm['policy_temperature_NLL'], key=lambda t: (fm['policy_temperature_NLL'][t], t))
            model.temperature.fill_(temperature)
            cells = [rollout(model, 6671002+10000000*int(smoke), h, 2 if smoke else 32)[0] for h in (32, 128)]
        torch.save(model.state_dict(), dest/'agent.pt')
        report = dict(arm=arm, seed=seed, spec=spec, lr=lrs[backend], status=status,
            updates=encoder_count, actor_updates=actor_counts[arm], encoder_shared=backend,
            encoder_sha256=encoder_hash(model), curves=encoder_curves[backend],
            actor_curves=actor_curves[arm], validation_fixed=fm, validation=cells,
            checkpoint_sha256=digest(dest/'agent.pt'), source=source)
        write(dest/'report.json', report); reports.append(report)
    assert len({r['encoder_sha256'] for r in reports if r['spec']['backend'] == 'tac'}) == 1
    if status == 'COMPLETE' and time.process_time()-start > protocol['group_cpu_cap_seconds']: status = 'CAPPED'
    summary = dict(seed=seed, status='SMOKE_ONLY' if smoke and status == 'COMPLETE' else status,
        cpu_seconds=time.process_time()-start, unique_encoder_fits=len(backends), actor_fits=len(specs), reports=reports)
    write(folder/'summary.json', summary); return summary


def run(directory, legacy_directory, smoke=False):
    torch.set_num_threads(1); start = time.process_time(); children = resource.getrusage(resource.RUSAGE_CHILDREN)
    root = Path(directory); root.mkdir(parents=True, exist_ok=False); protocol = json.loads(PROTOCOL.read_text())
    specs = configurations(); write(root/'specs.json', specs)
    if not smoke and subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=ROOT,text=True).strip():
        raise RuntimeError('Commit the registration/fitting source before training')
    offset = 10000000*int(smoke); dataset = root/'_datasets'; dataset.mkdir()
    training = corpus(6671001+offset, 8 if smoke else protocol['snapshots'])
    if not smoke:
        assert digest(ROOT/'research/dev065-semantic-stability/evidence-index.json') == protocol['legacy_index_sha256']
        assert digest(Path(legacy_directory)/'manifest.json') == protocol['legacy_manifest_sha256']
        visits, provenance = legacy_training(legacy_directory, protocol['legacy_snapshots'], 6671011)
        training += visits; write(dataset/'legacy-training-provenance.json', provenance)
    validation = corpus(6671002+offset, 8 if smoke else protocol['validation_snapshots'])
    gate = make('', next(iter(specs.values())), 0, gate_fit()).gate
    for name, records in [('train', training), ('validation', validation)]:
        torch.save(records, dataset/f'{name}.pt'); torch.save(prepare(records, gate), dataset/f'prepared-{name}.pt')
    del training, validation
    preparation = time.process_time()-start; pilots, fits, groups = [], [], []; candidate = None
    with ProcessPoolExecutor(max_workers=2, mp_context=multiprocessing.get_context('spawn')) as executor:
        jobs = [(96700+offset, {b:lr for b in ('tac','gru','no_casm')}, specs,
                 str(root/f'pilot-{index}'), str(dataset), smoke)
                for index, lr in enumerate(protocol['lrs'][:1] if smoke else protocol['lrs'])]
        for future in as_completed([executor.submit(group, job) for job in jobs]):
            g = future.result(); pilots.append(g); groups.append(g); write(root/'pilots.json', pilots)
            print('pilot', g['status'], round(g['cpu_seconds'],1), flush=True)
        complete = all(g['status'] == ('SMOKE_ONLY' if smoke else 'COMPLETE') for g in pilots)
        if complete:
            selected = {}
            for backend in ('tac','gru','no_casm'):
                candidates = [r for g in pilots for r in g['reports'] if r['spec']['backend'] == backend]
                selected[backend] = min(candidates, key=lambda r:(
                    max(r['validation_fixed']['p95_KL']/.05,r['validation_fixed']['p99_KL']/.1),
                    r['validation_fixed']['posterior_KL'],r['lr']))['lr']
            eligible = []
            for arm, spec in specs.items():
                report = next(r for g in pilots for r in g['reports'] if r['arm']==arm and r['lr']==selected[spec['backend']])
                measured = cell_metrics(report['validation']); fixed = report['validation_fixed']
                if fixed['p95_KL']<=.05 and fixed['p99_KL']<=.1 and measured['faults']==0 and all(cell_metrics([c])['completion']>=.95 for c in report['validation']):
                    eligible.append((round(measured['cost'],6), spec['parameters'], list(specs).index(arm), arm))
            candidate = min(eligible)[-1] if eligible else None
            write(root/'selection.json', dict(encoder_lrs=selected, candidate=candidate,
                decision='SMOKE_ONLY' if smoke else 'VALIDATION_ELIGIBLE' if candidate else 'VALIDATION_REJECTED_NO_TEST_OPEN'))
            if smoke or candidate is not None:
                seeds = protocol['seeds'][:1] if smoke else protocol['seeds']
                jobs = [(seed+offset, selected, specs, str(root/f'group-{seed+offset}'), str(dataset), smoke) for seed in seeds]
                for index in range(0,len(jobs),2):
                    if preparation+sum(g['cpu_seconds'] for g in groups)+len(jobs[index:index+2])*protocol['group_cpu_cap_seconds']>protocol['fit_cpu_cap_seconds']: break
                    for future in as_completed([executor.submit(group,job) for job in jobs[index:index+2]]):
                        g=future.result();fits.append(g);groups.append(g);write(root/'groups.json',fits)
                        print('group',g['seed'],g['status'],round(g['cpu_seconds'],1),flush=True)
    end = resource.getrusage(resource.RUSAGE_CHILDREN)
    cpu = time.process_time()-start+end.ru_utime+end.ru_stime-children.ru_utime-children.ru_stime
    complete = len(fits)==(1 if smoke else len(protocol['seeds'])) and all(g['status']==('SMOKE_ONLY' if smoke else 'COMPLETE') for g in fits) and cpu<=protocol['fit_cpu_cap_seconds']
    summary = dict(status='SMOKE_ONLY' if smoke and complete else 'INCOMPLETE' if smoke else 'FITS_COMPLETE' if complete else 'VALIDATION_REJECTED' if not fits and candidate is None and pilots and all(g['status']=='COMPLETE' for g in pilots) else 'INCOMPLETE',
        fits=len(fits)*len(specs), pilot_fits=len(pilots)*len(specs),
        unique_encoder_fits=sum(g['unique_encoder_fits'] for g in fits),
        unique_pilot_encoder_fits=sum(g['unique_encoder_fits'] for g in pilots),
        fitting_cpu_seconds=cpu, test_opened=False, protocol_sha256=digest(PROTOCOL))
    write(root/'summary.json',summary)
    if not smoke and complete:
        # Permanent marker precedes any test generation, including a partial failure.
        with (root/'TEST-EXPOSURE.json').open('x') as stream:
            json.dump(dict(protocol_sha256=digest(PROTOCOL), reason='All registered fresh fits complete; opening test once'),stream)
        summary.update(test_opened=True,status='EVALUATING');write(root/'summary.json',summary)
        fixed=corpus(6671005,512);torch.save(fixed,root/'test-fixed.pt')
        for g in sorted(fits,key=lambda g:g['seed']):
            for report in g['reports']:
                folder=root/f"group-{report['seed']}"/report['arm'];model=make(report['arm'],report['spec'],report['seed'],gate_fit()).double().eval()
                model.load_state_dict(torch.load(folder/'agent.pt',weights_only=True))
                fm=fixed_metrics(model,fixed,copy.deepcopy(model.gate).float())
                cells=[rollout(model,6671003,h,64)[0] for h in (32,128,512,1024)]
                cells += [rollout(model,6671004,32,64,error=e,inspection=price)[0] for e,price in ((.2,.1),(.1,.4))]
                write(folder/'test.json',dict(fixed=fm,cells=cells));print('tested',report['arm'],report['seed'],flush=True)
        summary.update(status='COMPLETE');write(root/'summary.json',summary)
    write(root/'manifest.json',{str(p.relative_to(root)):digest(p) for p in sorted(root.rglob('*')) if p.is_file() and p.name!='manifest.json'})


if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('--directory',required=True);p.add_argument('--legacy-directory',default='results/dev065-main');p.add_argument('--smoke',action='store_true')
    a=p.parse_args();run(a.directory,a.legacy_directory,a.smoke)
