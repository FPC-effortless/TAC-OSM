"""Frozen-source, staged public-outcome experiment; test access is gated."""
import argparse
import copy
import json
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import resource
import subprocess
import sys
import time
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'scripts')]
import numpy as np
import torch
from torch.nn import functional as F
from tac_osm.development.public_outcome import PublicOutcomeAgent, configurations
from tac_osm.development.belief_agent import choose
from data_public_outcome_dev068 import (PublicMaintenance, corpus, graph_from_public,
                                        state_batch, teacher_targets)
from run_semantic_dev065 import gate_fit
from run_learned_stack_dev055 import write, digest
PROTOCOL = ROOT/'research/dev068-public-outcome/protocol.json'
WORLD = ('actor.', 'future.', 'cost.')


def stage_parameters(model, stage):
    def selected(name):
        if stage == 'world': return name.startswith(WORLD)
        if stage == 'value': return name.startswith('policy.')
        return not name.startswith(WORLD+('policy.', 'gate.'))
    result = []
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(selected(name))
        if parameter.requires_grad: result.append(parameter)
    return result


@torch.no_grad()
def ema_update(target, model, decay=.995):
    for destination, source in zip(target.parameters(), model.parameters(), strict=True):
        if source.requires_grad: destination.mul_(decay).add_(source.to(destination.dtype), alpha=1-decay)


def make(spec, seed, gate):
    model = PublicOutcomeAgent(spec['backend'], spec['width'], spec['aux'], seed)
    model.gate.load_state_dict(gate)
    return model


@torch.no_grad()
def cached_states(model, data):
    return torch.cat([model.encode(x['events'], x['present'], x['keys'])[1]
        for x in (state_batch(data, torch.arange(i, min(i+128, len(data['keys']))), torch.float64)
                  for i in range(0, len(data['keys']), 128))])


@torch.no_grad()
def cached_features(model, data, states):
    features = []
    for start in range(0, len(states), 256):
        ids = torch.arange(start, min(start+256, len(states)))
        features.append(model.world_from_state(states[ids], data['keys'][ids].double(),
            data['candidates'][ids].double(), data['prices'][ids].double(),
            data['remaining'][ids].double())['features'])
    return torch.cat(features)


@torch.no_grad()
def public_metrics(model, data):
    q = cached_states(model, data)
    nll = F.nll_loss(q.clamp_min(1e-12).log(), data['label'])
    brier = ((q-F.one_hot(data['label'], 12))**2).sum(-1).mean()
    rows = torch.arange(len(data['state']))
    ids = data['state']
    out = model.world_from_state(q[ids], data['keys'][ids].double(),
        data['candidates'][ids].double(), data['prices'][ids].double(), data['remaining'][ids].double())
    future = F.cross_entropy(out['future'][rows, data['choice']], data['future_label'])
    cost = F.mse_loss(out['cost'][rows, data['choice']], data['physical_cost'].double()/40)
    return dict(measurement_NLL=float(nll), measurement_Brier=float(brier),
                factual_future_NLL=float(future), factual_cost_MSE=float(cost))


@torch.no_grad()
def rollout(model, seed, history, count, error=.1, inspection=.1, intervention=None):
    # PublicBelief is EVALUATION ONLY, not a model input or target in outcome arms.
    from data_belief_dev062 import PriceBelief
    worlds = [PublicMaintenance(seed, i, history, error, inspection) for i in range(count)]
    keys = torch.tensor(np.stack([w.keys() for w in worlds]), dtype=torch.float64)
    state = model.initial(keys)
    references, graphs = [], []
    for i, env in enumerate(worlds):
        events = env.observations(); graph = graph_from_public(events, env.keys()); graphs.append(graph)
        parents = [-1, -1]
        if graph == 1: parents[1] = 0
        elif graph == 2: parents[0] = 1
        belief = PriceBelief(parents, error, inspection=inspection)
        for event in events:
            if event[0] > .5: belief.condition(int((env.keys() @ event[6:22]).argmax()), int(event[38]))
        references.append(belief)
        value = torch.tensor(events, dtype=torch.float64)
        keep = model.gate(value[:, :6]).squeeze(-1) >= 0
        selected = value[keep][-64:]; state['events'][i, :len(selected)] = selected
        state['present'][i, :len(selected)] = True
    done = [False]*count; telemetry = [dict(actions=[], KL=[], margins=[], exact_flip=[]) for _ in worlds]
    for depth in range(10):
        if all(done): break
        payload = [w.choices() for w in worlds]
        candidates = keys.new_tensor(np.stack([p[0] for p in payload]))
        prices = keys.new_tensor([[error, inspection]]*count); remaining = keys.new_full((count, 1), (10-depth)/10)
        out = model(state['events'], state['present'], keys, candidates, prices, remaining)
        exact = out['posterior'].new_zeros(count, 12)
        for i, reference in enumerate(references): exact[i].reshape(4, 3)[:, graphs[i]] = keys.new_tensor(reference.p)
        exact_world = model.world_from_state(exact, keys, candidates, prices, remaining)
        exact_choice = choose(-model.policy(exact_world['features']).squeeze(-1), torch.ones(count, 5, dtype=torch.bool))
        if intervention is not None:
            if intervention == 'exact': posterior = exact
            elif intervention == 'blurred': posterior = .5*exact+torch.ones_like(exact)/24
            else: raise ValueError(intervention)
            world = model.world_from_state(posterior, keys, candidates, prices, remaining)
            out['logits'] = -model.policy(world['features']).squeeze(-1)
        choices = choose(out['logits'], torch.ones(count, 5, dtype=torch.bool))
        pending = []
        for i, env in enumerate(worlds):
            if done[i]: pending.append([]); continue
            posterior = out['posterior'][i].clamp_min(1e-12)
            kl = F.kl_div(posterior.log(), exact[i], reduction='sum')
            top = out['logits'][i].topk(2).values
            action = int(payload[i][1][int(choices[i])])
            receipt, events = env.execute(action); references[i].update(action, receipt)
            telemetry[i]['actions'].append(action); telemetry[i]['KL'].append(float(kl))
            telemetry[i]['margins'].append(float(top[0]-top[1]))
            telemetry[i]['exact_flip'].append(bool(choices[i] != exact_choice[i]))
            done[i] = action == 0; pending.append(events)
        for tick in range(max(map(len, pending), default=0)):
            present = torch.tensor([tick < len(e) for e in pending])
            event = keys.new_tensor(np.stack([e[tick] if tick < len(e) else np.zeros(46) for e in pending]))
            state = model.observe(state, event, present)
    rows = [dict(env.verify(), **trace) for env, trace in zip(worlds, telemetry, strict=True)]
    kls = [v for row in rows for v in row['KL']]
    return dict(seed=seed, history=history, error=error, inspection=inspection, rows=rows,
        completion=float(np.mean([r['completion'] for r in rows])), faults=sum(r['faults'] for r in rows),
        cost=float(np.mean([r['cost'] for r in rows])), p95_KL=float(np.quantile(kls, .95)),
        p99_KL=float(np.quantile(kls, .99)))


def group(job):
    seed, lrs, specs, folder, directory, smoke = job
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    start = time.process_time(); folder = Path(folder); folder.mkdir(parents=True, exist_ok=False)
    protocol = json.loads(PROTOCOL.read_text()); dataset = Path(directory)
    source = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    data = torch.load(dataset/'train.pt', mmap=True, weights_only=True)
    validation = torch.load(dataset/'validation.pt', mmap=True, weights_only=True)
    # Only the explicitly named teacher control may load this separate tensor.
    teacher = torch.load(dataset/'teacher-only.pt', weights_only=True) if any(s['teacher'] for s in specs.values()) else None
    gate = torch.load(dataset/'gate.pt', weights_only=True)
    models = {a: make(s, seed, gate) for a, s in specs.items()}
    curves = {a: [] for a in specs}; reports = []; status = 'RUNNING'
    sizes = dict(encoder=64, world=128, value=256)
    generator = torch.Generator().manual_seed(seed)

    def check_budget():
        if time.process_time()-start > protocol['group_cpu_cap_seconds']: raise TimeoutError('Registered seed CPU cap')

    try:
        for arm, model in models.items():
            params = stage_parameters(model, 'encoder')
            opt = torch.optim.AdamW(params, lr=lrs[arm], weight_decay=1e-4)
            ema = copy.deepcopy(model).double().eval().requires_grad_(False)
            steps = 4 if smoke else protocol['encoder_updates']
            for step in range(1, steps+1):
                ids = torch.randint(len(data['keys']), (sizes['encoder'],), generator=generator)
                x = state_batch(data, ids, torch.float32)
                latent, _ = model.encode(x['events'], x['present'], x['keys'])
                logp = model.belief(latent).log_softmax(-1)
                loss = (F.kl_div(logp.double(), teacher[ids], reduction='batchmean')
                        if specs[arm]['teacher'] else F.nll_loss(logp, x['label']))
                assert torch.isfinite(loss)
                opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(params, 1., error_if_nonfinite=True)
                opt.step(); ema_update(ema, model)
                if step in (1, steps) or step % 512 == 0:
                    v = state_batch(validation, torch.arange(min(512, len(validation['keys']))), torch.float64)
                    with torch.no_grad():
                        _, pred = ema.encode(v['events'], v['present'], v['keys'])
                        score = F.nll_loss(pred.clamp_min(1e-12).log(), v['label'])
                    curves[arm].append(dict(stage='encoder', update=step, loss=float(loss.detach()), validation_measurement_NLL=float(score)))
                    write(folder/f'{arm}-progress.json', curves[arm]); check_budget()
            model.double(); model.load_state_dict(ema.state_dict())
            cached = cached_states(model, data)
            params = stage_parameters(model, 'world')
            opt = torch.optim.AdamW(params, lr=.001, weight_decay=1e-4)
            ema = copy.deepcopy(model).eval().requires_grad_(False)
            steps = 4 if smoke else protocol['world_updates']
            for step in range(1, steps+1):
                ids = torch.randint(len(data['state']), (sizes['world'],), generator=generator)
                si = data['state'][ids]; rows = torch.arange(len(ids))
                out = model.world_from_state(cached[si], data['keys'][si].double(),
                    data['candidates'][si].double(), data['prices'][si].double(), data['remaining'][si].double())
                future_loss = F.cross_entropy(out['future'][rows, data['choice'][ids]], data['future_label'][ids])
                cost_loss = F.mse_loss(out['cost'][rows, data['choice'][ids]], data['physical_cost'][ids].double()/40)
                loss = future_loss+cost_loss
                assert torch.isfinite(loss)
                opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(params, 1., error_if_nonfinite=True)
                opt.step(); ema_update(ema, model)
                if step in (1, steps) or step % 512 == 0:
                    curves[arm].append(dict(stage='world', update=step, loss=float(loss.detach()), NLL=float(future_loss.detach()), MSE=float(cost_loss.detach())))
                    write(folder/f'{arm}-progress.json', curves[arm]); check_budget()
            model.load_state_dict(ema.state_dict())
            features = cached_features(model, data, cached)
            params = stage_parameters(model, 'value')
            opt = torch.optim.AdamW(params, lr=.001, weight_decay=1e-4)
            ema = copy.deepcopy(model).eval().requires_grad_(False)
            steps = 4 if smoke else protocol['value_updates']
            for step in range(1, steps+1):
                ids = torch.randint(len(data['state']), (sizes['value'],), generator=generator)
                si = data['state'][ids]; ni = data['next'][ids]; rows = torch.arange(len(ids))
                q = model.policy(features[si]).squeeze(-1)
                with torch.no_grad():
                    action = model.policy(features[ni]).squeeze(-1).argmin(-1)
                    continuation = ema.policy(features[ni]).squeeze(-1)[rows, action]
                    target = data['train_cost'][ids].double()/40+(~data['done'][ids]).double()*continuation
                observed = q[rows, data['choice'][ids]]
                bellman = F.mse_loss(observed, target)
                conservative = (torch.logsumexp(-q, -1)+observed).mean()
                loss = bellman+protocol['CQL_alpha']*conservative
                assert torch.isfinite(loss)
                opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(params, 1., error_if_nonfinite=True)
                opt.step(); ema_update(ema, model)
                if step in (1, steps) or step % 512 == 0:
                    curves[arm].append(dict(stage='value', update=step, loss=float(loss.detach()), Bellman_MSE=float(bellman.detach()), CQL=float(conservative.detach())))
                    write(folder/f'{arm}-progress.json', curves[arm]); check_budget()
            model.load_state_dict(ema.state_dict()); model.eval().requires_grad_(False)
            torch.save(model.state_dict(), folder/f'{arm}.pt')
            validation_fixed = public_metrics(model, validation)
            cells = [rollout(model, protocol['validation_seed']+10000000*int(smoke), h, 2 if smoke else 32) for h in (32, 128)]
            report = dict(arm=arm, spec=specs[arm], seed=seed, lr=lrs[arm], source=source,
                checkpoint_sha256=digest(folder/f'{arm}.pt'), curves=curves[arm],
                validation_fixed=validation_fixed, validation=cells)
            reports.append(report); write(folder/f'{arm}-report.json', report)
            print('fit', seed, arm, round(time.process_time()-start, 1), flush=True)
            check_budget()
        status = 'COMPLETE'
    except Exception:
        import traceback
        status = 'CAPPED' if time.process_time()-start > protocol['group_cpu_cap_seconds'] else 'FAILED'
        write(folder/'failure.json', dict(error=traceback.format_exc(), fitted_arms=list(r['arm'] for r in reports)))
    summary = dict(seed=seed, status='SMOKE_ONLY' if smoke and status == 'COMPLETE' else status,
        cpu_seconds=time.process_time()-start, reports=reports)
    write(folder/'summary.json', summary)
    return summary


def eligible(report, prior_nll):
    return (not report['spec']['teacher'] and
            report['validation_fixed']['measurement_NLL'] <= prior_nll-.02 and
            all(c['completion'] >= .95 and c['faults'] == 0 for c in report['validation']))


def run(directory, smoke=False):
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    protocol = json.loads(PROTOCOL.read_text()); start = time.process_time()
    root = Path(directory); root.mkdir(parents=True, exist_ok=False)
    if not smoke and subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=no'], cwd=ROOT, text=True).strip():
        raise RuntimeError('Commit source/registration before fitting')
    source = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    specs = configurations(); write(root/'specs.json', specs)
    dataset = root/'_datasets'; dataset.mkdir(); gate = gate_fit(); torch.save(gate, dataset/'gate.pt')
    prototype = make(specs['tac_outcome'], 96800, gate)
    for name, seed, count in [('train', protocol['train_seed'], protocol['transitions']),
                              ('validation', protocol['validation_seed'], protocol['validation_transitions'])]:
        data, episodes, coverage = corpus(seed+10000000*int(smoke), 24 if smoke else count, prototype.gate)
        torch.save(data, dataset/f'{name}.pt'); write(dataset/f'{name}-traces.json', episodes)
        write(dataset/f'{name}-coverage.json', coverage)
        if name == 'train': torch.save(teacher_targets(episodes, len(data['keys'])), dataset/'teacher-only.pt')
    data = torch.load(dataset/'train.pt', weights_only=True, mmap=True)
    val = torch.load(dataset/'validation.pt', weights_only=True, mmap=True)
    # Empirical train-only graph-conditioned paired-measurement prior.
    counts = torch.ones(3, 4, dtype=torch.float64)
    for label in data['label']: counts[int(label)%3, int(label)//3] += 1
    probabilities = counts/counts.sum(-1, keepdim=True)
    prior_nll = float(-probabilities[val['label']%3, val['label']//3].log().mean())
    write(root/'prior.json', dict(counts=counts.tolist(), validation_measurement_NLL=prior_nll))
    write(root/'data-manifest.json', {p.name: digest(p) for p in dataset.iterdir()})
    preparation = time.process_time()-start; groups = []
    offset = 10000000*int(smoke)
    with ProcessPoolExecutor(max_workers=2, mp_context=multiprocessing.get_context('spawn')) as executor:
        jobs = [(96800+offset, {a: lr for a in specs}, specs, str(root/f'pilot-{i}'), str(dataset), smoke)
                for i, lr in enumerate(protocol['encoder_lrs'][:1] if smoke else protocol['encoder_lrs'])]
        pilots = []
        for f in as_completed([executor.submit(group, j) for j in jobs]):
            result = f.result(); pilots.append(result); groups.append(result)
            write(root/'pilots.json', pilots)
            print('pilot', result['status'], round(result['cpu_seconds'], 1), flush=True)
        complete = smoke or all(g['status'] == 'COMPLETE' for g in pilots)
        candidates = [r for g in pilots for r in g['reports'] if eligible(r, prior_nll)] if complete else []
        if candidates or smoke:
            lrs = {a: min([r for g in pilots for r in g['reports'] if r['arm'] == a],
                key=lambda r: (r['validation_fixed']['measurement_NLL'], r['lr']))['lr'] for a in specs}
            # Each arm's selected LR must pass its screen for an outcome candidate.
            winners = [r for r in candidates if r['lr'] == lrs[r['arm']]]
            if not smoke and not winners:
                status = 'VALIDATION_REJECTED'
            else:
                primary = 'tac_outcome' if smoke else min(winners, key=lambda r:
                    (sum(c['cost'] for c in r['validation']), r['arm']))['arm']
                write(root/'selection.json', dict(lrs=lrs, primary=primary, prior_NLL=prior_nll,
                    locked_before_fresh_fits=True, eligible_pilot_arms=sorted({r['arm'] for r in winners})))
                seeds = protocol['seeds'][:1] if smoke else protocol['seeds']
                jobs = [(s+offset, lrs, specs, str(root/f'group-{s+offset}'), str(dataset), smoke) for s in seeds]
                fresh = []
                for start_id in range(0, len(jobs), 2):
                    used = preparation+sum(g['cpu_seconds'] for g in groups)
                    if used+len(jobs[start_id:start_id+2])*protocol['group_cpu_cap_seconds'] > protocol['fit_cpu_cap_seconds']:
                        break
                    for f in as_completed([executor.submit(group, j) for j in jobs[start_id:start_id+2]]):
                        result = f.result(); fresh.append(result); groups.append(result)
                        write(root/'fresh.json', fresh); print('fresh', result['seed'], result['status'], flush=True)
                complete = len(fresh) == len(seeds) and all(g['status'] in ('COMPLETE', 'SMOKE_ONLY') for g in fresh)
                passed = [r for g in fresh for r in g['reports'] if r['arm'] == primary and eligible(r, prior_nll)]
                status = 'VALIDATION_REJECTED' if complete and len(passed) < min(18, len(seeds)) else 'READY_FOR_TEST' if complete else 'INCOMPLETE'
                if status == 'READY_FOR_TEST' and not smoke:
                    write(root/'TEST-OPENED.json', dict(source=source, selection=digest(root/'selection.json'), fresh=digest(root/'fresh.json')))
                    for g in fresh:
                        for arm, spec in specs.items():
                            model = make(spec, g['seed'], gate).double().eval()
                            model.load_state_dict(torch.load(root/f"group-{g['seed']}"/f'{arm}.pt', weights_only=True))
                            cells = [rollout(model, protocol['test_seed'], h, protocol['test_worlds']) for h in protocol['histories']]
                            secondary = [rollout(model, protocol['secondary_seed'], h, 32, error=.2, inspection=.4) for h in (128, 1024)]
                            write(root/f"group-{g['seed']}"/f'{arm}-test.json', dict(primary=cells, secondary=secondary))
                    status = 'TEST_COMPLETE'
        else: status = 'VALIDATION_REJECTED'
    if smoke: status = 'SMOKE_ONLY'
    summary = dict(status=status, source=source, preparation_cpu_seconds=preparation,
        fit_cpu_seconds=sum(g['cpu_seconds'] for g in groups), fits=sum(len(g['reports']) for g in groups),
        tests_opened=(root/'TEST-OPENED.json').exists(), commercial_qualified=False)
    write(root/'summary.json', summary)
    write(root/'manifest.json', {str(p.relative_to(root)): digest(p) for p in sorted(root.rglob('*'))
                                if p.is_file() and p.name != 'manifest.json'})
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--directory', required=True)
    parser.add_argument('--smoke', action='store_true'); args = parser.parse_args()
    run(args.directory, args.smoke)
