"""DEV-040A bounded development pilot; frozen-head/probe controls, no tuning."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import random
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts')]
import torch  # noqa: E402
from torch import nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from tac_osm.development.bounded_mtsk import build_bounded  # noqa: E402
from tac_osm.development.entity_aging import EntityAgingControl  # noqa: E402
from tac_osm.development.memory_baselines import MemoryBaseline  # noqa: E402
from tac_osm.development.finite_clipping import clip_finite_cpu_gradients_  # noqa: E402
from tac_osm.development.schema_maintenance import make_pair, ROLES, DEADLINES, execute, exact_public_targets  # noqa: E402
from tac_osm.development.delayed_maintenance import Action  # noqa: E402
from tac_osm.development.noisy_memory import parameter_hash  # noqa: E402
from run_learned_schema_dev import public_stream_tensors, public_queries, labels  # noqa: E402

ARMS = ('bounded_mtsk', 'masked_local', 'schema41')
HEADS = ('original', 'mlp_uniform', 'mlp_balanced', 'linear_uniform', 'linear_balanced')
PROBES = ('trained', 'random', 'reset_writer', 'query', 'current', 'shuffled')
THRESHOLDS = (0., .05, .1, .15, .2, .25, .3, .4, .5, .6, .7, .75, .8, .9, 1.)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(arm, seed):
    if arm == 'bounded_mtsk':
        return build_bounded(seed)
    if arm == 'masked_local':
        return EntityAgingControl('masked_local', seed)
    if arm == 'schema41':
        return MemoryBaseline('schema41', seed)
    raise ValueError('unregistered architecture')


def weights(y, conditions, balanced):
    if not balanced:
        return torch.ones_like(y, dtype=torch.float32) / y.numel()
    focus = torch.zeros_like(y, dtype=torch.bool)
    for i, c in enumerate(conditions):
        if c['deadline'] != 'before':
            focus[i, c['target_entity'], 0] = True
    if not focus.any() or focus.all():
        raise ValueError('both focus and background labels required')
    return .5 * focus.float() / focus.sum() + .5 * (~focus).float() / (~focus).sum()


def train_backbone(model, seed, spec, smoke, output, arm):
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=.002, weight_decay=.0001)
    fps, curve = [], []
    start = time.monotonic()
    try:
        for step in range(2 if smoke else spec['backbone_updates']):
            cases = make_pair(seed * 100000 + step, 32 if smoke else 512,
                              (32, 128, 512)[(step // 4) % 3], ROLES[step % 4],
                              DEADLINES[(step // 12) % 3], domain='dev040a-backbone-train-v1')
            fps.extend(c.fingerprint for c in cases)
            x = public_stream_tensors([c.observations for c in cases])
            q = public_queries([c.tick for c in cases], [c.horizon for c in cases])
            y = labels(cases)
            receipts = [(c.world.act(Action('inspect', random.Random(f'dev040a:{seed}:{step}:{b}').randrange(8))),)
                        for b, c in enumerate(cases)]
            obs = public_stream_tensors(receipts)
            optimizer.zero_grad(set_to_none=True)
            state = model.encode(*x)
            pre = model.decode(state, q)
            post = model.decode(model.update(state, *(v[:, 0] for v in obs)), q)
            loss = (F.cross_entropy(pre.reshape(-1, 2), y.reshape(-1)) +
                    F.cross_entropy(post.reshape(-1, 2), y.reshape(-1))) / 2
            if not torch.isfinite(loss):
                raise FloatingPointError('nonfinite backbone loss')
            loss.backward()
            norm = clip_finite_cpu_gradients_(model.parameters(), 1.)
            optimizer.step()
            if step % 36 == 0:
                curve.append(dict(update=step + 1, loss=float(loss.detach()), norm=norm))
                print('PROGRESS ' + json.dumps(dict(seed=seed, arm=model.arm, **curve[-1])), flush=True)
    except Exception as error:
        torch.save(dict(model=model.state_dict(), optimizer=optimizer.state_dict(), rng=torch.get_rng_state()),
                   output / f'{arm}-failure.pt')
        (output / f'{arm}-failure.json').write_text(json.dumps(dict(arm=arm, seed=seed, error=str(error),
            attempted_update=step + 1, history=32 if smoke else 512)) + '\n')
        raise
    return dict(fingerprints=fps, curve=curve, updates=2 if smoke else spec['backbone_updates'],
                seconds=time.monotonic() - start, optimizer=optimizer.state_dict())


def batches(split, spec, smoke):
    repetitions = 1 if smoke else spec['repetitions'][split]
    for delay in ((32,) if smoke else (32, 128, 512)):
        for role in ROLES:
            for deadline in (('at',) if smoke else DEADLINES):
                cases, conditions = [], []
                for rep in range(repetitions):
                    pair = make_pair(400000 + rep, 32 if smoke else 512, delay, role, deadline,
                                     domain=f'dev040a-{split}-v1')
                    cases.extend(pair)
                    conditions.extend(dict(delay=delay, role=role, deadline=deadline, rep=rep, partner=b,
                                           history=len(c.observations), target_entity=c.target_entity,
                                           fingerprint=c.fingerprint) for b, c in enumerate(pair))
                yield cases, conditions


@torch.no_grad()
def feature_input(model, state, queries):
    captured = []
    handle = model.head.register_forward_pre_hook(lambda module, args: captured.append(args[0].detach().clone()))
    logits = model.decode(state, queries)
    handle.remove()
    if len(captured) != 1 or captured[0].shape[-1] != 44:
        raise AssertionError('exact existing 44D pre-head interface required')
    features = captured[0].reshape(state.shape[0], 8, 3, 44)
    if not torch.equal(model.head(features), logits):
        raise AssertionError('head capture differs from original forward path')
    return features


@torch.no_grad()
def gather(models, split, spec, smoke):
    collected = {k: [] for k in ('trained', 'random', 'reset_writer', 'query', 'current')}
    targets, conditions, reset, swapped = [], [], [], []
    for cases, cs in batches(split, spec, smoke):
        x = public_stream_tensors([c.observations for c in cases])
        q = public_queries([c.tick for c in cases], [c.horizon for c in cases])
        for name, model in models.items():
            model.eval()
            state = model.encode(*x)
            collected[name].append(feature_input(model, state, q))
            if name == 'trained':
                if state[0].numel() != model.state_scalars:
                    raise AssertionError('persistent state budget mismatch')
                zero = model.memory.initial(model.event_encoder(x[0][:, 0])) if hasattr(model.memory, 'bank_scalars') else torch.zeros_like(state)
                reset.append(model.decode(zero, q))
                pair_swap = state.reshape(-1, 2, *state.shape[1:]).flip(1).reshape_as(state)
                swapped.append(model.decode(pair_swap, q))
        collected['query'].append(q)
        recent = x[0][:, -1, None, None].expand(-1, 8, 3, -1)
        collected['current'].append(torch.cat((recent, q), -1))
        targets.append(labels(cases))
        conditions.extend(cs)
    return dict(features={k: torch.cat(v) for k, v in collected.items()},
                labels=torch.cat(targets), conditions=conditions,
                reset_logits=torch.cat(reset), swapped_logits=torch.cat(swapped))


def fit_head(head, data, balanced, lr, steps):
    x, y = data['features']['trained'], data['labels']
    w = weights(y, data['conditions'], balanced)
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=.0001)
    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        loss = (F.cross_entropy(head(x).reshape(-1, 2), y.reshape(-1), reduction='none').reshape_as(y) * w).sum()
        if not torch.isfinite(loss):
            raise FloatingPointError('nonfinite head loss')
        loss.backward()
        opt.step()
    return float(loss.detach())


def critical_indices(data):
    return [i for i, c in enumerate(data['conditions']) if c['role'] == 'reissued_cancel' and c['deadline'] != 'before']


def probe_values(data, field):
    name = 'trained' if field == 'shuffled' else field
    return torch.stack([data['features'][name][i, data['conditions'][i]['target_entity'], 0]
                        for i in critical_indices(data)])


def fit_probe(data, field, seed, steps):
    x = probe_values(data, field)
    indices = critical_indices(data)
    y = torch.tensor([int(data['labels'][i, data['conditions'][i]['target_entity'], 0]) for i in indices], dtype=torch.float32)
    if field == 'shuffled':
        y = y.clone()
        rng = random.Random(seed + 40404)
        for i in range(0, len(y), 2):
            if rng.randrange(2):
                y[i:i + 2] = 1 - y[i:i + 2]
    mean, scale = x.mean(0), x.std(0, unbiased=False).clamp_min(1e-3)
    torch.manual_seed(seed + 40000)
    head = nn.Linear(x.shape[-1], 1)
    optimizer = torch.optim.Adam(head.parameters(), lr=.03)
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        loss = F.binary_cross_entropy_with_logits(head((x - mean) / scale).flatten(), y) + .01 * head.weight.square().sum()
        loss.backward()
        optimizer.step()
    return dict(weights=head.state_dict(), mean=mean, scale=scale, dimensions=x.shape[-1],
                train_labels=y, train_examples=len(y))


def services(probabilities, threshold):
    return tuple(e for e, p in enumerate(probabilities) if p[0] - p[1] > threshold)


def labeled_cost(probs, labels_, threshold):
    return sum(len(services(p, threshold)) + 4 * sum(row[e][1 if e in services(p, threshold) else 0]
               for e in range(8)) for p, row in zip(probs, labels_))


def calibrate(head, data):
    with torch.no_grad():
        probs = head(data['features']['trained']).softmax(-1)[..., 1].tolist()
    scores = {str(t): labeled_cost(probs, data['labels'].tolist(), t) for t in THRESHOLDS}
    threshold = min(THRESHOLDS, key=lambda t: (scores[str(t)], abs(t - .25), t))
    return dict(threshold=threshold, scores=scores, probabilities=probs)


def prepare(arm, seed, spec, smoke, output):
    model = build(arm, seed + 40000)
    initial = copy.deepcopy(model.state_dict())
    training = train_backbone(model, seed, spec, smoke, output, arm)
    frozen_hash = parameter_hash(model)
    random_model = build(arm, seed + 40000)
    random_model.load_state_dict(initial)
    hybrid = copy.deepcopy(model)
    hybrid.memory.load_state_dict(random_model.memory.state_dict())
    models = dict(trained=model, random=random_model, reset_writer=hybrid)
    for item in models.values():
        for p in item.parameters():
            p.requires_grad_(False)
    train = gather(models, 'head-train', spec, smoke)
    dev = gather(models, 'calibration', spec, smoke)
    head_steps = 2 if smoke else spec['head_updates']
    heads = dict(original=copy.deepcopy(model.head))
    losses = {}
    for name in HEADS[1:]:
        torch.manual_seed(seed + 41000)
        head = nn.Linear(44, 2) if name.startswith('linear') else copy.deepcopy(model.head)
        for p in head.parameters():
            p.requires_grad_(True)
        losses[name] = fit_head(head, train, name.endswith('balanced'), .02 if name.startswith('linear') else .002, head_steps)
        heads[name] = head
    calibrations = {k: calibrate(h, dev) for k, h in heads.items()}
    probes = {k: fit_probe(train, k, seed, head_steps) for k in PROBES}
    if parameter_hash(model) != frozen_hash:
        raise AssertionError('head/probe fitting mutated frozen backbone')
    checkpoint = dict(arm=arm, seed=seed, trained=model.state_dict(), initial=initial,
                      heads={k: h.state_dict() for k, h in heads.items()}, probes=probes,
                      calibrations=calibrations, backbone_optimizer=training.pop('optimizer'))
    path = output / f'{arm}.pt'
    torch.save(checkpoint, path)
    features_path = output / f'{arm}-fit-features.pt'
    torch.save(dict(train=train, calibration=dev), features_path)
    train_fps = {c['fingerprint'] for c in train['conditions']}
    dev_fps = {c['fingerprint'] for c in dev['conditions']}
    if train_fps & dev_fps or (train_fps | dev_fps) & set(training['fingerprints']):
        raise AssertionError('backbone/head/calibration overlap')
    return dict(models=models, heads=heads, checkpoint=checkpoint,
                report=dict(training=training, frozen_model_hash=frozen_hash, head_losses=losses,
                  checkpoint_sha256=sha(path), fit_features_sha256=sha(features_path),
                  head_train_fingerprints=sorted(train_fps), calibration_fingerprints=sorted(dev_fps),
                  state_scalars=model.state_scalars, parameters=sum(p.numel() for p in model.parameters()),
                  calibrations=calibrations, probe_train_examples={k: p['train_examples'] for k, p in probes.items()}))


@torch.no_grad()
def evaluate(prepared, spec, smoke):
    data = gather(prepared['models'], 'evaluation', spec, smoke)
    head_logits = {k: h(data['features']['trained']).tolist() for k, h in prepared['heads'].items()}
    probabilities = {k: torch.tensor(v).softmax(-1)[..., 1].tolist() for k, v in head_logits.items()}
    probe_predictions = {}
    for field, p in prepared['checkpoint']['probes'].items():
        head = nn.Linear(p['dimensions'], 1)
        head.load_state_dict(p['weights'])
        logits = head((probe_values(data, field) - p['mean']) / p['scale']).flatten()
        probe_predictions[field] = (logits >= 0).int().tolist()
    critical_lookup = {j: i for i, j in enumerate(critical_indices(data))}
    rows, index = [], 0
    for cases, conditions in batches('evaluation', spec, smoke):
        for case, condition in zip(cases, conditions):
            if condition != data['conditions'][index] or exact_public_targets(case.observations, case.tick, case.horizon) != case.wait_targets:
                raise AssertionError('public cohort mismatch')
            cache = {}
            def outcome(chosen):
                chosen = tuple(chosen)
                if chosen not in cache:
                    cache[chosen] = execute(case.world, case.horizon, chosen, 0)
                return cache[chosen]
            controls = {}
            for name in HEADS:
                p = probabilities[name][index]
                threshold = prepared['report']['calibrations'][name]['threshold']
                controls[name] = dict(fixed=outcome(services(p, .25)), calibrated=outcome(services(p, threshold)))
            reference = dict(never=outcome(()), always=outcome(range(8)),
                             public=outcome(e for e, value in enumerate(case.wait_targets) if value))
            row = dict(**condition, labels=data['labels'][index].tolist(),
                logits={k: v[index] for k, v in head_logits.items()},
                probabilities={k: v[index] for k, v in probabilities.items()},
                controls=controls, references=reference,
                reset_prediction=int(data['reset_logits'][index, case.target_entity, 0].argmax()),
                swapped_prediction=int(data['swapped_logits'][index, case.target_entity, 0].argmax()))
            if index in critical_lookup:
                row['probes'] = {k: v[critical_lookup[index]] for k, v in probe_predictions.items()}
            rows.append(row)
            index += 1
    return rows, data


def restore(arm, seed, output, report):
    path = output / f'{arm}.pt'
    if sha(path) != report['checkpoint_sha256']:
        raise AssertionError('checkpoint changed')
    ck = torch.load(path, map_location='cpu', weights_only=True)
    models = {k: build(arm, seed + 40000) for k in ('trained', 'random', 'reset_writer')}
    models['trained'].load_state_dict(ck['trained'])
    if parameter_hash(models['trained']) != report['frozen_model_hash']:
        raise AssertionError('frozen model tensor hash mismatch')
    models['random'].load_state_dict(ck['initial'])
    models['reset_writer'].load_state_dict(ck['trained'])
    models['reset_writer'].memory.load_state_dict(models['random'].memory.state_dict())
    heads = {}
    for name in HEADS:
        head = nn.Linear(44, 2) if name.startswith('linear') else copy.deepcopy(models['trained'].head)
        head.load_state_dict(ck['heads'][name])
        heads[name] = head
    return dict(models=models, heads=heads, checkpoint=ck, report=report)


def run(seed, output, smoke=False, replay=False):
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    spec = json.loads((ROOT / 'research/dev040/pilot-protocol.json').read_text())
    if seed not in spec['seeds']:
        raise ValueError('unregistered seed')
    source = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    if subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=no'], cwd=ROOT, text=True).strip():
        raise ValueError('freeze source before execution')
    output.mkdir(parents=True, exist_ok=True)
    path = output / 'report.json'
    if replay:
        report = json.loads(path.read_text())
        if (report['source'], report['protocol_sha256'], report['seed'], report['smoke']) != (source, sha(ROOT / 'research/dev040/pilot-protocol.json'), seed, smoke):
            raise AssertionError('source/seed/protocol changed')
        prepared = {a: restore(a, seed, output, report['arms'][a]) for a in ARMS}
    else:
        if path.exists():
            raise ValueError('preserve prior result')
        prepared = {a: prepare(a, seed, spec, smoke, output) for a in ARMS}
        report = dict(study='DEV-040A', status='ADAPTIVE_DEVELOPMENT_NOT_CONFIRMATORY', source=source,
                      protocol_sha256=sha(ROOT / 'research/dev040/pilot-protocol.json'), seed=seed, smoke=smoke,
                      arms={a: p['report'] for a, p in prepared.items()}, rows={},
                      all_three_architectures_frozen_before_evaluation=True)
        (output / 'fit-manifest.json').write_text(json.dumps({a: p['report']['checkpoint_sha256'] for a, p in prepared.items()}, indent=2) + '\n')
    from audit_head_controls_dev040a import metrics
    for arm, item in prepared.items():
        raw, data = evaluate(item, spec, smoke)
        fps = {r['fingerprint'] for r in raw}
        prior = report['arms'][arm]
        if fps & (set(prior['training']['fingerprints']) | set(prior['head_train_fingerprints']) | set(prior['calibration_fingerprints'])):
            raise AssertionError('evaluation overlap')
        if replay:
            if raw != report['rows'][arm] or metrics(raw) != report['arms'][arm]['metrics']:
                raise AssertionError('full prediction/probe/action/cost replay failed')
            if sha(output / f'{arm}-fit-features.pt') != prior['fit_features_sha256']:
                raise AssertionError('fit features changed')
            fitting = torch.load(output / f'{arm}-fit-features.pt', map_location='cpu', weights_only=True)
            for split, key in (('head-train', 'train'), ('calibration', 'calibration')):
                regenerated = gather(item['models'], split, spec, smoke)
                if regenerated['conditions'] != fitting[key]['conditions']:
                    raise AssertionError('fit condition replay mismatch')
                torch.testing.assert_close(regenerated['labels'], fitting[key]['labels'], rtol=0, atol=0)
                for field, value in regenerated['features'].items():
                    torch.testing.assert_close(value, fitting[key]['features'][field], rtol=0, atol=0)
            for name in HEADS[1:]:
                torch.manual_seed(seed + 41000)
                head = nn.Linear(44, 2) if name.startswith('linear') else copy.deepcopy(item['models']['trained'].head)
                for p in head.parameters():
                    p.requires_grad_(True)
                fit_head(head, fitting['train'], name.endswith('balanced'), .02 if name.startswith('linear') else .002, 2 if smoke else spec['head_updates'])
                for key, value in head.state_dict().items():
                    torch.testing.assert_close(value, item['checkpoint']['heads'][name][key], rtol=0, atol=0)
                if calibrate(head, fitting['calibration']) != prior['calibrations'][name]:
                    raise AssertionError('calibrator replay mismatch')
            for field in PROBES:
                reproduced = fit_probe(fitting['train'], field, seed, 2 if smoke else spec['head_updates'])
                for key in ('mean', 'scale', 'train_labels'):
                    torch.testing.assert_close(reproduced[key], item['checkpoint']['probes'][field][key], rtol=0, atol=0)
                for key, value in reproduced['weights'].items():
                    torch.testing.assert_close(value, item['checkpoint']['probes'][field]['weights'][key], rtol=0, atol=0)
        else:
            report['rows'][arm] = raw
            prior['metrics'] = metrics(raw)
            torch.save(data, output / f'{arm}-evaluation-features.pt')
            prior['evaluation_features_sha256'] = sha(output / f'{arm}-evaluation-features.pt')
    if replay:
        print(f'REPLAY VERIFIED seed={seed} three_backbones=3 heads=15 probe_fits=18', flush=True)
    else:
        path.write_text(json.dumps(report, allow_nan=False) + '\n')
        print(f'TRAINING AND EVALUATION COMPLETE seed={seed}; outcomes archived before publication', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', required=True, type=int)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--replay', action='store_true')
    args = parser.parse_args()
    run(args.seed, args.output, args.smoke, args.replay)
