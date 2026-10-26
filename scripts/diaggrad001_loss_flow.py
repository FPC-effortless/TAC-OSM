"""FIT-only frozen ME106 gradient-flow diagnostic; never steps an optimizer.

Per-loss gradients and analytic shadow directions describe local geometry,
not finite-step loss changes or a newly fitted model. Original parameters,
optimizer moments and checkpoint files remain immutable.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import torch

from scripts import me106_priority_panel as panel
from scripts.me106_gradient_roles import objective_parts, project_gradient_lists

ROOT = Path(__file__).resolve().parents[1]
DOC = ROOT / 'research/mathematical-engineering/diaggrad001'
OUT = ROOT / 'research/replay-bundles/DIAGGRAD001'
TERMS = ('nll', 'point', 'location_mse', 'reconstruction_nll', 'interval')
CONTEXTS = (8, 64, 256)
BLOCKS = ('source_encoder', 'source_admission', 'fast', 'slow', 'address',
          'content', 'memory_gates', 'query', 'future_decoder', 'belief',
          'reconstruction', 'width', 'median')


def block_name(name):
    prefix = name.split('.')[0]
    return {'station': 'source_encoder', 'encoder': 'source_encoder',
            'source_admission': 'source_admission', 'fast_map': 'fast',
            'slow_map': 'slow', 'address': 'address', 'content': 'content',
            'memory_gates': 'memory_gates', 'query': 'query',
            'readout': 'future_decoder', 'retrieved_delta': 'future_decoder',
            'state_noise': 'belief', 'state_response': 'belief',
            'current_head': 'reconstruction', 'width_head': 'width',
            'median_head': 'median'}[prefix]


def dot(a, b):
    return sum(float((x.double() * y.double()).sum()) for x, y in zip(a, b))


def geometry(a, b):
    aa, bb, ab = dot(a, a), dot(b, b), dot(a, b)
    return dict(norm_a=aa**.5, norm_b=bb**.5, dot=ab,
                cosine=ab/(aa*bb)**.5 if aa > 0 and bb > 0 else None)


def gradient(loss, params):
    gs = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
    return [torch.zeros_like(p) if g is None else g.detach() for p, g in zip(params, gs)]


def shadow_adamw(params, grads, checkpoint, named):
    """Read-only next-update algebra with saved moments and global clip=1.

    Group order is reconstructed exactly from ME106's optimizer grouping.
    No optimizer object is stepped and no parameter tensor is modified.
    """
    norm = dot(grads, grads)**.5
    clip = min(1., 1./(norm + 1e-6))
    saved = checkpoint['optimizer']
    inherited = [i for i, (name, _) in enumerate(named)
                 if not name.startswith(('width_head.', 'median_head.'))]
    additions = [i for i in range(len(named)) if i not in inherited]
    deltas = [None] * len(params)
    group_sgd = [None] * len(params)
    for indices, group in zip((inherited, additions), saved['param_groups']):
        if len(indices) != len(group['params']):
            raise AssertionError('optimizer group ordering mismatch')
        beta1, beta2 = group['betas']
        for i, state_id in zip(indices, group['params']):
            state = saved['state'][state_id]
            step = int(state['step']) + 1
            g = grads[i].double() * clip
            m = beta1*state['exp_avg'].double() + (1-beta1)*g
            v = beta2*state['exp_avg_sq'].double() + (1-beta2)*g.square()
            update = (m/(1-beta1**step)) / ((v/(1-beta2**step)).sqrt()+group['eps'])
            deltas[i] = -group['lr'] * (update + group['weight_decay']*params[i].detach().double())
            group_sgd[i] = -group['lr'] * g
    return deltas, group_sgd, clip


def probe(model, checkpoint, batch, rawscale, recipe):
    named = list(model.named_parameters())
    params = [p for _, p in named]
    blocks = {b: [i for i, (name, _) in enumerate(named) if block_name(name) == b] for b in BLOCKS}
    blocks['all'] = list(range(len(params)))
    prediction, _, trace = model(*batch[:6], return_trace=True)
    _, parts = objective_parts(model, prediction, trace, batch, rawscale, panel.RECIPES[recipe])
    grads = {term: gradient(parts[term], params) for term in TERMS}
    weights = dict(nll=1., point=panel.RECIPES[recipe]['point'], location_mse=.1,
                   reconstruction_nll=.3, interval=.25)
    point = [weights['point']*g for g in grads['point']]
    aux = [sum(weights[t]*grads[t][i] for t in TERMS if t != 'point') for i in range(len(params))]
    combined, projection = project_gradient_lists(point, aux)
    projected_aux = [g-p for g, p in zip(combined, point)]
    adam, lr_sgd, clip = shadow_adamw(params, combined, checkpoint, named)
    result = dict(losses={k: float(v.detach()) for k, v in parts.items()},
                  weights=weights, clip_coefficient=clip,
                  projection={k: float(v) for k, v in projection.items()}, blocks={})
    for name, indices in blocks.items():
        take = lambda vector: [vector[i] for i in indices]
        result['blocks'][name] = dict(
            norms={t: dot(take(grads[t]), take(grads[t]))**.5 for t in TERMS},
            weighted_norms={t: abs(weights[t])*dot(take(grads[t]), take(grads[t]))**.5 for t in TERMS},
            pairs={f'{a}__{b}': geometry(take(grads[a]), take(grads[b]))
                   for a, b in itertools.combinations(TERMS, 2)},
            priority_aux_before=geometry(take(point), take(aux)),
            priority_aux_after=geometry(take(point), take(projected_aux)),
            # Positive dot(g_point, displacement) predicts local point increase.
            point_dot_group_sgd=dot(take(point), take(lr_sgd)),
            point_dot_saved_adamw=dot(take(point), take(adam)),
        )
    # Compare point/NLL within currently observed vs missing forecast channels.
    result['target_subsets'] = {}
    for label, select in (('observed', batch[1][:, -1, :6]), ('missing', ~batch[1][:, -1, :6])):
        subset = list(batch)
        subset[7] = batch[7] & select
        _, subset_parts = objective_parts(model, prediction, trace, subset, rawscale, panel.RECIPES[recipe])
        pg, ng = gradient(subset_parts['point'], params), gradient(subset_parts['nll'], params)
        result['target_subsets'][label] = dict(count=int(subset[7].sum()),
            blocks={b: geometry([pg[i] for i in ix], [ng[i] for i in ix]) for b, ix in blocks.items()})
    protected = trace['protected_floor_scale']
    missing = ~batch[1][:, -1, :6]
    result['floor_activity'] = dict(
        missing_targets=int(missing.sum()),
        left_at_floor=int(((prediction.left_scale <= protected) & missing).sum()),
        right_at_floor=int(((prediction.right_scale <= protected) & missing).sum()),
        reconstruction_targets=int(batch[9].sum()),
        median_tanh_saturated=int((trace['median_correction'].abs() > .49).sum()))
    if not all(torch.isfinite(g).all() for gs in grads.values() for g in gs):
        raise AssertionError('nonfinite gradient')
    return result


def summarize(rows):
    result = {}
    for block in (*BLOCKS, 'all'):
        bs = [r['measurement']['blocks'][block] for r in rows]
        pairs = {}
        for pair in bs[0]['pairs']:
            cosines = [b['pairs'][pair]['cosine'] for b in bs if b['pairs'][pair]['cosine'] is not None]
            pairs[pair] = dict(nonzero_pairs=len(cosines), negative=sum(c < -1e-8 for c in cosines),
                               median_cosine=float(np.median(cosines)) if cosines else None)
        result[block] = dict(pairs=pairs,
            median_weighted_norm={t: float(np.median([b['weighted_norms'][t] for b in bs])) for t in TERMS},
            opposing_aux_before=sum(b['priority_aux_before']['dot'] < -1e-10 for b in bs),
            opposing_aux_after=sum(b['priority_aux_after']['dot'] < -1e-10 for b in bs),
            locally_uphill_group_sgd=sum(b['point_dot_group_sgd'] > 1e-12 for b in bs),
            locally_uphill_saved_adamw=sum(b['point_dot_saved_adamw'] > 1e-12 for b in bs),
            probes=len(bs))
    return result


def run(replay=False):
    from scripts.validate_research_parity import check_manifest
    registration = json.loads((DOC/'FAIR_COMPARISON.json').read_text())
    check_manifest(registration, 'preflight')
    panel._frozen([DOC/'FAIR_COMPARISON.json', Path(__file__)])
    if (OUT/'RESULTS.json').exists() and not replay:
        raise RuntimeError('original diagnostic already exists; replay only')
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    d = panel.fit_data()  # FIT only. REPORT and protected rows never loaded.
    rawscale = torch.tensor(np.nanstd(d['raw'][..., :6].astype(np.float64), axis=(0, 1)).clip(1e-4), dtype=torch.float32)
    index = json.loads((panel.OUT/'FIT_INDEX.json').read_text())
    rows = []
    for row in index['models']:
        recipe, seed = row['architecture'], int(row['seed'])
        path = panel.OUT/recipe/f'seed-{seed}'/'final.pt'
        before = panel.sha(path)
        if before != row['checkpoint_sha256']:
            raise AssertionError('original checkpoint hash mismatch')
        cp = torch.load(path, map_location='cpu', weights_only=False)
        model = panel.initialize(d, seed)
        model.load_state_dict(cp['model'], strict=True)
        if sum(p.numel() for p in model.parameters()) != 100349:
            raise AssertionError('unexpected architecture')
        versions = model.versions()
        for length, population, repeat in itertools.product(CONTEXTS, ('general', 'gap'), range(2)):
            # Same station/time selection and augmentation for all recipes/seeds.
            rng = np.random.default_rng(107000 + length + 1000*repeat + (10000 if population == 'gap' else 0))
            origins = np.asarray(d[population])[rng.integers(len(d[population]), size=16)]
            batch = panel.tensor_batch(d, origins, length, rng)
            measurements = probe(model, cp, batch, rawscale, recipe)
            rows.append(dict(recipe=recipe, seed=seed, history=length, origin_population=population,
                             repeat=repeat, input_sha256=hashlib.sha256(b''.join(t.numpy().tobytes() for t in batch)).hexdigest(),
                             measurement=measurements))
        if panel.sha(path) != before or model.versions() != versions:
            raise AssertionError('diagnostic mutated original weights')
        print(f'{recipe}/{seed}: frozen gradient probes complete', flush=True)
    result = dict(study_id='DIAGGRAD001', no_new_model_fits=True, no_optimizer_updates=True,
        source_commit='e6ac736a', original_fit_index_sha256=panel.sha(panel.OUT/'FIT_INDEX.json'),
        source_files={p: panel.sha(ROOT/p) for p in (*panel.SOURCES, 'scripts/diaggrad001_loss_flow.py')},
        registration_sha256=panel.sha(DOC/'FAIR_COMPARISON.json'),
        cohort='2013-2014 FIT only; natural origin selection plus original causal artificial masking',
        rows=rows, summary=summarize(rows),
        by_history={str(n): summarize([r for r in rows if r['history']==n]) for n in CONTEXTS},
        by_population={p: summarize([r for r in rows if r['origin_population']==p]) for p in ('general','gap')})
    if replay:
        old = json.loads((OUT/'RESULTS.json').read_text())
        panel._compare_numeric(result, old)
        print('DIAGGRAD001_REPLAY_PASS')
    else:
        OUT.mkdir(parents=True, exist_ok=False)
        panel.write(OUT/'RESULTS.json', result)
        panel.write(OUT/'SUMMARY.json', {k: v for k, v in result.items() if k != 'rows'})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=('run', 'replay'))
    run(parser.parse_args().mode == 'replay')
