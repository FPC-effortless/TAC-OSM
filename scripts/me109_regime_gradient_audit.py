"""POSTHOC_FIT_ONLY descriptive regime-gradient audit; NO_NEW_MODEL_FITS.

Cosines and local displacement dots do not establish causes, realized loss
changes, architecture superiority, or a new training/evaluation protocol.
"""
import argparse
import itertools
import json

import numpy as np
import torch

from scripts import me109_residual_training as run
from tac_osm.role_optimizer import BlockProtectedAdamW

OUT = run.OUT / 'POSTHOC_FIT_REGIME_GRADIENTS.json'
LABELS = ('whole_point', 'gap_pm', 'general_pm', 'nll', 'interval', 'reconstruction_nll', 'location_mse')


def dot(a, b, indices):
    return sum(float((a[i].double() * b[i].double()).sum()) for i in indices)


def geometry(gradients, indices):
    norms = {name: dot(g, g, indices) ** .5 for name, g in gradients.items()}
    cosines = {}
    for a, b in itertools.combinations(LABELS, 2):
        denominator = norms[a] * norms[b]
        cosines[a + '__' + b] = dot(gradients[a], gradients[b], indices) / denominator if denominator else None
    return dict(norms=norms, cosines=cosines)


def audit():
    if OUT.exists():
        raise RuntimeError('original posthoc FIT audit exists; no overwrite')
    torch.set_num_threads(1)
    index = run.budget_audit()
    d = run.fit_data()
    if d['phase'] != 'fit':
        raise RuntimeError('FIT only')
    rawscale = torch.tensor(np.nanstd(d['raw'][..., :6].astype(np.float64), axis=(0, 1)).clip(1e-4), dtype=torch.float32)
    batches = []
    for batch_id in range(3):
        rng = np.random.default_rng(109900 + batch_id)
        origins = np.asarray([d['gap'][i] for i in rng.integers(len(d['gap']), size=16)]
                             + [d['general'][i] for i in rng.integers(len(d['general']), size=16)])
        x = run.tensor_batch(d, origins, 256, rng)
        batches.append((origins, x))
    rows = []
    for identity in index['models']:
        seed = identity['seed']
        path = run.OUT / 'residual_guarded' / f'seed-{seed}' / 'final.pt'
        before_hash = run.sha(path)
        assert before_hash == identity['checkpoint_sha256']
        cp = torch.load(path, map_location='cpu', weights_only=False)
        assert cp['steps'] == 4096 and cp['seed'] == seed and cp['architecture'] == 'residual_guarded'
        model = run.initialize(d, seed)
        model.load_state_dict(cp['model'], strict=True)
        params = tuple(model.parameters())
        assert len(params) == 45
        _, blockmap = run.blocks(model)
        opt = BlockProtectedAdamW(run.parameter_groups(model), blocks=list(blockmap.values()))
        opt.load_state_dict(cp['optimizer'])
        assert len(opt.state) == 45 and {int(s['step']) for s in opt.state.values()} == {4096}
        versions = model.versions()
        old = [p.detach().clone() for p in params]
        old_moments = [{k: v.clone() for k, v in opt.state[p].items()} for p in params]
        for batch_id, (origins, x) in enumerate(batches):
            forecast, _, trace = model(*x[:6], return_trace=True)
            total, parts = run.role_objective(model, forecast, trace, x, rawscale)
            pm_error = (model.raw_quantiles(forecast)[:, 0, 1] - x[6][:, 0]).abs() / rawscale[0]
            def pm_loss(start, stop):
                mask = x[7][start:stop, 0]
                assert int(mask.sum()) > 0
                return torch.where(mask, pm_error[start:stop], torch.zeros_like(pm_error[start:stop])).sum() / mask.sum()
            losses = dict(whole_point=.5 * parts['point'], gap_pm=pm_loss(0,16), general_pm=pm_loss(16,32),
                          nll=parts['nll'], interval=parts['interval'],
                          reconstruction_nll=parts['reconstruction_nll'], location_mse=parts['location_mse'])
            gradients = {}
            for name in LABELS:
                g = torch.autograd.grad(losses[name], params, retain_graph=True, allow_unused=True)
                gradients[name] = [torch.zeros_like(p) if v is None else v.detach() for p, v in zip(params, g)]
            combined = torch.autograd.grad(total, params, allow_unused=True)
            for p, g in zip(params, combined):
                p.grad = torch.zeros_like(p) if g is None else g.detach().clone()
                assert torch.isfinite(p.grad).all()
            preclip_norm = float(torch.nn.utils.clip_grad_norm_(params, 1.))
            proposed, _, receipt = opt.propose(gradients['whole_point'])
            displacement = [candidate.double() - p.detach().double() for candidate, p in zip(proposed, params)]
            applied = {}
            for name, indices in blockmap.items():
                applied[name] = {role: dot(gradients[role], displacement, indices)
                                 for role in ('whole_point', 'gap_pm', 'general_pm')}
                assert applied[name]['whole_point'] <= 0
            assert all(r['applied_dot'] <= 0 for r in receipt)
            assert model.versions() == versions
            for p, original, moments in zip(params, old, old_moments):
                assert torch.equal(p, original)
                for key in moments: assert torch.equal(opt.state[p][key], moments[key])
            assert {int(s['step']) for s in opt.state.values()} == {4096}
            assert run.sha(path) == before_hash
            overall = {role: dot(gradients[role], displacement, range(len(params)))
                       for role in ('whole_point', 'gap_pm', 'general_pm')}
            assert overall['whole_point'] <= 0
            rows.append(dict(seed=seed, batch=batch_id, batch_seed=109900+batch_id,
                origins=origins.tolist(), input_sha256=run.digest([run.sha_tensor(v) for v in x]),
                gap_pm_target_count=int(x[7][:16,0].sum()), general_pm_target_count=int(x[7][16:,0].sum()),
                losses={name: float(loss.detach()) for name, loss in losses.items()},
                preclip_total_gradient_norm=preclip_norm,
                geometry_overall=geometry(gradients, range(len(params))),
                geometry_blocks={name: geometry(gradients, indices) for name, indices in blockmap.items()},
                displacement_dots_overall=overall, displacement_dots_blocks=applied,
                unchanged_checkpoint_sha256=before_hash, original_moment_step=4096,
                no_optimizer_step=True))
    counts = {}
    for role in ('gap_pm', 'general_pm'):
        positive = sum(row['displacement_dots_overall'][role] > 0 for row in rows)
        block_positive = sum(block[role] > 0 for row in rows for block in row['displacement_dots_blocks'].values())
        counts[role] = dict(overall_positive_count=positive, overall_total=len(rows),
            overall_positive_fraction=positive/len(rows), block_positive_count=block_positive,
            block_total=13*len(rows), block_positive_fraction=block_positive/(13*len(rows)))
    result = dict(status='POSTHOC_FIT_ONLY', policy='NO_NEW_MODEL_FITS', models=5, batches_per_model=3,
        batch_size=32, natural_gap_origins=16, general_origins=16, context=256,
        causal_augmentation='original me102_gap_data.augmentation',
        interpretation='Descriptive local gradients and proposals only; no causal or winner claim.',
        unweighted_auxiliary_gradients=True, whole_point_coefficient=.5,
        proposal_total='original full role_objective, global clip norm 1',
        report_or_protected_access=False, no_optimizer_updates=True, original_steps_unchanged=4096,
        positive_regime_dot_summary=counts, checks=rows)
    run.write(OUT, result)
    print(json.dumps(counts, sort_keys=True))
    print('ME109_POSTHOC_FIT_REGIME_GRADIENTS_PASS')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=('run',))
    parser.parse_args()
    audit()
