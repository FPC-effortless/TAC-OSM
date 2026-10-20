"""REAL004 fixed factual-outcome benchmark, no counterfactual repair policy."""
import argparse
import copy
import json
from pathlib import Path
import subprocess
import sys
import time
ROOT = Path(__file__).resolve().parents[1]; sys.path[:0] = [str(ROOT/'src'), str(ROOT/'scripts')]
import numpy as np
import torch
from tac_osm.development.repair_outcome import RepairOutcomeForecaster
from data_real_maintenance_004 import source, split, fit_schema, features, labeled, normalize
from run_learned_stack_dev055 import write, digest
PROTOCOL = ROOT/'research/real-maint-004/protocol.json'


def reference_fit(x, y, rows, ids, regularization):
    center = x.mean(0); scale = x.std(0); scale[scale < 1e-8] = 1
    design = np.column_stack([np.ones(len(x)), (x-center)/scale])
    penalty = np.eye(design.shape[1])*regularization; penalty[0, 0] = 0
    weights = np.linalg.solve(design.T@design+penalty, design.T@np.log(y))
    categories = {}
    for i, value in zip(ids, y, strict=True): categories.setdefault(normalize(rows[i]['functional_status_before']), []).append(float(value))
    return dict(center=center.tolist(), scale=scale.tolist(), weights=weights.tolist(),
        median=float(np.median(y)), categories={k: float(np.median(v)) for k, v in categories.items()}, ridge=regularization)


def reference_predict(ref, x, rows, ids, kind):
    if kind == 'median': return np.full(len(x), ref['median'])
    if kind == 'prestatus':
        return np.array([ref['categories'].get(normalize(rows[i]['functional_status_before']), ref['median']) for i in ids])
    design = np.column_stack([np.ones(len(x)), (x-np.array(ref['center']))/np.array(ref['scale'])])
    return np.exp(design@np.array(ref['weights']))


def fit(seed, arm, x, y, vx, vy, recipe, folder, updates):
    folder.mkdir(parents=True, exist_ok=False)
    center = x.mean(0); scale = x.std(0); scale[scale < 1e-8] = 1
    model = RepairOutcomeForecaster(x.shape[1], center, scale, float(np.median(y)), seed)
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    opt = torch.optim.AdamW(model.parameters(), lr=recipe['lr'], weight_decay=recipe['decay'])
    generator = torch.Generator().manual_seed(seed); tx = torch.tensor(x); ty = torch.tensor(y)
    scale_loss = max(1., float(np.median(y))); curve = []; start = time.process_time()
    for step in range(1, updates+1):
        ids = torch.randint(len(x), (32,), generator=generator)
        pred = model(tx[ids]); loss = (pred-ty[ids]).abs().mean()/scale_loss
        assert torch.isfinite(loss)
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True); opt.step()
        with torch.no_grad():
            for target, raw in zip(ema.parameters(), model.parameters(), strict=True): target.mul_(.995).add_(raw, alpha=.005)
        if step in (1, updates) or step % 256 == 0:
            with torch.no_grad(): val = float((ema(torch.tensor(vx))-torch.tensor(vy)).abs().mean())
            curve.append(dict(update=step, train_scaled_MAE=float(loss.detach()), validation_MAE_seconds=val))
    torch.save(model.state_dict(), folder/'raw.pt'); torch.save(ema.state_dict(), folder/'model.pt')
    with torch.no_grad(): pred = ema(torch.tensor(vx)).numpy()
    report = dict(seed=seed, arm=arm, recipe=recipe, updates=updates, features=x.shape[1], width=16,
        parameters=sum(p.numel() for p in model.parameters()), validation_prediction_seconds=pred.tolist(),
        validation_MAE_seconds=float(np.abs(pred-vy).mean()), curve=curve,
        checkpoint_sha256=digest(folder/'model.pt'), cpu_seconds=time.process_time()-start)
    write(folder/'report.json', report)
    return report


def run(directory, input_path):
    torch.set_num_threads(1); torch.set_deterministic_debug_mode('error'); start = time.process_time()
    root = Path(directory); root.mkdir(parents=True, exist_ok=False)
    if subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=no'], cwd=ROOT, text=True).strip():
        raise RuntimeError('Freeze actual source and protocol before fitting')
    source_commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    protocol = json.loads(PROTOCOL.read_text()); rows = source(input_path)
    assignment, groups, metadata = split(rows); write(root/'split.json', metadata)
    write(root/'assignment.json', {k: v for k, v in assignment.items()})
    # No test target parsed or summarized before the explicit marker below.
    train, y, train_bad = labeled(rows, assignment['train'])
    val, vy, val_bad = labeled(rows, assignment['validation'])
    if len(train) < 80 or len(val) < 20 or len(np.unique(y)) < 5 or metadata['groups']['test'] < 20:
        status = 'SOURCE_HEADROOM_REJECTED'; reports = []
    else:
        schemas = {arm: fit_schema(rows, train, arm) for arm in ('prestate', 'repair')}
        matrices = {arm: (features(rows, train, schema), features(rows, val, schema)) for arm, schema in schemas.items()}
        write(root/'schemas.json', schemas)
        torch.save(dict(train_ids=train, validation_ids=val,
            train_targets_seconds=torch.tensor(y), validation_targets_seconds=torch.tensor(vy),
            train_groups=[groups[i] for i in train], validation_groups=[groups[i] for i in val],
            **{arm: dict(train=torch.tensor(pair[0]), validation=torch.tensor(pair[1]))
               for arm, pair in matrices.items()}), root/'prepared.pt')
        x, vx = matrices['repair']; references = {}
        for kind, ridge in [('median', 1), ('prestatus', 1), ('ridge_0.1', .1), ('ridge_1', 1), ('ridge_10', 10)]:
            ref = reference_fit(x, y, rows, train, ridge); prediction = reference_predict(ref, vx, rows, val, kind)
            references[kind] = dict(fit=ref, validation_prediction_seconds=prediction.tolist(), validation_MAE_seconds=float(np.abs(prediction-vy).mean()))
        reference = min(references, key=lambda k: (references[k]['validation_MAE_seconds'], k))
        write(root/'references.json', references)
        reports = []; pilots = []
        for arm in ('prestate', 'repair'):
            for i, recipe in enumerate(protocol['recipes']):
                for seed in protocol['pilot_seeds']:
                    x, vx = matrices[arm]
                    report = fit(seed, arm, x, y, vx, vy, recipe, root/f'pilot-{arm}-{i}-{seed}', protocol['updates'])
                    reports.append(report); pilots.append((i, report))
                    print('pilot', arm, i, seed, round(report['validation_MAE_seconds'], 5), flush=True)
                    if time.process_time()-start > protocol['cpu_cap_seconds']: raise TimeoutError('Registered total CPU cap')
        selected = {}
        for arm in ('prestate', 'repair'):
            selected[arm] = min(range(len(protocol['recipes'])), key=lambda i:
                (np.mean([r['validation_MAE_seconds'] for j, r in pilots if j == i and r['arm'] == arm]), i))
        candidate_mean = float(np.mean([r['validation_MAE_seconds'] for j, r in pilots if r['arm'] == 'repair' and j == selected['repair']]))
        screen = candidate_mean <= .95*references[reference]['validation_MAE_seconds']
        write(root/'selection.json', dict(recipes=selected, reference=reference, pilot_repair_MAE=candidate_mean,
            validation_screen_pass=screen, locked_before_fresh_fits=True))
        if not screen: status = 'VALIDATION_REJECTED'
        else:
            fresh = []
            for seed in protocol['seeds']:
                for arm in ('prestate', 'repair'):
                    x, vx = matrices[arm]; recipe = protocol['recipes'][selected[arm]]
                    report = fit(seed, arm, x, y, vx, vy, recipe, root/f'fresh-{arm}-{seed}', protocol['updates'])
                    reports.append(report); fresh.append(report)
                    print('fresh', arm, seed, round(report['validation_MAE_seconds'], 5), flush=True)
                    if time.process_time()-start > protocol['cpu_cap_seconds']: raise TimeoutError('Registered total CPU cap')
            reliable = sum(r['arm'] == 'repair' and r['validation_MAE_seconds'] <= references[reference]['validation_MAE_seconds'] for r in fresh)
            if reliable < 18: status = 'VALIDATION_REJECTED'
            else:
                write(root/'TEST-OPENED.json', dict(source=source_commit, selection_sha256=digest(root/'selection.json'), passing_validation_seeds=reliable))
                test, ty, test_bad = labeled(rows, assignment['test'])
                # Eligibility is fixed, no filtering by error magnitude or action.
                write(root/'test-labels.json', dict(ids=test, target_seconds=ty.tolist(), rejected_nonpositive_or_malformed=test_bad,
                    groups=[groups[i] for i in test], unit='seconds_per_20L', source=digest(input_path)))
                if len({groups[i] for i in test}) < 20: status = 'TEST_SOURCE_INSUFFICIENT'
                else:
                    ref_predictions = {}
                    for name, item in references.items():
                        ref_predictions[name] = reference_predict(item['fit'], features(rows, test, schemas['repair']), rows, test, name).tolist()
                    write(root/'test-references.json', ref_predictions)
                    torch.save({arm: torch.tensor(features(rows, test, schema))
                                for arm, schema in schemas.items()}, root/'test-inputs.pt')
                    for report in fresh:
                        arm = report['arm']; folder = root/f"fresh-{arm}-{report['seed']}"
                        x, _ = matrices[arm]
                        model = RepairOutcomeForecaster(x.shape[1], np.zeros(x.shape[1]), np.ones(x.shape[1]), 1.)
                        model.load_state_dict(torch.load(folder/'model.pt', weights_only=True))
                        with torch.no_grad(): prediction = model(torch.tensor(features(rows, test, schemas[arm]))).numpy()
                        residual = np.abs(np.asarray(report['validation_prediction_seconds'])-vy)
                        group_errors = [float(residual[[groups[i] == group for i in val]].max()) for group in sorted({groups[i] for i in val})]
                        rank = min(len(group_errors), int(np.ceil((len(group_errors)+1)*.9)))
                        radius = sorted(group_errors)[rank-1]
                        write(folder/'test.json', dict(prediction_seconds=prediction.tolist(), MAE_seconds=float(np.abs(prediction-ty).mean()),
                            interval90_radius_seconds=radius, empirical_row_coverage=float(np.mean(np.abs(prediction-ty) <= radius)),
                            lower_seconds=np.maximum(0, prediction-radius).tolist(), upper_seconds=(prediction+radius).tolist()))
                    status = 'TEST_COMPLETE'
    write(root/'summary.json', dict(status=status, source=source_commit, fit_count=len(reports),
        training_rows=len(train), validation_rows=len(val), rejected_train=train_bad, rejected_validation=val_bad,
        cpu_seconds=time.process_time()-start, test_opened=(root/'TEST-OPENED.json').exists(),
        source_sha256=digest(input_path), outcome_unit='seconds_per_20L', commercial_qualified=False))
    write(root/'manifest.json', {str(p.relative_to(root)): digest(p) for p in sorted(root.rglob('*')) if p.is_file() and p.name != 'manifest.json'})
    print((root/'summary.json').read_text(), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--directory', required=True); parser.add_argument('--source', required=True)
    args = parser.parse_args()
    try: run(args.directory, args.source)
    except Exception:
        import traceback
        root = Path(args.directory)
        if root.exists():
            write(root/'failure.json', dict(error=traceback.format_exc(),
                tests_opened=(root/'TEST-OPENED.json').exists(), source_sha256=digest(args.source),
                fitted_reports=len(list(root.glob('*/report.json'))), commercial_qualified=False))
        raise
