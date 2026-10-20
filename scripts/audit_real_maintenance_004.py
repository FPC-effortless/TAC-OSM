"""Reconstruct public inputs and replay every recorded forecast; never refit."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
ROOT = Path(__file__).resolve().parents[1]; sys.path[:0] = [str(ROOT/'src'), str(ROOT/'scripts')]
import numpy as np
import torch
from tac_osm.development.repair_outcome import RepairOutcomeForecaster
from data_real_maintenance_004 import source, split, labeled, fit_schema, features
from run_real_maintenance_004 import reference_predict
from run_learned_stack_dev055 import write, digest
from audit_identity_dev0615 import compare


def export(model, folder, report, schema, rows, validation_ids, destination, qualified):
    destination.mkdir(parents=True, exist_ok=False)
    shutil.copy(folder/'model.pt', destination/'model.pt')
    shutil.copy(ROOT/'src/tac_osm/development/repair_outcome.py', destination/'model.py')
    shutil.copy(ROOT/'scripts/runtime_repair_outcome_004.py', destination/'runtime.py')
    write(destination/'metadata.json', dict(arm=report['arm'], seed=report['seed'], schema=schema,
        recipe=report['recipe'], fitting_source_commit=json.loads((folder.parent/'summary.json').read_text())['source'],
        features=report['features'], checkpoint_sha256=report['checkpoint_sha256'],
        unit='seconds_per_20L', evidence_stage='FRESH_REGISTERED_FIT' if folder.name.startswith('fresh') else 'DIAGNOSTIC_PILOT_ONLY',
        local_numerical_qualification=qualified, commercial_qualified=False,
        source_dataset='openwashdata/floodchlorinationsurvey DOI10.5281/zenodo.15846578 CC BY4.0',
        target='reported post-rehabilitation bucket fill seconds; not a causal repair benefit'))
    allowed = schema['categorical']+schema['numeric']
    demo = [dict((key, rows[i][key]) for key in allowed) for i in validation_ids[:5]]
    request = json.dumps(dict(records=demo))+'\n'
    (destination/'demo-request.jsonl').write_text(request)
    expected = dict(time_to_fill_20L_seconds=model(torch.tensor(features(rows, validation_ids[:5], schema))).tolist())
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
        input=request, capture_output=True, text=True)
    assert response.returncode == 0, response.stderr
    compare(json.loads(response.stdout), expected, dict(values=0, max_abs=0.))
    (destination/'demo-output.jsonl').write_text(response.stdout)
    (destination/'NOTICE.txt').write_text('Data attribution: Emmanuel Mhango and contributors, floodchlorinationsurvey0.1.0, USAID survey Mulanje Malawi2019–2020; CC BY4.0; DOI10.5281/zenodo.15846578.\nTAC-OSM project code metadata declares Apache-2.0. No geographic/personal identifiers are inference inputs.\n')
    (destination/'README.txt').write_text('Python + PyTorch only. python runtime.py < demo-request.jsonl\nInput JSON: records containing the technical prestatus/recorded replacement/chlorination fields in metadata schema. Missing numeric fields remain explicitly flagged unknown.\nOutput: time_to_fill_20L_seconds, a factual point forecast conditioned on the recorded bundle. Units are SECONDS PER20L, not strokes or liters/second.\nNo entity identity, date, coordinates, text, after-functionality or observed bucket/stroke value enters inference.\nThis small same-survey model does not select repairs, identify causal benefits, price work, certify repair success or establish commercial reliability. Check metadata/model card for actual qualification status.\n')
    return len(demo)


@torch.no_grad()
def run(directory, source_path, output, exports=None):
    torch.set_num_threads(1); torch.set_deterministic_debug_mode('error'); start = time.monotonic()
    root = Path(directory); summary = json.loads((root/'summary.json').read_text())
    assert summary['status'] in ('VALIDATION_REJECTED', 'SOURCE_HEADROOM_REJECTED', 'TEST_COMPLETE', 'TEST_SOURCE_INSUFFICIENT')
    for name, sha in json.loads((root/'manifest.json').read_text()).items(): assert digest(root/name) == sha, name
    frozen = summary['source']
    names = ['src/tac_osm/development/repair_outcome.py', 'scripts/data_real_maintenance_004.py',
        'scripts/run_real_maintenance_004.py', 'research/real-maint-004/PROTOCOL.md', 'research/real-maint-004/protocol.json']
    for name in names: assert subprocess.check_output(['git', 'show', frozen+':'+name], cwd=ROOT) == (ROOT/name).read_bytes(), name
    rows = source(source_path); assignment, groups, metadata = split(rows)
    assert metadata == json.loads((root/'split.json').read_text())
    assert assignment == json.loads((root/'assignment.json').read_text())
    train, y, train_bad = labeled(rows, assignment['train']); val, vy, val_bad = labeled(rows, assignment['validation'])
    assert train_bad == summary['rejected_train'] and val_bad == summary['rejected_validation']
    if not summary['test_opened']:
        assert not (root/'TEST-OPENED.json').exists() and not (root/'test-labels.json').exists()
    result = dict(status='RUNNING', checkpoints=0, predictions=0, portable_predictions=0,
        test_opened=summary['test_opened'], source=frozen, telemetry=dict(values=0, max_abs=0.), commercial_qualified=False)
    if not (root/'prepared.pt').exists():
        assert summary['status'] == 'SOURCE_HEADROOM_REJECTED'
    else:
        prepared = torch.load(root/'prepared.pt', weights_only=True)
        assert train == prepared['train_ids'] and val == prepared['validation_ids']
        torch.testing.assert_close(torch.tensor(y), prepared['train_targets_seconds'], atol=0, rtol=0)
        torch.testing.assert_close(torch.tensor(vy), prepared['validation_targets_seconds'], atol=0, rtol=0)
        schemas = json.loads((root/'schemas.json').read_text())
        for arm, schema in schemas.items():
            assert schema == fit_schema(rows, train, arm)
            torch.testing.assert_close(torch.tensor(features(rows, train, schema)), prepared[arm]['train'], atol=0, rtol=0)
            torch.testing.assert_close(torch.tensor(features(rows, val, schema)), prepared[arm]['validation'], atol=0, rtol=0)
        references = json.loads((root/'references.json').read_text())
        for name, ref in references.items():
            pred = reference_predict(ref['fit'], features(rows, val, schemas['repair']), rows, val, name)
            compare(pred.tolist(), ref['validation_prediction_seconds'], result['telemetry'])
        test_inputs = None
        if summary['test_opened']:
            labels = json.loads((root/'test-labels.json').read_text())
            test, ty, test_bad = labeled(rows, assignment['test'])
            assert test == labels['ids'] and ty.tolist() == labels['target_seconds'] and test_bad == labels['rejected_nonpositive_or_malformed']
            assert labels['groups'] == [groups[i] for i in test]
            if summary['status'] == 'TEST_COMPLETE':
                test_inputs = torch.load(root/'test-inputs.pt', weights_only=True)
                for arm, schema in schemas.items():
                    torch.testing.assert_close(torch.tensor(features(rows, test, schema)), test_inputs[arm], atol=0, rtol=0)
        reports = sorted(root.glob('*/report.json')); assert len(reports) == summary['fit_count']
        for path in reports:
            report = json.loads(path.read_text()); folder = path.parent; arm = report['arm']
            assert digest(folder/'model.pt') == report['checkpoint_sha256']
            model = RepairOutcomeForecaster(report['features'], [0]*report['features'], [1]*report['features'], 1.)
            model.load_state_dict(torch.load(folder/'model.pt', weights_only=True)); model.eval()
            pred = model(prepared[arm]['validation']).tolist()
            compare(pred, report['validation_prediction_seconds'], result['telemetry'])
            compare(float(np.abs(np.array(pred)-vy).mean()), report['validation_MAE_seconds'], result['telemetry'])
            result['predictions'] += len(pred)
            if (folder/'test.json').exists():
                expected = json.loads((folder/'test.json').read_text())
                compare(model(test_inputs[arm]).tolist(), expected['prediction_seconds'], result['telemetry'])
                result['predictions'] += len(test_inputs[arm])
            chosen = (folder.name == f'fresh-{arm}-97401' if list(root.glob('fresh-*')) else folder.name == f'pilot-{arm}-0-97400')
            if exports and chosen:
                result['portable_predictions'] += export(model, folder, report, schemas[arm], rows, val, Path(exports)/arm, qualified=False)
            result['checkpoints'] += 1; write(output, result)
    result.update(status='COMPLETE', elapsed_seconds=time.monotonic()-start,
        source_files_sha256={name: digest(ROOT/name) for name in names})
    write(output, result); print(json.dumps(result), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--directory', required=True); parser.add_argument('--source', required=True)
    parser.add_argument('--output', required=True); parser.add_argument('--exports'); args = parser.parse_args()
    run(args.directory, args.source, args.output, args.exports)
