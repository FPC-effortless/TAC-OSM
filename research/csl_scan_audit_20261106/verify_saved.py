"""Read-only archive/status replay; no Torch import or operator calls."""
import base64
import gzip
import hashlib
import io
import itertools
import json
import math
from pathlib import Path
import zipfile


def require(condition, label):
    if not condition:
        raise ValueError(label)


def verify():
    folder = Path(__file__).resolve().parent
    root = folder.parents[1]
    receipt = json.loads((folder / 'HOSTED_RECEIPT.json').read_text())
    raw = base64.decodebytes((folder / 'hosted/ORIGINAL_ARTIFACT.zip.b64').read_bytes())
    require(hashlib.sha256(raw).hexdigest() == receipt['zip_sha256'], 'original ZIP hash')
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        result = json.loads(archive.read('RESULTS.json'))
    summary = json.loads((folder / 'HOSTED_SUMMARY.json').read_text())
    expected_failed = [{k: row[k] for k in ('case', 'chunk', 'objective', 'failed_checks')}
                       for row in result['rows'] if row['failed_checks']]
    require(summary['failed_case_identities'] == expected_failed, 'summary hides failed cases')
    require(summary['audit_status'] == result['audit_status'], 'summary gate relabeled')
    expected_receipts = [{'case': row['case'], 'chunk': row['chunk'], 'objective': row['objective'],
                         'checks': {k: row['checks'][k] for k in row['failed_checks']}}
                        for row in result['rows'] if row['failed_checks']]
    require(summary['failed_check_receipts'] == expected_receipts, 'summary errors changed')
    require(summary['case_rows'] == result['case_rows']
            and summary['failed_rows'] == result['failed_rows'], 'summary count changed')
    protocol = json.loads((folder / 'PROTOCOL.json').read_text())
    for path, digest in protocol['source_sha256'].items():
        require(hashlib.sha256((root / path).read_bytes()).hexdigest() == digest, 'original operator changed')
    expected = set(itertools.product(protocol['cases'], (1, 7, 32, 64),
                                    ('linear_state', 'normalized_read', 'normalized_read_source_default')))
    rows = result['rows']
    observed = {(r['case'], r['chunk'], r['objective']) for r in rows}
    require(expected == observed and len(rows) == len(expected) == 84, 'case coverage or duplicate')
    checks = {'serial_vs_decimal', 'numpy_vs_decimal', 'stable_future_poison', 'candidate_future_poison'}
    for name in ('stable', 'candidate'):
        checks.update({name + '_forward', name + '_final'})
        checks.update(name + '_grad_' + arg for arg in ('z', 'carry', 'rate', 'gate'))
    checks.update('candidate_vs_stable_grad_' + arg for arg in ('z', 'carry', 'rate', 'gate'))
    failures = 0
    for row in rows:
        require(set(row['checks']) == checks, 'missing/extra numerical check')
        for check in row['checks'].values():
            scaled = check['max_scaled']
            if scaled is not None:
                require(math.isfinite(scaled) and scaled >= 0, 'invalid scaled error')
                require(check['status'] == ('PASS' if scaled <= 1 else 'FAIL_TOLERANCE'), 'false numerical status')
            else:
                require(check['status'] in ('FAIL_NONFINITE', 'FAIL_SHAPE_OR_EMPTY'), 'unknown check passed')
        failed = [k for k, v in row['checks'].items() if v['status'] != 'PASS']
        require(len(row['failed_checks']) == len(failed)
                and set(row['failed_checks']) == set(failed), 'failed checks hidden')
        require(row['status'] == ('FAIL_NUMERICAL_GATE' if failed else 'PASS_BOUNDED_FIXTURE'), 'false row status')
        failures += bool(failed)
    require(failures == result['failed_rows'] == receipt['failed_rows'] == 11, 'failed row count')
    require(result['audit_status'] == 'COMPLETE_WITH_FAILURES', 'failed gate relabeled')
    require(result['kernel_promotion_authorized'] is False, 'promotion was granted')
    for name, entry in json.loads((folder / 'local_history/SHA256.json').read_text()).items():
        compressed = base64.decodebytes((folder / 'local_history' / (name + '.gz.b64')).read_bytes())
        require(hashlib.sha256(compressed).hexdigest() == entry['gzip_sha256'], 'old compressed ledger changed')
        require(hashlib.sha256(gzip.decompress(compressed)).hexdigest() == entry['original_sha256'], 'old local ledger changed')
    return {'status': 'PASS_SAVED_EVIDENCE_ARITHMETIC', 'numerical_gate': 'FAIL',
            'case_rows': 84, 'failed_rows': failures, 'operator_executions': 0,
            'qualification': False}


if __name__ == '__main__':
    print(json.dumps(verify()))
