#!/usr/bin/env python3
"""Bounded publisher TRAIN-only acquisition; no archive, holdout, fit, or model.

Run the inherited source audit after two exact CSVs are acquired. Network and
source failures are receipts, not a reason to weaken hash/schema rules. Raw data
stays in the runner scratch directory; only metadata/audit receipts are exported.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

READOUT_SHA256 = 'e01cb0bd87dfab4c9dbad215d51d81282fc0d413be96d6f819ea872fb7a3c715'
TTE_SHA256 = 'd8c2379ed7c95a575dd869730b2b3b96d660317f49e57de300518ff3b08d53a5'
ALLOWED = {'train_operational_readouts.csv': (1_300_000_000, READOUT_SHA256),
           'train_tte.csv': (10_000_000, TTE_SHA256)}
HOSTS = {'api.researchdata.se'}


def check_url(url: str, filename: str) -> None:
    p = urllib.parse.urlsplit(url)
    if (filename not in ALLOWED or p.scheme != 'https' or p.hostname not in HOSTS
            or p.username or p.password or p.port not in (None, 443)
            or p.query or p.fragment
            or p.path != '/dataset/2024-34/3/file/data/' + filename):
        raise ValueError('URL outside exact publisher TRAIN-only allowlist')


class AllowedRedirect(urllib.request.HTTPRedirectHandler):
    def __init__(self, filename: str):
        self.filename = filename
        super().__init__()

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        try:
            check_url(newurl, self.filename)
        except ValueError as exc:
            parsed = urllib.parse.urlsplit(newurl)
            # No token/query values: retain only enough public route metadata to
            # review a redirect before adding it to the allowlist.
            route = {'scheme': parsed.scheme, 'host': parsed.hostname,
                     'path': parsed.path, 'query_keys': sorted(urllib.parse.parse_qs(parsed.query))}
            raise ValueError('rejected publisher redirect: ' + json.dumps(route, sort_keys=True)) from exc
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')
    tmp.replace(path)


def acquire(filename: str, directory: Path, attempts: list, opener_factory=None) -> dict:
    if filename not in ALLOWED:
        raise ValueError('not an allowed TRAIN file')
    directory.mkdir(parents=True, exist_ok=True)
    limit, expected = ALLOWED[filename]
    target = directory / filename
    partial = directory / (filename + '.partial')
    urls = [f'https://api.researchdata.se/dataset/2024-34/3/file/data/{filename}']
    for url in urls:
        check_url(url, filename)
        entry = {'requested_url': url, 'filename': filename, 'status': 'STARTED'}
        attempts.append(entry)
        h, count, started = hashlib.sha256(), 0, time.monotonic()
        try:
            opener = (urllib.request.build_opener(AllowedRedirect(filename))
                      if opener_factory is None else opener_factory(filename))
            request = urllib.request.Request(url, headers={
                'User-Agent': 'Woyengi-DecisionSYS1-003-SourceAudit/1.0',
                'Accept-Encoding': 'identity'})
            with opener.open(request, timeout=60) as response:
                check_url(response.geturl(), filename)
                if response.status != 200:
                    raise ValueError(f'unexpected HTTP status {response.status}')
                entry['final_url'] = response.geturl()
                entry['content_type'] = response.headers.get('Content-Type', '')
                if any(t in entry['content_type'].lower() for t in ('text/html', 'application/json')):
                    raise ValueError('publisher returned a document rather than CSV')
                with partial.open('wb') as out:
                    while True:
                        block = response.read(1024 * 1024)
                        if not block:
                            break
                        count += len(block)
                        if count > limit or time.monotonic() - started > 900:
                            raise ValueError('download byte/time budget exceeded')
                        h.update(block)
                        out.write(block)
            digest = h.hexdigest()
            entry.update(bytes=count, sha256=digest, elapsed_seconds=time.monotonic()-started)
            if count == 0:
                raise ValueError('empty source')
            if expected is not None and digest != expected:
                raise ValueError('original-source SHA256 mismatch; no substitute accepted')
            with partial.open('rb') as f:
                header = f.readline(4096).decode('utf-8-sig').strip()
            columns = header.split(',')
            if filename == 'train_tte.csv' and set(columns) != {
                    'vehicle_id', 'length_of_study_time_step', 'in_study_repair'}:
                raise ValueError('unexpected TRAIN outcome header')
            if filename == 'train_operational_readouts.csv' and len(columns) != 107:
                raise ValueError('unexpected readout header')
            partial.replace(target)
            entry['status'] = 'ACQUIRED'
            entry['hash_attestation'] = ('matches_historical_and_publisher' if filename == 'train_operational_readouts.csv' else
                                        'matches_publisher_catalogue_not_independent_attestation')
            print(json.dumps({'file': filename, 'bytes': count, 'sha256': digest,
                              'status': 'ACQUIRED'}), flush=True)
            return entry
        except Exception as exc:
            entry.update(status='FAILED', error=f'{type(exc).__name__}: {exc}',
                         bytes_received=count, elapsed_seconds=time.monotonic()-started)
            partial.unlink(missing_ok=True)
    raise RuntimeError(f'no allowed individual-file endpoint succeeded for {filename}')


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-dir', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    report = {'study': 'DECISION-SYS1-003', 'stage': 'TRAIN_SOURCE_ACQUISITION',
              'attempts': [], 'optimizer_steps': 0, 'real_model_predictions': 0,
              'validation_opened': False, 'test_opened': False,
              'archive_requested': False, 'training_admitted': False}
    try:
        # No labels are requested until authenticated readout bytes exist.
        acquire('train_operational_readouts.csv', a.data_dir, report['attempts'])
        acquire('train_tte.csv', a.data_dir, report['attempts'])
        write_json(a.output/'acquisition.json', report)
        from audit_decision_sys1_train_source_002 import audit
        source = audit(a.data_dir/'train_operational_readouts.csv',
                       a.data_dir/'train_tte.csv', a.output/'source')
        report['status'] = 'SOURCE_AUDIT_COMPLETED'
        report['source_summary'] = source
        return_code = 0
    except Exception as exc:
        report['status'] = 'BLOCKED'
        report['error'] = f'{type(exc).__name__}: {exc}'
        return_code = 1
    write_json(a.output/'acquisition.json', report)
    print('DECISION_SYS1_003_RECEIPT=' + json.dumps(report, sort_keys=True), flush=True)
    return return_code


if __name__ == '__main__':
    raise SystemExit(main())
