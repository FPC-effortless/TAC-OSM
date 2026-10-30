#!/usr/bin/env python3
"""Authenticate immutable publisher bytes for an offline private Kaggle bundle."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import subprocess

HERE = Path(__file__).resolve().parent

def gh_raw(repo, commit, path):
    return subprocess.check_output(['gh', 'api', f'repos/{repo}/contents/{path}?ref={commit}',
                                    '-H', 'Accept: application/vnd.github.raw+json'])

def digest(raw):
    return hashlib.sha256(raw).hexdigest()

def prepare(out):
    reg = json.loads((HERE / 'REGISTER.json').read_text())
    transfer = json.loads((HERE / 'TRANSFER_SOURCES.json').read_text())
    rows = []
    for r in reg['train']['files']:
        rows.append(dict(r, repo=reg['train']['repo'], commit=reg['train']['commit'],
                         partition='FIT', group='cpython', license='PSF-2.0'))
    licenses = {'requests': 'Apache-2.0', 'httpx': 'BSD-3-Clause', 'click': 'BSD-3-Clause'}
    for r in reg['exposed_dev']['files']:
        rows.append(dict(r, partition='DEV', license=licenses[r['group']]))
    for r in transfer['files']:
        rows.append(dict(r, partition='TRANSFER_DEV', group=r['repo'].lower(),
                         license=r['license']['spdx_id']))
    out.mkdir(parents=True, exist_ok=True)
    def fetch(pair):
        i, row = pair
        raw = gh_raw(row['repo'], row['commit'], row['path'])
        blob = hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()
        if len(raw) != row['size'] or blob != row['git_blob_sha1']:
            raise ValueError('publisher source mismatch: '+row['repo']+':'+row['path'])
        filename = f'{i:02d}.bin'
        (out / filename).write_bytes(raw)
        return {k: v for k, v in dict(row, sha256=digest(raw), file=filename).items()
                if k != 'status'}, raw
    with ThreadPoolExecutor(max_workers=8) as ex:
        fetched = list(ex.map(fetch, enumerate(rows)))
    originals = [raw for r, raw in fetched if r['partition'] == 'FIT']
    fit_hashes = {digest(b) for b in originals}
    # Coarse source-duplication screen; cannot prove semantic disjointness.
    # Shared 64-byte shingles sampled every32 bytes, compared to FIT only.
    fit_shingles = {digest(b[i:i+64]) for b in originals for i in range(0,len(b)-63,32)}
    audits = []
    for r, b in fetched:
        if r['partition'] == 'FIT': continue
        shingles = {digest(b[i:i+64]) for i in range(0,len(b)-63,32)}
        overlap = len(shingles & fit_shingles) / max(1, len(shingles))
        if digest(b) in fit_hashes or overlap > .10:
            raise ValueError(f'FIT overlap exclusion triggered: {r["path"]} {overlap}')
        audits.append({'file':r['file'], 'exact_fit_duplicate':False,
                       'sampled64byte_shingle_overlap_fraction':overlap})
    # Archive original license text at each publisher commit. No generated grant.
    license_paths = {'python/cpython':'LICENSE','psf/requests':'LICENSE',
                     'encode/httpx':'LICENSE.md','pallets/click':'LICENSE.txt',
                     'django/django':'LICENSE','fastapi/fastapi':'LICENSE',
                     'Textualize/rich':'LICENSE','pydantic/pydantic':'LICENSE',
                     'pytest-dev/pytest':'LICENSE'}
    archived = []
    for repo, commit in sorted({(r['repo'],r['commit']) for r in rows}):
        path = license_paths[repo]
        raw = gh_raw(repo, commit, path)
        filename = 'license-'+repo.replace('/','-')+'.txt'
        (out/filename).write_bytes(raw)
        archived.append({'repo':repo,'commit':commit,'path':path,'file':filename,'sha256':digest(raw)})
    receipt = {'study':'CODEVALUE003', 'files':[r for r,b in fetched],
               'license_files':archived, 'duplicate_screen':audits,
               'exposure':'DEV internally measured, no access-separated blind TEST claim',
               'historical_exposure_audit':'INCONCLUSIVE: not every prior branch/archive audited',
               'duplicate_screen_limits':'sampled lexical screen only; semantic duplication not excluded'}
    encoded = json.dumps(receipt, sort_keys=True, separators=(',',':')).encode()
    receipt['cohort_sha256'] = digest(encoded)
    (out/'SOURCE_RECEIPT.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps({'status':'SOURCE_VERIFIED','files':len(rows),
                      'cohort_sha256':receipt['cohort_sha256']}))

if __name__ == '__main__':
    p=argparse.ArgumentParser(); p.add_argument('--out',type=Path,required=True)
    prepare(p.parse_args().out)
