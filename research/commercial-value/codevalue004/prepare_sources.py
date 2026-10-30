#!/usr/bin/env python3
"""Freeze publisher metadata, then authenticate bytes and archive original licenses."""
from concurrent.futures import ThreadPoolExecutor
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
EXPANSION=('encode/uvicorn','aio-libs/aiohttp','pallets/jinja','pypa/pip',
           'python/mypy','psf/black','sqlalchemy/sqlalchemy','urllib3/urllib3')
TRANSFER=('astropy/astropy','sympy/sympy','networkx/networkx','scrapy/scrapy','bokeh/bokeh')

def api(path,raw=False):
    cmd=['gh','api',path]
    if raw:cmd+=['-H','Accept: application/vnd.github.raw+json']
    b=subprocess.check_output(cmd)
    return b if raw else json.loads(b)

def digest(b):return hashlib.sha256(b).hexdigest()

def original_rows():
    old=json.loads((ROOT/'research/commercial-value/codevalue003/SOURCE_RECEIPT.json').read_text())
    return [dict(r,partition='BASE_FIT' if r['partition']=='FIT' else 'SELECT_DEV')
            for r in old['files'] if r['partition'] in ('FIT','DEV')]

def eligible(path,size):
    denied={'test','tests','testing','docs','doc','examples','example','benchmarks','benchmark',
            'vendored','_vendor','vendor','third_party','tools','scripts'}
    parts=Path(path).parts
    return (path.endswith('.py') and 8192<=size<=65536 and not denied.intersection(parts)
            and not Path(path).name.startswith('test_') and not Path(path).name.endswith('_test.py'))

def metadata(repo):
    commit=api('repos/'+repo+'/commits/HEAD')['sha']
    tree=api('repos/'+repo+'/git/trees/'+commit+'?recursive=1')
    if tree.get('truncated'):raise ValueError('publisher tree truncated: '+repo)
    license=api('repos/'+repo+'/license')
    spdx=license['license']['spdx_id']
    reviewed={'python/mypy':'MIT','sympy/sympy':'BSD-3-Clause','networkx/networkx':'BSD-3-Clause'}
    if spdx in ('NOASSERTION','NONE'):
        if repo not in reviewed:raise ValueError('unresolved license: '+repo)
        spdx=reviewed[repo]  # Original license texts were read; mixed notices confuse API detection.
    license['source_reviewed_spdx']=spdx
    candidates=[r for r in tree['tree'] if r['type']=='blob' and eligible(r['path'],r.get('size',0))]
    candidates.sort(key=lambda r:digest((repo+':'+r['path']).encode()))
    cap=225000 if repo in EXPANSION else 131072
    chosen=[];total=0
    for row in candidates:
        if total+row['size']>cap:continue
        chosen.append(dict(repo=repo,commit=commit,path=row['path'],size=row['size'],
            git_blob_sha1=row['sha'],license=spdx,
            partition='EXPANSION_FIT' if repo in EXPANSION else 'TRANSFER_DEV',group=repo.lower()))
        total+=row['size']
        if len(chosen)>=(12 if repo in EXPANSION else 2):break
    if len(chosen)<2:raise ValueError('insufficient eligible source files: '+repo)
    return chosen,{'repo':repo,'commit':commit,'path':license['path'],'spdx_id':spdx,'license_git_blob_sha1':license['sha'],
        'license_interpretation':'Original text read for NOASSERTION mypy(MIT .py), SymPy(BSD notices), NetworkX(explicit BSD3); all complete notices archived' if repo in reviewed else 'GitHub SPDX metadata plus archived original text'}

def prepare(out):
    if out.exists():raise FileExistsError('do not overwrite frozen publisher bundle')
    out.mkdir(parents=True)
    rows=original_rows();licenses=[]
    with ThreadPoolExecutor(max_workers=8) as ex:
        for chosen,license in ex.map(metadata,EXPANSION+TRANSFER):
            rows.extend(chosen);licenses.append(license)
    # Freeze identities and selection/exclusion policy before decoding any new bytes.
    for i,row in enumerate(rows):row['file']=f'{i:03d}.bin'
    policy={'path_selection':'sha256(repo:path) ascending; .py8..64KiB; exclude tests/docs/examples/vendor; no performance-based choice',
            'expansion_limit_bytes_per_publisher':225000,'max_expansion_files_per_publisher':12,
            'transfer_files_per_publisher':2,'overlap_fail_threshold':.10,
            'overlap_screen':'dense FIT64-byte windows, heldout64-byte windows sampled every16 bytes',
            'failure':'Fail closed on exact file duplication or lexical overlap>10%; no post-score exclusion',
            'access':'Internal source audit, not an access-separated blind TEST'}
    frozen={'study':'CODEVALUE004','files':rows,'new_publisher_licenses':licenses,'policy':policy}
    (HERE/'SOURCE_IDENTITIES.json').write_text(json.dumps(frozen,indent=2)+'\n')
    def fetch(row):
        b=api(f'repos/{row["repo"]}/contents/{row["path"]}?ref={row["commit"]}',True)
        blob=hashlib.sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest()
        if len(b)!=row['size'] or blob!=row['git_blob_sha1']:raise ValueError('source mismatch: '+row['path'])
        (out/row['file']).write_bytes(b)
        return dict(row,sha256=digest(b)),b
    with ThreadPoolExecutor(max_workers=8) as ex:files=list(ex.map(fetch,rows))
    fits=[b for r,b in files if r['partition'] in ('BASE_FIT','EXPANSION_FIT')]
    exact={digest(b) for b in fits}
    shingles={b[i:i+64] for b in fits for i in range(len(b)-63)}
    audits=[]
    for row,b in files:
        if row['partition'] not in ('SELECT_DEV','TRANSFER_DEV'):continue
        probes=[b[i:i+64] for i in range(0,len(b)-63,16)]
        overlap=sum(s in shingles for s in probes)/len(probes)
        if digest(b) in exact or overlap>.10:raise ValueError('heldout source overlap exclusion: '+row['repo']+':'+row['path']+' '+str(overlap))
        audits.append({'file':row['file'],'exact_fit_duplicate':False,'sampled_overlap_fraction':overlap})
    del shingles
    # Inherit original license text identities from CODEVALUE003, preserving bytes.
    old=json.loads((ROOT/'research/commercial-value/codevalue003/SOURCE_RECEIPT.json').read_text())
    for lic in old['license_files']:
        if lic['repo'] in {r['repo'] for r in rows if r['partition'] in ('BASE_FIT','SELECT_DEV')}:
            licenses.append({k:lic[k] for k in ('repo','commit','path')})
    archived=[]
    for lic in licenses:
        b=api(f'repos/{lic["repo"]}/contents/{lic["path"]}?ref={lic["commit"]}',True)
        name='license-'+lic['repo'].replace('/','-')+'.txt';(out/name).write_bytes(b)
        archived.append(dict(lic,file=name,sha256=digest(b)))
    receipt={'study':'CODEVALUE004','files':[r for r,b in files],
        'license_files':archived,'partition_audit':audits,'policy':policy,
        'historical_exposure':'INCONCLUSIVE across all historical studies; no strict blind claim',
        'semantic_duplicate_audit':'INCONCLUSIVE; lexical screen is not semantic verification'}
    receipt['cohort_sha256']=digest(json.dumps(receipt,sort_keys=True,separators=(',',':')).encode())
    for path in (out/'SOURCE_RECEIPT.json',HERE/'SOURCE_RECEIPT.json'):
        path.write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps({'cohort_sha256':receipt['cohort_sha256'],
        'partitions':{p:{'files':sum(r['partition']==p for r,b in files),
            'bytes':sum(len(b) for r,b in files if r['partition']==p)}
            for p in ('BASE_FIT','EXPANSION_FIT','SELECT_DEV','TRANSFER_DEV')}}))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True)
    prepare(p.parse_args().out)
