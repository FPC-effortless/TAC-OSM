#!/usr/bin/env python3
"""Authenticate a FIT-only mixed-code/documentation tokenizer pilot corpus."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

HERE=Path(__file__).resolve().parent


def sha(raw):return hashlib.sha256(raw).hexdigest()
def blob(raw):return hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()

def api(route,raw=False):
    args=['gh','api',route]
    if raw:args+=['-H','Accept: application/vnd.github.raw+json']
    result=subprocess.check_output(args)
    return result if raw else json.loads(result)


def prepare(code_receipt,code_data,out):
    if out.exists():raise FileExistsError('do not overwrite original corpus')
    old=json.loads(code_receipt.read_text())
    code=[r for r in old['files'] if r['partition'] in ('BASE_FIT','EXPANSION_FIT')]
    if len(code)!=93:raise ValueError('expected frozen93-file CODEVALUE004 broad FIT')
    out.mkdir(parents=True);(out/'code').mkdir();(out/'text').mkdir();(out/'licenses').mkdir()
    sources=[];identities=[]
    for i,row in enumerate(code):
        raw=(code_data/row['file']).read_bytes();raw.decode('utf-8',errors='strict')
        if sha(raw)!=row['sha256'] or blob(raw)!=row['git_blob_sha1']:raise ValueError('original FIT code mismatch')
        path=f'code/{i:03d}.py';(out/path).write_bytes(raw)
        sources.append(dict(path=path,partition='FIT',modality='code',sha256=sha(raw),bytes=len(raw),
            publisher=row['repo'],repo=row['repo'],commit=row['commit'],source_path=row['path'],
            git_blob_sha1=row['git_blob_sha1'],license=row['license']))
    approved={r['repo']:r for r in old['license_files'] if r['repo'] in {x['repo'] for x in code}}
    prose=[]
    # Reuse exact already-authenticated FIT publisher versions; no DEV publisher.
    def metadata(item):
        repo,lic=item
        tree=api(f'repos/{repo}/git/trees/{lic["commit"]}?recursive=1')
        if tree.get('truncated'):raise ValueError('truncated source tree')
        candidates=[]
        for row in tree['tree']:
            path=Path(row['path']);parts=set(path.parts)
            if row['type']!='blob' or path.suffix.lower() not in ('.rst','.md','.txt'):continue
            if not 2048<=row.get('size',0)<=65536:continue
            if parts & {'test','tests','testing','_vendor','vendor','third_party','changelog.d','news','newsfragments'}:continue
            if any(w in path.name.lower() for w in ('license','copying','change','history','notice')):continue
            if not (parts & {'doc','docs','documentation'} or path.name.lower().startswith('readme')):continue
            candidates.append(row)
        candidates.sort(key=lambda r:sha((repo+':'+r['path']).encode()))
        selected=[];total=0
        for row in candidates:
            if total+row['size']>225000:continue
            selected.append(dict(repo=repo,commit=lic['commit'],path=row['path'],size=row['size'],
                git_blob_sha1=row['sha'],license=next(r['license'] for r in code if r['repo']==repo)))
            total+=row['size']
            if len(selected)==16:break
        for row in selected:
            if repo=='python/cpython' and row['path'].startswith('Modules/_decimal/libmpdec/'):
                row['license']='BSD-2-Clause; CPython Doc/license.rst additional libmpdec notice'
        return selected
    with ThreadPoolExecutor(max_workers=6) as ex:
        for rows in ex.map(metadata,sorted(approved.items())):prose.extend(rows)
    if not prose:raise ValueError('no authenticated prose candidates')
    identities={'study':'LANG10M001_TOKENIZER_PILOT','code_receipt_sha256':sha(code_receipt.read_bytes()),
       'code_source_cohort_sha256':old['cohort_sha256'],'prose':prose,
       'policy':'FIT publishers only, deterministic sha256(repo:path) order,2..64KiB docs/readme,225kB/publisher, no changelog/license/tests/vendor; no content/performance-based choice',
       'selection_DEV_and_transfer_included':False,'adequate_for10M_pretraining_claim':False}
    (out/'SOURCE_IDENTITIES.json').write_text(json.dumps(identities,indent=2)+'\n')
    def fetch(pair):
        i,row=pair;raw=api(f'repos/{row["repo"]}/contents/{row["path"]}?ref={row["commit"]}',True)
        raw.decode('utf-8',errors='strict')
        if len(raw)!=row['size'] or blob(raw)!=row['git_blob_sha1']:raise ValueError('publisher prose mismatch')
        path=f'text/{i:03d}{Path(row["path"]).suffix}'
        (out/path).write_bytes(raw)
        return dict(path=path,partition='FIT',modality='documentation_prose_with_code_examples',
            sha256=sha(raw),bytes=len(raw),publisher=row['repo'],repo=row['repo'],commit=row['commit'],
            source_path=row['path'],git_blob_sha1=row['git_blob_sha1'],license=row['license'])
    with ThreadPoolExecutor(max_workers=8) as ex:sources.extend(ex.map(fetch,enumerate(prose)))
    archives=[]
    for repo,lic in sorted(approved.items()):
        raw=(code_data/lic['file']).read_bytes()
        if sha(raw)!=lic['sha256']:raise ValueError('original license mismatch')
        name='licenses/'+lic['file'];(out/name).write_bytes(raw)
        archives.append(dict(repo=repo,commit=lic['commit'],path=lic['path'],file=name,sha256=sha(raw)))
    if 'python/cpython' in approved:
        commit=approved['python/cpython']['commit'];raw=api(f'repos/python/cpython/contents/Doc/license.rst?ref={commit}',True)
        name='licenses/license-python-cpython-additional-notices.rst';(out/name).write_bytes(raw)
        archives.append({'repo':'python/cpython','commit':commit,'path':'Doc/license.rst','file':name,'sha256':sha(raw)})
    manifest={'variant':'LANG10M001','partition':'FIT','purpose':'Tokenizer pilot and no-update capacity profile only; not a claim of adequate unique data for10M pretraining',
        'source_policy':identities['policy'],'license_archives':archives,'sources':sources,
        'dev_accessed':False,'test_accessed':False,'neural_fits':0}
    manifest['cohort_sha256']=sha(json.dumps(manifest,sort_keys=True,separators=(',',':')).encode())
    (out/'FIT_MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({'sources':len(sources),'code_bytes':sum(r['bytes'] for r in sources if r['modality']=='code'),
        'prose_bytes':sum(r['bytes'] for r in sources if r['modality']!='code'),'cohort_sha256':manifest['cohort_sha256']}))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--code-receipt',type=Path,required=True)
    p.add_argument('--code-data',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();prepare(a.code_receipt,a.code_data,a.out)
