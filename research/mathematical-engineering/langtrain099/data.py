"""Authenticated real byte corpus plus frozen097 episodic FIT/DEV.

All four097 DEV code owners are removed from broad-corpus FIT. Two documents
per publisher are held out deterministically before any099 model prediction.
Training packages contain FIT only; actor/judge data are packaged separately.
"""
import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import urllib.request
import numpy as np

ROOT=Path(__file__).resolve().parents[3]
HERE=Path(__file__).resolve().parent
PILOT=ROOT/'research/language-scaling/lang10m/tokenizer/FIT_MANIFEST.json'
EPISODIC=ROOT/'research/mathematical-engineering/langacquire097'

def sha(raw):return hashlib.sha256(raw).hexdigest()
def encoded(value):return (json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n').encode()
def read(path):return json.loads(Path(path).read_text())
def save(path,value):
    path=Path(path);raw=encoded(value);path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():
        if path.read_bytes()!=raw:raise ValueError('immutable input differs: '+str(path))
    else:path.write_bytes(raw)
def load_episodic():
    spec=importlib.util.spec_from_file_location('langtrain099_epidata',EPISODIC/'data.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module
def owner(row):return row['repo']+'@'+row['commit']+':'+row['source_path']

def authenticated(row,cache):
    path=cache/'originals'/row['path'];path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():raw=path.read_bytes()
    else:
        url='https://raw.githubusercontent.com/'+row['repo']+'/'+row['commit']+'/'+row['source_path']
        with urllib.request.urlopen(url,timeout=45) as response:raw=response.read(4*1024*1024+1)
    blob=hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()
    if len(raw)!=row['bytes'] or sha(raw)!=row['sha256'] or blob!=row['git_blob_sha1']:
        raise ValueError('publisher file bytes differ: '+owner(row))
    raw.decode('utf-8',errors='strict')
    if not path.exists():path.write_bytes(raw)
    return row,raw

def prepare(cache):
    cache=Path(cache);cache.mkdir(parents=True,exist_ok=True)
    d=load_episodic();epicache=cache/'episodic';episode_manifest=d.prepare(epicache)
    if episode_manifest.get('status')!='COHORT_READY':raise ValueError('original episodic corpus invalid')
    original=read(PILOT);rows=original['sources']
    with ThreadPoolExecutor(max_workers=8) as pool:corpus=list(pool.map(lambda r:authenticated(r,cache),rows))
    documents=defaultdict(list)
    for r,raw in corpus:
        if r['modality']=='documentation_prose_with_code_examples':documents[r['repo']].append(r)
    if len(documents)!=9:raise ValueError("expected all9 licensed documentation publishers")
    selected_docs=set()
    for publisher,rs in sorted(documents.items()):
        if len(rs)<2:raise ValueError('two held-out docs per publisher unavailable')
        selected_docs.update(owner(r) for r in sorted(rs,key=lambda r:sha(('LANGTRAIN099-DOC-DEV:'+owner(r)).encode()))[:2])
    if len(selected_docs)!=18:raise ValueError('exact18 held-out docs required')
    dev_code={(r['repo'],r['path']) for r in d.contract()['cpython_files'] if r['partition']=='DEV'}
    dev_hash={r['sha256'] for r,raw in corpus if owner(r) in selected_docs or (r['repo'],r['source_path']) in dev_code}
    fit_root=cache/'fit-package';dev_root=cache/'eval-package'
    (fit_root/'corpus').mkdir(parents=True,exist_ok=True);(dev_root/'language').mkdir(parents=True,exist_ok=True)
    fit=[];dev=[]
    for r,raw in corpus:
        dev_selected=owner(r) in selected_docs or (r['repo'],r['source_path']) in dev_code
        # Deduplicate whole-file identities/content across partitions, not only paths.
        if not dev_selected and r['sha256'] in dev_hash:continue
        identifier=sha(owner(r).encode())
        target=(dev_root/'language' if dev_selected else fit_root/'corpus')/(identifier+'.bin')
        if target.exists() and target.read_bytes()!=raw:raise ValueError('original package bytes changed')
        target.write_bytes(raw)
        record={'id':identifier,'owner':owner(r),'modality':r['modality'],'sha256':r['sha256'],'bytes':len(raw),
                'file':target.relative_to(dev_root if dev_selected else fit_root).as_posix(),
                'publisher':r['repo'],'commit':r['commit'],'source_path':r['source_path'],'license':r['license']}
        (dev if dev_selected else fit).append(record)
    if len({r['owner'] for r in fit}&{r['owner'] for r in dev}) or {r['sha256'] for r in fit}&{r['sha256'] for r in dev}:
        raise ValueError('whole-file/content leakage between FIT/DEV')
    if {r['owner'] for r in fit}&set(episode_manifest['dev_owners']):raise ValueError('episodic DEV owner in broad FIT')
    episodes=read(epicache/'FIT.json');save(fit_root/'FIT_EPISODES.json',episodes)
    for name in ('DEV_ACTOR.json','DEV_JUDGE.json','DEV_CONTROL_ACTOR.json','DEV_DONORS.json'):
        shutil.copyfile(epicache/name,dev_root/name)
    lm_cases=[]
    for r in dev:
        # Eight nonoverlapping source windows per owner, source-independent hash order.
        starts=list(range(0,r['bytes']-256,256))
        starts.sort(key=lambda i:sha(('LANGTRAIN099-LM-DEV:'+r['id']+':'+str(i)).encode()))
        for offset in starts[:8]:
            lm_cases.append({'id':sha((r['id']+':'+str(offset)).encode()),'source_id':r['id'],'file':r['file'],
                             'offset':offset,'bytes':257,'modality':r['modality'],'owner':r['owner']})
    save(dev_root/'LANGUAGE_CASES.json',lm_cases)
    licenses=[]
    for r in original['license_archives']:
        p=PILOT.parent/r['file'];raw=p.read_bytes()
        if sha(raw)!=r['sha256']:raise ValueError('original license archive changed')
        for base in (fit_root,dev_root):
            target=base/'licenses'/p.name;target.parent.mkdir(exist_ok=True);target.write_bytes(raw)
        licenses.append(r)
    # Preserve all publisher license/notice payloads, including the old pilot's
    # supplemental CPython notices. The baseline is FIT-only and never a NN input.
    notice_hashes={}
    for notice in sorted((PILOT.parent/'licenses').iterdir()):
        if not notice.is_file():continue
        content=notice.read_bytes();notice_hashes[notice.name]=sha(content)
        for base in (fit_root,dev_root):(base/'licenses'/notice.name).write_bytes(content)
    bigram=np.zeros((256,256),dtype=np.int64)
    for r in fit:
        values=np.frombuffer((fit_root/r['file']).read_bytes(),dtype=np.uint8).astype(np.int32)
        bigram+=np.bincount(values[:-1]*256+values[1:],minlength=256*256).reshape(256,256)
    with (dev_root/'FIT_BIGRAM_COUNTS.npy').open('wb') as handle:np.save(handle,bigram,allow_pickle=False)
    manifest={'schema':'langtrain099.authenticated-data.v1','parent_pilot_manifest_sha256':sha(PILOT.read_bytes()),
              'parent_episode_manifest_sha256':sha((epicache/'MANIFEST.json').read_bytes()),
              'fit_sources':fit,'language_dev_sources':dev,'fit_episode_count':len(episodes),
              'original_episode_counts':episode_manifest['counts'],'fit_bytes':sum(r['bytes'] for r in fit),
              'removed_episodic_DEV_owners':sorted(r+'@'+d.contract()['cpython_files'][0]['commit']+':'+p for r,p in dev_code),
              'doc_DEV_selection':'sha256(LANGTRAIN099-DOC-DEV:owner), first2/publisher',
              'language_DEV_windows':len(lm_cases),'licenses':licenses,'additional_license_payloads':notice_hashes,'fit_bigram_counts_sha256':sha((dev_root/'FIT_BIGRAM_COUNTS.npy').read_bytes()),
              'episode_actor_sha256':sha((dev_root/'DEV_ACTOR.json').read_bytes()),
              'episode_judge_sha256':sha((dev_root/'DEV_JUDGE.json').read_bytes()),
              'no_TEST':True,'scope':'Public exposed development; source-disjoint new-model FIT/DEV, not independent blind qualification.'}
    save(cache/'MANIFEST.json',manifest)
    fit_manifest={'source_manifest_sha256':sha(encoded(manifest)),'sources':fit,
                  'episodes_sha256':sha((fit_root/'FIT_EPISODES.json').read_bytes())}
    eval_manifest={'source_manifest_sha256':sha(encoded(manifest)),'language_sources':dev,
                   'files':{p.relative_to(dev_root).as_posix():sha(p.read_bytes()) for p in dev_root.rglob('*') if p.is_file()}}
    save(fit_root/'FIT_MANIFEST.json',fit_manifest);save(dev_root/'EVAL_MANIFEST.json',eval_manifest)
    save(cache/'PACKAGE_HASHES.json',{'MANIFEST.json':sha(encoded(manifest)),
           'FIT_MANIFEST.json':sha(encoded(fit_manifest)),'EVAL_MANIFEST.json':sha(encoded(eval_manifest))})
    return manifest

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--cache',type=Path,required=True);a=p.parse_args()
    m=prepare(a.cache);print(json.dumps({'fit_files':len(m['fit_sources']),'fit_bytes':m['fit_bytes'],
                                      'language_dev_files':len(m['language_dev_sources']),'language_dev_windows':m['language_DEV_windows']}))
