#!/usr/bin/env python3
"""Whitelist exact no-update profile source, tokenizer and authenticated FIT assets."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import zipfile

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]

BOOTSTRAP='''import hashlib,json,runpy,shutil,sys,zipfile
from pathlib import Path
EXPECTED_ARCHIVE={archive_sha!r}
EXPECTED_MANIFEST={manifest_sha!r}
root=Path('/kaggle/working/lang10m-offline')
if root.exists():raise RuntimeError('refuse to overwrite reviewed bundle')
archives=[p for p in Path('/kaggle/input').rglob('bundle.zip') if hashlib.sha256(p.read_bytes()).hexdigest()==EXPECTED_ARCHIVE]
if len(archives)==1:
    root.mkdir()
    with zipfile.ZipFile(archives[0]) as z:
        for entry in z.infolist():
            if not (root/entry.filename).resolve().is_relative_to(root.resolve()):raise RuntimeError('unsafe bundle path')
        z.extractall(root)
else:
    manifests=[p for p in Path('/kaggle/input').rglob('BUNDLE_MANIFEST.json') if hashlib.sha256(p.read_bytes()).hexdigest()==EXPECTED_MANIFEST]
    if len(manifests)!=1:raise RuntimeError('expected exact reviewed auto-unpacked manifest')
    shutil.copytree(manifests[0].parent,root)
manifest=root/'BUNDLE_MANIFEST.json'
if hashlib.sha256(manifest.read_bytes()).hexdigest()!=EXPECTED_MANIFEST:raise RuntimeError('manifest changed')
for name,digest in json.loads(manifest.read_text())['files'].items():
    path=(root/name).resolve()
    if not path.is_relative_to(root.resolve()) or not path.is_file():raise RuntimeError('missing/escaped asset')
    if hashlib.sha256(path.read_bytes()).hexdigest()!=digest:raise RuntimeError('source/tokenizer/FIT asset changed')
script=root/'repo/research/language-scaling/lang10m/capacity_profile.py'
sys.path.insert(0,str(script.parent))
module=runpy.run_path(str(script))
module['run'](root/'tokens/FIT_TOKENS.npz',Path('/kaggle/working/LANG10M001_CAPACITY'),max_seconds=300)
'''


def source_files():
    files=list(HERE.glob('*.py'))+list(HERE.glob('*.json'))
    files+=list((HERE/'tokenizer').glob('*'))
    files+=[ROOT/p for p in ('research/mathematical-engineering/context026/context_transition.py',
        'research/mathematical-engineering/langlayer038/recurrence_only.py',
        'research/mathematical-engineering/lang10m001_capacity/FAIR_COMPARISON.json')]
    return sorted(set(p for p in files if p.is_file()))

def bootstrap(archive):
    archive=Path(archive)
    with zipfile.ZipFile(archive) as z:manifest_sha=hashlib.sha256(z.read('BUNDLE_MANIFEST.json')).hexdigest()
    return BOOTSTRAP.format(archive_sha=hashlib.sha256(archive.read_bytes()).hexdigest(),manifest_sha=manifest_sha)

def package(corpus,tokens,out,user,require_clean=True):
    if require_clean and subprocess.check_output(['git','status','--porcelain'],cwd=ROOT):raise ValueError('commit exact reviewed source first')
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    dataset=out/'dataset';kernel=out/'kernel';dataset.mkdir(parents=True,exist_ok=False);kernel.mkdir(parents=True,exist_ok=False)
    assets={str(Path('repo')/p.relative_to(ROOT)):p.read_bytes() for p in source_files()}
    source_manifest=json.loads((corpus/'FIT_MANIFEST.json').read_text())
    if source_manifest.get('partition')!='FIT' or any(r.get('partition')!='FIT' for r in source_manifest['sources']):
        raise ValueError('capacity inputs must be authenticated FIT only')
    allowed={'FIT_MANIFEST.json':None,'SOURCE_IDENTITIES.json':None}
    allowed.update({r['path']:r['sha256'] for r in source_manifest['sources']})
    allowed.update({r['file']:r['sha256'] for r in source_manifest.get('license_archives',[])})
    for name,digest in sorted(allowed.items()):
        p=corpus/name
        if p.is_symlink() or not p.resolve().is_relative_to(corpus.resolve()) or not p.is_file():
            raise ValueError('missing/escaped source data')
        raw=p.read_bytes()
        if digest and hashlib.sha256(raw).hexdigest()!=digest:raise ValueError('original source/license changed')
        assets['corpus/'+name]=raw
    for name in ('FIT_TOKENS.npz','TOKENS_RECEIPT.json'):assets['tokens/'+name]=(tokens/name).read_bytes()
    manifest={'study':'LANG10M001_CAPACITY','source_commit':commit,
        'files':{n:hashlib.sha256(b).hexdigest() for n,b in assets.items()}}
    assets['BUNDLE_MANIFEST.json']=(json.dumps(manifest,indent=2,sort_keys=True)+'\n').encode()
    archive=dataset/'bundle.zip'
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED) as z:
        for name,raw in sorted(assets.items()):
            info=zipfile.ZipInfo(name,date_time=(2026,10,30,0,0,0));info.compress_type=zipfile.ZIP_DEFLATED;z.writestr(info,raw)
    meta={'id':user+'/lang10m001-fit-capacity-v1','title':'LANG10M001 private FIT tokenizer capacity',
          'licenses':[{'name':'other'}]}
    (dataset/'dataset-metadata.json').write_text(json.dumps(meta,indent=2)+'\n')
    script=bootstrap(archive);(kernel/'main.py').write_text(script)
    k={'id':user+'/lang10m001-no-update-capacity-t4x2','title':'LANG10M001 no update capacity T4x2',
       'code_file':'main.py','language':'python','kernel_type':'script','is_private':'true','enable_gpu':'true',
       'enable_internet':'false','machine_shape':'NvidiaTeslaT4','dataset_sources':[meta['id']],
       'kernel_sources':[],'competition_sources':[],'model_sources':[]}
    (kernel/'kernel-metadata.json').write_text(json.dumps(k,indent=2)+'\n')
    receipt={'study':'LANG10M001_CAPACITY','source_commit':commit,'status':'PACKAGED_NOT_SUBMITTED',
       'dataset':meta['id'],'kernel':k['id'],'private':True,'internet':False,'accelerator':'NvidiaTeslaT4',
       'archive_sha256':hashlib.sha256(archive.read_bytes()).hexdigest(),'files':len(assets),
       'bootstrap_sha256':hashlib.sha256(script.encode()).hexdigest(),'archive_bytes':archive.stat().st_size,
       'optimizer_steps':0,'neural_training':False,'max_aggregate_GPU_minutes':12,'global_deadline_seconds':300}
    (out/'PACKAGE_RECEIPT.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt));return receipt

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--corpus',type=Path,required=True)
    p.add_argument('--tokens',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--user',required=True);a=p.parse_args();package(a.corpus,a.tokens,a.out,a.user)
