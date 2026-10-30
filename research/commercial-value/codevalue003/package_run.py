#!/usr/bin/env python3
"""Build a credential-free offline dataset and exact-hash Kaggle bootstrap."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import zipfile

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]

BOOTSTRAP='''"""CODEVALUE003 private offline exact-source dual-T4 run."""
import hashlib
from pathlib import Path
import runpy
import sys
import zipfile

EXPECTED_ARCHIVE_SHA256 = {archive_sha!r}
candidates=list(Path('/kaggle/input').rglob('bundle.zip'))
matching=[p for p in candidates if hashlib.sha256(p.read_bytes()).hexdigest()==EXPECTED_ARCHIVE_SHA256]
if len(matching)!=1:raise RuntimeError('expected one exact reviewed offline dataset archive')
root=Path('/kaggle/working/offline-bundle')
if root.exists():raise RuntimeError('refuse to overwrite extracted original bundle')
root.mkdir()
with zipfile.ZipFile(matching[0]) as z:
    for info in z.infolist():
        if not (root/info.filename).resolve().is_relative_to(root.resolve()):
            raise RuntimeError('unsafe archive path')
    z.extractall(root)
script=root/'repo/research/kaggle_workloads/codevalue003.py'
sys.argv=[str(script),'--bundle-root',str(root),'--output','/kaggle/working/CODEVALUE003']
runpy.run_path(str(script),run_name='__main__')
'''


def files():
    paths=list(HERE.glob('*.py'))+list(HERE.glob('*.json'))
    paths.extend(ROOT/p for p in (
        'research/commercial-value/codevalue002/train.py',
        'research/commercial-value/codevalue002/REGISTER.json',
        'research/mathematical-engineering/langscale070/scaled_recurrence.py',
        'research/mathematical-engineering/context026/context_transition.py',
        'research/mathematical-engineering/langlayer038/recurrence_only.py',
        'research/mathematical-engineering/langhead094/model094.py',
        'research/mathematical-engineering/langhead093/model093.py',
        'research/mathematical-engineering/langmem086/compact_tac.py',
        'research/mathematical-engineering/codevalue003/FAIR_COMPARISON.json',
        'research/kaggle_workloads/codevalue003.py',
        'scripts/validate_research_parity.py',
        'src/tac_osm/research_execution.py','src/tac_osm/__init__.py'))
    return sorted(set(paths))

def package(data,out,user,require_clean=True):
    if require_clean and subprocess.check_output(['git','status','--porcelain'],cwd=ROOT):
        raise ValueError('commit and push exact reviewed source before packaging')
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    dataset=out/'dataset';kernel=out/'kernel'
    dataset.mkdir(parents=True,exist_ok=False);kernel.mkdir(parents=True,exist_ok=False)
    items={str(Path('repo')/p.relative_to(ROOT)):p.read_bytes() for p in files()}
    for p in sorted(data.iterdir()):
        if not p.is_file() or p.is_symlink():raise ValueError('unexpected source asset')
        if p.suffix not in ('.bin','.json','.txt'):raise ValueError('unreviewed source type')
        items[str(Path('data')/p.name)]=p.read_bytes()
    manifest={'study':'CODEVALUE003','git_commit':commit,
              'files':{n:hashlib.sha256(b).hexdigest() for n,b in items.items()}}
    items['BUNDLE_MANIFEST.json']=(json.dumps(manifest,sort_keys=True,indent=2)+'\n').encode()
    archive=dataset/'bundle.zip'
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED) as z:
        for name,raw in sorted(items.items()):
            info=zipfile.ZipInfo(name,date_time=(2026,10,30,0,0,0))
            info.compress_type=zipfile.ZIP_DEFLATED;z.writestr(info,raw)
    archive_sha=hashlib.sha256(archive.read_bytes()).hexdigest()
    dataset_meta={'id':user+'/codevalue003-offline-code-v1',
                  'title':'CODEVALUE003 private offline code and runner',
                  'licenses':[{'name':'other'}]}
    (dataset/'dataset-metadata.json').write_text(json.dumps(dataset_meta,indent=2)+'\n')
    (kernel/'main.py').write_text(BOOTSTRAP.format(archive_sha=archive_sha))
    metadata={'id':user+'/tacosm-codevalue003-long-t4x2-v1','title':'CODEVALUE003 longer500K code language T4x2',
              'code_file':'main.py','language':'python','kernel_type':'script',
              'is_private':'true','enable_gpu':'true','enable_internet':'false',
              'machine_shape':'NvidiaTeslaT4','dataset_sources':[dataset_meta['id']],
              'competition_sources':[],'kernel_sources':[],'model_sources':[]}
    (kernel/'kernel-metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    receipt={'study':'CODEVALUE003','source_commit':commit,'status':'PACKAGED_NOT_SUBMITTED',
             'bundle_sha256':archive_sha,'files':len(items),'compressed_bytes':archive.stat().st_size,
             'dataset':dataset_meta['id'],'kernel':metadata['id'],
             'private':True,'internet':False,'accelerator':'NvidiaTeslaT4','required_physical_GPUs':2}
    (out/'PACKAGE_RECEIPT.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt))
    return receipt

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--user',required=True)
    a=p.parse_args();package(a.data,a.out,a.user)
