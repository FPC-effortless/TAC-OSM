#!/usr/bin/env python3
"""Credential-free exact-source packaging with ZIP and Kaggle-unpacked support."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import zipfile

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]


def files():
    own=list(HERE.glob('*.py'))+list(HERE.glob('*.json'))
    needed=[
        'research/commercial-value/codevalue003/scoring.py',
        'research/commercial-value/codevalue002/train.py',
        'research/commercial-value/codevalue002/REGISTER.json',
        'research/mathematical-engineering/langscale070/scaled_recurrence.py',
        'research/mathematical-engineering/context026/context_transition.py',
        'research/mathematical-engineering/langlayer038/recurrence_only.py',
        'research/mathematical-engineering/langhead094/model094.py',
        'research/mathematical-engineering/langhead093/model093.py',
        'research/mathematical-engineering/langmem086/compact_tac.py',
        'research/mathematical-engineering/codevalue004/FAIR_COMPARISON.json',
        'research/kaggle_workloads/codevalue004.py',
        'scripts/validate_research_parity.py','src/tac_osm/research_execution.py','src/tac_osm/__init__.py']
    return sorted(set(own+[ROOT/p for p in needed]))

def bootstrap(archive):
    spec=importlib.util.spec_from_file_location('codevalue003_package_bootstrap',HERE.parent/'codevalue003/package_run.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    text=module.bootstrap(archive,max_seconds=7020)
    return text.replace('CODEVALUE003','CODEVALUE004').replace(
        "repo/research/commercial-value/codevalue003/coordinator.py",
        "repo/research/commercial-value/codevalue004/run004.py")

def package(data,out,user,require_clean=True):
    if require_clean and subprocess.check_output(['git','status','--porcelain'],cwd=ROOT):raise ValueError('commit exact reviewed source first')
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    dataset=out/'dataset';kernel=out/'kernel'
    dataset.mkdir(parents=True,exist_ok=False);kernel.mkdir(parents=True,exist_ok=False)
    assets={str(Path('repo')/p.relative_to(ROOT)):p.read_bytes() for p in files()}
    for p in sorted(data.iterdir()):
        if p.is_symlink() or not p.is_file() or p.suffix not in ('.bin','.json','.txt'):raise ValueError('unexpected data asset')
        assets['data/'+p.name]=p.read_bytes()
    manifest={'study':'CODEVALUE004','git_commit':commit,
        'files':{n:hashlib.sha256(b).hexdigest() for n,b in assets.items()}}
    assets['BUNDLE_MANIFEST.json']=(json.dumps(manifest,indent=2,sort_keys=True)+'\n').encode()
    archive=dataset/'bundle.zip'
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED) as z:
        for name,raw in sorted(assets.items()):
            info=zipfile.ZipInfo(name,date_time=(2026,10,30,0,0,0));info.compress_type=zipfile.ZIP_DEFLATED
            z.writestr(info,raw)
    meta={'id':user+'/codevalue004-offline-generalization-v1','title':'CODEVALUE004 private offline generalization',
          'licenses':[{'name':'other'}]}
    (dataset/'dataset-metadata.json').write_text(json.dumps(meta,indent=2)+'\n')
    title='CODEVALUE004 corpus schedule T4x2'
    kernel_meta={'id':user+'/codevalue004-corpus-schedule-t4x2','title':title,
       'code_file':'main.py','language':'python','kernel_type':'script','is_private':'true',
       'enable_gpu':'true','enable_internet':'false','machine_shape':'NvidiaTeslaT4',
       'dataset_sources':[meta['id']],'competition_sources':[],'kernel_sources':[],'model_sources':[]}
    script=bootstrap(archive);(kernel/'main.py').write_text(script)
    (kernel/'kernel-metadata.json').write_text(json.dumps(kernel_meta,indent=2)+'\n')
    receipt={'study':'CODEVALUE004','source_commit':commit,'dataset':meta['id'],'kernel':kernel_meta['id'],
       'status':'PACKAGED_NOT_SUBMITTED','private':True,'internet':False,'accelerator':'NvidiaTeslaT4',
       'physical_GPUs_required':2,'archive_sha256':hashlib.sha256(archive.read_bytes()).hexdigest(),
       'bootstrap_sha256':hashlib.sha256(script.encode()).hexdigest(),'archive_bytes':archive.stat().st_size,
       'files':len(assets),'max_allocated_GPU_minutes':240,'runtime_deadline_seconds':7020}
    (out/'PACKAGE_RECEIPT.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt));return receipt

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--user',required=True)
    a=p.parse_args();package(a.data,a.out,a.user)
