#!/usr/bin/env python3
"""Package the explicitly bounded six-arm tranche, not the full tuning grid."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import textwrap
ROOT=Path(__file__).resolve().parents[1]


def package(output,source,username):
    if not re.fullmatch('[0-9a-f]{40}',source):raise ValueError('full source SHA required')
    if not re.fullmatch('[A-Za-z0-9_-]{1,50}',username):raise ValueError('invalid username')
    subprocess.run(['git','cat-file','-e',source+':scripts/run_mixed_tranche_dev031.py'],cwd=ROOT,check=True,capture_output=True)
    output=Path(output)
    if output.exists():raise ValueError('fresh package required')
    output.mkdir(parents=True)
    code=textwrap.dedent(f'''\
        import json, pathlib, subprocess, sys, time
        SOURCE={source!r}
        root=pathlib.Path('/tmp/tac-osm-dev031-source')
        if not root.exists():
            root.mkdir()
            subprocess.run(['git','init',str(root)],check=True,capture_output=True)
            subprocess.run(['git','-C',str(root),'remote','add','origin','https://github.com/FPC-effortless/TAC-OSM.git'],check=True)
            subprocess.run(['git','-C',str(root),'fetch','--depth=1','origin',SOURCE],check=True)
            subprocess.run(['git','-C',str(root),'checkout','--detach','FETCH_HEAD'],check=True)
        if subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD'],text=True).strip()!=SOURCE:
            raise RuntimeError('source mismatch; use a fresh session')
        output=pathlib.Path('/kaggle/working')/('dev031-tranche-'+str(time.time_ns()))
        log_path=pathlib.Path(str(output)+'.log')
        command=[sys.executable,str(root/'scripts/run_mixed_tranche_dev031.py'),'--output',str(output)]
        print('Starting bounded 30-minute execution window; all six arms retained; no full grid or test access.',flush=True)
        with log_path.open('w') as log:
            process=subprocess.Popen(command,cwd=root,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
            for line in process.stdout:
                print(line,end='',flush=True);log.write(line);log.flush()
            status=process.wait()
        print('Retained checkpoint/optimizer/receipt/cost evidence:',output,flush=True)
        if status:
            raise RuntimeError('Tranche or strict replay incomplete; retain every partial file and failure log.')
        print(json.dumps(json.loads((output/'execution-manifest.json').read_text())['status'],indent=2))
        print('Execution and replay complete. Independent full-matrix audit, recipe selection and test remain pending.')
    ''')
    notebook=dict(nbformat=4,nbformat_minor=5,metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python'},'accelerator':'GPU'},cells=[
        dict(cell_type='markdown',id='registration',metadata={},source=['# DEV-031 bounded mixed-device tranche\n','All six arms, seed311/LR0.002/H512/B2, 216 and648 checkpoints. GRU on CUDA, other arms on CPU.\n','This is six of72 tuning slots, not recipe selection or scientific test evaluation. All outcomes/failures retained.\n']),
        dict(cell_type='code',id='execution',metadata={},execution_count=None,outputs=[],source=code.splitlines(keepends=True))])
    name='dev031_mixed_tranche.ipynb';(output/name).write_text(json.dumps(notebook,indent=2)+'\n')
    metadata=dict(id=username+'/tac-osm-dev031-mixed-device-tranche',title='TAC OSM DEV031 Mixed Device Tranche',code_file=name,language='python',kernel_type='notebook',is_private=True,enable_gpu=True,enable_internet=True,machine_shape='NvidiaTeslaT4',dataset_sources=[],competition_sources=[],kernel_sources=[],model_sources=[])
    (output/'kernel-metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    manifest=dict(source=source,experiment='DEV-031',private=True,full_grid_launched=False,test_accessed=False,credentials_embedded=False,files={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in output.iterdir()})
    (output/'package-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');print(json.dumps(manifest))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--source',required=True);p.add_argument('--username',required=True)
    a=p.parse_args();package(a.output,a.source,a.username)
