#!/usr/bin/env python3
"""Build a private Kaggle notebook package pinned to a real git commit.

No credentials are embedded. Without --username the notebook can be imported
manually, but API submission metadata remains an explicit unsendable template.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import textwrap
import zipfile

ROOT=Path(__file__).resolve().parents[1]


def package(output,source,username=None):
    if not re.fullmatch(r'[0-9a-f]{40}',source):raise ValueError('source must be a full commit SHA')
    if username is not None and not re.fullmatch(r'[A-Za-z0-9_-]{1,50}',username):raise ValueError('invalid Kaggle username')
    subprocess.run(['git','cat-file','-e',source+':scripts/run_compute_preflight_dev030.py'],cwd=ROOT,check=True,capture_output=True)
    output=Path(output)
    if output.exists():raise ValueError('preserve existing package; choose a fresh destination')
    output.mkdir(parents=True)
    bootstrap=textwrap.dedent(f'''\
        import json, os, pathlib, subprocess, sys, time
        import torch
        SOURCE = {source!r}
        ROOT = pathlib.Path('/kaggle/working/tac-osm-source')
        EVIDENCE = pathlib.Path('/kaggle/working/dev030-evidence')
        EVIDENCE.mkdir(parents=True, exist_ok=True)
        if not torch.cuda.is_available():
            raise RuntimeError('Enable a GPU in Kaggle Notebook settings. No CPU fallback is allowed for this GPU preflight.')
        if not ROOT.exists():
            ROOT.mkdir()
            subprocess.run(['git', 'init', str(ROOT)], check=True, capture_output=True)
            subprocess.run(['git', '-C', str(ROOT), 'remote', 'add', 'origin', 'https://github.com/FPC-effortless/TAC-OSM.git'], check=True)
            subprocess.run(['git', '-C', str(ROOT), 'fetch', '--depth=1', 'origin', SOURCE], check=True)
            subprocess.run(['git', '-C', str(ROOT), 'checkout', '--detach', 'FETCH_HEAD'], check=True)
        actual = subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip()
        if actual != SOURCE:
            raise RuntimeError('Checkout differs from the frozen source; use a fresh session.')
        run_id = str(time.time_ns())
        destination = EVIDENCE / ('preflight-' + run_id + '.json')
        command = [sys.executable, str(ROOT / 'scripts/run_compute_preflight_dev030.py'), '--output', str(destination)]
        (EVIDENCE / ('submission-' + run_id + '.json')).write_text(json.dumps(dict(source=SOURCE, command=command, torch_version=str(torch.__version__), cuda_version=torch.version.cuda, scientific_training_run=False, test_accessed=False), indent=2))
        with (EVIDENCE / ('execution-' + run_id + '.log')).open('w') as log:
            process = subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            for line in process.stdout:
                print(line, end='', flush=True)
                log.write(line); log.flush()
            status = process.wait()
        print('Evidence directory:', EVIDENCE)
        if status:
            raise RuntimeError('Preflight failed; retain the partial JSON and execution log. Full training was not launched.')
        result = json.loads(destination.read_text())
        print(json.dumps(result['recommendation'], indent=2))
        print('Preflight only: no scientific training matrix or test set was opened.')
    ''')
    notebook=dict(nbformat=4,nbformat_minor=5,
        metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},
                  'language_info':{'name':'python'},'accelerator':'GPU'},
        cells=[dict(cell_type='markdown',id='introduction',metadata={},source=[
            '# TAC-OSM DEV-030: GPU preflight\n',
            'Private engineering preflight for the registered DEV-029 workload. Enable GPU and Internet in Kaggle settings.\n',
            'No secrets are needed inside this notebook. Source is pinned; no tests or full tuning grid are launched.\n',
            'Outputs stay under `/kaggle/working/dev030-evidence`. Save a notebook version with outputs to preserve them.\n',
            'FP32, batch two, unchanged task/optimizer; table state stays CPU and its learned head uses CUDA.\n']),
            dict(cell_type='code',id='preflight',metadata={},execution_count=None,outputs=[],source=bootstrap.splitlines(keepends=True))])
    notebook_path=output/'tac_osm_dev030_preflight.ipynb'
    notebook_path.write_text(json.dumps(notebook,indent=2)+'\n')
    owner=username or 'YOUR_KAGGLE_USERNAME'
    metadata=dict(id=owner+'/tac-osm-dev030-gpu-preflight',title='TAC OSM DEV030 GPU Preflight',
        code_file=notebook_path.name,language='python',kernel_type='notebook',is_private=True,
        enable_gpu=True,enable_internet=True,dataset_sources=[],competition_sources=[],kernel_sources=[],model_sources=[])
    # Title/slug matching is required by Kaggle: tac-osm-dev030-gpu-preflight.
    (output/'kernel-metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    manifest=dict(experiment='DEV-030',source=source,private=True,api_metadata_ready=username is not None,
        source_repository='FPC-effortless/TAC-OSM',
        files={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in output.iterdir() if p.is_file()},
        automatic_full_training=False,scientific_test_access=False,credentials_embedded=False)
    (output/'package-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    archive=output.with_suffix('.zip')
    if archive.exists():raise ValueError('archive already exists')
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for path in output.iterdir():z.write(path,path.name)
    print(json.dumps(dict(package=str(output),archive=str(archive),source=source,
        api_metadata_ready=username is not None,archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest())))
    return manifest


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--source',required=True)
    p.add_argument('--username');a=p.parse_args();package(a.output,a.source,a.username)
