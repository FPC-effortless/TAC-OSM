#!/usr/bin/env python3
"""One existing DEV-029 grid slot; retain failures and replay saved checkpoints."""
import argparse
import json
from pathlib import Path
import re
import textwrap


def package(output,username):
    if not re.fullmatch('[A-Za-z0-9_-]{1,50}',username):raise ValueError('invalid username')
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    code=textwrap.dedent('''\
        import json,pathlib,subprocess,sys,time
        source='4a8728ef3f08e01d9c542374cf1e722bf8701240'
        root=pathlib.Path('/tmp/dev029-frozen');root.mkdir()
        subprocess.run(['git','init',str(root)],check=True,capture_output=True)
        subprocess.run(['git','-C',str(root),'remote','add','origin','https://github.com/FPC-effortless/TAC-OSM.git'],check=True)
        subprocess.run(['git','-C',str(root),'fetch','--depth=1','origin',source],check=True)
        subprocess.run(['git','-C',str(root),'checkout','--detach','FETCH_HEAD'],check=True)
        assert subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD'],text=True).strip()==source
        output=pathlib.Path('/kaggle/working/dev032-registered-gru');output.mkdir()
        command=[sys.executable,str(root/'scripts/run_optimization_study_dev029.py'),
            '--arm','gru768','--seed','311','--learning-rate','0.0005','--curriculum','fixed512',
            '--until','2160','--device','cuda:0','--output',str(output/'trial')]
        start=time.monotonic();status=None;timed_out=False
        with (output/'training.log').open('w') as log:
            try:status=subprocess.run(command,cwd=root,stdout=log,stderr=subprocess.STDOUT,timeout=420).returncode
            except subprocess.TimeoutExpired:timed_out=True
        print((output/'training.log').read_text(),flush=True)
        replays=[]
        for budget in (216,648,2160):
            report=output/'trial'/f'{budget}.json'
            if not report.exists():continue
            replay_path=output/f'{budget}-replay.json'
            result=subprocess.run([sys.executable,str(root/'scripts/replay_tuning_checkpoint_dev031.py'),
                '--report',str(report),'--output',str(replay_path)],cwd=root,capture_output=True,text=True,timeout=120)
            (output/f'{budget}-replay.log').write_text(result.stdout+result.stderr)
            replays.append(dict(budget=budget,exit_code=result.returncode,replay_passed=result.returncode==0))
        manifest=dict(source=source,arm='gru768',seed=311,learning_rate=.0005,curriculum='fixed512',
            training_exit_code=status,timed_out=timed_out,wall_seconds=time.monotonic()-start,
            replays=replays,full_matrix_complete=False,recipe_selected=False,test_accessed=False,
            original_high_learning_rate_trial_remains_failed=True)
        (output/'execution-manifest.json').write_text(json.dumps(manifest,indent=2)+'\\n')
        print(json.dumps(manifest),flush=True)
        if status!=0 or len(replays)!=3 or not all(r['replay_passed'] for r in replays):
            raise RuntimeError('Registered slot incomplete; all failures and partial evidence preserved')
    ''')
    notebook=dict(nbformat=4,nbformat_minor=5,metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'accelerator':'GPU'},cells=[dict(cell_type='code',id='registered-grid-slot',metadata={},execution_count=None,outputs=[],source=code.splitlines(keepends=True))])
    name='dev032_registered_gru.ipynb';(output/name).write_text(json.dumps(notebook,indent=2)+'\n')
    metadata=dict(id=username+'/tac-osm-dev032-registered-gru-slot',title='TAC OSM DEV032 Registered GRU Slot',code_file=name,language='python',kernel_type='notebook',is_private=True,enable_gpu=True,enable_internet=True,machine_shape='NvidiaTeslaT4',dataset_sources=[],competition_sources=[],kernel_sources=[],model_sources=[])
    (output/'kernel-metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print(json.dumps(dict(private=True,source='4a8728ef3f08e01d9c542374cf1e722bf8701240',credentials_embedded=False,test_accessed=False)))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--username',required=True)
    a=p.parse_args();package(a.output,a.username)
