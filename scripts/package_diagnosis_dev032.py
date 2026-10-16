#!/usr/bin/env python3
"""Private frozen-failure notebook. Authentication belongs outside this package."""
import argparse
import json
from pathlib import Path
import re
import textwrap


def package(output,source,username):
    if not re.fullmatch('[0-9a-f]{40}',source):raise ValueError('full source SHA required')
    if not re.fullmatch('[A-Za-z0-9_-]{1,50}',username):raise ValueError('invalid username')
    output=Path(output);output.mkdir(exist_ok=False,parents=True)
    code=textwrap.dedent(f'''\
        import json,pathlib,subprocess,sys
        tools_source={source!r}
        frozen_source='4a8728ef3f08e01d9c542374cf1e722bf8701240'
        roots=[]
        for name,source in [('dev032-tools',tools_source),('dev031-frozen',frozen_source)]:
            root=pathlib.Path('/tmp')/name
            subprocess.run(['git','init',str(root)],check=True,capture_output=True)
            subprocess.run(['git','-C',str(root),'remote','add','origin','https://github.com/FPC-effortless/TAC-OSM.git'],check=True)
            subprocess.run(['git','-C',str(root),'fetch','--depth=1','origin',source],check=True)
            subprocess.run(['git','-C',str(root),'checkout','--detach','FETCH_HEAD'],check=True)
            assert subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD'],text=True).strip()==source
            roots.append(root)
        candidates=list(pathlib.Path('/kaggle/input').rglob('216.json'))
        assert len(candidates)==1,'Exactly one original private GRU216 report required'
        output=pathlib.Path('/kaggle/working/dev032-gradient-diagnosis')
        command=[sys.executable,str(roots[0]/'scripts/diagnose_frozen_gru_dev032.py'),'--checkout',str(roots[1]),'--report',str(candidates[0]),'--output',str(output)]
        # The original RuntimeError must remain a failure; successful diagnosis
        # means its exception was captured, never that training was repaired.
        result=subprocess.run(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,timeout=1200)
        (pathlib.Path('/kaggle/working')/'dev032-diagnosis.log').write_text(result.stdout)
        print(result.stdout,flush=True)
        status=json.loads((output/'diagnostic-status.json').read_text())
        assert result.returncode!=0 and status['failure_captured'],'Frozen failure not captured; no reproduction claim'
        health=json.loads((output/'gradient-health.json').read_text())
        print(json.dumps(dict(diagnosis_completed=True,scientific_trial_remains_failed=True,
            classification=health['classification'],attempted_update=health['attempted_update'])),flush=True)
        subprocess.run([sys.executable,str(roots[0]/'scripts/compare_failure_backends_dev032.py'),
            '--checkout',str(roots[1]),'--evidence',str(output),
            '--output','/kaggle/working/dev032-backend-comparison'],check=True,timeout=600)
    ''')
    notebook=dict(nbformat=4,nbformat_minor=5,metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'accelerator':'GPU'},cells=[dict(cell_type='code',id='frozen-diagnosis',metadata={},execution_count=None,outputs=[],source=code.splitlines(keepends=True))])
    name='dev032_diagnosis.ipynb';(output/name).write_text(json.dumps(notebook,indent=2)+'\n')
    metadata=dict(id=username+'/tac-osm-dev032-frozen-gru-diagnosis',title='TAC OSM DEV032 Frozen GRU Diagnosis',code_file=name,language='python',kernel_type='notebook',is_private=True,enable_gpu=True,enable_internet=True,machine_shape='NvidiaTeslaT4',dataset_sources=[username+'/tac-osm-dev032-frozen-gru216'],competition_sources=[],kernel_sources=[],model_sources=[])
    (output/'kernel-metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print(json.dumps(dict(source=source,private=True,credentials_embedded=False,test_accessed=False)))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--source',required=True);p.add_argument('--username',required=True)
    a=p.parse_args();package(a.output,a.source,a.username)
