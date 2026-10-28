"""Constructed byte fixtures only: these never establish genuine model replay."""
import copy
import hashlib
import os

import pytest
from tac_osm.replay_intake import inspect


def fixture(tmp_path):
    def record(name, **extra):
        data=name.encode();(tmp_path/name).write_bytes(data)
        return dict(path=name, sha256=hashlib.sha256(data).hexdigest(), **extra)
    code=record('model.py')
    contract=dict(schema='tac-osm.replay-intake.v1', study_id='fixture',
                  execution_level='internal_research', protected_TEST=False,new_fits=False,
                  source_counts={'fit':1,'dev':1},checkpoint_families=['native'],seeds=[1],
                  required_assets=['protocol'],source_files={code['path']:code['sha256']})
    inventory=dict(schema='tac-osm.replay-inventory.v1',study_id='fixture',
                   publisher=dict(url='https://example.invalid',license='fixture',license_evidence='fixture'),
                   sources=[record('fit.txt',partition='fit'),record('dev.txt',partition='dev')],
                   checkpoints=[record('checkpoint.pt',family='native',seed=1)],
                   assets=[record('protocol.json',role='protocol')])
    return contract,inventory,{p.name for p in tmp_path.iterdir()}


def test_complete_is_readiness_never_replay(tmp_path):
    c,i,t=fixture(tmp_path);r=inspect(c,i,tmp_path,t)
    assert r['state']=='READY_FOR_REPLAY' and r['REPO_REPLAY']=='BLOCKED'
    assert r['model_executions']==0


def test_missing_inventory_is_blocked(tmp_path):
    c,_,_=fixture(tmp_path)
    assert inspect(c,None,tmp_path,set())['state']=='BLOCKED'


@pytest.mark.parametrize('field',['sources','checkpoints','assets'])
def test_incomplete_coverage(tmp_path,field):
    c,i,t=fixture(tmp_path);i[field].pop()
    assert inspect(c,i,tmp_path,t)['state']=='BLOCKED'


def test_changed_original_bytes_fail(tmp_path):
    c,i,t=fixture(tmp_path);(tmp_path/'checkpoint.pt').write_bytes(b'replacement')
    assert inspect(c,i,tmp_path,t)['state']=='FAIL'


def test_untracked_and_missing_bytes_block(tmp_path):
    c,i,t=fixture(tmp_path);t.remove('fit.txt');(tmp_path/'checkpoint.pt').unlink()
    r=inspect(c,i,tmp_path,t)
    assert r['state']=='BLOCKED' and len(r['missing'])==2


def test_test_partition_rejected_before_hashing(tmp_path,monkeypatch):
    c,i,t=fixture(tmp_path);i['sources'][1]['partition']='test'
    monkeypatch.setattr('tac_osm.replay_intake.sha256',lambda _:pytest.fail('payload opened'))
    with pytest.raises(ValueError,match='FIT/DEV'):inspect(c,i,tmp_path,t)


@pytest.mark.parametrize('path',['../outside','/absolute','a/../b','a\\b','./fit.txt','a//b'])
def test_escape_rejected(tmp_path,path):
    c,i,t=fixture(tmp_path);i['sources'][0]['path']=path
    with pytest.raises(ValueError):inspect(c,i,tmp_path,t)


def test_symlink_rejected(tmp_path):
    c,i,t=fixture(tmp_path);(tmp_path/'fit.txt').unlink();(tmp_path/'fit.txt').symlink_to('dev.txt')
    with pytest.raises(ValueError,match='symlink'):inspect(c,i,tmp_path,t)


def test_duplicate_and_hardlink_rejected(tmp_path):
    c,i,t=fixture(tmp_path);other=copy.deepcopy(i);other['sources'][1]['path']='fit.txt'
    with pytest.raises(ValueError,match='duplicate'):inspect(c,other,tmp_path,t)
    (tmp_path/'dev.txt').unlink();os.link(tmp_path/'fit.txt',tmp_path/'dev.txt')
    with pytest.raises(ValueError,match='aliased'):inspect(c,i,tmp_path,t)


def test_unknown_hash_and_missing_license(tmp_path):
    c,i,t=fixture(tmp_path);i['publisher']={}
    assert inspect(c,i,tmp_path,t)['state']=='BLOCKED'
    i['checkpoints'][0]['sha256']=None
    with pytest.raises(ValueError,match='sha256'):inspect(c,i,tmp_path,t)


def test_cli_missing_inventory_records_blocker(tmp_path):
    import json
    import subprocess
    import sys
    from pathlib import Path
    root=Path(__file__).resolve().parents[1]
    result=subprocess.run([sys.executable,str(root/'scripts/check_original_replay_intake.py'),
        '--contract',str(root/'research/replay-readiness/language_graph006/CONTRACT.json'),
        '--output',str(tmp_path/'receipt.json')],capture_output=True,text=True)
    assert result.returncode==2
    receipt=json.loads((tmp_path/'receipt.json').read_text())
    assert receipt['state']=='BLOCKED' and receipt['model_executions']==0


def test_cli_changed_inventory_rejected_before_read(tmp_path):
    import json
    import subprocess
    import sys
    from pathlib import Path
    root=Path(__file__).resolve().parents[1]
    subprocess.run(['git','init',str(tmp_path)],check=True,capture_output=True)
    (tmp_path/'inventory.json').write_text('not even valid json')
    subprocess.run(['git','-C',str(tmp_path),'add','inventory.json'],check=True)
    subprocess.run(['git','-C',str(tmp_path),'-c','user.name=Fixture','-c',
                    'user.email=fixture@example.invalid','commit','-m','fixture'],
                   check=True,capture_output=True)
    output=tmp_path/'receipt.json'
    result=subprocess.run([sys.executable,str(root/'scripts/check_original_replay_intake.py'),
        '--contract',str(root/'research/replay-readiness/language_graph006/CONTRACT.json'),
        '--root',str(tmp_path),'--inventory','inventory.json',
        '--inventory-sha256','0'*64,'--output',str(output)],capture_output=True,text=True)
    assert result.returncode==1
    assert 'inventory hash differs' in json.loads(output.read_text())['reason']
