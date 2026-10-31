"""Training source/batch/gate falsifiers; never construct or run an optimizer."""
import copy
import hashlib
import json
from pathlib import Path
import pytest
import torch
from tac_osm.adaptive_token_training import load_corpus, batch_at, sample_order_digest, learning_rate
from scripts.run_sense_tok_training import main


def corpus(tmp_path):
    rows=[]
    for split, raw in [('FIT',b'abcdefghi'),('DEV',b'123456789'),('TEST',b'SEALED')]:
        if split!='TEST':
            (tmp_path/split).write_bytes(raw)
        rows.append({'split':split,'source_id':split,'group_id':split,'path':split,
                     'sha256':hashlib.sha256(raw).hexdigest(),'size_bytes':len(raw)})
    return {'files':rows}


def test_verified_windows_never_read_or_stat_sealed_test(tmp_path, monkeypatch):
    data=corpus(tmp_path)
    original=Path.read_bytes
    def guarded(path):
        assert path.name!='TEST'
        return original(path)
    monkeypatch.setattr(Path,'read_bytes',guarded)
    result=load_corpus(tmp_path,data,4)
    assert [v[2] for v in result['FIT']]==[b'abcd',b'efgh']
    assert not (tmp_path/'TEST').exists()


@pytest.mark.parametrize('damage',['group','hash','escape','duplicate'])
def test_corpus_contamination_and_changed_bytes_are_rejected(tmp_path,damage):
    data=corpus(tmp_path)
    if damage=='group': data['files'][1]['group_id']='FIT'
    if damage=='hash': (tmp_path/'FIT').write_bytes(b'XXXXXXXXX')
    if damage=='escape': data['files'][0]['path']='../outside'
    if damage=='duplicate': data['files'][1]['sha256']=data['files'][0]['sha256']
    with pytest.raises(ValueError): load_corpus(tmp_path,data,4)


def test_batches_replay_target_shift_and_identical_order_digest(tmp_path):
    windows=load_corpus(tmp_path,corpus(tmp_path),4)['FIT']
    a=batch_at(windows,seed=1,step=2,batch_size=3,device='cpu')
    b=batch_at(windows,seed=1,step=2,batch_size=3,device='cpu')
    assert all(torch.equal(x,y) for x,y in zip(a[:3],b[:3]))
    assert torch.equal(a[0][:,1:],a[1][:,:-1])
    assert (a[0][:,0]==256).all()
    hasher=hashlib.sha256()
    for step in range(3):
        receipt=batch_at(windows,seed=1,step=step,batch_size=3,device='cpu')[3]
        hasher.update((json.dumps(receipt,separators=(',',':'))+'\n').encode())
    assert sample_order_digest(windows,seed=1,steps=3,batch_size=3)==hasher.hexdigest()


def test_schedule_uses_registered_endpoints():
    c={'steps_per_arm_seed':6,'optimizer':{'learning_rate':.1},'learning_rate_schedule':{'warmup_steps':2,'final_learning_rate':.01}}
    assert learning_rate(0,c)==pytest.approx(.05)
    assert learning_rate(1,c)==pytest.approx(.1)
    assert learning_rate(5,c)==pytest.approx(.01)
    with pytest.raises(ValueError): learning_rate(6,c)


def test_current_no_fit_registration_cannot_reach_optimizer(tmp_path,monkeypatch,capsys):
    root=Path(__file__).resolve().parents[1]
    def forbidden(*args,**kwargs): raise AssertionError('optimizer constructed')
    monkeypatch.setattr(torch.optim,'AdamW',forbidden)
    manifest=root/'research/mathematical-engineering/sense_tok_006/FAIR_COMPARISON.json'
    argv=['--manifest',str(manifest),'--evidence',str(tmp_path/'absent'), '--bundle',str(tmp_path/'absent2'),
          '--source-root',str(tmp_path),'--output',str(tmp_path.parent/(tmp_path.name+'-result')),'--execute']
    assert main(argv)==2
    assert not (tmp_path.parent/(tmp_path.name+'-result')).exists()
    assert 'no-fit preparation cannot launch' in capsys.readouterr().out
