import hashlib
import importlib.util
import io
from pathlib import Path
import zipfile
import pytest

path=Path(__file__).resolve().parents[1]/'.github/scripts/download-langmath101-artifacts.py'
spec=importlib.util.spec_from_file_location('lm101_download_test',path)
D=importlib.util.module_from_spec(spec);spec.loader.exec_module(D)

def test_complete_pagination_beyond_first100(monkeypatch):
    rows=[{'id':i,'name':'fit-'+str(i)} for i in range(138)]+[{'id':139,'name':'other'}]
    calls=[]
    def api(repo,suffix):
        calls.append(suffix);page=int(suffix.rsplit('=',1)[-1])
        return {'total_count':139,'artifacts':rows[(page-1)*100:page*100]}
    monkeypatch.setattr(D,'api',api)
    actual=D.records('owner/repo',1,'fit-',138)
    assert len(actual)==138 and actual[-1]['id']==137
    assert len(calls)==2 and 'page=2' in calls[-1]

def test_incomplete_registry_fails_before_download():
    with pytest.raises(ValueError,match='exactly69'):
        D.records('owner/repo',1,'',69,{'run_id':1,'archives':[]})

def archive(name):
    stream=io.BytesIO()
    with zipfile.ZipFile(stream,'w') as z:z.writestr(name,b'original bytes')
    return stream.getvalue()

def test_verify_original_zip_and_checkpoint_bytes(tmp_path):
    raw=archive('INITIAL_MODEL.pt')
    row={'id':1,'name':'original','digest':'sha256:'+hashlib.sha256(raw).hexdigest(),
         'initial_checkpoint_sha256':hashlib.sha256(b'original bytes').hexdigest()}
    D.extract_verified(raw,row,tmp_path)
    assert (tmp_path/'original/INITIAL_MODEL.pt').read_bytes()==b'original bytes'
    with pytest.raises(ValueError,match='ZIP digest differs'):
        D.extract_verified(raw+b'changed',row,tmp_path/'bad')

def test_traversal_fails_even_with_matching_zip_digest(tmp_path):
    raw=archive('../escape')
    row={'id':1,'name':'original','digest':'sha256:'+hashlib.sha256(raw).hexdigest()}
    with pytest.raises(ValueError,match='Unsafe archive member'):
        D.extract_verified(raw,row,tmp_path)
    assert not (tmp_path/'escape').exists()
