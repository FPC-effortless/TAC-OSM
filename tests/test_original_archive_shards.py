"""Constructed byte round trips only, never original scientific replay."""
import hashlib
import json
from pathlib import Path
import pytest
from tac_osm import original_archive_shards as a


def make(tmp_path,monkeypatch):
    monkeypatch.setattr(a,'CHUNK_BYTES',8)
    raw=tmp_path/'input.zip';raw.write_bytes(b'0123456789abcdefghijkl')
    digest=hashlib.sha256(raw.read_bytes()).hexdigest()
    folder=tmp_path/'shards'
    a.split_original(raw,folder,expected_sha=digest)
    return raw,digest,folder


def test_exact_roundtrip(tmp_path,monkeypatch):
    raw,digest,folder=make(tmp_path,monkeypatch)
    output=tmp_path/'restored.zip';r=a.restore_original(folder,output,expected_sha=digest)
    assert output.read_bytes()==raw.read_bytes() and r['shards']==3
    assert r['status']=='ORIGINAL_ARCHIVE_BYTES_RESTORED_NOT_SCORE_REPLAY'


def test_wrong_original_produces_no_output(tmp_path):
    raw=tmp_path/'wrong.zip';raw.write_bytes(b'not original')
    with pytest.raises(ValueError,match='size/hash'):a.split_original(raw,tmp_path/'dest')
    assert not (tmp_path/'dest').exists()


@pytest.mark.parametrize('change',['reorder','traverse','wrong_size','wrong_parent','coverage'])
def test_manifest_poison(tmp_path,monkeypatch,change):
    _,digest,folder=make(tmp_path,monkeypatch)
    path=folder/'MANIFEST.json';m=json.loads(path.read_text())
    if change=='reorder':m['chunks'].reverse()
    if change=='traverse':m['chunks'][0]['path']='../input.zip'
    if change=='wrong_size':m['chunks'][0]['bytes']=True
    if change=='wrong_parent':m['archive_sha256']='0'*64
    if change=='coverage':m['chunks'].pop()
    path.write_text(json.dumps(m));out=tmp_path/'restored.zip'
    with pytest.raises(ValueError):a.restore_original(folder,out,expected_sha=digest)
    assert not out.exists()


def test_corrupt_and_symlink_shards(tmp_path,monkeypatch):
    _,digest,folder=make(tmp_path,monkeypatch);out=tmp_path/'restore.zip'
    shard=folder/'archive-0000.part';shard.write_bytes(b'XXXXXXXX')
    with pytest.raises(ValueError,match='hash differs'):a.restore_original(folder,out,expected_sha=digest)
    shard.unlink();shard.symlink_to(tmp_path/'input.zip')
    with pytest.raises(ValueError):a.restore_original(folder,out,expected_sha=digest)
    assert not out.exists()


def test_full_parent_hash_rechecked_not_only_shards(tmp_path,monkeypatch):
    _,digest,folder=make(tmp_path,monkeypatch)
    shard=folder/'archive-0000.part';shard.write_bytes(b'XXXXXXXX')
    p=folder/'MANIFEST.json';m=json.loads(p.read_text());m['chunks'][0]['sha256']=a.file_sha(shard);p.write_text(json.dumps(m))
    with pytest.raises(ValueError,match='restored original archive hash'):a.restore_original(folder,tmp_path/'out',expected_sha=digest)


def test_existing_and_inplace_outputs_refused(tmp_path,monkeypatch):
    raw,digest,folder=make(tmp_path,monkeypatch)
    with pytest.raises(ValueError,match='already exists'):a.split_original(raw,folder,expected_sha=digest)
    with pytest.raises(ValueError,match='outside'):a.restore_original(folder,folder/'restore.zip',expected_sha=digest)
    with pytest.raises(ValueError):a.restore_original(folder,raw,expected_sha=digest)
    assert raw.read_bytes()==b'0123456789abcdefghijkl'
